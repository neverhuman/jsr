//! Witness & integration tests for `H48d` non-mutating speculative onset peek.

mod common;

use std::sync::Arc;

use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::peek::{
    reconcile_preview_words, speculative_preview, PeekConfig, PeekEvent, PeekSession, PeekStats,
};
use ultra_engine::{Engine, Isa, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn wave(n: usize, s: f32, p: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * p).sin() * s).collect()
}
fn lin(n: usize, k: usize, s: f32, p: f32, b: Option<&[f32]>) -> Linear {
    Linear::from_fp32(&wave(n * k, s, p), n, k, b)
}
/// Synthetic model: emits `<blk>` on zero-PCM silence (even after prior tokens) and word tokens on tones.
fn tiny_model() -> Arc<Model> {
    let mut fb = vec![0u8; 257 * 80 * 4];
    for m in 0..80 {
        for bin in 1..40 {
            let o = (bin * 80 + m) * 4;
            fb[o..o + 4].copy_from_slice(&0.25f32.to_le_bytes());
        }
    }
    let cmvn = GlobalCmvn::new(&[-13.815_511; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_le_bytes(&fb, 1e-6, cmvn).expect("mel"));
    let toks = "<blk> 0\n<unk> 1\n\u{2581}hello 2\n\u{2581}world 3\n\u{2581}yes 4\n\u{2581}no 5\n\u{2581}go 6\n\u{2581}ok 7\n";
    let tokens = Arc::new(TokenTable::from_str_checked(toks, 8).expect("toks"));
    let mut jw = wave(32, 2.0, 0.71);
    jw[..4].fill(0.0);
    let jb = [0.8, -2.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2];
    let c2 = Linear::from_fp32(&[0.15; 2 * 18], 2, 18, None);
    let stem_out = Linear::from_fp32(&[0.15; 4 * 38], 4, 38, None);
    let arch = Arch {
        n_mels: 80,
        d: 4,
        n_blocks: 0,
        lower: 0,
        mid: 0,
        fin: 0,
        heads: 1,
        dh: 4,
        ffn: 16,
        kernel: 3,
        stem_ch: 2,
        f1: 39,
        f2: 19,
        chunk: 4,
        left: 16,
        post_block_norm: false,
        vocab: 8,
        pred_emb: 4,
        joiner: 4,
        blank: 0,
        max_symbols: 1,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: String::new(),
        weights_sha256: String::new(),
        qat_stem_int8: false,
    };
    Arc::new(Model {
        arch,
        root: Default::default(),
        c1_w: vec![0.15; 18],
        c1_b: vec![0.0; 2],
        c2,
        stem_out,
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: lin(4, 4, 0.5, 0.43, None),
        emb: wave(32, 0.05, 0.37),
        pred_proj: lin(4, 8, 0.05, 0.53, None),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&jw, 8, 4, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0, 0.1],
        mel_cfg,
        tokens,
        ctc_head: None,
        int8_prepared: None,
    })
}
fn tone(n: usize, hz: f32) -> Vec<i16> {
    let k = std::f32::consts::TAU * hz / 16e3;
    (0..n)
        .map(|i| ((k * i as f32).sin() * 12e3) as i16)
        .collect()
}
fn plain(eng: &Engine, pcm: &[i16]) -> (Vec<u32>, String) {
    let (mut s, mut ws) = (eng.create_stream(), Workspace::default());
    for c in pcm.chunks(320) {
        s.accept_i16(c).expect("accept");
        while eng.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
    }
    s.input_finished();
    while eng.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
    (s.tokens().to_vec(), eng.text(&s))
}
fn feed(sess: &mut PeekSession, pcm: &[i16], probe: bool) -> Vec<PeekEvent> {
    let (mut evs, mut pws) = (Vec::new(), Workspace::default());
    for pkt in pcm.chunks(320) {
        evs.extend(sess.accept_i16(pkt).expect("accept"));
        if probe {
            let _ = sess.peek_immutable(&mut pws).expect("peek");
        }
    }
    evs.extend(sess.finish().expect("finish"));
    evs
}

#[test]
fn peek_committed_text_is_bit_identical_to_plain_stream() {
    let (model, pcm) = (tiny_model(), tone(8_000, 440.0));
    for beam in [1usize, 4] {
        let eng = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
            .unwrap()
            .with_beam(beam);
        let (rt, rx) = plain(&eng, &pcm);
        assert!(!rt.is_empty());
        for onset_only in [true, false] {
            let mut cfg = PeekConfig::default();
            (cfg.max_peek_frames, cfg.onset_only) = (3, onset_only);
            let mut s = PeekSession::new(eng.clone(), cfg);
            let evs = feed(&mut s, &pcm, true);
            let has_p = evs.iter().any(|e| matches!(e, PeekEvent::Preview { seq, base_commit_seq, through_sample, .. } if seq > base_commit_seq && *through_sample > 0));
            assert!(has_p && s.committed_tokens() == rt && s.committed_text() == rx);
        }
    }
    let eng = Engine::new(model, Isa::Scalar, Precision::F32).unwrap();
    let (mut st, mut pws) = (eng.create_stream(), Workspace::default());
    st.accept_i16(&pcm[..1280]).unwrap();
    let p = speculative_preview(&eng, &st, 2, &mut pws).unwrap();
    assert!(p.is_some() && st.enc.t == 0 && st.tokens().is_empty());
}

#[test]
fn peek_proxy_telemetry_accounting_is_exact() {
    assert_eq!(PeekStats::default().word_mismatch_proxy(), 0.0);
    assert_eq!(PeekStats::default().mean_first_lead_ms_proxy(), 0.0);
    for (b, p, a, w) in [
        ("", "hello world", "hello world", (2, 0)),
        ("", "hello dog", "hello cat", (2, 1)),
        ("", "hello", "", (1, 1)),
        ("hi", "hi there friend", "hi there buddy", (2, 1)),
        ("hi", "hi", "hi", (0, 0)),
    ] {
        assert_eq!(reconcile_preview_words(b, p, a), w);
    }
    let mut s = PeekStats::default();
    (
        s.preview_events,
        s.preview_superseded,
        s.preview_matched_next_commit,
    ) = (3, 2, 2);
    (
        s.preview_words_checked,
        s.preview_words_mismatched,
        s.preview_first_lead_ms_sum,
    ) = (4, 1, 120.0);
    assert!((s.word_mismatch_proxy() - 0.25).abs() < 1e-12);
    assert!((s.mean_first_lead_ms_proxy() - 60.0).abs() < 1e-12);
}

#[test]
fn peek_onset_only_boundary_and_clean_reset() {
    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    let mut s = PeekSession::new(eng.clone(), PeekConfig::default());
    let (mut p_ms, mut c_ms, mut after) = (None, None, 0usize);
    for ev in feed(&mut s, &tone(8_000, 440.0), false) {
        match ev {
            PeekEvent::Preview { .. } if c_ms.is_some() => after += 1,
            PeekEvent::Preview { audio_ms, .. } => _ = p_ms.get_or_insert(audio_ms),
            PeekEvent::Committed { text, audio_ms, .. } if !text.is_empty() => {
                _ = c_ms.get_or_insert(audio_ms)
            }
            PeekEvent::Committed { .. } => {}
        }
    }
    assert!(p_ms.unwrap() < c_ms.unwrap() && after == 0);
    let prev_n = s.stats().preview_events;
    s.reset();
    assert!(s.committed_tokens().is_empty() && s.committed_text().is_empty());
    let pcm2 = tone(6_400, 770.0);
    let (rt, rx) = plain(&eng, &pcm2);
    feed(&mut s, &pcm2, false);
    assert!(s.committed_tokens() == rt && s.committed_text() == rx);
    assert!(s.stats().preview_events > prev_n);
}

#[test]
fn peek_arm_onset_multi_turn_stays_armed_until_new_token() {
    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    let mut s = PeekSession::new(eng.clone(), PeekConfig::default());
    let (t1, flush, sil, t2) = (
        tone(3_200, 440.0),
        vec![0i16; 2_560],
        vec![0i16; 5_120],
        tone(4_800, 770.0),
    );
    let mut full = Vec::new();
    for pkt in t1
        .iter()
        .chain(&flush)
        .copied()
        .collect::<Vec<_>>()
        .chunks(320)
    {
        s.accept_i16(pkt).unwrap();
    }
    full.extend_from_slice(&t1);
    full.extend_from_slice(&flush);
    let tok_after_t1 = s.committed_tokens().len();
    assert!(tok_after_t1 > 0, "turn 1 must emit committed tokens");

    // Re-arm onset after turn 1 WITHOUT resetting the stream; feed 2 blank chunks (320 ms silence),
    // then turn 2 speech. Window must stay armed across blank chunks until a *new* token commits.
    s.arm_onset();
    for pkt in sil.chunks(320) {
        s.accept_i16(pkt).unwrap();
    }
    full.extend_from_slice(&sil);
    assert_eq!(
        s.committed_tokens().len(),
        tok_after_t1,
        "silence must emit 0 new tokens"
    );

    let (mut t2_previews, mut t2_committed, mut after_t2_commit) = (0usize, false, 0usize);
    for pkt in t2.chunks(320) {
        for ev in s.accept_i16(pkt).unwrap() {
            match ev {
                PeekEvent::Preview { .. } if t2_committed => after_t2_commit += 1,
                PeekEvent::Preview { .. } => t2_previews += 1,
                PeekEvent::Committed { .. } if s.committed_tokens().len() > tok_after_t1 => {
                    t2_committed = true;
                }
                PeekEvent::Committed { .. } => {}
            }
        }
    }
    full.extend_from_slice(&t2);
    s.finish().unwrap();
    assert!(
        t2_previews > 0,
        "re-armed onset must emit preview on turn 2 after 2 blank chunks"
    );
    assert!(
        t2_committed && after_t2_commit == 0,
        "onset window must close once new token commits"
    );
    let (rt, rx) = plain(&eng, &full);
    assert!(s.committed_tokens() == rt && s.committed_text() == rx);
}

#[test]
fn peek_real_weights_int8_smoke_parity() {
    let (w, npz) = (common::weights_dir(), common::dev_clean_npz());
    if !common::have(&[&w, &npz]) {
        return;
    }
    let mut m = Model::load(&w, false).unwrap();
    m.quantize_int8();
    let eng = Engine::new(Arc::new(m), Isa::detect(), Precision::Int8).unwrap();
    let eng = eng.with_chunk(4).unwrap();
    let mut s = PeekSession::new(eng.clone(), PeekConfig::default());
    let clips = load_npz_audio_only(&npz, 0, Some(4)).unwrap();
    for (i, u) in clips.iter().enumerate() {
        let (rt, rx) = plain(&eng, &u.pcm);
        s.reset();
        feed(&mut s, &u.pcm, false);
        assert!(
            s.committed_tokens() == rt && s.committed_text() == rx,
            "{i}"
        );
    }
    let st = s.stats();
    let (mis, lead) = (st.word_mismatch_proxy(), st.mean_first_lead_ms_proxy());
    eprintln!("H48d INT8 (4 clips, c=4): {st:?} mismatch_proxy={mis:.3} lead_proxy={lead:.1}ms");
}

#[test]
fn peek_preserves_width3_context_and_beam_reset_ctx_start() {
    use ultra_engine::beam::BeamState;

    let mut base = Arc::try_unwrap(tiny_model()).expect("unique model");
    base.arch.ctx_init = vec![0, 0, 0];
    base.enc_proj = Linear::from_fp32(&[0.0; 16], 4, 4, None);
    let mut emb = vec![0.0f32; 8 * 4];
    emb[2 * 4] = 1.0; // token 2 ("hello") -> dim 0
    emb[3 * 4 + 1] = 1.0; // token 3 ("world") -> dim 1
    emb[4 * 4 + 2] = 1.0; // token 4 ("yes")   -> dim 2
    base.emb = emb;

    let mut pw = vec![0.0f32; 4 * 12];
    pw[0] = 4.0; // row 0 reads slot 0, dim 0 (token 2 in oldest of 3 context slots)
    pw[12 + 10] = 4.0; // row 1 reads slot 2, dim 2 (token 4 in newest context slot)
    base.pred_proj = Linear::from_fp32(&pw, 4, 12, Some(&[0.0, 0.0, 4.0, 0.0]));

    let mut jw = vec![0.0f32; 8 * 4];
    jw[5 * 4 + 2] = 5.0; // token 5 ("no"): wins on fresh reset context [0, 0, 0]
    jw[6 * 4] = 10.0; // token 6 ("go"): wins iff oldest context slot 0 == token 2
    jw[7 * 4] = -10.0;
    jw[7 * 4 + 1] = 10.0; // token 7 ("ok"): would win if slot 0 were truncated away
    base.joiner_out = Linear::from_fp32(&jw, 8, 4, Some(&[0.0; 8]));

    let model = Arc::new(base);
    let eng = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .unwrap()
        .with_beam(4);
    let mut st = eng.create_stream();
    let mut beam = BeamState::new(&model);
    beam.hyps[0].tokens = vec![2, 3, 4];
    st.dec.tokens = vec![2, 3, 4];
    st.dec.beam = Some(beam);
    st.accept_i16(&tone(1_600, 440.0)).unwrap();

    // Before reset: full width-3 context [2, 3, 4] must emit "go" (not truncated "ok") while
    // keeping committed tokens intact and leaving `st` unmutated.
    let mut pws = Workspace::default();
    let before = speculative_preview(&eng, &st, 1, &mut pws)
        .expect("width-3 preview")
        .expect("ready for 1-frame peek");
    assert_eq!(before, "hello world yes go");
    assert_eq!(st.tokens(), &[2, 3, 4]);
    assert!(st.dec.beam.is_some());

    // After beam reset: `ctx_start == 3` must reconstruct [0, 0, 0] -> emits "no", while still
    // preserving the committed prefix "hello world yes" and dropping only the clone's beam state.
    st.dec.reset_context(&model);
    let after = speculative_preview(&eng, &st, 1, &mut pws)
        .expect("post-reset preview")
        .expect("ready for 1-frame peek");
    assert_eq!(after, "hello world yes no");
    assert_eq!(st.tokens(), &[2, 3, 4]);
    assert!(st.dec.beam.is_some());
}
