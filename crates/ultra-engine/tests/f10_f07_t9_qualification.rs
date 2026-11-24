//! Qualification tests for `F01/F10` (`EffectiveDecodeConfig`), `F07` (`DecodeProgress`
//! endpoint activity), `F09` (`run_commit_pcm_timed`), and `T9` (`beam=1` width-1 fused greedy LM).

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::beam::{BeamState, Hyp};
use ultra_engine::commit::{run_commit_pcm_timed, CommitConfig};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::peek::{speculative_preview, PeekConfig, PeekSession};
use ultra_engine::{Engine, Isa, LmConfig, NgramLm, Precision, Workspace};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const VOCAB4: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n";

/// Synthetic LM strongly preferring token 2 (`▁cat`) over token 1 (`▁the`).
const ARPA_PREFER_CAT: &str = "\
\\data\\
ngram 1=4
ngram 2=3

\\1-grams:
-4.5\t<unk>
-99.0\t<s>\t-0.1
-3.0\t1\t-0.1
-0.05\t2\t-0.05

\\2-grams:
-0.05\t<s> 2
-0.05\t2 2
-3.0\t<s> 1

\\end\\
";

const ARPA_LODR_FLAT: &str = "\
\\data\\
ngram 1=4
ngram 2=2

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.1
-1.5\t1
-1.5\t2

\\2-grams:
-1.5\t<s> 1
-1.5\t<s> 2

\\end\\
";

fn synthetic_model(joiner_bias: [f32; 4], max_symbols: usize) -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (vocab, pe, jn) = (4, 4, 8);
    let z = |n, m| Linear::from_fp32(&vec![0.0; n * m], n, m, None);
    let nm = |n| Norm {
        g: vec![1.0; n],
        b: vec![0.0; n],
    };
    let blk = Block {
        ff1_norm: nm(d),
        ff1_w1: z(ffn, d),
        ff1_w2: z(d, ffn),
        att_norm: nm(d),
        qkv: z(3 * d, d),
        att_out: z(d, d),
        conv_norm: nm(d),
        pw1: z(2 * d, d),
        dw_w: vec![0.0; k * d],
        dw_b: vec![0.0; d],
        dnorm: nm(d),
        pw2: z(d, d),
        ff2_norm: nm(d),
        ff2_w1: z(ffn, d),
        ff2_w2: z(d, ffn),
        post_norm: None,
    };
    let tok = TokenTable::from_str_checked(VOCAB4, vocab).expect("tok");
    let arch = Arch {
        n_mels: 80,
        d,
        n_blocks: 1,
        lower: 1,
        mid: 0,
        fin: 0,
        heads: h,
        dh,
        ffn,
        kernel: k,
        stem_ch: sc,
        f1,
        f2,
        chunk: 4,
        left: 4,
        post_block_norm: false,
        vocab,
        pred_emb: pe,
        joiner: jn,
        blank: 0,
        max_symbols,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: "a".repeat(64),
        weights_sha256: "b".repeat(64),
        qat_stem_int8: false,
    };
    Arc::new(Model {
        arch,
        root: PathBuf::new(),
        c1_w: vec![0.0; 9 * sc],
        c1_b: vec![0.0; sc],
        c2: z(sc, 9 * sc),
        stem_out: z(d, f2 * sc),
        blocks: vec![blk],
        final_norm: Some(nm(d)),
        pair: None,
        restore: None,
        enc_proj: z(jn, d),
        emb: vec![0.0; vocab * pe],
        pred_proj: z(jn, 2 * pe),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&vec![0.0; vocab * jn], vocab, jn, Some(&joiner_bias)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        ctc_head: None,
        int8_prepared: None,
    })
}

fn stream_decode_steps(engine: &Engine, pcm: &[i16]) -> (Vec<Vec<u32>>, Vec<u32>, String) {
    let mut s = engine.create_stream();
    let mut ws = Workspace::default();
    let mut history = Vec::new();
    for pkt in pcm.chunks(320) {
        s.accept_i16(pkt).expect("accept");
        while engine.decode(&mut [&mut s], &mut ws).expect("decode") > 0 {
            history.push(s.tokens().to_vec());
        }
    }
    s.input_finished();
    while engine.decode(&mut [&mut s], &mut ws).expect("decode") > 0 {
        history.push(s.tokens().to_vec());
    }
    let text = engine.text(&s);
    (history, s.tokens().to_vec(), text)
}

#[test]
fn t9_beam1_no_lm_matches_legacy_greedy_and_beam1_with_lm_is_append_only() {
    let model = synthetic_model([-2.0, 2.0, 1.5, -4.0], 1);
    let lm = Arc::new(
        NgramLm::from_arpa_str(ARPA_PREFER_CAT, Some(ARPA_LODR_FLAT), 4, false).expect("lm"),
    );
    let lm_cfg = LmConfig {
        lm_weight: 0.8,
        lodr_weight: 0.6,
        blank_penalty: 0.0,
    };
    let pcm = vec![0i16; 6_400];

    // 1. `beam=1, lm=None`: must stay on legacy greedy path (`!uses_beam_state()`).
    let eng_plain = Engine::new(model.clone(), Isa::Scalar, Precision::F32).expect("plain");
    let eng_beam1 = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .expect("beam1")
        .with_beam(1);
    assert!(!eng_plain.uses_beam_state());
    assert!(!eng_beam1.uses_beam_state());
    let (_, tok_plain, txt_plain) = stream_decode_steps(&eng_plain, &pcm);
    let (_, tok_b1, txt_b1) = stream_decode_steps(&eng_beam1, &pcm);
    assert_eq!(tok_plain, tok_b1);
    assert_eq!(txt_plain, txt_b1);
    assert!(!tok_plain.is_empty() && tok_plain.iter().all(|&t| t == 1));
    assert!(txt_plain.split_whitespace().all(|w| w == "the"));

    // 2. `beam=1, lm=Some(...)`: activates width-1 fused greedy LM, strictly changes top output
    //    from "the" to "cat", and is strictly append-only (0 retractions) across every chunk.
    let eng_lm1 = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .expect("lm1")
        .with_beam(1)
        .with_beam_max_symbols(1)
        .with_lm(lm.clone(), lm_cfg)
        .expect("with_lm");
    assert!(eng_lm1.uses_beam_state());
    assert_ne!(
        eng_plain.effective_decode_config().policy_sha256(),
        eng_lm1.effective_decode_config().policy_sha256()
    );
    let (hist1, tok_lm1, txt_lm1) = stream_decode_steps(&eng_lm1, &pcm);
    assert!(!tok_lm1.is_empty());
    assert_ne!(txt_lm1, txt_plain, "beam=1 LM must change decoded output");
    assert!(
        tok_lm1.iter().all(|&t| t == 2),
        "LM must steer beam=1 to token 2 (cat), got {tok_lm1:?}"
    );
    for w in hist1.windows(2) {
        assert!(
            w[1].starts_with(&w[0]),
            "beam=1 fused greedy LM must be strictly append-only (0 retractions): {:?} -> {:?}",
            w[0],
            w[1]
        );
    }

    // 3. Explicit `beam_max_symbols = 2` works cleanly at `beam=1` and stays strictly append-only.
    let eng_lm1_s2 = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .expect("lm1_s2")
        .with_beam(1)
        .with_beam_max_symbols(2)
        .with_lm(lm.clone(), lm_cfg)
        .expect("with_lm");
    assert_eq!(
        eng_lm1_s2.effective_decode_config().beam_max_symbols,
        2,
        "EffectiveDecodeConfig must reflect explicit beam_max_symbols at beam=1"
    );
    let (hist2, tok_lm1_s2, _) = stream_decode_steps(&eng_lm1_s2, &pcm);
    assert_eq!(
        tok_lm1_s2.len(),
        tok_lm1.len() * 2,
        "beam_max_symbols=2 should emit 2 symbols per non-blank frame"
    );
    assert!(tok_lm1_s2.iter().all(|&t| t == 2));
    for w in hist2.windows(2) {
        assert!(
            w[1].starts_with(&w[0]),
            "beam=1 max_symbols=2 must stay strictly append-only: {:?} -> {:?}",
            w[0],
            w[1]
        );
    }

    // 4. `speculative_preview` and `PeekSession` handle `engine.lm.is_some()` at beam=1 & beam=4
    //    without panicking or corrupting committed state.
    for beam in [1usize, 4] {
        let eng = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
            .expect("eng")
            .with_beam(beam)
            .with_lm(lm.clone(), lm_cfg)
            .expect("lm");
        let (_, ref_toks, ref_txt) = stream_decode_steps(&eng, &pcm);
        let mut sess = PeekSession::new(eng.clone(), PeekConfig::default());
        let mut pws = Workspace::default();
        for pkt in pcm.chunks(320) {
            sess.accept_i16(pkt).expect("peek accept");
            let _ = sess.peek_immutable(&mut pws).expect("peek_immutable");
        }
        sess.finish().expect("peek finish");
        assert_eq!(sess.committed_tokens(), ref_toks);
        assert_eq!(sess.committed_text(), ref_txt);

        // Direct speculative_preview on a stream that already has committed tokens must preserve them.
        let mut st = eng.create_stream();
        let mut ws = Workspace::default();
        st.accept_i16(&pcm[..3_200]).expect("accept");
        while eng.decode(&mut [&mut st], &mut ws).expect("dec") > 0 {}
        let before = st.tokens().to_vec();
        assert!(!before.is_empty());
        st.accept_i16(&pcm[3_200..4_480]).expect("accept partial");
        let prev = speculative_preview(&eng, &st, 2, &mut pws)
            .expect("preview")
            .expect("ready");
        assert!(
            prev.starts_with(&eng.text(&st)),
            "preview `{prev}` must preserve committed prefix `{}`",
            eng.text(&st)
        );
        assert_eq!(st.tokens(), before);
    }
}

#[test]
fn f07_equal_length_beam_substitution_and_contraction_prevent_false_endpoint_reset() {
    // Build a model where the joiner strongly prefers blank (0), EXCEPT predictor context [0, 3]
    // shifts the blank logit slightly HIGHER than predictor context [0, 1] so rival hyp [2, 3]
    // overtakes incumbent hyp [1, 1] via pure blank scoring on an equal-length substitution!
    let mut base = Arc::try_unwrap(synthetic_model([5.0, -5.0, -5.0, -5.0], 1)).expect("unique");
    let (vocab, pe, jn) = (base.arch.vocab, base.arch.pred_emb, base.arch.joiner);
    // Token 3 embedding sets dim 0 = 1.0; all other tokens = 0.0.
    let mut emb = vec![0.0f32; vocab * pe];
    emb[3 * pe] = 1.0;
    base.emb = emb;
    // Predictor slot 1 (newest token), dim 0 -> joiner hidden dim 0 = +2.0.
    let mut pw = vec![0.0f32; jn * 2 * pe];
    pw[pe] = 2.0; // row 0, col (slot1 * pe + 0)
    base.pred_proj = Linear::from_fp32(&pw, jn, 2 * pe, Some(&vec![0.0; jn]));
    // Joiner row 0 (blank) gets +4.0 weight on dim 0; non-blank rows stay 0.0 ->
    // hyp ending in token 3 gets much higher blank log-prob than hyp ending in token 1!
    let mut jw = vec![0.0f32; vocab * jn];
    jw[0] = 4.0;
    base.joiner_out = Linear::from_fp32(&jw, vocab, jn, Some(&[2.0, 0.5, 0.5, 0.5]));
    let model = Arc::new(base);

    // Endpoint reset threshold = 160 ms (= 4 frames = 1 chunk).
    let eng = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .expect("eng")
        .with_beam(2)
        .with_endpoint_reset_ms(160);

    // Case 1: Equal-length substitution ([1, 1] -> [2, 3], len 2 -> len 2).
    let mut s = eng.create_stream();
    let mut ws = Workspace::default();
    let mut beam = BeamState::new(&model);
    let mut rival = beam.hyps[0].clone();
    beam.hyps[0].tokens = vec![1, 1];
    beam.hyps[0].score = 0.0;
    rival.tokens = vec![2, 3]; // ends in token 3 -> higher blank log-prob, overtakes [1, 1]!
    rival.score = -0.05;
    beam.hyps.push(rival);
    s.dec.tokens = vec![1, 1];
    s.dec.beam = Some(beam);
    // Feed 1 chunk (4 frames = 160 ms, exact threshold).
    s.accept_features(&vec![[0.0; 80]; 20]).expect("feat1");
    assert_eq!(eng.decode(&mut [&mut s], &mut ws).expect("dec1"), 1);
    assert_eq!(s.tokens(), &[2, 3], "rival must overtake with equal length");
    let p1 = s.last_progress();
    assert!(!p1.emitted_new_tokens, "length stayed 2");
    assert!(
        p1.hypothesis_changed,
        "equal-length swap must set hypothesis_changed"
    );
    assert!(
        p1.has_activity(),
        "equal-length swap is real decode activity"
    );
    assert_eq!(
        s.endpoint_resets(),
        0,
        "equal-length beam substitution must NOT trigger false endpoint reset"
    );

    // Case 2: Contraction ([1, 1, 1] -> [3], len 3 -> len 1).
    let mut s2 = eng.create_stream();
    let mut beam2 = BeamState::new(&model);
    let mut short_rival = beam2.hyps[0].clone();
    beam2.hyps[0].tokens = vec![1, 1, 1];
    beam2.hyps[0].score = 0.0;
    short_rival.tokens = vec![3];
    short_rival.score = -0.05;
    beam2.hyps.push(short_rival);
    s2.dec.tokens = vec![1, 1, 1];
    s2.dec.beam = Some(beam2);
    s2.accept_features(&vec![[0.0; 80]; 20]).expect("feat2");
    assert_eq!(eng.decode(&mut [&mut s2], &mut ws).expect("dec2"), 1);
    assert_eq!(s2.tokens(), &[3], "shorter rival must overtake");
    let p2 = s2.last_progress();
    assert!(!p2.emitted_new_tokens);
    assert!(p2.hypothesis_changed);
    assert!(p2.has_activity());
    assert_eq!(
        s2.endpoint_resets(),
        0,
        "beam contraction must NOT trigger false endpoint reset"
    );

    // Case 3: Genuine pure-blank silence chunk AFTER activity DOES trigger endpoint reset.
    // Collapse beam on `s` to single hyp [2, 3] so next pure-blank chunk has zero change.
    s.dec.beam.as_mut().expect("beam").hyps.truncate(1);
    s.accept_features(&vec![[0.0; 80]; 20]).expect("feat3");
    assert_eq!(eng.decode(&mut [&mut s], &mut ws).expect("dec3"), 1);
    assert!(!s.last_progress().has_activity());
    assert_eq!(
        s.endpoint_resets(),
        1,
        "genuine 160 ms blank chunk after armed activity must trigger endpoint reset"
    );
    // Keep unused import check clean: Hyp is imported to ensure public visibility.
    let _: Option<Hyp> = None;
}

#[test]
fn f09_run_commit_pcm_timed_reports_separate_finite_wall_and_cpu_time() {
    let model = synthetic_model([-2.0, 2.0, 1.5, -4.0], 1);
    let eng = Engine::new(model, Isa::Scalar, Precision::F32).expect("eng");
    let cfg = CommitConfig {
        commit_chunk: 4,
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let pcm = vec![0i16; 6_400];
    let mut ws = Workspace::default();
    let (txt, idx, timing) = run_commit_pcm_timed(&eng, cfg, &pcm, &mut ws).expect("timed");
    assert!(!txt.is_empty());
    assert_eq!(idx, 0);
    assert!(timing.wall_s.is_finite() && timing.wall_s >= 0.0);
    assert!(timing.cpu_s.is_finite() && timing.cpu_s >= 0.0);
    assert!((timing.wall_ms() - timing.wall_s * 1000.0).abs() < 1e-9);
    assert!((timing.cpu_ms() - timing.cpu_s * 1000.0).abs() < 1e-9);
}
