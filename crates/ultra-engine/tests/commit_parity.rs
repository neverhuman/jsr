//! H45 parity and transcript-ledger contract tests addressing `@codex-lead` review blockers:
//! 1. Exact SentencePiece BPE-1024 retokenization parity (`fevered`, `threatened`, DEV hypothesis).
//! 2. Real `ultra_core::events::Ledger` active-segment pre-commit, sample continuity across
//!    endpoint/replay, and bounded PCM retention.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::events::{Applied, Body, Ledger, SegmentStatus};
use ultra_engine::commit::{
    ctc_rescore_text, CommitConfig, CommitPassSession, CtcVocab, DEFAULT_TAIL_PAD_SAMPLES,
};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const BPE1024_SHA256: &str = "557a3a28cf7895686227c973e5a7d00a04bf7e0bc1300fc00e4670c6e436c2d4";

fn synthetic_model(bias_token: usize) -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (v, pe, jn) = (4, 4, 8);
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
    let mut jb = vec![-2.0f32; v];
    jb[bias_token] = 2.0;
    let tok =
        TokenTable::from_str_checked("<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n", v)
            .expect("tok");
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
        vocab: v,
        pred_emb: pe,
        joiner: jn,
        blank: 0,
        max_symbols: 2,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: String::new(),
        weights_sha256: String::new(),
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
        emb: vec![0.0; v * pe],
        pred_proj: z(jn, 2 * pe),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&vec![0.0; v * jn], v, jn, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        ctc_head: None,
        int8_prepared: None,
    })
}

#[test]
fn commit_ctc_spm_bpe1024_real_hypothesis_parity() {
    let base = std::env::var_os("JSR_DATA")
        .map_or_else(|| PathBuf::from("/mnt/xbabe3-external/jsr"), PathBuf::from);
    let spm_path = base.join("tokenizers/librispeech_train_bpe1024.model");
    if !spm_path.exists() {
        return;
    }
    let vocab = CtcVocab::from_spm_model_file(&spm_path, 0).expect("load bpe1024");
    assert_eq!(vocab.model_sha256(), Some(BPE1024_SHA256));
    assert_eq!(vocab.vocab_size(), 1025);
    assert_eq!(
        vocab.retokenize_pieces("fevered").expect("fev"),
        vec!["\u{2581}fe", "ve", "red"]
    );
    assert_eq!(
        vocab.retokenize_pieces("threatened").expect("thr"),
        vec!["\u{2581}th", "reat", "ened"]
    );

    let hyp = "HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE'S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE";
    let want: Vec<usize> = vec![
        32, 60, 36, 4, 173, 57, 116, 85, 209, 25, 596, 9, 1012, 29, 24, 5, 290, 114, 79, 967, 1020,
        1005, 644, 73, 30, 233, 477, 24, 18, 183, 326, 79, 505, 348, 19, 51, 247,
    ];
    let got = vocab.retokenize_text(hyp).expect("retokenize hyp");
    assert_eq!(got, want);

    let frames = got.len() + 4;
    let mut lp = vec![-10.0f32; frames * vocab.vocab_size()];
    for (t, &id) in got.iter().enumerate() {
        lp[t * vocab.vocab_size() + id] = -0.05;
    }
    for t in got.len()..frames {
        lp[t * vocab.vocab_size()] = -0.05;
    }
    let score = ctc_rescore_text(&vocab, &lp, frames, hyp);
    assert!(score.is_finite() && score > -5.0);
}

#[test]
fn commit_ledger_precommitted_active_segment_and_sample_continuity() {
    let eng = Engine::new(synthetic_model(1), Isa::Scalar, Precision::F32).expect("engine");
    let cfg = CommitConfig {
        tail_pad_samples: DEFAULT_TAIL_PAD_SAMPLES,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassSession::with_stream_id(eng, cfg, "h45-s1", "a1-c16").expect("sess");
    let mut ledger = Ledger::new();

    sess.accept_i16(&[0i16; 3_200]).expect("seg0_a");
    assert_eq!(
        ledger
            .apply(&sess.preview_event(100).expect("p0"))
            .expect("ap"),
        Applied::New
    );
    let prov0 = sess.provisional_text();
    let n = prov0.find(' ').unwrap_or(prov0.len());
    assert!(n > 0);
    assert_eq!(
        ledger
            .apply(&sess.precommit_active_prefix(200, n).expect("c0"))
            .expect("ac"),
        Applied::New
    );
    let pre = ledger.segment(0).expect("seg0").immutable.clone();
    assert_eq!(pre, &prov0[..n]);

    sess.accept_i16(&[0i16; 1_600]).expect("seg0_b");
    let (r0, ev0) = sess.commit_endpoint_event(300).expect("f0");
    assert_eq!(ledger.apply(&ev0).expect("af0"), Applied::New);
    assert!(r0.committed.starts_with(&pre));
    assert_eq!(
        ledger.segment(0).expect("seg0").status,
        SegmentStatus::Final
    );
    let Body::Final {
        update: u0,
        processing: p0,
    } = &ev0.body
    else {
        panic!("Final")
    };
    assert_eq!(
        (u0.ownership_start_sample, u0.ownership_end_sample),
        (0, 4_800)
    );
    assert_eq!(
        (
            p0.accepted_through_sample,
            p0.required_processed_through_sample,
            p0.flushed
        ),
        (4_800, 4_800, true)
    );

    sess.accept_i16(&[0i16; 3_200]).expect("seg1");
    assert_eq!(sess.active_immutable(), "");
    let (r1, ev1) = sess.commit_endpoint_event(400).expect("f1");
    assert_eq!(ledger.apply(&ev1).expect("af1"), Applied::New);
    assert_eq!(r1.committed, ledger.segment(1).expect("seg1").immutable);
    let Body::Final { update: u1, .. } = &ev1.body else {
        panic!("Final")
    };
    assert_eq!(
        (
            u1.ownership_start_sample,
            u1.ownership_end_sample,
            u1.immutable_before_utf8_bytes
        ),
        (4_800, 8_000, 0)
    );
}

#[test]
fn commit_bounded_pcm_retention_caps_memory_and_keeps_ledger_valid() {
    let eng = Engine::new(synthetic_model(1), Isa::Scalar, Precision::F32).expect("engine");
    let cfg = CommitConfig {
        max_pcm_samples: 1_600,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassSession::with_stream_id(eng, cfg, "h45-cap", "a1-c16").expect("sess");
    let mut ledger = Ledger::new();
    for _ in 0..5 {
        sess.accept_i16(&[0i16; 1_600]).expect("accept");
        assert!(sess.buffered_samples() <= 1_600);
    }
    assert!(
        sess.pcm_truncated()
            && sess.dropped_pcm_samples() == 6_400
            && sess.total_samples() == 8_000
    );
    let (res, ev) = sess.commit_endpoint_event(500).expect("commit capped");
    assert!(res.pcm_truncated && !res.revised && !res.committed.is_empty());
    assert_eq!(ledger.apply(&ev).expect("apply capped final"), Applied::New);
    assert_eq!(sess.buffered_samples(), 0);
}
