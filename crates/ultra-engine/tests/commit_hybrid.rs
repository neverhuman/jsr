//! Tests for H45 hybrid `(live_model, final_model)` endpoint commit pass.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::events::{Applied, Body, Ledger, SegmentStatus};
use ultra_engine::commit::{CommitConfig, CommitPassEngine};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, EngineError, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn synthetic_model_with_vocab(bias_token: usize, tokens_str: &str, vocab: usize) -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (pe, jn) = (4, 8);
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
    let mut jb = vec![-2.0f32; vocab];
    jb[bias_token] = 2.0;
    let tok = TokenTable::from_str_checked(tokens_str, vocab).expect("tok");
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
        max_symbols: 2,
        ctx_init: [0, 0],
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
        emb: vec![0.0; vocab * pe],
        pred_proj: z(jn, 2 * pe),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&vec![0.0; vocab * jn], vocab, jn, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        int8_prepared: None,
    })
}

const VOCAB4: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n";
const VOCAB4_ALT: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}dog 2\n\u{2581}sat 3\n";
const VOCAB5: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n\u{2581}mat 4\n";

#[test]
fn commit_hybrid_same_model_is_bit_identical_to_single_model_new() {
    let model = synthetic_model_with_vocab(1, VOCAB4, 4);
    let eng = Engine::new(model, Isa::Scalar, Precision::F32).expect("eng");
    let cfg = CommitConfig {
        tail_pad_samples: 640,
        ..CommitConfig::default()
    };
    let mut single = CommitPassEngine::new(eng.clone(), cfg).expect("single");
    let mut hybrid = CommitPassEngine::new_hybrid(eng.clone(), eng, cfg).expect("hybrid");
    assert!(!single.is_hybrid() && !hybrid.is_hybrid());
    assert!(Arc::ptr_eq(
        &hybrid.provisional_engine().model,
        &hybrid.commit_engine().model
    ));

    let pcm = vec![0i16; 3_200];
    single.accept_i16(&pcm).expect("s accept");
    hybrid.accept_i16(&pcm).expect("h accept");
    let (rs, es) = single.commit_endpoint_event(100).expect("s commit");
    let (rh, eh) = hybrid.commit_endpoint_event(100).expect("h commit");
    assert_eq!(rs.provisional, rh.provisional);
    assert_eq!(rs.committed, rh.committed);
    assert_eq!(rs.committed_prefix, rh.committed_prefix);
    assert_eq!(rs.revised, rh.revised);
    assert_eq!(es.body, eh.body);
}

#[test]
fn commit_hybrid_routes_live_provisional_and_final_commit_with_ledger_continuity() {
    let live = Engine::new(
        synthetic_model_with_vocab(1, VOCAB4, 4),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("live");
    let final_m = Engine::new(
        synthetic_model_with_vocab(2, VOCAB4, 4),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("final");
    let cfg = CommitConfig {
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassEngine::with_stream_id_hybrid(live, final_m, cfg, "h45-hyb", "hyb")
        .expect("hybrid sess");
    assert!(sess.is_hybrid());
    let mut ledger = Ledger::new();

    // Segment 0: live produces "the ...", pre-commit first word, final produces "cat ...".
    sess.accept_i16(&[0i16; 3_200]).expect("seg0");
    assert!(sess
        .provisional_text()
        .split_whitespace()
        .all(|w| w == "the"));
    assert_eq!(
        ledger
            .apply(&sess.preview_event(10).expect("p0"))
            .expect("ap0"),
        Applied::New
    );
    assert_eq!(
        ledger
            .apply(&sess.precommit_active_prefix(20, 3).expect("c0"))
            .expect("ac0"),
        Applied::New
    );
    let (r0, ev0) = sess.commit_endpoint_event(30).expect("f0");
    assert_eq!(ledger.apply(&ev0).expect("af0"), Applied::New);
    assert!(r0.revised);
    assert!(r0.committed.starts_with("the"));
    assert!(r0.committed.contains("cat"));
    assert_eq!(
        ledger.segment(0).expect("seg0").status,
        SegmentStatus::Final
    );

    // Segment 1: no active pre-commit -> provisional all "the" (live), committed all "cat" (final).
    sess.accept_i16(&[0i16; 3_200]).expect("seg1");
    let (r1, ev1) = sess.commit_endpoint_event(40).expect("f1");
    assert_eq!(ledger.apply(&ev1).expect("af1"), Applied::New);
    assert!(r1.revised);
    assert!(r1.provisional.split_whitespace().all(|w| w == "the"));
    assert!(r1.committed.split_whitespace().all(|w| w == "cat"));
    let Body::Final { update: u1, .. } = &ev1.body else {
        panic!("expected Final")
    };
    assert_eq!(
        (u1.ownership_start_sample, u1.ownership_end_sample),
        (3_200, 6_400)
    );
}

#[test]
fn commit_hybrid_rejects_mismatched_vocab_size_and_pieces() {
    let live = Engine::new(
        synthetic_model_with_vocab(1, VOCAB4, 4),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("live");
    let bad_size = Engine::new(
        synthetic_model_with_vocab(1, VOCAB5, 5),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("bad_size");
    let err_sz = CommitPassEngine::new_hybrid(live.clone(), bad_size, CommitConfig::default())
        .expect_err("must reject vocab size mismatch");
    assert!(matches!(err_sz, EngineError::Format(ref m) if m.contains("vocab size mismatch")));

    let bad_pieces = Engine::new(
        synthetic_model_with_vocab(1, VOCAB4_ALT, 4),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("bad_pieces");
    let err_pc = CommitPassEngine::new_hybrid(live, bad_pieces, CommitConfig::default())
        .expect_err("must reject sp_pieces mismatch");
    assert!(matches!(err_pc, EngineError::Format(ref m) if m.contains("sp_pieces differ")));
}
