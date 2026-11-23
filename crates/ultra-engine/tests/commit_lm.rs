//! Tests for H45 endpoint-only LM+LODR beam decoding (`CommitLmConfig` + `beam_chunk_with_lm`).

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::events::{Applied, Body, Ledger, SegmentStatus};
use ultra_engine::commit::{CommitConfig, CommitLmConfig, CommitPassSession};
use ultra_engine::lm::NgramLm;
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, EngineError, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const VOCAB4: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n";

/// Synthetic token n-gram LM strongly preferring token 2 (`▁cat`) over token 1 (`▁the`).
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

fn synthetic_near_tie_model(bias1: f32, bias2: f32) -> Arc<Model> {
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
    let jb = vec![-2.0, bias1, bias2, -4.0];
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

fn roundtrip_v2_ulm(beam: usize) -> CommitLmConfig {
    let lm = NgramLm::from_arpa_str(ARPA_PREFER_CAT, Some(ARPA_LODR_FLAT), 4, false).expect("arpa");
    let path =
        std::env::temp_dir().join(format!("ultra_commit_lm_{}_{beam}.ulm", std::process::id()));
    lm.save(&path).expect("save v2 ulm");
    let cfg = CommitLmConfig::from_file(&path, beam, 0.8, 0.6, 0.0, 1).expect("load v2 ulm");
    let _ = std::fs::remove_file(&path);
    cfg
}

#[test]
fn commit_lm_none_is_bit_identical_to_default_commit_pass() {
    let eng = Engine::new(
        synthetic_near_tie_model(2.0, 1.5),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("eng");
    let cfg = CommitConfig {
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let mut base = CommitPassSession::new(eng.clone(), cfg).expect("base");
    let mut none_lm = CommitPassSession::new(eng, cfg)
        .expect("n")
        .with_commit_lm(None)
        .expect("nl");
    assert!(!none_lm.has_commit_lm());
    let pcm = vec![0i16; 3_200];
    base.accept_i16(&pcm).expect("b accept");
    none_lm.accept_i16(&pcm).expect("n accept");
    let (rb, eb) = base.commit_endpoint_event(100).expect("b commit");
    let (rn, en) = none_lm.commit_endpoint_event(100).expect("n commit");
    assert_eq!(
        (rb.provisional, rb.committed, rb.revised),
        (rn.provisional, rn.committed, rn.revised)
    );
    assert_eq!(eb.body, en.body);
}

#[test]
fn commit_lm_v2_pack_guides_endpoint_commit_while_live_stays_greedy() {
    let eng = Engine::new(
        synthetic_near_tie_model(2.0, 1.5),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("eng");
    let cfg = CommitConfig {
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassSession::with_stream_id(eng, cfg, "h45-lm", "b0-m5")
        .expect("sess")
        .with_commit_lm(Some(roundtrip_v2_ulm(4)))
        .expect("with_lm");
    assert!(sess.has_commit_lm());
    assert_eq!(sess.provisional_engine().beam, 1);
    assert!(sess.provisional_engine().lm.is_none());
    assert_eq!(sess.commit_engine().beam, 4);
    assert!(sess.commit_engine().lm.is_some());

    let mut ledger = Ledger::new();
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
    assert!(r0.revised && r0.committed.starts_with("the") && r0.committed.contains("cat"));
    assert_eq!(
        ledger.segment(0).expect("seg0").status,
        SegmentStatus::Final
    );

    sess.accept_i16(&[0i16; 3_200]).expect("seg1");
    let (r1, ev1) = sess.commit_endpoint_event(40).expect("f1");
    assert_eq!(ledger.apply(&ev1).expect("af1"), Applied::New);
    assert!(r1.revised && r1.provisional.split_whitespace().all(|w| w == "the"));
    assert!(r1.committed.split_whitespace().all(|w| w == "cat"));
    let Body::Final { update: u1, .. } = &ev1.body else {
        panic!("Final")
    };
    assert_eq!(
        (u1.ownership_start_sample, u1.ownership_end_sample),
        (3_200, 6_400)
    );
}

#[test]
fn commit_lm_supports_beam16_and_rejects_invalid_beam_or_vocab() {
    let eng = Engine::new(
        synthetic_near_tie_model(2.0, 1.5),
        Isa::Scalar,
        Precision::F32,
    )
    .expect("eng");
    let cfg = CommitConfig {
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let lm = Arc::new(
        NgramLm::from_arpa_str(ARPA_PREFER_CAT, Some(ARPA_LODR_FLAT), 4, true).expect("q8"),
    );
    let mut sess16 = CommitPassSession::new(eng.clone(), cfg)
        .expect("s16")
        .with_commit_lm(Some(CommitLmConfig::new(lm.clone(), 16, 0.8, 0.6)))
        .expect("lm16");
    assert_eq!(sess16.commit_engine().beam, 16);
    assert_eq!(sess16.provisional_engine().beam, 1);
    sess16.accept_i16(&[0i16; 3_200]).expect("accept");
    let res = sess16.commit_endpoint().expect("commit");
    assert!(res.revised && res.committed.split_whitespace().all(|w| w == "cat"));

    let bad_beam = CommitPassSession::new(eng.clone(), cfg)
        .expect("s")
        .with_commit_lm(Some(CommitLmConfig::new(lm, 1, 0.8, 0.6)))
        .expect_err("reject beam 1");
    assert!(matches!(bad_beam, EngineError::Shape(_)));
    let wrong_v = Arc::new(NgramLm::from_arpa_str(ARPA_PREFER_CAT, None, 8, false).expect("v8"));
    let bad_v = CommitPassSession::new(eng, cfg)
        .expect("s")
        .with_commit_lm(Some(CommitLmConfig::new(wrong_v, 4, 0.8, 0.6)))
        .expect_err("reject vocab");
    assert!(matches!(bad_v, EngineError::Shape(_)));
}
