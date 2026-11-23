//! H55 Wave 1A tests: optional `ctc_head`, duplicate-segmentation `logaddexp` aggregation,
//! temperature-scaled word-MBR, `beam=32` validation, and bit-exact default preservation.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::commit::{
    aggregate_duplicate_hyps, mbr_select, mbr_select_with_temp, CommitConfig, CommitPassSession,
};
use ultra_engine::model::{Arch, Block, Linear, Norm};
use ultra_engine::{Engine, Isa, Model, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn tiny_model(ctc_head: Option<Linear>) -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2, v, pe, jn) = (4, 1, 4, 8, 3, 2, 39, 19, 5, 4, 4);
    let z = |n, kk| Linear::from_fp32(&vec![0.0; n * kk], n, kk, None);
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
    let tok = TokenTable::from_str_checked("<blk> 0\n\u{2581}A 1\n\u{2581}B 2\nA 3\nB 4\n", v)
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
        joiner_out: z(v, jn),
        ctc_head,
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        int8_prepared: None,
    })
}

#[test]
fn duplicate_bpe_segmentations_aggregate_via_logaddexp_before_mbr() {
    // Two distinct BPE segmentations produce "THE CAT" with -1.1 each -> logaddexp = -1.1 + ln(2) = -0.407,
    // beating single hypothesis "THE DOG" at -0.8.
    let hyps = vec![
        ("THE DOG".to_owned(), -0.8f32),
        ("THE CAT".to_owned(), -1.1f32),
        ("THE  CAT ".to_owned(), -1.1f32),
    ];
    let merged = aggregate_duplicate_hyps(&hyps);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].0, vec!["THE", "DOG"]);
    assert_eq!(merged[1].0, vec!["THE", "CAT"]);
    let expected = (-1.1f64 + 2.0f64.ln()) as f32;
    assert!((merged[1].2 - expected).abs() < 1e-5);
    assert_eq!(merged[1].3, 1);
    let cands: Vec<(Vec<String>, f32)> =
        merged.iter().map(|(w, _, s, _)| (w.clone(), *s)).collect();
    assert_eq!(mbr_select(&cands), 1);
}

#[test]
fn mbr_temperature_scaling_controls_posterior_sharpness() {
    let w = |s: &str| s.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
    // Top hyp 0 ("X Y Z") has slightly higher score (-1.0), while 3 consensus hyps near "A B C"
    // sit at -1.5, -1.6, -1.7. Sharp temp=4.0 picks hyp 0; soft temp=0.5 picks consensus hyp 1.
    let cands = vec![
        (w("X Y Z"), -1.0f32),
        (w("A B C"), -1.5f32),
        (w("A B D"), -1.6f32),
        (w("A E C"), -1.7f32),
    ];
    assert_eq!(mbr_select_with_temp(&cands, 4.0), 0);
    assert_eq!(mbr_select_with_temp(&cands, 0.5), 1);
    assert_eq!(mbr_select(&cands), mbr_select_with_temp(&cands, 1.0));
}

#[test]
fn ctc_log_probs_exact_and_optional_ctc_head_behavior() {
    let w = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 2.0, 0.0, 0.0, // row 1
        0.0, 0.0, -1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 0.5, // row 3
        0.5, 0.5, 0.5, 0.5, // row 4
    ];
    let b = vec![0.1, -0.2, 0.3, 0.0, -0.5];
    let head = Linear::from_fp32(&w, 5, 4, Some(&b));
    let eng_with = Engine::new(tiny_model(Some(head)), Isa::Scalar, Precision::F32).expect("eng");
    let enc = vec![1.0, 2.0, -1.0, 0.5];
    let lp = eng_with.ctc_log_probs(&enc, 1).expect("ctc_lp");
    assert_eq!(lp.len(), 5);
    let raw = [1.1f64, 3.8, 1.3, 0.25, 0.75];
    let mx = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = mx + raw.iter().map(|&x| (x - mx).exp()).sum::<f64>().ln();
    for (i, &got) in lp.iter().enumerate() {
        let want = (raw[i] - lse) as f32;
        assert!((got - want).abs() < 1e-5, "row {i}: {got} vs {want}");
    }

    let eng_without = Engine::new(tiny_model(None), Isa::Scalar, Precision::F32).expect("eng");
    assert!(eng_without.ctc_log_probs(&enc, 1).is_err());

    let cfg = CommitConfig {
        beam_size: 32,
        ctc_weight: 0.2,
        mbr_enabled: true,
        mbr_temp: 1.5,
        ..CommitConfig::default()
    };
    assert!(cfg.validate().is_ok());
    let mut sess_no_head = CommitPassSession::new(eng_without, cfg).expect("sess");
    sess_no_head.accept_i16(&vec![0i16; 3200]).expect("acc");
    assert!(
        !sess_no_head
            .commit_endpoint()
            .expect("no-head fallback")
            .pcm_truncated
    );

    let mut sess_ok = CommitPassSession::new(eng_with, cfg).expect("sess");
    sess_ok.accept_i16(&vec![0i16; 3200]).expect("acc");
    let res = sess_ok.commit_endpoint().expect("commit with ctc_head");
    assert!(!res.pcm_truncated);
}
