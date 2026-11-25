//! Unit & parity tests for Wave 1B: Exact Stateless-Predictor ILME, FfnnLm / M5+FFNN
//! interpolation, bit-identical Ngram+LODR beam parity, and acoustic-confidence margin gating.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::beam::{beam_chunk_with_lm, BeamState};
use ultra_engine::commit::lm_gate::{
    acoustic_top1_margin, decode_enc_gated, CommitTokenLm, InternalLm, LmGateConfig,
};
use ultra_engine::decoder::DecWorkspace;
use ultra_engine::lm::{FfnnActivation, FfnnLm, LmConfig, NgramLm, StatelessIlme};
use ultra_engine::model::{Arch, Linear};
use ultra_engine::{Isa, Model, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn synthetic_model(vocab: usize, pred_emb: usize, joiner: usize) -> Model {
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
        stem_ch: 4,
        f1: 39,
        f2: 19,
        chunk: 4,
        left: 16,
        post_block_norm: false,
        vocab,
        pred_emb,
        joiner,
        blank: 0,
        max_symbols: 4,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: String::new(),
        weights_sha256: "test_weights_sha".into(),
        qat_stem_int8: false,
    };
    let mut tok_str = String::from("<blk> 0\n");
    for i in 1..vocab {
        tok_str.push_str(&format!("t{i} {i}\n"));
    }
    let tokens = Arc::new(TokenTable::from_str_checked(&tok_str, vocab).expect("tokens"));
    let cmvn = GlobalCmvn::new(&[0.0; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(
        TrainerMelConfig::from_fbank_le_bytes(&vec![0u8; 257 * 80 * 4], 1e-6, cmvn).expect("mel"),
    );
    let emb: Vec<f32> = (0..vocab * pred_emb)
        .map(|i| (((i * 17 + 3) % 23) as f32 - 11.0) * 0.05)
        .collect();
    let pred_w: Vec<f32> = (0..joiner * 2 * pred_emb)
        .map(|i| (((i * 13 + 5) % 19) as f32 - 9.0) * 0.04)
        .collect();
    let pred_b: Vec<f32> = (0..joiner).map(|i| (i as f32 - 3.0) * 0.02).collect();
    let join_w: Vec<f32> = (0..vocab * joiner)
        .map(|i| (((i * 29 + 7) % 31) as f32 - 15.0) * 0.06)
        .collect();
    let join_b: Vec<f32> = (0..vocab)
        .map(|i| if i == 0 { 0.3 } else { -(i as f32) * 0.1 })
        .collect();
    let d = Linear::from_fp32(&[0.0], 1, 1, None);
    Model {
        arch,
        root: PathBuf::new(),
        c1_w: vec![],
        c1_b: vec![],
        c2: d.clone(),
        stem_out: d.clone(),
        blocks: vec![],
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: d,
        ctc_head: None,
        emb,
        pred_proj: Linear::from_fp32(&pred_w, joiner, 2 * pred_emb, Some(&pred_b)),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&join_w, vocab, joiner, Some(&join_b)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![],
        mel_cfg,
        tokens,
        assets: Default::default(),
        int8_prepared: None,
    }
}

fn sample_ngram(vocab: usize) -> NgramLm {
    let lm = "\\data\\\nngram 1=6\nngram 2=4\n\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\t-0.2\n\
              -0.7\t1\t-0.1\n-0.9\t2\t-0.1\n-1.1\t3\t-0.1\n-1.3\t4\t-0.1\n\n\\2-grams:\n\
              -0.3\t<s> 1\n-0.4\t1 2\n-0.35\t2 3\n-0.5\t3 4\n\n\\end\\\n";
    let lodr = "\\data\\\nngram 1=5\nngram 2=2\n\n\\1-grams:\n-3.5\t<unk>\n-99.0\t<s>\t-0.1\n\
                -0.8\t1\t-0.1\n-1.0\t2\t-0.1\n-1.2\t3\n\n\\2-grams:\n-0.6\t<s> 1\n-0.5\t1 2\n\n\\end\\\n";
    NgramLm::from_arpa_str(lm, Some(lodr), vocab, false).expect("ngram")
}

#[test]
fn stateless_ilme_matches_reference_formula_and_roundtrips_json() {
    let m = synthetic_model(6, 4, 8);
    let e_avg: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.1).collect();
    let ilme = StatelessIlme::from_model(&m, e_avg.clone()).expect("ilme");
    // Exact Python reference vectors from python/ultra/train/ilme.py::stateless_ilme_logprobs.
    let py_ref = [
        (
            (0, 0),
            [-1.988015f32, -0.800832, -2.193847, -1.551907, -2.399679],
        ),
        (
            (0, 2),
            [-1.930854, -0.834761, -2.138079, -1.573596, -2.345303],
        ),
        (
            (2, 5),
            [-2.125161, -0.710571, -2.353228, -1.521039, -2.581294],
        ),
    ];
    for ((p0, p1), expected) in py_ref {
        let mut out = vec![0.0f32; 6];
        ilme.score_ctx_isa(Isa::Scalar, p0, p1, &mut out);
        assert_eq!(out[0], f32::NEG_INFINITY);
        let max_diff = out[1..]
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Python-vs-Rust ILME parity diff {max_diff} >= 1e-5"
        );
        let prob_sum: f32 = out[1..].iter().map(|&lp| lp.exp()).sum();
        assert!((prob_sum - 1.0).abs() < 1e-5, "non-blank sum {prob_sum}");
    }
    let tmp = std::env::temp_dir().join(format!("ultra_ilme_test_{}.json", std::process::id()));
    StatelessIlme::save_eavg_json(&tmp, &e_avg, 100, 420, "test_weights_sha").expect("save");
    let loaded = StatelessIlme::from_eavg_file(&m, &tmp).expect("load");
    let (mut a, mut b) = (vec![0.0f32; 6], vec![0.0f32; 6]);
    ilme.score_ctx_isa(Isa::Scalar, 1, 3, &mut a);
    loaded.score_ctx_isa(Isa::Scalar, 1, 3, &mut b);
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let bad = std::fs::read_to_string(&tmp)
        .expect("read")
        .replace("0.05", "0.99");
    std::fs::write(&tmp, bad).expect("tamper");
    assert!(StatelessIlme::from_eavg_file(&m, &tmp).is_err());
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn gated_decoder_matches_beam_chunk_with_lm_and_supports_ffnn_interp_ilme() {
    let m = synthetic_model(6, 4, 8);
    let ng = Arc::new(sample_ngram(6));
    let ff = Arc::new(
        FfnnLm::from_parts(
            6,
            2,
            2,
            2,
            true,
            FfnnActivation::Silu,
            vec![10; 12],
            vec![0.05; 6],
            vec![5; 8],
            vec![0.05; 2],
            vec![0.0; 2],
            Vec::new(),
            Vec::new(),
            vec![0.0, 0.3, 0.2, 0.1, -0.1, -0.2],
        )
        .expect("ffnn"),
    );
    let enc: Vec<f32> = (0..6 * 8)
        .map(|i| ((i * 11 % 17) as f32 - 8.0) * 0.25)
        .collect();
    let mut bs = BeamState::new(&m);
    let mut ws = DecWorkspace::default();
    let lm_cfg = LmConfig {
        lm_weight: 0.8,
        lodr_weight: 0.6,
        blank_penalty: 0.0,
    };
    beam_chunk_with_lm(
        &m,
        Isa::Scalar,
        Precision::F32,
        4,
        1,
        0.0,
        Some((&ng, lm_cfg)),
        &mut [&mut bs],
        &[6],
        &enc,
        &mut ws,
    )
    .expect("ref");
    let gate_cfg = LmGateConfig {
        beam: 4,
        lm_scale: 0.8,
        ilm_scale: 0.6,
        token_bonus: 0.0,
        max_sym_per_frame: 1,
        conf_margin: 0.0,
    };
    let got = decode_enc_gated(
        &m,
        Isa::Scalar,
        Precision::F32,
        &enc,
        6,
        Some(&CommitTokenLm::Ngram(ng.clone())),
        &InternalLm::Lodr(ng.clone()),
        &gate_cfg,
    )
    .expect("gated");
    assert_eq!(got, bs.best());

    let ilme = Arc::new(StatelessIlme::from_zero_enc(&m).expect("zero ilme"));
    let interp = CommitTokenLm::Interpolated {
        ngram: ng,
        ffnn: ff,
        alpha: 0.5,
    };
    let out_interp = decode_enc_gated(
        &m,
        Isa::Scalar,
        Precision::F32,
        &enc,
        6,
        Some(&interp),
        &InternalLm::StatelessIlme(ilme),
        &gate_cfg,
    )
    .expect("interp");
    assert!(out_interp.iter().all(|&t| (t as usize) < 6 && t != 0));
}

#[test]
fn gated_decoder_matches_beam_chunk_with_lm_int8() {
    let mut m = synthetic_model(6, 4, 8);
    m.quantize_int8();
    let ng = Arc::new(sample_ngram(6));
    let enc: Vec<f32> = (0..6 * 8)
        .map(|i| ((i * 11 % 17) as f32 - 8.0) * 0.25)
        .collect();
    let lm_cfg = LmConfig {
        lm_weight: 0.8,
        lodr_weight: 0.6,
        blank_penalty: 0.0,
    };
    let gate_cfg = LmGateConfig {
        beam: 4,
        lm_scale: 0.8,
        ilm_scale: 0.6,
        token_bonus: 0.0,
        max_sym_per_frame: 1,
        conf_margin: 0.0,
    };
    for isa in [Isa::Scalar, Isa::detect()] {
        let mut bs = BeamState::new(&m);
        let mut ws = DecWorkspace::default();
        beam_chunk_with_lm(
            &m,
            isa,
            Precision::Int8,
            4,
            1,
            0.0,
            Some((&ng, lm_cfg)),
            &mut [&mut bs],
            &[6],
            &enc,
            &mut ws,
        )
        .expect("int8 beam ref");
        let got = decode_enc_gated(
            &m,
            isa,
            Precision::Int8,
            &enc,
            6,
            Some(&CommitTokenLm::Ngram(ng.clone())),
            &InternalLm::Lodr(ng.clone()),
            &gate_cfg,
        )
        .expect("int8 gated");
        assert_eq!(got, bs.best(), "INT8 parity mismatch on {isa:?}");
    }
}

#[test]
fn acoustic_confidence_margin_gate_blocks_adversarial_lm_override() {
    let mut m = synthetic_model(5, 4, 8);
    let strong_b = [-2.0f32, 0.0, -3.0, -3.0, 6.5];
    m.joiner_out = Linear::from_fp32(&[0.0f32; 5 * 8], 5, 8, Some(&strong_b));
    let (k1, margin) = acoustic_top1_margin(&strong_b);
    assert_eq!(k1, 4);
    assert!((margin - 6.5).abs() < 1e-6);

    let adv_arpa = "\\data\\\nngram 1=6\n\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\n-0.1\t1\n-4.0\t2\n-4.0\t3\n-5.0\t4\n\n\\end\\\n";
    let adv_lm = Arc::new(NgramLm::from_arpa_str(adv_arpa, None, 5, false).expect("adv"));
    let enc = vec![0.0f32; 8];
    let ungated = LmGateConfig {
        beam: 4,
        lm_scale: 1.0,
        ilm_scale: 0.0,
        token_bonus: 0.0,
        max_sym_per_frame: 1,
        conf_margin: 0.0,
    };
    assert_eq!(
        decode_enc_gated(
            &m,
            Isa::Scalar,
            Precision::F32,
            &enc,
            1,
            Some(&CommitTokenLm::Ngram(adv_lm.clone())),
            &InternalLm::None,
            &ungated
        )
        .expect("ungated"),
        vec![1]
    );
    let gated = LmGateConfig {
        conf_margin: 4.0,
        ..ungated
    };
    assert_eq!(
        decode_enc_gated(
            &m,
            Isa::Scalar,
            Precision::F32,
            &enc,
            1,
            Some(&CommitTokenLm::Ngram(adv_lm)),
            &InternalLm::None,
            &gated
        )
        .expect("gated"),
        vec![4]
    );
}
