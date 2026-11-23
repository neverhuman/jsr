//! Integration tests for opt-in D4 Transformer rescoring in `CommitPassSession`.

use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use ultra_engine::commit::{
    CommitConfig, CommitLmConfig, CommitPassSession, CommitTfRescoreConfig, DEFAULT_RESCORE_M5_SUB,
    DEFAULT_RESCORE_WEIGHT,
};
use ultra_engine::lm::{NgramLm, TfDirection, TfRescorer, TF_MAGIC, TF_VERSION};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn synthetic_commit_model() -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2, vocab, pe, jn) = (8, 2, 4, 16, 3, 4, 39, 19, 4, 4, 8);
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
    let tok = TokenTable::from_str_checked(
        "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n",
        vocab,
    )
    .expect("tok");
    Arc::new(Model {
        arch: Arch {
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
        },
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
        joiner_out: Linear::from_fp32(
            &vec![0.0; vocab * jn],
            vocab,
            jn,
            Some(&[-2.0, 2.0, 1.5, -4.0]),
        ),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        ctc_head: None,
        int8_prepared: None,
    })
}

fn tiny_utfr(dir: TfDirection) -> Arc<TfRescorer> {
    let (v, d, f) = (4usize, 8usize, 16usize);
    let mut buf = Vec::new();
    buf.extend_from_slice(TF_MAGIC);
    let hdr = [
        TF_VERSION,
        v as u32,
        d as u32,
        1u32,
        2u32,
        f as u32,
        16u32,
        u32::from(dir == TfDirection::R2l),
        1u32,
        10000.0f32.to_bits(),
        1e-6f32.to_bits(),
    ];
    hdr.iter()
        .for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
    let push_f32 = |b: &mut Vec<u8>, xs: &[f32]| {
        xs.iter()
            .for_each(|x| b.extend_from_slice(&x.to_le_bytes()))
    };
    let mut emb = vec![0.0f32; v * d];
    emb[2 * d] = 1.0; // bias output head gently toward token 2
    push_f32(&mut buf, &emb);
    push_f32(&mut buf, &vec![1.0f32; d]); // n1
    buf.extend(vec![0u8; 3 * d * d]);
    push_f32(&mut buf, &vec![0.0f32; 3 * d]);
    buf.extend(vec![0u8; d * d]);
    push_f32(&mut buf, &vec![0.0f32; d]);
    push_f32(&mut buf, &vec![1.0f32; d]); // n2
    buf.extend(vec![0u8; 2 * f * d]);
    push_f32(&mut buf, &vec![0.0f32; 2 * f]);
    buf.extend(vec![0u8; d * f]);
    push_f32(&mut buf, &vec![0.0f32; d]);
    push_f32(&mut buf, &vec![1.0f32; d]); // final norm
    buf.extend_from_slice(&Sha256::digest(&buf));
    Arc::new(TfRescorer::from_bytes(&buf).expect("tiny utfr"))
}

#[test]
fn commit_session_tf_rescorer_off_is_bit_identical_and_opt_in_runs() {
    let eng = Engine::new(synthetic_commit_model(), Isa::Scalar, Precision::F32).expect("eng");
    let arpa = "\\data\\\nngram 1=4\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\t-0.1\n-1.0\t1\n-1.0\t2\n\\end\\\n";
    let lm = Arc::new(NgramLm::from_arpa_str(arpa, None, 4, false).expect("lm"));
    let cfg = CommitConfig {
        tail_pad_samples: 0,
        ..CommitConfig::default()
    };
    let mut base = CommitPassSession::new(eng.clone(), cfg)
        .expect("b")
        .with_commit_lm(Some(CommitLmConfig::new(lm.clone(), 4, 0.8, 0.0)))
        .expect("blm");
    let mut off = CommitPassSession::new(eng.clone(), cfg)
        .expect("o")
        .with_commit_lm(Some(CommitLmConfig::new(lm.clone(), 4, 0.8, 0.0)))
        .expect("olm")
        .with_commit_tf_rescorer(None)
        .expect("off");
    assert!(!off.has_commit_tf_rescorer());
    base.accept_i16(&[0i16; 3_200]).expect("ba");
    off.accept_i16(&[0i16; 3_200]).expect("oa");
    let (rb, ro) = (
        base.commit_endpoint().expect("bc"),
        off.commit_endpoint().expect("oc"),
    );
    assert_eq!(
        (rb.provisional, rb.committed, rb.revised, rb.selected_hyp),
        (ro.provisional, ro.committed, ro.revised, ro.selected_hyp)
    );

    assert!(CommitTfRescoreConfig::from_files(
        None,
        None,
        DEFAULT_RESCORE_WEIGHT,
        DEFAULT_RESCORE_M5_SUB
    )
    .expect("none")
    .is_none());

    let l2r = tiny_utfr(TfDirection::L2r);
    let r2l = tiny_utfr(TfDirection::R2l);
    assert!(CommitTfRescoreConfig::new(Some(r2l.clone()), None, 0.38, 0.50).is_err());
    assert!(CommitTfRescoreConfig::new(None, Some(l2r.clone()), 0.38, 0.50).is_err());

    let tf_cfg = CommitTfRescoreConfig::new(Some(l2r), Some(r2l), 0.38, 0.50)
        .expect("valid")
        .with_r2l_top_k(2);
    let mut on = CommitPassSession::new(eng, cfg)
        .expect("on")
        .with_commit_lm(Some(CommitLmConfig::new(lm, 4, 0.8, 0.0)))
        .expect("onlm")
        .with_commit_tf_rescorer(Some(tf_cfg))
        .expect("ontf");
    assert!(on.has_commit_tf_rescorer());
    on.accept_i16(&[0i16; 3_200]).expect("ona");
    let ron = on.commit_endpoint().expect("onc");
    assert!(!ron.committed.is_empty());
}
