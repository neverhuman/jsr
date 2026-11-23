//! Unit and integration tests for the H45 endpoint commit pass (`ultra_engine::commit`).

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::commit::{
    combine_ctc_rnnt_scores, ctc_forward_logp, ctc_rescore_text, mbr_select, pad_tail_silence,
    word_edit_distance, CommitConfig, CommitEvent, CommitPassSession, CtcVocab,
    DEFAULT_TAIL_PAD_SAMPLES,
};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::npz::load_npz_audio_only;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_owned).collect()
}

fn synthetic_model(bias_token: usize) -> Arc<Model> {
    let (d, heads, dh, ffn, kernel, stem_ch, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (vocab, pred_emb, joiner) = (4, 4, 8);
    let zlin = |n, k| Linear::from_fp32(&vec![0.0; n * k], n, k, None);
    let norm = |n| Norm {
        g: vec![1.0; n],
        b: vec![0.0; n],
    };
    let block = Block {
        ff1_norm: norm(d),
        ff1_w1: zlin(ffn, d),
        ff1_w2: zlin(d, ffn),
        att_norm: norm(d),
        qkv: zlin(3 * d, d),
        att_out: zlin(d, d),
        conv_norm: norm(d),
        pw1: zlin(2 * d, d),
        dw_w: vec![0.0; kernel * d],
        dw_b: vec![0.0; d],
        dnorm: norm(d),
        pw2: zlin(d, d),
        ff2_norm: norm(d),
        ff2_w1: zlin(ffn, d),
        ff2_w2: zlin(d, ffn),
        post_norm: None,
    };
    let mut j_bias = vec![-2.0f32; vocab];
    j_bias[bias_token] = 2.0;
    let tokens = TokenTable::from_str_checked(
        "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n",
        vocab,
    )
    .expect("tokens");
    Arc::new(Model {
        arch: Arch {
            n_mels: 80,
            d,
            n_blocks: 1,
            lower: 1,
            mid: 0,
            fin: 0,
            heads,
            dh,
            ffn,
            kernel,
            stem_ch,
            f1,
            f2,
            chunk: 4,
            left: 4,
            post_block_norm: false,
            vocab,
            pred_emb,
            joiner,
            blank: 0,
            max_symbols: 2,
            ctx_init: [0, 0],
            log_floor: 1e-6,
            checkpoint_sha256: String::new(),
            weights_sha256: String::new(),
            qat_stem_int8: false,
        },
        root: PathBuf::new(),
        c1_w: vec![0.0; 9 * stem_ch],
        c1_b: vec![0.0; stem_ch],
        c2: zlin(stem_ch, 9 * stem_ch),
        stem_out: zlin(d, f2 * stem_ch),
        blocks: vec![block],
        final_norm: Some(norm(d)),
        pair: None,
        restore: None,
        enc_proj: zlin(joiner, d),
        emb: vec![0.0; vocab * pred_emb],
        pred_proj: zlin(joiner, 2 * pred_emb),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&vec![0.0; vocab * joiner], vocab, joiner, Some(&j_bias)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tokens),
        int8_prepared: None,
    })
}

#[test]
fn commit_mbr_select_prefers_consensus_over_top1_outlier() {
    let cands = vec![
        (words("dog ran away"), -1.00),
        (words("the cat sat"), -1.10),
        (words("the cat sat"), -1.15),
        (words("the cat sat down"), -1.20),
    ];
    assert_eq!(mbr_select(&cands), 1);
    let with_inf = vec![
        (words("the cat sat"), f32::NEG_INFINITY),
        (words("the cat sat"), -3.0),
    ];
    assert_eq!(mbr_select(&with_inf), 1);
    assert_eq!(mbr_select(&[]), 0);
    assert_eq!(mbr_select(&[(words("only"), -0.5)]), 0);
    assert_eq!(
        word_edit_distance(&words("the cat"), &words("a cat sat")),
        2
    );
}

#[test]
fn commit_ctc_retokenizes_separate_vocab_and_penalizes_impossible_paths() {
    let char_vocab = CtcVocab::new(0, vec!["<blk>".into(), "a".into(), "b".into(), " ".into()])
        .expect("char ctc vocab");
    assert_eq!(char_vocab.retokenize_text("a b"), Some(vec![1, 3, 2]));
    assert_eq!(char_vocab.retokenize_text("ab"), Some(vec![1, 2]));
    assert_eq!(char_vocab.retokenize_text("cat"), None);

    // Separate SentencePiece CTC vocab with permuted IDs vs RNN-T.
    let sp_vocab = CtcVocab::new(
        0,
        vec!["<blk>".into(), "\u{2581}sat".into(), "\u{2581}the".into()],
    )
    .expect("sp ctc vocab");
    assert_eq!(sp_vocab.retokenize_text("the sat"), Some(vec![2, 1]));
    assert_eq!(sp_vocab.retokenize_text("the unknown"), None);

    let lp = vec![-10.0, -0.05, -10.0, -10.0, -10.0, -10.0, -0.05, -10.0];
    let good = ctc_rescore_text(&char_vocab, &lp, 2, "ab");
    assert!(good.is_finite() && good > -0.2);

    // Impossible paths MUST receive -inf (infinite cost): OOV, repeated label, too long, blank.
    assert_eq!(
        ctc_rescore_text(&char_vocab, &lp, 2, "z"),
        f32::NEG_INFINITY
    );
    assert_eq!(
        ctc_rescore_text(&char_vocab, &lp, 2, "aa"),
        f32::NEG_INFINITY
    );
    assert_eq!(
        ctc_rescore_text(&char_vocab, &lp, 2, "a b"),
        f32::NEG_INFINITY
    );
    assert_eq!(ctc_forward_logp(&lp, 2, 4, 0, &[0]), f32::NEG_INFINITY);

    assert_eq!(combine_ctc_rnnt_scores(-2.0, f32::NEG_INFINITY, 0.0), -2.0);
    assert_eq!(
        combine_ctc_rnnt_scores(-2.0, f32::NEG_INFINITY, 0.3),
        f32::NEG_INFINITY
    );
    assert!((combine_ctc_rnnt_scores(-2.0, -1.0, 0.5) - (-2.5)).abs() < 1e-6);
}

#[test]
fn commit_tail_pad_silence_invariance_and_session_api() {
    let raw = vec![120i16, -45, 300, 10];
    let padded = pad_tail_silence(&raw, DEFAULT_TAIL_PAD_SAMPLES);
    assert_eq!(padded.len(), raw.len() + DEFAULT_TAIL_PAD_SAMPLES);
    assert_eq!(&padded[..raw.len()], &raw[..]);
    assert!(padded[raw.len()..].iter().all(|&s| s == 0));
    assert_eq!(pad_tail_silence(&raw, 0), raw);

    let eng = Engine::new(synthetic_model(0), Isa::Scalar, Precision::F32).expect("engine");
    assert!(CommitPassSession::new(
        eng.clone(),
        CommitConfig {
            commit_chunk: 0,
            ..CommitConfig::default()
        }
    )
    .is_err());

    let cfg_pad = CommitConfig {
        commit_chunk: 16,
        tail_pad_samples: DEFAULT_TAIL_PAD_SAMPLES,
        beam_size: 2,
        ctc_weight: 0.0,
        mbr_enabled: true,
    };
    let mut sess_pad = CommitPassSession::new(eng.clone(), cfg_pad).expect("session");
    assert!(Arc::ptr_eq(
        &sess_pad.provisional_engine().model,
        &sess_pad.commit_engine().model
    ));
    assert_eq!(sess_pad.provisional_engine().chunk, 4);
    assert_eq!(sess_pad.commit_engine().chunk, 16);

    let pcm = vec![100i16; 3_200];
    let ev = sess_pad.accept_i16(&pcm).expect("accept");
    assert!(
        matches!(ev, CommitEvent::Provisional { audio_ms, .. } if (audio_ms - 200.0).abs() < 1e-6)
    );
    let res_pad = sess_pad.commit_endpoint().expect("commit");
    assert_eq!(sess_pad.buffered_samples(), 0);
    assert!(res_pad.commit_wall_ms >= 0.0);

    let mut sess_pre = CommitPassSession::new(
        eng,
        CommitConfig {
            tail_pad_samples: 0,
            ..cfg_pad
        },
    )
    .expect("pre");
    sess_pre
        .accept_i16(&pad_tail_silence(&pcm, DEFAULT_TAIL_PAD_SAMPLES))
        .expect("accept pre");
    assert_eq!(
        res_pad.committed,
        sess_pre.commit_endpoint().expect("commit pre").committed
    );
}

#[test]
fn commit_preserves_immutable_committed_prefix_and_ctc_rescore() {
    let eng = Engine::new(synthetic_model(1), Isa::Scalar, Precision::F32).expect("engine");
    let mut sess = CommitPassSession::new(
        eng,
        CommitConfig {
            commit_chunk: 16,
            tail_pad_samples: 640,
            beam_size: 2,
            ctc_weight: 0.3,
            mbr_enabled: false,
        },
    )
    .expect("session");

    sess.accept_i16(&[0i16; 1_600]).expect("utt1");
    let r1 = sess.commit_endpoint().expect("commit1");
    assert!(!r1.committed.is_empty());
    assert_eq!(r1.committed_prefix, r1.committed);

    let ev2 = sess.accept_i16(&[0i16; 1_600]).expect("utt2");
    assert!(
        matches!(ev2, CommitEvent::Provisional { ref prefix, .. } if prefix == &r1.committed_prefix)
    );
    let ctc_vocab = CtcVocab::new(0, vec!["<blk>".into(), "\u{2581}the".into()]).expect("ctc");
    let r2 = sess
        .commit_endpoint_with_ctc(&ctc_vocab, &[-0.5f32; 16], 8)
        .expect("commit2");
    assert!(r2.committed_prefix.starts_with(&r1.committed_prefix));
    assert!(r2.committed_prefix.len() > r1.committed_prefix.len());
    assert!(matches!(r2.into_event(), CommitEvent::Committed { .. }));
}

#[test]
fn commit_real_weights_smoke_when_data_drive_present() {
    let (weights, npz) = (common::weights_dir(), common::dev_clean_npz());
    if !common::have(&[&weights, &npz]) {
        return;
    }
    let mut model = Model::load(&weights, false).expect("load");
    model.quantize_int8();
    let eng = Engine::new(Arc::new(model), Isa::detect(), Precision::Int8).expect("engine");
    let clip = load_npz_audio_only(&npz, 0, Some(1))
        .expect("npz")
        .remove(0);
    let mut sess = CommitPassSession::new(eng, CommitConfig::default()).expect("session");
    for pkt in clip.pcm.chunks(320) {
        sess.accept_i16(pkt).expect("accept");
    }
    let res = sess.commit_endpoint().expect("commit");
    assert!(!res.committed.trim().is_empty() && res.commit_wall_ms > 0.0);
}
