//! Integration tests for zero-spike streaming `.ulm` loading and `commit_engine.max_chunks == 4`.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::commit::{run_commit_pcm, CommitConfig, CommitPassSession};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, NgramLm, Precision, Workspace};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn tiny_model() -> Arc<Model> {
    let (d, heads, dh, ffn, kernel, stem_ch, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (vocab, pred_emb, joiner) = (4, 4, 8);
    let lin = |n: usize, k: usize, seed: usize| {
        let w: Vec<f32> = (0..n * k)
            .map(|i| (((i + seed) % 13) as f32 - 6.0) * 0.03)
            .collect();
        Linear::from_fp32(&w, n, k, None)
    };
    let norm = |n| Norm {
        g: vec![1.0; n],
        b: vec![0.0; n],
    };
    let block = Block {
        ff1_norm: norm(d),
        ff1_w1: lin(ffn, d, 1),
        ff1_w2: lin(d, ffn, 2),
        att_norm: norm(d),
        qkv: lin(3 * d, d, 3),
        att_out: lin(d, d, 4),
        conv_norm: norm(d),
        pw1: lin(2 * d, d, 5),
        dw_w: vec![0.05; kernel * d],
        dw_b: vec![0.0; d],
        dnorm: norm(d),
        pw2: lin(d, d, 6),
        ff2_norm: norm(d),
        ff2_w1: lin(ffn, d, 7),
        ff2_w2: lin(d, ffn, 8),
        post_norm: None,
    };
    let j_bias = vec![0.1, 0.3, -0.2, 0.25];
    let j_w: Vec<f32> = (0..vocab * joiner)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
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
            left: 8,
            post_block_norm: false,
            vocab,
            pred_emb,
            joiner,
            blank: 0,
            max_symbols: 2,
            ctx_init: vec![0, 0],
            log_floor: 1e-6,
            checkpoint_sha256: String::new(),
            weights_sha256: String::new(),
            qat_stem_int8: false,
        },
        root: PathBuf::new(),
        c1_w: vec![0.02; 9 * stem_ch],
        c1_b: vec![0.0; stem_ch],
        c2: lin(stem_ch, 9 * stem_ch, 9),
        stem_out: lin(d, f2 * stem_ch, 10),
        blocks: vec![block],
        final_norm: Some(norm(d)),
        pair: None,
        restore: None,
        enc_proj: lin(joiner, d, 11),
        emb: vec![0.04; vocab * pred_emb],
        pred_proj: lin(joiner, 2 * pred_emb, 12),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&j_w, vocab, joiner, Some(&j_bias)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tokens),
        ctc_head: None,
        int8_prepared: None,
    })
}

#[test]
fn commit_engine_defaults_to_max_chunks_4_and_matches_max_chunks_1_bit_for_bit() {
    let model = tiny_model();
    let eng_default = Engine::new(model.clone(), Isa::detect(), Precision::F32).expect("eng");
    assert_eq!(eng_default.max_chunks, 1);

    let cfg = CommitConfig {
        commit_chunk: 8,
        tail_pad_samples: 1600,
        beam_size: 4,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassSession::new(eng_default.clone(), cfg).expect("sess");
    assert_eq!(sess.commit_engine().max_chunks, 4);
    assert_eq!(sess.provisional_engine().max_chunks, 1);

    // Caller override > 4 is preserved.
    let sess_hi = CommitPassSession::new(eng_default.clone().with_max_chunks(8), cfg).expect("hi");
    assert_eq!(sess_hi.commit_engine().max_chunks, 8);

    // Multi-chunk utterance (> 4 commit chunks = > 32 frames = > 1.28 s) produces identical output
    // for max_chunks=4 vs explicit max_chunks=1.
    let pcm: Vec<i16> = (0..32_000)
        .map(|i| (((i * 37) % 1000) as i16 - 500).saturating_mul(8))
        .collect();
    for pkt in pcm.chunks(320) {
        sess.accept_i16(pkt).expect("accept");
    }
    let res_mc4 = sess.commit_endpoint().expect("commit mc4");

    let ce_mc1 = eng_default
        .with_chunk(cfg.commit_chunk)
        .expect("chunk")
        .with_max_chunks(1)
        .with_beam(cfg.beam_size);
    let mut ws = Workspace::default();
    let (txt_mc1, idx_mc1, _) = run_commit_pcm(&ce_mc1, cfg, &pcm, &mut ws).expect("commit mc1");
    assert_eq!(res_mc4.committed, txt_mc1);
    assert_eq!(res_mc4.selected_hyp, idx_mc1);
}

#[test]
fn ngram_load_and_load_streaming_are_identical_across_v1_and_v2() {
    let arpa = "\\data\\\nngram 1=4\nngram 2=2\n\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\t-0.2\n-1.0\t1\t-0.1\n-1.5\t2\n\n\\2-grams:\n-0.5\t<s> 1\n-0.7\t1 2\n\n\\end\\\n";
    let lm = NgramLm::from_arpa_str(arpa, Some(arpa), 4, true).expect("build");
    let p = std::env::temp_dir().join(format!("ultra_stream_int_{}.ulm", std::process::id()));
    lm.save(&p).expect("save v2");
    let a = NgramLm::load(&p).expect("load");
    let b = NgramLm::load_streaming(&p).expect("load_streaming");
    assert_eq!(a.memory_bytes(), b.memory_bytes());
    let ca = a.ctx_state(&[1]);
    let cb = b.ctx_state(&[1]);
    for k in 1..4 {
        assert_eq!(a.lm_logp(&ca, k).to_bits(), b.lm_logp(&cb, k).to_bits());
        assert_eq!(a.lodr_logp(&ca, k).to_bits(), b.lodr_logp(&cb, k).to_bits());
    }
    let _ = std::fs::remove_file(&p);
}
