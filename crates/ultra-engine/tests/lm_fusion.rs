//! Integration tests for pure-Rust token n-gram shallow fusion + LODR inside beam search.
//!
//! Verifies:
//! 1. Exact equivalence between branch-and-bound pruned beam search and brute-force full-vocab
//!    reference scoring across multi-frame sequences with `blank_penalty`, `lm_weight`, and
//!    `lodr_weight` (both exact fp32 and 8-bit quantized LMs).
//! 2. `lm = None` is bit-identical to default beam search.
//! 3. Batch invariance (`B = 1` vs `B = 4` mixed lengths) with Micro-Beam-4 + LM + LODR active.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::beam::{beam_chunk_with_lm, beam_chunk_with_opts, BeamState};
use ultra_engine::decoder::DecWorkspace;
use ultra_engine::encoder::Precision;
use ultra_engine::lm::{LmConfig, NgramLm};
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::Isa;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const ARPA_LM: &str = "\
\\data\\
ngram 1=7
ngram 2=8
ngram 3=4

\\1-grams:
-4.5\t<unk>
-99.0\t<s>\t-0.25
-1.1\t2\t-0.20
-1.3\t3\t-0.30
-1.7\t4\t-0.15
-2.0\t5\t-0.10
-2.2\t6\t-0.05

\\2-grams:
-0.3\t<s> 2\t-0.12
-1.4\t<s> 3\t-0.08
-0.2\t2 3\t-0.18
-0.9\t2 4\t-0.06
-0.4\t3 4\t-0.22
-0.8\t3 5\t-0.11
-0.5\t4 5\t-0.07
-0.6\t5 6

\\3-grams:
-0.1\t<s> 2 3
-0.15\t2 3 4
-0.25\t3 4 5
-0.2\t4 5 6

\\end\\
";

const ARPA_LODR: &str = "\
\\data\\
ngram 1=7
ngram 2=5

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.15
-1.2\t2\t-0.10
-1.5\t3\t-0.20
-1.6\t4\t-0.05
-1.9\t5\t-0.05
-2.1\t6

\\2-grams:
-0.9\t<s> 2
-0.8\t2 3
-0.7\t3 4
-0.6\t4 5
-0.5\t5 6

\\end\\
";

struct TinyWeights {
    model: Model,
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    join_w: Vec<f32>,
    join_b: Vec<f32>,
}

/// Build a tiny deterministic `Model` with predictor + joiner weights (vocab=8, joiner=4).
fn tiny_model() -> TinyWeights {
    let vocab = 8usize;
    let joiner = 4usize;
    let pred_emb = 4usize;

    let emb: Vec<f32> = (0..vocab * pred_emb)
        .map(|i| ((i as f32 * 0.37).sin()) * 0.4)
        .collect();
    let pred_w: Vec<f32> = (0..joiner * (2 * pred_emb))
        .map(|i| ((i as f32 * 0.53).cos()) * 0.35)
        .collect();
    let pred_b: Vec<f32> = (0..joiner).map(|i| (i as f32 - 1.5) * 0.1).collect();

    let join_w: Vec<f32> = (0..vocab * joiner)
        .map(|i| ((i as f32 * 0.71).sin()) * 0.6)
        .collect();
    let join_b: Vec<f32> = (0..vocab)
        .map(|k| {
            if k == 0 {
                0.4
            } else {
                -0.2 + (k as f32) * 0.05
            }
        })
        .collect();

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
        max_symbols: 1,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: String::new(),
        weights_sha256: String::new(),
        qat_stem_int8: false,
    };

    let tok_str = "<blk> 0\n<unk> 1\na 2\nb 3\nc 4\nd 5\ne 6\nf 7\n";
    let tokens = Arc::new(TokenTable::from_str_checked(tok_str, vocab).expect("tokens"));
    let cmvn = GlobalCmvn::new(&[0.0; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(
        TrainerMelConfig::from_fbank_le_bytes(&vec![0u8; 257 * 80 * 4], 1e-6, cmvn)
            .expect("mel_cfg"),
    );
    let dummy = Linear::from_fp32(&[0.0], 1, 1, None);
    let model = Model {
        arch,
        root: PathBuf::new(),
        c1_w: Vec::new(),
        c1_b: Vec::new(),
        c2: dummy.clone(),
        stem_out: dummy.clone(),
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: dummy,
        emb,
        pred_proj: Linear::from_fp32(&pred_w, joiner, 2 * pred_emb, Some(&pred_b)),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&join_w, vocab, joiner, Some(&join_b)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: Vec::new(),
        mel_cfg,
        tokens,
        ctc_head: None,
        int8_prepared: None,
    };
    TinyWeights {
        model,
        pred_w,
        pred_b,
        join_w,
        join_b,
    }
}

/// Brute-force full-vocab Python-style `modified_beam_search` reference (zero pruning).
fn ref_beam_with_lm(
    tw: &TinyWeights,
    enc: &[f32],
    t_len: usize,
    beam: usize,
    lm: &NgramLm,
    cfg: LmConfig,
) -> Vec<u32> {
    let m = &tw.model;
    let (jn, v, blank, e) = (m.arch.joiner, m.arch.vocab, m.arch.blank, m.arch.pred_emb);
    let mut hyps: Vec<(Vec<u32>, f32)> = vec![(Vec::new(), 0.0f32)];
    for t in 0..t_len {
        let ev = &enc[t * jn..(t + 1) * jn];
        let mut all_cands: Vec<(f32, usize, usize)> = Vec::new();
        for (h, (toks, base)) in hyps.iter().enumerate() {
            let n = toks.len();
            let c0 = if n > 1 { toks[n - 2] as usize } else { blank };
            let c1 = if n > 0 { toks[n - 1] as usize } else { blank };
            let mut pred_in = vec![0.0f32; 2 * e];
            pred_in[..e].copy_from_slice(&m.emb[c0 * e..(c0 + 1) * e]);
            pred_in[e..].copy_from_slice(&m.emb[c1 * e..(c1 + 1) * e]);
            let mut pred = vec![0.0f32; jn];
            for (row, p_slot) in pred.iter_mut().enumerate() {
                let mut s = tw.pred_b[row];
                for (col, &pv) in pred_in.iter().enumerate() {
                    s += tw.pred_w[row * (2 * e) + col] * pv;
                }
                *p_slot = s;
            }
            let hid: Vec<f32> = (0..jn).map(|j| (ev[j] + pred[j]).tanh()).collect();
            let mut logits = vec![0.0f32; v];
            for (k, l_slot) in logits.iter_mut().enumerate() {
                let mut s = tw.join_b[k];
                for (j, &hv) in hid.iter().enumerate() {
                    s += tw.join_w[k * jn + j] * hv;
                }
                *l_slot = s;
            }
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + logits.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
            let ctx = lm.ctx_state(toks);
            for (k, &x) in logits.iter().enumerate() {
                let lp = x - lse;
                let sc = if k == blank {
                    base + lp - cfg.blank_penalty
                } else {
                    base + lp + cfg.lm_weight * lm.lm_logp(&ctx, k)
                        - cfg.lodr_weight * lm.lodr_logp(&ctx, k)
                };
                all_cands.push((sc, h, k));
            }
        }
        all_cands.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        all_cands.truncate(beam);

        let mut merged: Vec<(Vec<u32>, f32)> = Vec::new();
        let mut idx_map: HashMap<Vec<u32>, usize> = HashMap::new();
        for (sc, h, k) in all_cands {
            let mut seq = hyps[h].0.clone();
            if k != blank {
                seq.push(k as u32);
            }
            if let Some(&pos) = idx_map.get(&seq) {
                let a = merged[pos].1;
                let mx = a.max(sc);
                merged[pos].1 = mx + (-(a - sc).abs()).exp().ln_1p();
            } else {
                idx_map.insert(seq.clone(), merged.len());
                merged.push((seq, sc));
            }
        }
        hyps = merged;
    }
    hyps.into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(t, _)| t)
        .unwrap_or_default()
}

#[test]
fn pruned_beam_matches_bruteforce_reference_exactly() {
    let tw = tiny_model();
    let m = &tw.model;
    let lm_f32 = NgramLm::from_arpa_str(ARPA_LM, Some(ARPA_LODR), 8, false).expect("f32 lm");
    let lm_q8 = NgramLm::from_arpa_str(ARPA_LM, Some(ARPA_LODR), 8, true).expect("q8 lm");
    let cfg = LmConfig {
        lm_weight: 0.6,
        lodr_weight: 0.3,
        blank_penalty: 0.25,
    };
    let t_len = 24usize;
    let enc: Vec<f32> = (0..t_len * m.arch.joiner)
        .map(|i| ((i as f32 * 0.41).sin() + (i as f32 * 0.19).cos()) * 1.1)
        .collect();

    // 1. None LM is bit-identical to beam_chunk_with_opts
    let mut s_opts = BeamState::new(m);
    let mut s_none = BeamState::new(m);
    let mut ws = DecWorkspace::default();
    beam_chunk_with_opts(
        m,
        Isa::Scalar,
        Precision::F32,
        4,
        1,
        0.0,
        &mut [&mut s_opts],
        &[t_len],
        &enc,
        &mut ws,
    )
    .expect("opts");
    beam_chunk_with_lm(
        m,
        Isa::Scalar,
        Precision::F32,
        4,
        1,
        0.0,
        None,
        &mut [&mut s_none],
        &[t_len],
        &enc,
        &mut ws,
    )
    .expect("none");
    assert_eq!(s_opts.best(), s_none.best());

    // 2. Branch-and-bound pruned beam == brute-force full-vocab reference (both fp32 & q8)
    for lm in [&lm_f32, &lm_q8] {
        let expected = ref_beam_with_lm(&tw, &enc, t_len, 4, lm, cfg);
        let mut st = BeamState::new(m);
        let mut ws = DecWorkspace::default();
        beam_chunk_with_lm(
            m,
            Isa::Scalar,
            Precision::F32,
            4,
            1,
            0.0,
            Some((lm, cfg)),
            &mut [&mut st],
            &[t_len],
            &enc,
            &mut ws,
        )
        .expect("beam_chunk_with_lm");
        assert_eq!(st.best(), expected.as_slice());
    }
}

#[test]
fn lm_fusion_is_batch_invariant() {
    let tw = tiny_model();
    let m = &tw.model;
    let lm = Arc::new(NgramLm::from_arpa_str(ARPA_LM, Some(ARPA_LODR), 8, true).expect("q8 lm"));
    let cfg = LmConfig {
        lm_weight: 0.6,
        lodr_weight: 0.3,
        blank_penalty: 0.25,
    };
    let lens = [10usize, 18, 7, 15];
    let encs: Vec<Vec<f32>> = lens
        .iter()
        .enumerate()
        .map(|(s, &t)| {
            (0..t * m.arch.joiner)
                .map(|i| (((i + s * 37) as f32 * 0.29).sin()) * 1.2)
                .collect()
        })
        .collect();

    let mut singles = Vec::new();
    for (s, &t) in lens.iter().enumerate() {
        let mut st = BeamState::new(m);
        let mut ws = DecWorkspace::default();
        beam_chunk_with_lm(
            m,
            Isa::Scalar,
            Precision::F32,
            4,
            1,
            3.5,
            Some((lm.as_ref(), cfg)),
            &mut [&mut st],
            &[t],
            &encs[s],
            &mut ws,
        )
        .expect("single");
        singles.push(st.best().to_vec());
    }

    let mut flat = Vec::new();
    for e in &encs {
        flat.extend_from_slice(e);
    }
    let mut sts: Vec<BeamState> = (0..lens.len()).map(|_| BeamState::new(m)).collect();
    let mut refs: Vec<&mut BeamState> = sts.iter_mut().collect();
    let mut ws = DecWorkspace::default();
    beam_chunk_with_lm(
        m,
        Isa::Scalar,
        Precision::F32,
        4,
        1,
        3.5,
        Some((lm.as_ref(), cfg)),
        &mut refs,
        &lens,
        &flat,
        &mut ws,
    )
    .expect("batched");
    for (s, st) in sts.iter().enumerate() {
        assert_eq!(
            st.best(),
            singles[s].as_slice(),
            "stream {s} batch mismatch"
        );
    }
}
