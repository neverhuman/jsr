//! Mathematical and end-to-end tests for `ultra_engine::fscore`.

use std::path::Path;
use std::sync::Arc;

use ultra_engine::fscore::{
    rnnt_forward_from_log_probs, score_lm_tokens, score_tokens, EncodedAudio, FscoreWorkspace,
};
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::{Engine, Isa, NgramLm, Precision, TokenLm};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

struct Lattice<'a> {
    logp_blank: &'a [f64],
    logp_tok: &'a [f64],
    t_len: usize,
    u_len: usize,
}

impl Lattice<'_> {
    fn brute_force_full(&self) -> f64 {
        fn dfs(lat: &Lattice<'_>, t: usize, u: usize, acc: f64) -> f64 {
            if t == lat.t_len {
                return if u == lat.u_len {
                    acc
                } else {
                    f64::NEG_INFINITY
                };
            }
            let cols = lat.u_len + 1;
            let mut total = dfs(lat, t + 1, u, acc + lat.logp_blank[t * cols + u]);
            if u < lat.u_len {
                total = logaddexp(
                    total,
                    dfs(lat, t, u + 1, acc + lat.logp_tok[t * lat.u_len + u]),
                );
            }
            total
        }
        dfs(self, 0, 0, 0.0)
    }

    fn brute_force_sync(&self) -> f64 {
        let cols = self.u_len + 1;
        let mut total = f64::NEG_INFINITY;
        for mask in 0u32..(1u32 << self.t_len) {
            if mask.count_ones() as usize != self.u_len {
                continue;
            }
            let (mut u, mut sc) = (0usize, 0.0f64);
            for t in 0..self.t_len {
                if ((mask >> t) & 1) == 1 {
                    sc += self.logp_tok[t * self.u_len + u];
                    u += 1;
                } else {
                    sc += self.logp_blank[t * cols + u];
                }
            }
            total = logaddexp(total, sc);
        }
        total
    }
}

#[test]
fn forward_recursions_match_brute_force_on_t4_u2_lattice() {
    let (t_len, u_len) = (4usize, 2usize);
    let cols = u_len + 1;
    let (mut logp_blank, mut logp_tok) = (vec![0.0f64; t_len * cols], vec![0.0f64; t_len * u_len]);
    for t in 0..t_len {
        for u in 0..cols {
            let pb = 0.35 + 0.07 * ((t * 3 + u) % 5) as f64;
            logp_blank[t * cols + u] = pb.ln();
            if u < u_len {
                let pt = (1.0 - pb) * (0.4 + 0.1 * ((t + u * 2) % 4) as f64);
                logp_tok[t * u_len + u] = pt.ln();
            }
        }
    }
    let lat = Lattice {
        logp_blank: &logp_blank,
        logp_tok: &logp_tok,
        t_len,
        u_len,
    };
    let (got_full, got_sync) = rnnt_forward_from_log_probs(&logp_blank, &logp_tok, t_len, u_len);
    assert!((got_full - lat.brute_force_full()).abs() < 1e-12);
    assert!((got_sync - lat.brute_force_sync()).abs() < 1e-12);
}

#[test]
fn empty_candidate_blank_sum_monotonicity_and_u_gt_t() {
    let blanks = [-0.2, -0.5, -0.1, -0.9];
    let (f0, s0) = rnnt_forward_from_log_probs(&blanks, &[], 4, 0);
    let want: f64 = blanks.iter().sum();
    assert!((f0 - want).abs() < 1e-12 && (s0 - want).abs() < 1e-12);

    let (f_hi, s_hi) = rnnt_forward_from_log_probs(&[-0.7; 8], &[-1.1; 6], 2, 3);
    assert!(f_hi.is_finite() && s_hi == f64::NEG_INFINITY);

    let (fa, sa) = rnnt_forward_from_log_probs(&[-0.8; 6], &[-2.0; 3], 3, 1);
    let (fb, sb) = rnnt_forward_from_log_probs(&[-0.8; 6], &[-1.0; 3], 3, 1);
    assert!(fb > fa && sb > sa);
}

#[test]
fn lm_and_lodr_scores_match_token_lm_exactly() {
    const LM_ARPA: &str = "\\data\\\nngram 1=6\nngram 2=4\nngram 3=2\n\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\t-0.3\n-1.0\t2\t-0.2\n-1.5\t3\t-0.4\n-2.0\t4\t-0.1\n-2.5\t5\n\n\\2-grams:\n-0.5\t<s> 2\t-0.15\n-0.8\t2 3\t-0.25\n-0.6\t3 4\t-0.35\n-1.2\t4 5\n\n\\3-grams:\n-0.2\t<s> 2 3\n-0.3\t2 3 4\n\n\\end\\\n";
    const LODR_ARPA: &str = "\\data\\\nngram 1=5\nngram 2=2\n\n\\1-grams:\n-3.5\t<unk>\n-99.0\t<s>\t-0.2\n-1.1\t2\t-0.3\n-1.4\t3\t-0.1\n-1.8\t4\n\n\\2-grams:\n-0.7\t<s> 2\n-0.5\t2 3\n\n\\end\\\n";
    let lm = NgramLm::from_arpa_str(LM_ARPA, Some(LODR_ARPA), 8, false).expect("parse");
    let toks = [2u32, 3, 4, 5];
    let (got_lm, got_lodr) = score_lm_tokens(Some(&lm), &toks);
    let (mut exp_lm, mut exp_lodr) = (0.0f64, 0.0f64);
    for (i, &t) in toks.iter().enumerate() {
        exp_lm += f64::from(lm.score_next(&toks[..i], t));
        exp_lodr += f64::from(lm.lodr_logp(&lm.ctx_state(&toks[..i]), t as usize));
    }
    assert!((got_lm - exp_lm).abs() < 1e-12 && (got_lodr - exp_lodr).abs() < 1e-12);
    assert_eq!(score_lm_tokens(None, &toks), (0.0, 0.0));
}

#[test]
fn score_tokens_matches_brute_force_with_predictor_and_joiner() {
    let (vocab, pred_emb, joiner, t_len) = (6usize, 4usize, 4usize, 4usize);
    let emb: Vec<f32> = (0..vocab * pred_emb)
        .map(|i| ((i as f32) * 0.37).sin() * 0.3)
        .collect();
    let pred_w: Vec<f32> = (0..joiner * 2 * pred_emb)
        .map(|i| ((i as f32) * 0.51).cos() * 0.25)
        .collect();
    let pred_b: Vec<f32> = (0..joiner).map(|i| (i as f32 - 1.5) * 0.1).collect();
    let join_w: Vec<f32> = (0..vocab * joiner)
        .map(|i| ((i as f32) * 0.67).sin() * 0.4)
        .collect();
    let join_b: Vec<f32> = vec![0.3, -0.5, 0.1, 0.2, -0.1, -0.3];

    let arch = Arch {
        n_mels: 80,
        d: 4,
        n_blocks: 0,
        lower: 0,
        mid: 0,
        fin: 0,
        heads: 1,
        dh: 4,
        ffn: 8,
        kernel: 3,
        stem_ch: 4,
        f1: 39,
        f2: 19,
        chunk: 4,
        left: 8,
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
    let tokens = Arc::new(
        TokenTable::from_str_checked("<blk> 0\n<unk> 1\na 2\nb 3\nc 4\nd 5\n", vocab).expect("t"),
    );
    let cmvn = GlobalCmvn::new(&[0.0; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(
        TrainerMelConfig::from_fbank_le_bytes(&vec![0u8; 257 * 80 * 4], 1e-6, cmvn).expect("mel"),
    );
    let dummy = Linear::from_fp32(&[0.0], 1, 1, None);
    let model = Arc::new(Model {
        arch,
        root: Path::new("").to_path_buf(),
        c1_w: Vec::new(),
        c1_b: Vec::new(),
        c2: dummy.clone(),
        stem_out: dummy.clone(),
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: dummy,
        emb: emb.clone(),
        pred_proj: Linear::from_fp32(&pred_w, joiner, 2 * pred_emb, Some(&pred_b)),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&join_w, vocab, joiner, Some(&join_b)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: Vec::new(),
        mel_cfg,
        tokens,
        int8_prepared: None,
    });
    let engine = Engine::new(model, Isa::Scalar, Precision::F32).expect("engine");
    let enc = EncodedAudio {
        frames: (0..t_len * joiner)
            .map(|i| ((i as f32) * 0.29).sin() * 0.5)
            .collect(),
        n_frames: t_len,
        joiner_dim: joiner,
    };
    let seq = [2u32, 3];
    let mut ws = FscoreWorkspace::default();
    let sc = score_tokens(&engine, &enc, &seq, None, &mut ws).expect("score");

    // Independent naive reference over the exact same predictor/joiner weights.
    let ctxs = [[0usize, 0], [0, 2], [2, 3]];
    let (mut lb, mut ly) = (vec![0.0f64; t_len * 3], vec![0.0f64; t_len * 2]);
    for t in 0..t_len {
        for (u, &[c0, c1]) in ctxs.iter().enumerate() {
            let mut pin = vec![0.0f32; 2 * pred_emb];
            pin[..pred_emb].copy_from_slice(&emb[c0 * pred_emb..(c0 + 1) * pred_emb]);
            pin[pred_emb..].copy_from_slice(&emb[c1 * pred_emb..(c1 + 1) * pred_emb]);
            let mut hid = vec![0.0f32; joiner];
            for j in 0..joiner {
                let mut p = pred_b[j];
                for c in 0..2 * pred_emb {
                    p += pred_w[j * 2 * pred_emb + c] * pin[c];
                }
                hid[j] = (enc.frames[t * joiner + j] + p).tanh();
            }
            let logits: Vec<f64> = (0..vocab)
                .map(|k| {
                    let s = (0..joiner).fold(join_b[k], |a, j| a + join_w[k * joiner + j] * hid[j]);
                    f64::from(s)
                })
                .collect();
            let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + logits.iter().map(|&x| (x - mx).exp()).sum::<f64>().ln();
            lb[t * 3 + u] = logits[0] - lse;
            if u < seq.len() {
                ly[t * 2 + u] = logits[seq[u] as usize] - lse;
            }
        }
    }
    let lat = Lattice {
        logp_blank: &lb,
        logp_tok: &ly,
        t_len,
        u_len: seq.len(),
    };
    assert!((sc.rnnt_full_logp - lat.brute_force_full()).abs() < 1e-6);
    assert!((sc.rnnt_sync_logp - lat.brute_force_sync()).abs() < 1e-6);
}
