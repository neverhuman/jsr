//! Pure-Rust INT8 causal Transformer N-best rescorer (`ultra.tf-rescore/v1`).
//!
//! Evaluates a batch of N-best hypotheses in ONE prefix-trie-deduplicated forward pass:
//! shared prefixes (L2R) and shared suffixes (R2L) are computed once, batched across
//! unique trie nodes through `quant::linear_i8`, with reusable scratch buffers (`TfWorkspace`).

use super::tf_rescore_trie::TrieBuf;
use crate::kernels::{self, Isa};
use crate::quant::{linear_i8, QuantScratch};
use crate::EngineError;

pub use super::tf_rescore_io::{TfConfig, TfDirection, TfRescorer, TF_MAGIC, TF_VERSION};

/// Sentence log-probability breakdown for one candidate sequence (natural log).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TfScore {
    pub tok_logp: f32,
    pub eos_logp: f32,
    pub total_logp: f32,
    pub n_tok: usize,
}

/// Reusable scratch buffers for zero-churn N-best Transformer rescoring.
#[derive(Debug, Default)]
pub struct TfWorkspace {
    pub(crate) q: QuantScratch,
    trie: TrieBuf,
    h: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    ff12: Vec<f32>,
    ff_mid: Vec<f32>,
    logits: Vec<f32>,
    lse: Vec<f32>,
    scores: Vec<f32>,
    path: Vec<usize>,
}

impl TfWorkspace {
    /// Enable or disable 7-bit paired-accumulation activations (`gemm_i8_a7` on AVX2).
    pub fn set_act7(&mut self, act7: bool) {
        self.q.act7 = act7;
    }
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

fn rms_norm(x: &[f32], w: &[f32], d: usize, eps: f32, out: &mut [f32]) {
    let inv_d = 1.0f32 / (d as f32);
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let ss = xr.iter().fold(0.0f32, |acc, &v| v.mul_add(v, acc));
        let inv = 1.0f32 / (ss * inv_d + eps).sqrt();
        for ((o, &xv), &wv) in or.iter_mut().zip(xr).zip(w) {
            *o = xv * inv * wv;
        }
    }
}

fn apply_rope(vec: &mut [f32], nh: usize, hd: usize, cs: &[f32], sn: &[f32]) {
    let half = hd / 2;
    for hh in 0..nh {
        let sl = &mut vec[hh * hd..(hh + 1) * hd];
        for j in 0..half {
            let (c, s, a, b) = (cs[j], sn[j], sl[2 * j], sl[2 * j + 1]);
            sl[2 * j] = a * c - b * s;
            sl[2 * j + 1] = a * s + b * c;
        }
    }
}

impl TfRescorer {
    fn trie_attention(&self, isa: Isa, n_nodes: usize, ws: &mut TfWorkspace) {
        let (d, nh, hd) = (self.cfg.d_model, self.cfg.n_heads, self.cfg.head_dim());
        let scale = 1.0f32 / (hd as f32).sqrt();
        let attn = grow(&mut ws.attn_out, n_nodes * d);
        attn.fill(0.0);
        for u in 0..n_nodes {
            ws.path.clear();
            let mut cur = u as i32;
            while cur >= 0 {
                ws.path.push(cur as usize);
                cur = ws.trie.parent[cur as usize];
            }
            let sc = grow(&mut ws.scores, ws.path.len());
            for hh in 0..nh {
                let q_uh = &ws.qkv[u * 3 * d + hh * hd..u * 3 * d + (hh + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for (idx, &anc) in ws.path.iter().enumerate() {
                    let k_ah = &ws.qkv[anc * 3 * d + d + hh * hd..anc * 3 * d + d + (hh + 1) * hd];
                    let s = kernels::dot(isa, q_uh, k_ah) * scale;
                    sc[idx] = s;
                    mx = mx.max(s);
                }
                let mut sum = 0.0f32;
                for s in sc.iter_mut() {
                    let e = kernels::scalar::exp_approx(*s - mx);
                    *s = e;
                    sum += e;
                }
                let inv_sum = 1.0f32 / sum;
                let out_uh = &mut attn[u * d + hh * hd..u * d + (hh + 1) * hd];
                for (&e, &anc) in sc.iter().zip(&ws.path) {
                    let v_ah =
                        &ws.qkv[anc * 3 * d + 2 * d + hh * hd..anc * 3 * d + 2 * d + (hh + 1) * hd];
                    kernels::axpy(isa, e * inv_sum, v_ah, out_uh);
                }
            }
        }
    }

    /// Score a single sequence using the best detected CPU ISA.
    pub fn score_sequence(
        &self,
        tokens: &[u32],
        ws: &mut TfWorkspace,
    ) -> Result<TfScore, EngineError> {
        let mut res = self.score_batch(Isa::detect(), &[tokens], ws)?;
        Ok(res.pop().unwrap_or_default())
    }

    /// Score a batch of N-best token sequences in ONE prefix-trie-deduplicated forward pass.
    pub fn score_batch(
        &self,
        isa: Isa,
        seqs: &[&[u32]],
        ws: &mut TfWorkspace,
    ) -> Result<Vec<TfScore>, EngineError> {
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        ws.trie
            .build(seqs, self.cfg.ctx - 1, self.cfg.vocab, self.cfg.direction);
        let n_nodes = ws.trie.tok.len();
        let (d, nh, hd, d_ff, vocab) = (
            self.cfg.d_model,
            self.cfg.n_heads,
            self.cfg.head_dim(),
            self.cfg.d_ff,
            self.cfg.vocab,
        );
        let half = hd / 2;
        let h = grow(&mut ws.h, n_nodes * d);
        for (u, &tok) in ws.trie.tok.iter().enumerate() {
            let t = usize::from(tok);
            h[u * d..(u + 1) * d].copy_from_slice(&self.emb[t * d..(t + 1) * d]);
        }
        for ly in &self.layers {
            rms_norm(
                &ws.h[..n_nodes * d],
                &ly.n1_w,
                d,
                self.cfg.rms_eps,
                grow(&mut ws.norm, n_nodes * d),
            );
            let qkv = grow(&mut ws.qkv, n_nodes * 3 * d);
            linear_i8(
                isa,
                &ly.qkv,
                &ws.norm[..n_nodes * d],
                n_nodes,
                qkv,
                &mut ws.q,
            );
            for (u, &pos) in ws.trie.pos.iter().enumerate() {
                let p = usize::from(pos);
                let cs = &self.rope_cos[p * half..(p + 1) * half];
                let sn = &self.rope_sin[p * half..(p + 1) * half];
                let row = &mut qkv[u * 3 * d..(u + 1) * 3 * d];
                apply_rope(&mut row[..d], nh, hd, cs, sn);
                apply_rope(&mut row[d..2 * d], nh, hd, cs, sn);
            }
            self.trie_attention(isa, n_nodes, ws);
            let proj = grow(&mut ws.proj_out, n_nodes * d);
            linear_i8(
                isa,
                &ly.wo,
                &ws.attn_out[..n_nodes * d],
                n_nodes,
                proj,
                &mut ws.q,
            );
            kernels::axpy(isa, 1.0, proj, &mut ws.h[..n_nodes * d]);
            rms_norm(
                &ws.h[..n_nodes * d],
                &ly.n2_w,
                d,
                self.cfg.rms_eps,
                grow(&mut ws.norm, n_nodes * d),
            );
            let ff12 = grow(&mut ws.ff12, n_nodes * 2 * d_ff);
            linear_i8(
                isa,
                &ly.w12,
                &ws.norm[..n_nodes * d],
                n_nodes,
                ff12,
                &mut ws.q,
            );
            let ff_mid = grow(&mut ws.ff_mid, n_nodes * d_ff);
            for u in 0..n_nodes {
                let (g, up) = ff12[u * 2 * d_ff..(u + 1) * 2 * d_ff].split_at_mut(d_ff);
                kernels::silu_inplace(isa, g);
                for ((o, &gv), &uv) in ff_mid[u * d_ff..(u + 1) * d_ff]
                    .iter_mut()
                    .zip(g.iter())
                    .zip(up.iter())
                {
                    *o = gv * uv;
                }
            }
            let proj = grow(&mut ws.proj_out, n_nodes * d);
            linear_i8(
                isa,
                &ly.w3,
                &ws.ff_mid[..n_nodes * d_ff],
                n_nodes,
                proj,
                &mut ws.q,
            );
            kernels::axpy(isa, 1.0, proj, &mut ws.h[..n_nodes * d]);
        }
        rms_norm(
            &ws.h[..n_nodes * d],
            &self.norm_w,
            d,
            self.cfg.rms_eps,
            grow(&mut ws.norm, n_nodes * d),
        );
        let logits = grow(&mut ws.logits, n_nodes * vocab);
        linear_i8(
            isa,
            &self.head,
            &ws.norm[..n_nodes * d],
            n_nodes,
            logits,
            &mut ws.q,
        );
        let lse = grow(&mut ws.lse, n_nodes);
        for (u, slot) in lse.iter_mut().enumerate() {
            let row = &logits[u * vocab..(u + 1) * vocab];
            let mut mx = f32::NEG_INFINITY;
            for &v in row {
                if !v.is_finite() {
                    return Err(EngineError::Output("non-finite TF rescorer logit".into()));
                }
                mx = mx.max(v);
            }
            *slot = mx + kernels::sum_exp_shifted(isa, row, mx).ln();
        }
        Ok(ws
            .trie
            .ranges
            .iter()
            .map(|&(start, end)| {
                let n_tok = (end - start).saturating_sub(1);
                let tok_logp: f32 = ws.trie.steps[start..start + n_tok]
                    .iter()
                    .map(|&(n, t)| {
                        ws.logits[(n as usize) * vocab + usize::from(t)] - ws.lse[n as usize]
                    })
                    .sum();
                let eos_logp = if end > start {
                    let (n, t) = ws.trie.steps[end - 1];
                    ws.logits[(n as usize) * vocab + usize::from(t)] - ws.lse[n as usize]
                } else {
                    0.0
                };
                TfScore {
                    tok_logp,
                    eos_logp,
                    total_logp: tok_logp + eos_logp,
                    n_tok,
                }
            })
            .collect())
    }
}
