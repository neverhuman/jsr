//! Endpoint commit-pass TokenLm (`NgramLm`, `FfnnLm`, prob-space `M5+FFNN` interpolation),
//! Exact Stateless-Predictor ILME / LODR subtraction, and Acoustic-Confidence Margin Gating.

use std::collections::HashMap;
use std::sync::Arc;

use crate::beam::context;
use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::lm::{FfnnLm, LmCtx, NgramLm, StatelessIlme, TokenLm};
use crate::model::Model;
use crate::quant::QuantScratch;
use crate::EngineError;

pub use super::lm::encode_pcm_frames;
pub use super::lm_gate_beam::decode_enc_gated;

/// External token LM backend for endpoint commit beam decoding.
#[derive(Debug, Clone)]
pub enum CommitTokenLm {
    Ngram(Arc<NgramLm>),
    Ffnn(Arc<FfnnLm>),
    /// Probability-space linear interpolation: `ln(alpha * P_ng + (1 - alpha) * P_ffnn)`.
    Interpolated {
        ngram: Arc<NgramLm>,
        ffnn: Arc<FfnnLm>,
        alpha: f32,
    },
}

impl CommitTokenLm {
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Ngram(n) => n.vocab,
            Self::Ffnn(f) => f.vocab,
            Self::Interpolated { ngram, .. } => ngram.vocab,
        }
    }

    pub fn validate(&self, model_vocab: usize) -> Result<(), EngineError> {
        match self {
            Self::Ngram(n) => check_vocab("NgramLm", n.vocab, model_vocab),
            Self::Ffnn(f) => check_vocab("FfnnLm", f.vocab, model_vocab),
            Self::Interpolated { ngram, ffnn, alpha } => {
                check_vocab("NgramLm", ngram.vocab, model_vocab)?;
                check_vocab("FfnnLm", ffnn.vocab, model_vocab)?;
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
                    return Err(EngineError::Format(format!(
                        "interp_alpha must be in [0, 1] (got {alpha})"
                    )));
                }
                Ok(())
            }
        }
    }
}

fn check_vocab(name: &str, got: usize, expect: usize) -> Result<(), EngineError> {
    if got != expect {
        return Err(EngineError::Shape(format!(
            "{name} vocab {got} != model vocab {expect}"
        )));
    }
    Ok(())
}

/// Internal language model subtracted during shallow fusion.
#[derive(Debug, Clone)]
pub enum InternalLm {
    None,
    Lodr(Arc<NgramLm>),
    StatelessIlme(Arc<StatelessIlme>),
}

/// Hyperparameters for gated commit-pass beam decoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmGateConfig {
    pub beam: usize,
    pub lm_scale: f32,
    pub ilm_scale: f32,
    pub token_bonus: f32,
    pub max_sym_per_frame: usize,
    /// When `> 0.0` and top-1 acoustic token `k1 != blank` beats runner-up by `>= conf_margin`
    /// (natural-log units), suppress LM+ILM adjustment on that hypothesis-frame row.
    pub conf_margin: f32,
}

impl Default for LmGateConfig {
    fn default() -> Self {
        Self {
            beam: 16,
            lm_scale: 0.80,
            ilm_scale: 0.60,
            token_bonus: 0.0,
            max_sym_per_frame: 1,
            conf_margin: 0.0,
        }
    }
}

impl LmGateConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(2..=32).contains(&self.beam) {
            return Err(EngineError::Shape(format!(
                "LmGateConfig beam {} not in 2..=32",
                self.beam
            )));
        }
        if !(1..=8).contains(&self.max_sym_per_frame) {
            return Err(EngineError::Shape(format!(
                "LmGateConfig max_sym_per_frame {} not in 1..=8",
                self.max_sym_per_frame
            )));
        }
        for (name, v) in [
            ("lm_scale", self.lm_scale),
            ("ilm_scale", self.ilm_scale),
            ("token_bonus", self.token_bonus),
            ("conf_margin", self.conf_margin),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(EngineError::Format(format!(
                    "LmGateConfig {name} must be finite >= 0 (got {v})"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct GatedHyp {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub dec: Arc<Vec<f32>>,
    pub dec_ok: bool,
    pub ngram_ctx: Option<LmCtx>,
    pub lm_row: Option<(Arc<Vec<f32>>, f32)>,
    pub ilme_row: Option<Arc<Vec<f32>>>,
}

#[inline]
pub(super) fn logaddexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// Compute top-1 index and `(top1 - top2)` margin over raw logits (equals log-prob margin).
#[inline]
pub fn acoustic_top1_margin(row: &[f32]) -> (usize, f32) {
    let (mut b1_i, mut b1_v, mut b2_v) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (k, &v) in row.iter().enumerate() {
        if v > b1_v {
            b2_v = b1_v;
            b1_v = v;
            b1_i = k;
        } else if v > b2_v {
            b2_v = v;
        }
    }
    (b1_i, b1_v - b2_v)
}

fn ffnn_key(tokens: &[u32], ctx: usize) -> [u16; 8] {
    let mut k = [0u16; 8];
    let n = ctx.min(8);
    let take = tokens.len().min(n);
    let pad = n - take;
    for (i, &t) in tokens[tokens.len() - take..].iter().enumerate() {
        k[pad + i] = t as u16;
    }
    k
}

pub(super) fn refresh_pred(model: &Model, isa: Isa, hyps: &mut [GatedHyp], q: &mut QuantScratch) {
    let (jn, e) = (model.arch.joiner, model.arch.pred_emb);
    let init = &model.arch.ctx_init;
    if let Some(pc) = &model.pred_cache {
        for hy in hyps.iter_mut().filter(|h| !h.dec_ok) {
            let mut it = context(&hy.tokens, init);
            let (t0, t1) = (it.next().expect("ctx0"), it.next().expect("ctx1"));
            let dec = Arc::make_mut(&mut hy.dec);
            for ((o, &a), &b) in dec
                .iter_mut()
                .zip(&pc.h0[t0 * jn..(t0 + 1) * jn])
                .zip(&pc.h1[t1 * jn..(t1 + 1) * jn])
            {
                *o = a + b;
            }
            hy.dec_ok = true;
        }
        return;
    }
    let (mut inp, mut out) = (vec![0.0f32; init.len() * e], vec![0.0f32; jn]);
    for hy in hyps.iter_mut().filter(|h| !h.dec_ok) {
        for (pos, tok) in context(&hy.tokens, init).enumerate() {
            inp[pos * e..(pos + 1) * e].copy_from_slice(&model.emb[tok * e..(tok + 1) * e]);
        }
        linear(isa, Precision::F32, &model.pred_proj, &inp, 1, &mut out, q);
        Arc::make_mut(&mut hy.dec).copy_from_slice(&out);
        hy.dec_ok = true;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refresh_lm_caches(
    isa: Isa,
    blank: usize,
    vocab: usize,
    lm: Option<&CommitTokenLm>,
    ilm: &InternalLm,
    hyps: &mut [GatedHyp],
    ffnn_cache: &mut HashMap<[u16; 8], (Arc<Vec<f32>>, f32)>,
    ilme_cache: &mut HashMap<(u16, u16), Arc<Vec<f32>>>,
) {
    let ng_for_ctx: Option<&NgramLm> = match (lm, ilm) {
        (Some(CommitTokenLm::Ngram(n)), _) => Some(n.as_ref()),
        (_, InternalLm::Lodr(n)) => Some(n.as_ref()),
        _ => None,
    };
    for hy in hyps.iter_mut() {
        if let Some(ng) = ng_for_ctx {
            if hy.ngram_ctx.is_none() {
                hy.ngram_ctx = Some(ng.ctx_state(&hy.tokens));
            }
        }
        if hy.lm_row.is_none() {
            match lm {
                Some(CommitTokenLm::Ffnn(ff)) => {
                    let entry = ffnn_cache
                        .entry(ffnn_key(&hy.tokens, ff.ctx))
                        .or_insert_with(|| {
                            let mut buf = vec![0.0f32; vocab];
                            ff.score_all_isa(isa, &hy.tokens, &mut buf);
                            let mx = buf
                                .iter()
                                .enumerate()
                                .filter(|&(k, _)| k != blank)
                                .map(|(_, &x)| x)
                                .fold(f32::NEG_INFINITY, f32::max);
                            (Arc::new(buf), mx)
                        });
                    hy.lm_row = Some(entry.clone());
                }
                Some(CommitTokenLm::Interpolated { ngram, ffnn, alpha }) => {
                    let ff_row = ffnn_cache
                        .entry(ffnn_key(&hy.tokens, ffnn.ctx))
                        .or_insert_with(|| {
                            let mut buf = vec![0.0f32; vocab];
                            ffnn.score_all_isa(isa, &hy.tokens, &mut buf);
                            (Arc::new(buf), 0.0)
                        });
                    let mut ng_row = vec![0.0f32; vocab];
                    ngram.score_all(&hy.tokens, &mut ng_row);
                    let (mut out, la, lb, mut mx) = (
                        vec![0.0f32; vocab],
                        alpha.ln(),
                        (1.0 - *alpha).ln(),
                        f32::NEG_INFINITY,
                    );
                    for k in 0..vocab {
                        if k == blank {
                            continue;
                        }
                        out[k] = if *alpha >= 1.0 {
                            ng_row[k]
                        } else if *alpha <= 0.0 {
                            ff_row.0[k]
                        } else {
                            logaddexp(la + ng_row[k], lb + ff_row.0[k])
                        };
                        mx = mx.max(out[k]);
                    }
                    hy.lm_row = Some((Arc::new(out), mx));
                }
                _ => {}
            }
        }
        if let InternalLm::StatelessIlme(ilme) = ilm {
            if hy.ilme_row.is_none() {
                let (p0, p1) = ilme.ctx_pair(&hy.tokens);
                let row = ilme_cache.entry((p0 as u16, p1 as u16)).or_insert_with(|| {
                    let mut buf = vec![0.0f32; vocab];
                    ilme.score_ctx_isa(isa, p0, p1, &mut buf);
                    Arc::new(buf)
                });
                hy.ilme_row = Some(Arc::clone(row));
            }
        }
    }
}
