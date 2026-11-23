//! Frame-synchronous modified beam search with `CommitTokenLm`, `InternalLm` (LODR or
//! exact Stateless-Predictor ILME), and acoustic-confidence margin gating.

use std::collections::HashMap;
use std::sync::Arc;

use super::lm_gate::{
    acoustic_top1_margin, logaddexp, refresh_lm_caches, refresh_pred, CommitTokenLm, GatedHyp,
    InternalLm, LmGateConfig,
};
use crate::encoder::{linear, Precision};
use crate::kernels::{self, Isa};
use crate::model::Model;
use crate::quant::QuantScratch;
use crate::EngineError;

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    h: usize,
    k: usize,
}

fn logsumexp(isa: Isa, row: &[f32]) -> Result<f32, EngineError> {
    let mut mx = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(EngineError::Output(format!(
                "non-finite joiner logit at {i}"
            )));
        }
        mx = mx.max(v);
    }
    Ok(mx + kernels::sum_exp_shifted(isa, row, mx).ln())
}

#[inline]
fn better(a: &Cand, b: &Cand) -> bool {
    a.score > b.score || (a.score == b.score && (a.h, a.k) < (b.h, b.k))
}

#[inline]
fn push_cand(cands: &mut Vec<Cand>, beam: usize, c: Cand) {
    if cands.len() < beam {
        cands.push(c);
        let mut i = cands.len() - 1;
        while i > 0 && better(&cands[i], &cands[i - 1]) {
            cands.swap(i, i - 1);
            i -= 1;
        }
    } else if better(&c, &cands[beam - 1]) {
        cands[beam - 1] = c;
        let mut i = beam - 1;
        while i > 0 && better(&cands[i], &cands[i - 1]) {
            cands.swap(i, i - 1);
            i -= 1;
        }
    }
}

fn merge_hyp(dst: &mut Vec<GatedHyp>, hy: GatedHyp) {
    if let Some(x) = dst.iter_mut().find(|x| x.tokens == hy.tokens) {
        x.score = logaddexp(x.score, hy.score);
    } else {
        dst.push(hy);
    }
}

fn advance_hyp(p: &GatedHyp, k: usize, blank: usize, score: f32) -> GatedHyp {
    if k == blank {
        GatedHyp {
            tokens: p.tokens.clone(),
            score,
            dec: Arc::clone(&p.dec),
            dec_ok: p.dec_ok,
            ngram_ctx: p.ngram_ctx,
            lm_row: p.lm_row.clone(),
            ilme_row: p.ilme_row.clone(),
        }
    } else {
        let mut tk = p.tokens.clone();
        tk.push(k as u32);
        GatedHyp {
            tokens: tk,
            score,
            dec: Arc::clone(&p.dec),
            dec_ok: false,
            ngram_ctx: None,
            lm_row: None,
            ilme_row: None,
        }
    }
}

/// Frame-synchronous modified beam search over projected encoder frames `enc` (`n_frames * joiner`)
/// with `CommitTokenLm`, `InternalLm` (LODR or exact Stateless ILME), and confidence margin gating.
#[allow(clippy::too_many_arguments)]
pub fn decode_enc_gated(
    model: &Model,
    isa: Isa,
    prec: Precision,
    enc: &[f32],
    n_frames: usize,
    lm: Option<&CommitTokenLm>,
    ilm: &InternalLm,
    cfg: &LmGateConfig,
) -> Result<Vec<u32>, EngineError> {
    cfg.validate()?;
    let (jn, v, blank) = (model.arch.joiner, model.arch.vocab, model.arch.blank);
    if enc.len() < n_frames * jn {
        return Err(EngineError::Shape(
            "enc shorter than n_frames * joiner".into(),
        ));
    }
    if let Some(l) = lm {
        l.validate(v)?;
    }
    let mut hyps = vec![GatedHyp {
        tokens: Vec::new(),
        score: 0.0,
        dec: Arc::new(vec![0.0; jn]),
        dec_ok: false,
        ngram_ctx: None,
        lm_row: None,
        ilme_row: None,
    }];
    let mut ffnn_cache: HashMap<[u16; 8], (Arc<Vec<f32>>, f32)> = HashMap::new();
    let mut ilme_cache: HashMap<(u16, u16), Arc<Vec<f32>>> = HashMap::new();
    let (mut hid, mut logits, mut q) = (Vec::new(), Vec::new(), QuantScratch::default());
    let mut cands: Vec<Cand> = Vec::with_capacity(cfg.beam);
    let mut done: Vec<GatedHyp> = Vec::new();
    let max_sym = cfg.max_sym_per_frame;
    for t in 0..n_frames {
        let ev = &enc[t * jn..(t + 1) * jn];
        if max_sym > 1 {
            done.clear();
        }
        for round in 0..max_sym {
            if hyps.is_empty() {
                break;
            }
            refresh_lm_caches(
                isa,
                blank,
                v,
                lm,
                ilm,
                &mut hyps,
                &mut ffnn_cache,
                &mut ilme_cache,
            );
            refresh_pred(model, isa, &mut hyps, &mut q);
            let nh = hyps.len();
            hid.resize(nh * jn, 0.0);
            for (h, hy) in hyps.iter().enumerate() {
                for ((o, &a), &b) in hid[h * jn..(h + 1) * jn]
                    .iter_mut()
                    .zip(ev)
                    .zip(hy.dec.iter())
                {
                    *o = (a + b).tanh();
                }
            }
            logits.resize(nh * v, 0.0);
            linear(
                isa,
                prec,
                &model.joiner_out,
                &hid[..nh * jn],
                nh,
                &mut logits[..nh * v],
                &mut q,
            );
            cands.clear();
            for h in 0..nh {
                let row = &logits[h * v..(h + 1) * v];
                let lse = logsumexp(isa, row)?;
                let base = hyps[h].score;
                let gated_off = cfg.conf_margin > 0.0 && {
                    let (k1, m) = acoustic_top1_margin(row);
                    k1 != blank && m >= cfg.conf_margin
                };
                let (lm_w, ilm_w) = if gated_off {
                    (0.0, 0.0)
                } else {
                    (cfg.lm_scale, cfg.ilm_scale)
                };
                let b_score = base + (row[blank] - lse) - cfg.token_bonus;
                if cands.len() < cfg.beam || b_score >= cands[cfg.beam - 1].score {
                    push_cand(
                        &mut cands,
                        cfg.beam,
                        Cand {
                            score: b_score,
                            h,
                            k: blank,
                        },
                    );
                }
                let hy = &hyps[h];
                let lm_ub = if lm_w > 0.0 {
                    match lm {
                        Some(CommitTokenLm::Ngram(_)) => {
                            lm_w * hy.ngram_ctx.expect("ctx").max_logp_ub
                        }
                        Some(_) => lm_w * hy.lm_row.as_ref().expect("lm_row").1,
                        None => 0.0,
                    }
                } else {
                    0.0
                };
                for (k, &x) in row.iter().enumerate() {
                    if k == blank {
                        continue;
                    }
                    let mut ub = base + (x - lse);
                    if ilm_w > 0.0 {
                        ub -= ilm_w
                            * match ilm {
                                InternalLm::None => 0.0,
                                InternalLm::Lodr(ng) => {
                                    ng.lodr_logp(hy.ngram_ctx.as_ref().expect("lodr"), k)
                                }
                                InternalLm::StatelessIlme(_) => {
                                    hy.ilme_row.as_ref().expect("ilme")[k]
                                }
                            };
                    }
                    if cands.len() == cfg.beam && (ub + lm_ub) < cands[cfg.beam - 1].score {
                        continue;
                    }
                    let score = if lm_w > 0.0 {
                        ub + lm_w
                            * match lm {
                                Some(CommitTokenLm::Ngram(ng)) => {
                                    ng.lm_logp(hy.ngram_ctx.as_ref().expect("ng"), k)
                                }
                                Some(_) => hy.lm_row.as_ref().expect("lm_row").0[k],
                                None => 0.0,
                            }
                    } else {
                        ub
                    };
                    if cands.len() == cfg.beam && score < cands[cfg.beam - 1].score {
                        continue;
                    }
                    push_cand(&mut cands, cfg.beam, Cand { score, h, k });
                }
            }
            if max_sym == 1 {
                let mut next = Vec::with_capacity(cfg.beam);
                for c in &cands {
                    merge_hyp(&mut next, advance_hyp(&hyps[c.h], c.k, blank, c.score));
                }
                hyps = next;
            } else {
                let mut expand = Vec::with_capacity(cfg.beam);
                for c in &cands {
                    let p = &hyps[c.h];
                    if c.k == blank {
                        merge_hyp(&mut done, advance_hyp(p, blank, blank, c.score));
                    } else {
                        merge_hyp(&mut expand, advance_hyp(p, c.k, blank, c.score));
                    }
                }
                hyps = expand;
                if round + 1 == max_sym {
                    break;
                }
            }
        }
        if max_sym > 1 {
            for hy in hyps.drain(..) {
                merge_hyp(&mut done, hy);
            }
            done.sort_by(|a, b| b.score.total_cmp(&a.score));
            done.truncate(cfg.beam);
            hyps = std::mem::take(&mut done);
        }
    }
    let mut best = 0usize;
    for (i, h) in hyps.iter().enumerate() {
        if h.score > hyps[best].score {
            best = i;
        }
    }
    Ok(hyps[best].tokens.clone())
}
