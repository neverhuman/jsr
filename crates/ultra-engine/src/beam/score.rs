//! Exact candidate scoring, branch-and-bound LM/LODR pruning, and hypothesis merging for beam search.

use super::{Cand, Hyp};
use crate::kernels::Isa;
use crate::EngineError;

/// `logsumexp` of one row in f32 plus the maximum non-blank logit in a single scan:
/// `max + ln(sum exp(x - max))`, the sum in the engine's fixed cross-ISA order
/// ([`kernels::sum_exp_shifted`]). Refuses NaN/Inf logits.
pub(super) fn logsumexp_and_max_nb(
    isa: Isa,
    row: &[f32],
    blank: usize,
) -> Result<(f32, f32), EngineError> {
    let mut mx = f32::NEG_INFINITY;
    let mut max_nb = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(EngineError::Output(format!(
                "non-finite joiner logit at {i}"
            )));
        }
        if v > mx {
            mx = v;
        }
        if i != blank && v > max_nb {
            max_nb = v;
        }
    }
    Ok((
        mx + crate::kernels::sum_exp_shifted(isa, row, mx).ln(),
        max_nb,
    ))
}

/// torch's `logaddexp` for finite f32: `max + log1p(exp(-|a - b|))`.
#[inline]
pub(super) fn logaddexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// `a` before `b` in the deterministic top-`beam` order.
#[inline]
fn better(a: &Cand, b: &Cand) -> bool {
    a.score > b.score || (a.score == b.score && (a.h, a.k) < (b.h, b.k))
}

/// Insert `h` into `dst`, merging equal token sequences via `logaddexp`.
#[inline]
pub(super) fn merge_hyp(dst: &mut Vec<Hyp>, hy: Hyp) {
    if let Some(x) = dst.iter_mut().find(|x| x.tokens == hy.tokens) {
        x.score = logaddexp(x.score, hy.score);
    } else {
        dst.push(hy);
    }
}

/// Insert candidate `c` into sorted top-`beam` `cands` using deterministic tie-breaking.
#[inline]
pub(super) fn push_cand(cands: &mut Vec<Cand>, beam: usize, c: Cand) {
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

/// Evaluate exact `best_nb` for the `beam_margin` blank-dominance gate, using O(1) `max_nb`
/// and `lodr_row_min` fast-path before touching the dense LODR row.
#[inline]
pub(super) fn is_confident_blank_margin(
    row: &[f32],
    max_nb: f32,
    blank: usize,
    blank_pen: f32,
    beam_margin: f32,
    lm: Option<(&crate::lm::NgramLm, crate::lm::LmConfig)>,
    ctx0: Option<crate::lm::LmCtx>,
) -> bool {
    let zb = row[blank] - blank_pen;
    if !zb.is_finite() || !max_nb.is_finite() {
        return false;
    }
    let Some((ngram, cfg)) = lm else {
        return (zb - max_nb) > beam_margin;
    };
    let ctx = ctx0.expect("lm_ctx refreshed");
    let lm_ub = cfg.lm_weight * ctx.max_logp_ub;
    let lodr_w = cfg.lodr_weight;
    if lodr_w == 0.0 || !ngram.has_lodr() {
        let best_nb = max_nb + lm_ub;
        return best_nb.is_finite() && (zb - best_nb) > beam_margin;
    }
    let row_min_lodr = ngram.lodr_row_min(&ctx);
    let slack = 1e-3f32.max(zb.abs() * 1e-5);
    let eff_ub = max_nb - lodr_w * row_min_lodr + lm_ub;
    if eff_ub.is_finite() && (zb - eff_ub) > beam_margin + slack {
        return true;
    }
    let lodr_row = ngram.lodr_row(&ctx);
    let mut best_nb = f32::NEG_INFINITY;
    let x_needed = (zb - beam_margin - lm_ub + lodr_w * row_min_lodr) - slack;
    for (k, &x) in row.iter().enumerate() {
        if k == blank || x < x_needed {
            continue;
        }
        let eff = x - lodr_w * lodr_row[k] + lm_ub;
        if eff > best_nb {
            best_nb = eff;
            let diff = zb - best_nb;
            if !diff.is_finite() || diff <= beam_margin {
                return false;
            }
        }
    }
    if best_nb.is_finite() {
        (zb - best_nb) > beam_margin
    } else {
        let mut exact_best = f32::NEG_INFINITY;
        for (k, &x) in row.iter().enumerate() {
            if k != blank {
                let eff = x - lodr_w * lodr_row[k] + lm_ub;
                if eff > exact_best {
                    exact_best = eff;
                }
            }
        }
        exact_best.is_finite() && (zb - exact_best) > beam_margin
    }
}

/// Score all non-blank candidates of one hypothesis `h` into `cands` with exact branch-and-bound:
/// - When `cands` is already full (`len == beam`, the common case after blanks-first seeding):
///   O(1) `max_nb` skip -> scalar logit threshold `x_thresh` -> best-first sorted shortlist with
///   early `break` before 4-gram LM trie lookup.
/// - When `cands` is not yet full (`len < beam`, e.g. `nh == 1`):
///   Pass 1 computes `ub` in L1 and finds the top-`beam` tokens by `ub` + `next_ub`; evaluates
///   `lm_logp` on those top-`beam` first to establish a tight cutoff, then only scans remaining
///   tokens if `next_ub + lm_ub_term >= cutoff`.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_nonblank_lm(
    h: usize,
    row: &[f32],
    lse: f32,
    max_nb: f32,
    base: f32,
    blank: usize,
    beam: usize,
    ngram: &crate::lm::NgramLm,
    cfg: crate::lm::LmConfig,
    ctx: &crate::lm::LmCtx,
    cands: &mut Vec<Cand>,
    shortlist: &mut Vec<(f32, usize)>,
    ub_buf: &mut Vec<f32>,
) {
    let lm_w = cfg.lm_weight;
    let lodr_w = cfg.lodr_weight;
    let lm_ub_term = lm_w * ctx.max_logp_ub;
    let lodr_row = if lodr_w > 0.0 {
        ngram.lodr_row(ctx)
    } else {
        &[]
    };
    let row_min_lodr = if lodr_w > 0.0 {
        ngram.lodr_row_min(ctx)
    } else {
        0.0
    };
    let slack = 1e-3f32.max(base.abs() * 1e-5);

    if cands.len() == beam {
        let cutoff = cands[beam - 1].score;
        let max_possible = base + (max_nb - lse) - lodr_w * row_min_lodr + lm_ub_term;
        if max_possible < cutoff - slack {
            return;
        }
        let x_thresh = (cutoff - base + lse - lm_ub_term + lodr_w * row_min_lodr) - slack;
        if lm_w == 0.0 {
            for (k, &x) in row.iter().enumerate() {
                if k == blank || x < x_thresh {
                    continue;
                }
                let mut score = base + (x - lse);
                if !lodr_row.is_empty() {
                    score -= lodr_w * lodr_row[k];
                }
                if score >= cands[beam - 1].score {
                    push_cand(cands, beam, Cand { score, h, k });
                }
            }
            return;
        }
        shortlist.clear();
        if lodr_row.is_empty() {
            for (k, &x) in row.iter().enumerate() {
                if k == blank || x < x_thresh {
                    continue;
                }
                let ub = base + (x - lse);
                if ub + lm_ub_term >= cutoff {
                    shortlist.push((ub, k));
                }
            }
        } else {
            for (k, &x) in row.iter().enumerate() {
                if k == blank || x < x_thresh {
                    continue;
                }
                let ub = base + (x - lse) - lodr_w * lodr_row[k];
                if ub + lm_ub_term >= cutoff {
                    shortlist.push((ub, k));
                }
            }
        }
        if shortlist.is_empty() {
            return;
        }
        if shortlist.len() > 1 {
            shortlist.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        }
        for &(ub, k) in shortlist.iter() {
            if ub + lm_ub_term < cands[beam - 1].score {
                break;
            }
            let score = ub + lm_w * ngram.lm_logp(ctx, k);
            if score >= cands[beam - 1].score {
                push_cand(cands, beam, Cand { score, h, k });
            }
        }
        return;
    }

    // `cands.len() < beam` (e.g. `nh == 1` after initial start or blank-margin collapse).
    if lm_w == 0.0 {
        for (k, &x) in row.iter().enumerate() {
            if k == blank {
                continue;
            }
            let mut score = base + (x - lse);
            if !lodr_row.is_empty() {
                score -= lodr_w * lodr_row[k];
            }
            if cands.len() < beam || score >= cands[beam - 1].score {
                push_cand(cands, beam, Cand { score, h, k });
            }
        }
        return;
    }

    let v = row.len();
    if ub_buf.len() < v {
        ub_buf.resize(v, f32::NEG_INFINITY);
    }
    let ubs = &mut ub_buf[..v];
    shortlist.clear();
    let mut next_ub = f32::NEG_INFINITY;
    for (k, &x) in row.iter().enumerate() {
        if k == blank {
            ubs[k] = f32::NEG_INFINITY;
            continue;
        }
        let mut ub = base + (x - lse);
        if !lodr_row.is_empty() {
            ub -= lodr_w * lodr_row[k];
        }
        ubs[k] = ub;
        if shortlist.len() < beam {
            shortlist.push((ub, k));
            let mut i = shortlist.len() - 1;
            while i > 0 && shortlist[i].0 > shortlist[i - 1].0 {
                shortlist.swap(i, i - 1);
                i -= 1;
            }
        } else if ub > shortlist[beam - 1].0 {
            let evicted = shortlist[beam - 1].0;
            if evicted > next_ub {
                next_ub = evicted;
            }
            shortlist[beam - 1] = (ub, k);
            let mut i = beam - 1;
            while i > 0 && shortlist[i].0 > shortlist[i - 1].0 {
                shortlist.swap(i, i - 1);
                i -= 1;
            }
        } else if ub > next_ub {
            next_ub = ub;
        }
    }
    for &(ub, k) in shortlist.iter() {
        ubs[k] = f32::NEG_INFINITY;
        if cands.len() == beam && ub + lm_ub_term < cands[beam - 1].score {
            continue;
        }
        let score = ub + lm_w * ngram.lm_logp(ctx, k);
        if cands.len() < beam || score >= cands[beam - 1].score {
            push_cand(cands, beam, Cand { score, h, k });
        }
    }
    if cands.len() == beam && next_ub + lm_ub_term < cands[beam - 1].score {
        return;
    }
    let cutoff = if cands.len() == beam {
        cands[beam - 1].score
    } else {
        f32::NEG_INFINITY
    };
    shortlist.clear();
    for (k, &ub) in ubs.iter().enumerate() {
        if ub.is_finite() && (cands.len() < beam || ub + lm_ub_term >= cutoff) {
            shortlist.push((ub, k));
        }
    }
    if shortlist.is_empty() {
        return;
    }
    if shortlist.len() > 1 {
        shortlist.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    }
    for &(ub, k) in shortlist.iter() {
        if cands.len() == beam && ub + lm_ub_term < cands[beam - 1].score {
            break;
        }
        let score = ub + lm_w * ngram.lm_logp(ctx, k);
        if cands.len() < beam || score >= cands[beam - 1].score {
            push_cand(cands, beam, Cand { score, h, k });
        }
    }
}

/// Apply an all-blank distinct-parent update in place with zero token clones or allocations.
#[inline]
pub(super) fn update_hyps_all_blank_inplace(
    hyps: &mut Vec<Hyp>,
    cands: &[Cand],
    old_buf: &mut Vec<Option<Hyp>>,
) {
    let mut identity = true;
    for (i, c) in cands.iter().enumerate() {
        if c.h != i {
            identity = false;
            break;
        }
    }
    if identity {
        for (hy, c) in hyps.iter_mut().zip(cands) {
            hy.score = c.score;
        }
        return;
    }
    old_buf.clear();
    old_buf.extend(hyps.drain(..).map(Some));
    for c in cands {
        let mut hy = old_buf[c.h].take().expect("distinct blank parent");
        hy.score = c.score;
        hyps.push(hy);
    }
}
