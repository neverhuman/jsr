//! CTC forward log-probability and word-level Minimum Bayes Risk (MBR) selection for H45.

use super::types::CommitConfig;
use super::vocab::CtcVocab;
use crate::{Engine, EngineStream};

/// Exact word-level Levenshtein edit distance.
pub fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return n + m;
    }
    let (mut prev, mut curr): (Vec<usize>, Vec<usize>) = ((0..=m).collect(), vec![0; m + 1]);
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            curr[j] = sub.min(prev[j] + 1).min(curr[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Exact word-level Levenshtein expected-WER Minimum Bayes Risk selector over log-prob weighted
/// N-best hypotheses. Deterministic tie-break: finite score, lower risk, higher score, lower index.
pub fn mbr_select(candidates: &[(Vec<String>, f32)]) -> usize {
    let n = candidates.len();
    if n <= 1 {
        return 0;
    }
    let max_s = candidates
        .iter()
        .map(|(_, s)| *s)
        .filter(|s| s.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_s.is_finite() {
        return 0;
    }
    let (mut probs, mut sum_p) = (vec![0.0f64; n], 0.0f64);
    for (i, (_, s)) in candidates.iter().enumerate() {
        if s.is_finite() {
            probs[i] = f64::from(*s - max_s).exp();
            sum_p += probs[i];
        }
    }
    if sum_p <= 0.0 {
        return 0;
    }
    probs.iter_mut().for_each(|p| *p /= sum_p);
    let mut risks = vec![0.0f64; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = word_edit_distance(&candidates[i].0, &candidates[j].0) as f64;
            risks[i] += probs[j] * d;
            risks[j] += probs[i] * d;
        }
    }
    let mut best = candidates
        .iter()
        .position(|(_, s)| s.is_finite())
        .unwrap_or(0);
    for i in (best + 1)..n {
        if !candidates[i].1.is_finite() {
            continue;
        }
        let d = risks[i] - risks[best];
        if d < -1e-12 || (d.abs() <= 1e-12 && candidates[i].1 > candidates[best].1) {
            best = i;
        }
    }
    best
}

/// Combine RNN-T and CTC log-probabilities. Impossible CTC paths (`-inf` / non-finite) yield
/// `f32::NEG_INFINITY` whenever `ctc_weight > 0.0`; `ctc_weight == 0.0` preserves `rnnt_logp`.
pub fn combine_ctc_rnnt_scores(rnnt_logp: f32, ctc_logp: f32, ctc_weight: f32) -> f32 {
    if !rnnt_logp.is_finite() {
        return f32::NEG_INFINITY;
    }
    if !ctc_weight.is_finite() || ctc_weight <= 0.0 {
        return rnnt_logp;
    }
    if !ctc_logp.is_finite() {
        return f32::NEG_INFINITY;
    }
    rnnt_logp + ctc_weight * ctc_logp
}

fn logaddexp_f64(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// Exact CTC forward log-probability over `[num_frames][vocab_size]` log-probs.
pub fn ctc_forward_logp(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
    targets: &[usize],
) -> f32 {
    if vocab_size == 0 || blank_id >= vocab_size || log_probs.len() != num_frames * vocab_size {
        return f32::NEG_INFINITY;
    }
    if targets.iter().any(|&id| id >= vocab_size || id == blank_id) {
        return f32::NEG_INFINITY;
    }
    let repeats = targets.windows(2).filter(|w| w[0] == w[1]).count();
    if num_frames < targets.len() + repeats {
        return f32::NEG_INFINITY;
    }
    if num_frames == 0 {
        return 0.0;
    }
    let ext_len = 2 * targets.len() + 1;
    let ext = |s: usize| {
        if s.is_multiple_of(2) {
            blank_id
        } else {
            targets[s / 2]
        }
    };
    let emit = |t: usize, id: usize| {
        let v = log_probs[t * vocab_size + id];
        if v.is_finite() {
            f64::from(v)
        } else {
            f64::NEG_INFINITY
        }
    };
    let mut prev = vec![f64::NEG_INFINITY; ext_len];
    let mut curr = vec![f64::NEG_INFINITY; ext_len];
    prev[0] = emit(0, blank_id);
    if !targets.is_empty() {
        prev[1] = emit(0, targets[0]);
    }
    for t in 1..num_frames {
        curr.fill(f64::NEG_INFINITY);
        for s in 0..ext_len {
            let mut acc = prev[s];
            if s > 0 {
                acc = logaddexp_f64(acc, prev[s - 1]);
            }
            if s >= 2 && ext(s) != blank_id && ext(s) != ext(s - 2) {
                acc = logaddexp_f64(acc, prev[s - 2]);
            }
            let e = emit(t, ext(s));
            curr[s] = if acc.is_finite() && e.is_finite() {
                acc + e
            } else {
                f64::NEG_INFINITY
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let total = if targets.is_empty() {
        prev[0]
    } else {
        logaddexp_f64(prev[ext_len - 2], prev[ext_len - 1])
    };
    if total.is_finite() {
        total as f32
    } else {
        f32::NEG_INFINITY
    }
}

/// Retokenize candidate text into `vocab` (separate from RNN-T IDs) and score via CTC forward.
pub fn ctc_rescore_text(vocab: &CtcVocab, log_probs: &[f32], num_frames: usize, text: &str) -> f32 {
    let Some(ids) = vocab.retokenize_text(text) else {
        return f32::NEG_INFINITY;
    };
    ctc_forward_logp(
        log_probs,
        num_frames,
        vocab.vocab_size(),
        vocab.blank_id(),
        &ids,
    )
}

pub(crate) fn select_commit_hyp(
    eng: &Engine,
    stream: &EngineStream,
    cfg: CommitConfig,
    ctc: Option<(&CtcVocab, &[f32], usize)>,
) -> (String, usize) {
    if ctc.is_none() && !cfg.mbr_enabled {
        return (eng.text(stream), 0);
    }
    let Some(beam) = &stream.dec.beam else {
        return (eng.text(stream), 0);
    };
    if beam.hyps.len() <= 1 {
        return (eng.text(stream), 0);
    }
    let mut texts = Vec::with_capacity(beam.hyps.len());
    let mut cands = Vec::with_capacity(beam.hyps.len());
    for hy in &beam.hyps {
        let ids: Vec<i64> = hy.tokens.iter().map(|&t| i64::from(t)).collect();
        let text = eng.model.tokens.decode_ids(&ids);
        let score = ctc.map_or(hy.score, |(vocab, lp, frames)| {
            let ctc_lp = ctc_rescore_text(vocab, lp, frames, &text);
            combine_ctc_rnnt_scores(hy.score, ctc_lp, cfg.ctc_weight)
        });
        texts.push(text.clone());
        cands.push((text.split_whitespace().map(str::to_owned).collect(), score));
    }
    let idx = if cfg.mbr_enabled {
        mbr_select(&cands)
    } else {
        cands
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.1.total_cmp(&b.1).then_with(|| ib.cmp(ia)))
            .map_or(0, |(i, _)| i)
    };
    (texts.swap_remove(idx), idx)
}
