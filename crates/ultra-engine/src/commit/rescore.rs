//! Separate-vocabulary CTC forward rescoring and word-level Minimum Bayes Risk (MBR) selection
//! for the H45 endpoint commit pass (`crate::commit`).
//!
//! Correctness invariants:
//! 1. **Separate CTC vocabulary**: CTC rescoring always retokenizes candidate text into the
//!    checkpoint's own [`CtcVocab`]; RNN-T token IDs are never passed into CTC forward scoring.
//! 2. **Impossible CTC paths receive infinite cost**: OOV text, blank targets, or sequences with
//!    `num_frames < targets.len() + adjacent_repeats` return `f32::NEG_INFINITY` log-prob.

use std::collections::BTreeMap;

use crate::EngineError;

/// Separate CTC vocabulary used to retokenize candidate text before CTC forward scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtcVocab {
    blank_id: usize,
    vocab_size: usize,
    uses_sp_marker: bool,
    pieces: BTreeMap<String, usize>,
    max_piece_bytes: usize,
}

impl CtcVocab {
    pub fn new(blank_id: usize, tokens: Vec<String>) -> Result<Self, EngineError> {
        let vocab_size = tokens.len();
        if vocab_size == 0 || blank_id >= vocab_size {
            return Err(EngineError::Format(format!(
                "invalid CTC vocab: blank_id {blank_id} for {vocab_size} entries"
            )));
        }
        let mut pieces = BTreeMap::new();
        let (mut max_piece_bytes, mut uses_sp_marker) = (0usize, false);
        for (id, piece) in tokens.into_iter().enumerate() {
            if id == blank_id || piece.is_empty() {
                continue;
            }
            uses_sp_marker |= piece.contains('\u{2581}');
            max_piece_bytes = max_piece_bytes.max(piece.len());
            if pieces.insert(piece, id).is_some() {
                return Err(EngineError::Format(format!(
                    "duplicate non-blank CTC piece at id {id}"
                )));
            }
        }
        Ok(Self {
            blank_id,
            vocab_size,
            uses_sp_marker,
            pieces,
            max_piece_bytes,
        })
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Retokenize candidate text into separate CTC token IDs (greedy longest-match).
    /// Returns `None` if any span is OOV in the CTC vocabulary.
    pub fn retokenize_text(&self, text: &str) -> Option<Vec<usize>> {
        let norm: Vec<&str> = text.split_whitespace().collect();
        if norm.is_empty() {
            return Some(Vec::new());
        }
        let encoded = if self.uses_sp_marker {
            format!("\u{2581}{}", norm.join("\u{2581}"))
        } else {
            norm.join(" ")
        };
        let (mut out, mut pos) = (Vec::new(), 0usize);
        while pos < encoded.len() {
            let mut end = (pos + self.max_piece_bytes).min(encoded.len());
            let mut matched = None;
            while end > pos {
                if encoded.is_char_boundary(end) {
                    if let Some(&id) = self.pieces.get(&encoded[pos..end]) {
                        matched = Some((id, end));
                        break;
                    }
                }
                end -= 1;
            }
            let (id, next_pos) = matched?;
            out.push(id);
            pos = next_pos;
        }
        Some(out)
    }
}

/// Exact word-level Levenshtein edit distance.
pub fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return n + m;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
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
    let mut probs = vec![0.0f64; n];
    let mut sum_p = 0.0f64;
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
        let r_diff = risks[i] - risks[best];
        if r_diff < -1e-12 || (r_diff.abs() <= 1e-12 && candidates[i].1 > candidates[best].1) {
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
/// Assigns `f32::NEG_INFINITY` (infinite cost) to any impossible path (invalid target, blank in
/// targets, or `num_frames < targets.len() + adjacent_repeats`).
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
    prev[0] = emit(0, blank_id);
    if !targets.is_empty() {
        prev[1] = emit(0, targets[0]);
    }
    let mut curr = vec![f64::NEG_INFINITY; ext_len];
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
