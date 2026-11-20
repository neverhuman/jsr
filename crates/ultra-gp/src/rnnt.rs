//! SentencePiece (`U+2581`) token table, sliding causal feature ring, chunk boundary
//! arithmetic, and per-stream cached greedy RNN-T state (`ultra.r1-export/v1`).

use std::path::Path;

use ultra_frontend::MEL_BINS;

use crate::GpError;

/// Validated `0..vocab_size` SentencePiece token table (`0 == "<blk>"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTable {
    pieces: Vec<String>,
}

impl TokenTable {
    pub fn from_file(path: &Path, expected_vocab: usize) -> Result<Self, GpError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str_checked(&content, expected_vocab)
    }

    pub fn from_str_checked(content: &str, expected_vocab: usize) -> Result<Self, GpError> {
        let mut slots: Vec<Option<String>> = vec![None; expected_vocab];
        let mut count = 0usize;
        for (line_no, raw) in content.lines().enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let (piece, idx_s) = line.rsplit_once(' ').ok_or_else(|| {
                GpError::Tokens(format!("malformed tokens.txt line {}", line_no + 1))
            })?;
            let idx: usize = idx_s.parse().map_err(|_| {
                GpError::Tokens(format!(
                    "non-integer token id '{}' on line {}",
                    idx_s,
                    line_no + 1
                ))
            })?;
            if idx >= expected_vocab {
                return Err(GpError::Tokens(format!(
                    "token id {idx} out of range 0..{expected_vocab}"
                )));
            }
            if slots[idx].replace(piece.to_owned()).is_some() {
                return Err(GpError::Tokens(format!("duplicate token id {idx}")));
            }
            count += 1;
        }
        if count != expected_vocab {
            return Err(GpError::Tokens(format!(
                "tokens.txt has {count} entries, expected {expected_vocab}"
            )));
        }
        let mut pieces = Vec::with_capacity(expected_vocab);
        for (idx, opt) in slots.into_iter().enumerate() {
            let p = opt.ok_or_else(|| GpError::Tokens(format!("missing token id {idx}")))?;
            pieces.push(p);
        }
        if pieces.first().map(String::as_str) != Some("<blk>") {
            return Err(GpError::Tokens(format!(
                "token 0 must be '<blk>', got {:?}",
                pieces.first()
            )));
        }
        Ok(Self { pieces })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Detokenize non-blank IDs identically to `bundle_infer.py`: concatenate pieces,
    /// replace `U+2581` (`'▁'`) with `' '`, and strip leading/trailing whitespace.
    pub fn decode_ids(&self, ids: &[i64]) -> String {
        let mut raw = String::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Ok(u) = usize::try_from(id) {
                if u > 0 {
                    if let Some(p) = self.pieces.get(u) {
                        raw.push_str(p);
                    }
                }
            }
        }
        raw.replace('\u{2581}', " ").trim().to_owned()
    }
}

/// Exact encoder chunk readiness rule from `bundle_infer.py:_bounds`:
/// - While open: chunk `[t0, t0 + chunk)` is ready when feature frame `4 * (t0 + chunk) - 4` exists.
/// - After `finished`: total encoder frames `T = ceil(ceil(F / 2) / 2) = ((F + 1) / 2 + 1) / 2`;
///   emits `(t0, min(t0 + chunk, T))` while `t0 < T`.
#[inline]
pub fn chunk_bounds(
    t_next: usize,
    n_feat: usize,
    chunk_frames: usize,
    finished: bool,
) -> Option<(usize, usize)> {
    let t0 = t_next;
    if finished {
        let total = (n_feat.div_ceil(2)).div_ceil(2);
        (t0 < total).then(|| (t0, (t0 + chunk_frames).min(total)))
    } else {
        let t1 = t0 + chunk_frames;
        let needed_last = 4 * t1 - 4;
        (n_feat > needed_last).then_some((t0, t1))
    }
}

/// Bounded per-stream buffer of normalized 80-bin mel frames supporting the `4*t0 - 6 ..= 4*(t0+n) - 4`
/// causal stem window with zero-padding for negative feature indices.
#[derive(Debug, Clone, Default)]
pub struct FeatureRing {
    rows: Vec<[f32; MEL_BINS]>,
    feat_base: usize,
    n_feat: usize,
}

impl FeatureRing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_feat(&self) -> usize {
        self.n_feat
    }

    pub fn push_rows(&mut self, new_rows: &[[f32; MEL_BINS]]) {
        self.rows.extend_from_slice(new_rows);
        self.n_feat += new_rows.len();
    }

    /// Write `4 * n + 3` rows (`[4*t0 - 6 ..= 4*(t0+n) - 4]`) into `dst` (which has length `(4*n + 3) * 80`)
    /// and discard frames older than the next chunk's `4*(t0+n) - 6` left overlap.
    pub fn extract_chunk_into(&mut self, t0: usize, n: usize, dst: &mut [f32]) {
        let rows_needed = 4 * n + 3;
        debug_assert_eq!(dst.len(), rows_needed * MEL_BINS);
        dst.fill(0.0);

        let lo: isize = 4 * (t0 as isize) - 6;
        let hi: isize = 4 * ((t0 + n) as isize) - 4;
        let a = lo.max(self.feat_base as isize);
        if a <= hi {
            let dst_row_start = (a - lo) as usize;
            let src_row_start = (a as usize) - self.feat_base;
            let count = ((hi - a) as usize + 1).min(self.rows.len().saturating_sub(src_row_start));
            for r in 0..count {
                let d_off = (dst_row_start + r) * MEL_BINS;
                dst[d_off..d_off + MEL_BINS].copy_from_slice(&self.rows[src_row_start + r]);
            }
        }

        let keep_from = (4 * ((t0 + n) as isize) - 6).max(0) as usize;
        if keep_from > self.feat_base {
            let drop_n = (keep_from - self.feat_base).min(self.rows.len());
            self.rows.drain(..drop_n);
            self.feat_base += drop_n;
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.rows.capacity() * MEL_BINS * 4
    }
}

/// Per-stream transducer decoder state (`ctx`, emitted non-blank `tokens`, and `forced_advances`).
#[derive(Debug, Clone)]
pub struct RnntStreamState {
    pub ctx: [i64; 2],
    pub tokens: Vec<i64>,
    pub forced_advances: u64,
}

impl RnntStreamState {
    pub fn new(ctx_init: [i64; 2]) -> Self {
        Self {
            ctx: ctx_init,
            tokens: Vec::new(),
            forced_advances: 0,
        }
    }
}

/// Checked first-index argmax matching `numpy.argmax` tie-breaking while rejecting empty or
/// non-finite (`NaN`/`±Inf`) logit slices rather than silently returning `<blk>` (`0`).
#[inline]
pub fn checked_argmax_f32(logits: &[f32]) -> Result<usize, GpError> {
    if logits.is_empty() {
        return Err(GpError::Output("empty joiner logits slice".into()));
    }
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(GpError::Output(format!(
                "non-finite joiner logit at index {i}: {v}"
            )));
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(best_i)
}

/// First-index argmax matching `numpy.argmax` tie-breaking on finite inputs.
#[inline]
pub fn argmax_f32(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}
