//! Pure-Rust token n-gram language model with LODR bigram subtraction for frame-synchronous
//! RNN-T beam search (`ultra.lm-ngram/v2`, backward-compatible with `v1`).
//!
//! - **Bit-packed reverse-context SoA CSR trie:** sorted `tok` keys packed at exact
//!   `ceil(log2(vocab + 1))` bits (9 bits for BPE-500 → 1.125 B/node, 2.125 B/leaf with 8-bit
//!   quantile LUT) and bit-packed CSR `child_off` pointers.
//! - **Single-walk hypothesis backoff cache:** each hypothesis computes `bow_sum[1..=order]`
//!   once per token emission; scoring any candidate token `k` needs one reverse-trie walk plus
//!   one addition.
//! - **Dense `f32` LODR bigram table:** `(vocab + 1) x vocab` (~1.0 MB for `vocab = 501`),
//!   exact unquantized natural-log bigram scores including `<s>`, O(1) L2 load.
//! - **Zero C++ FFI, 100% safe Rust, bit-exact across x86 and ARM.**

mod arpa;
mod bitvec;
pub mod ffnn;
mod ffnn_io;
pub mod ilme;
mod io;
pub mod traits;

pub use ffnn::{FfnnActivation, FfnnLm};
pub use ilme::{IlmeArtifact, StatelessIlme, ILME_SCHEMA};
pub use traits::TokenLm;

use std::path::Path;

use crate::EngineError;
use bitvec::BitVec;

/// Maximum supported n-gram order (`1..=8`).
pub const MAX_ORDER: usize = 8;

/// Shallow-fusion + LODR hyperparameters (`train.lm_fusion` parity).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LmConfig {
    /// Weight on external token n-gram log-probability (`λ1`, natural log).
    pub lm_weight: f32,
    /// Weight subtracted for internal/LODR bigram log-probability (`λ2`, natural log).
    pub lodr_weight: f32,
    /// Penalty subtracted from the blank log-probability at every frame (`>= 0.0`).
    pub blank_penalty: f32,
}

impl LmConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        for (name, val) in [
            ("lm_weight", self.lm_weight),
            ("lodr_weight", self.lodr_weight),
            ("blank_penalty", self.blank_penalty),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(EngineError::Format(format!(
                    "{name} must be finite and >= 0 (got {val})"
                )));
            }
        }
        Ok(())
    }
}

/// Precomputed per-hypothesis LM context (cached on each [`crate::beam::Hyp`], updated only when
/// a non-blank token is emitted).
#[derive(Debug, Clone, Copy, Default)]
pub struct LmCtx {
    /// Context tokens in reverse chronological order: `rev[0]` is newest (`w_t` or `<s>`).
    pub rev: [u16; MAX_ORDER],
    /// Number of valid context tokens in `rev[..len]` (`0..=order-1`).
    pub len: u8,
    /// Cumulative backoff weight added when a candidate matches up to length `m` (`1..=len+1`).
    pub bow_sum: [f32; MAX_ORDER + 1],
    /// Previous token index into the dense LODR bigram table (`0..=vocab`, `vocab` = `<s>`).
    pub lodr_prev: u16,
    /// Airtight upper bound on `lm_logp(ctx, k)` for any `k` (exact branch-and-bound pruning).
    pub max_logp_ub: f32,
}

#[derive(Debug, Clone)]
pub(super) enum LevelVals {
    Q8 {
        prob_lut: Box<[f32; 256]>,
        bow_lut: Box<[f32; 256]>,
        q_prob: Vec<u8>,
        q_bow: Vec<u8>,
    },
    F32 {
        prob: Vec<f32>,
        bow: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub(super) struct TrieLevel {
    pub(in crate::lm) tok: BitVec,
    pub(in crate::lm) vals: LevelVals,
    pub(in crate::lm) child_off: BitVec,
}

impl TrieLevel {
    pub(super) fn len(&self) -> usize {
        self.tok.len()
    }

    #[inline]
    pub(super) fn prob(&self, idx: usize) -> f32 {
        match &self.vals {
            LevelVals::Q8 {
                prob_lut, q_prob, ..
            } => prob_lut[usize::from(q_prob[idx])],
            LevelVals::F32 { prob, .. } => prob[idx],
        }
    }

    #[inline]
    pub(super) fn bow(&self, idx: usize) -> f32 {
        match &self.vals {
            LevelVals::Q8 { bow_lut, q_bow, .. } => {
                if q_bow.is_empty() {
                    0.0
                } else {
                    bow_lut[usize::from(q_bow[idx])]
                }
            }
            LevelVals::F32 { bow, .. } => {
                if bow.is_empty() {
                    0.0
                } else {
                    bow[idx]
                }
            }
        }
    }

    #[inline]
    pub(super) fn find_child(&self, lo: usize, hi: usize, target: u16) -> Option<usize> {
        self.tok.find_sorted_u16(lo, hi, target)
    }

    #[inline]
    pub(super) fn child_range(&self, idx: usize) -> (usize, usize) {
        (
            self.child_off.get_u32(idx) as usize,
            self.child_off.get_u32(idx + 1) as usize,
        )
    }

    pub(super) fn memory_bytes(&self) -> usize {
        let vals_bytes = match &self.vals {
            LevelVals::Q8 { q_prob, q_bow, .. } => 2048 + q_prob.len() + q_bow.len(),
            LevelVals::F32 { prob, bow } => (prob.len() + bow.len()) * 4,
        };
        self.tok.memory_bytes() + self.child_off.memory_bytes() + vals_bytes
    }
}

/// Pure-Rust token n-gram LM + optional dense LODR bigram table (`ultra.lm-ngram/v1`).
#[derive(Debug, Clone)]
pub struct NgramLm {
    pub vocab: usize,
    pub order: usize,
    pub quantized: bool,
    pub has_eos: bool,
    pub(super) uni_prob: Vec<f32>,
    pub(super) uni_bow: Vec<f32>,
    pub(super) seen: Vec<bool>,
    pub(super) uni_child_off: Vec<u32>,
    pub(super) levels: Vec<TrieLevel>,
    pub(super) lodr_bigram: Vec<f32>,
    pub(super) max_lm_logp: f32,
}

impl NgramLm {
    #[inline]
    pub fn max_lm_logp(&self) -> f32 {
        self.max_lm_logp
    }

    #[inline]
    pub fn has_lodr(&self) -> bool {
        !self.lodr_bigram.is_empty()
    }

    #[inline]
    pub fn has_eos(&self) -> bool {
        self.has_eos
    }

    #[inline]
    pub fn bos_id(&self) -> u16 {
        self.vocab as u16
    }

    #[inline]
    pub fn eos_id(&self) -> Option<u16> {
        self.has_eos.then_some((self.vocab + 1) as u16)
    }

    pub fn memory_bytes(&self) -> usize {
        let mut total = self.uni_prob.len() * 4
            + self.uni_bow.len() * 4
            + self.seen.len()
            + self.uni_child_off.len() * 4
            + self.lodr_bigram.len() * 4;
        for lv in &self.levels {
            total += lv.memory_bytes();
        }
        total
    }

    #[inline]
    fn map_tok(&self, tok: usize) -> u16 {
        if tok < self.seen.len() && self.seen[tok] {
            tok as u16
        } else {
            0
        }
    }

    /// Compute context + cumulative backoff sums for `tokens` (KenLM `BeginSentence` semantics).
    pub fn ctx_state(&self, tokens: &[u32]) -> LmCtx {
        let mut ctx = LmCtx::default();
        let bos = self.bos_id();
        ctx.lodr_prev = tokens.last().map_or(bos, |&t| {
            let tu = t as usize;
            if tu < self.vocab {
                tu as u16
            } else {
                0
            }
        });
        let max_ctx = self.order.saturating_sub(1);
        if max_ctx == 0 {
            ctx.max_logp_ub = self.max_lm_logp;
            return ctx;
        }
        let mut len = 0usize;
        for &t in tokens.iter().rev().take(max_ctx) {
            ctx.rev[len] = self.map_tok(t as usize);
            len += 1;
        }
        if len < max_ctx {
            ctx.rev[len] = bos;
            len += 1;
        }
        ctx.len = len as u8;

        let mut bows = [0.0f32; MAX_ORDER];
        let first = usize::from(ctx.rev[0]);
        bows[0] = self.uni_bow[first];
        if len > 1 && !self.levels.is_empty() {
            let mut lo = self.uni_child_off[first] as usize;
            let mut hi = self.uni_child_off[first + 1] as usize;
            for (step, slot) in bows.iter_mut().enumerate().take(len).skip(1) {
                let lv = &self.levels[step - 1];
                if let Some(idx) = lv.find_child(lo, hi, ctx.rev[step]) {
                    *slot = lv.bow(idx);
                    if step + 1 < len && step < self.levels.len() {
                        (lo, hi) = lv.child_range(idx);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        ctx.bow_sum[len + 1] = 0.0;
        let mut max_bow = 0.0f32;
        for m in (1..=len).rev() {
            ctx.bow_sum[m] = ctx.bow_sum[m + 1] + bows[m - 1];
            max_bow = max_bow.max(ctx.bow_sum[m]);
        }
        ctx.max_logp_ub = max_bow + self.max_lm_logp;
        ctx
    }

    #[inline]
    pub fn lodr_logp(&self, ctx: &LmCtx, tok: usize) -> f32 {
        if self.lodr_bigram.is_empty() {
            return 0.0;
        }
        let k = if tok < self.vocab { tok } else { 0 };
        self.lodr_bigram[usize::from(ctx.lodr_prev) * self.vocab + k]
    }

    #[inline]
    fn lookup_logp(&self, ctx: &LmCtx, k: usize) -> f32 {
        let mut best_prob = self.uni_prob[k];
        let mut matched = 1usize;
        let clen = usize::from(ctx.len);
        if clen > 0 && !self.levels.is_empty() {
            let mut lo = self.uni_child_off[k] as usize;
            let mut hi = self.uni_child_off[k + 1] as usize;
            for step in 0..clen {
                let lv = &self.levels[step];
                if let Some(idx) = lv.find_child(lo, hi, ctx.rev[step]) {
                    best_prob = lv.prob(idx);
                    matched = step + 2;
                    if step + 1 < clen && step + 1 < self.levels.len() {
                        (lo, hi) = lv.child_range(idx);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        ctx.bow_sum[matched] + best_prob
    }

    #[inline]
    pub fn lm_logp(&self, ctx: &LmCtx, tok: usize) -> f32 {
        let k = usize::from(self.map_tok(tok));
        self.lookup_logp(ctx, k)
    }

    /// Natural-log end-of-sentence probability `ln P(</s> | ctx)` (`0.0` if `!self.has_eos`).
    #[inline]
    pub fn eos_logp(&self, ctx: &LmCtx) -> f32 {
        if !self.has_eos {
            return 0.0;
        }
        self.lookup_logp(ctx, self.vocab + 1)
    }

    /// Natural-log end-of-sentence probability `ln P(</s> | history)` (`0.0` if `!self.has_eos`).
    #[inline]
    pub fn score_eos(&self, history: &[u32]) -> f32 {
        if !self.has_eos {
            return 0.0;
        }
        let ctx = self.ctx_state(history);
        self.eos_logp(&ctx)
    }

    /// Select the highest-scoring hypothesis index after adding `eos_weight * score_eos(tokens)`.
    pub fn select_best_with_eos(&self, hyps: &[(&[u32], f32)], eos_weight: f32) -> Option<usize> {
        if hyps.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, &(toks, score)) in hyps.iter().enumerate() {
            let s = if self.has_eos && eos_weight != 0.0 {
                score + eos_weight * self.score_eos(toks)
            } else {
                score
            };
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        Some(best_idx)
    }

    pub fn from_arpa_str(
        lm_arpa: &str,
        lodr_arpa: Option<&str>,
        vocab: usize,
        quantize_bits8: bool,
    ) -> Result<Self, EngineError> {
        arpa::from_arpa_str_with_eos(lm_arpa, lodr_arpa, vocab, quantize_bits8, false)
    }

    pub fn from_arpa_str_with_eos(
        lm_arpa: &str,
        lodr_arpa: Option<&str>,
        vocab: usize,
        quantize_bits8: bool,
        include_eos: bool,
    ) -> Result<Self, EngineError> {
        arpa::from_arpa_str_with_eos(lm_arpa, lodr_arpa, vocab, quantize_bits8, include_eos)
    }

    pub fn from_arpa_files(
        lm_path: &Path,
        lodr_path: Option<&Path>,
        vocab: usize,
        quantize_bits8: bool,
    ) -> Result<Self, EngineError> {
        arpa::from_arpa_files_with_eos(lm_path, lodr_path, vocab, quantize_bits8, false)
    }

    pub fn from_arpa_files_with_eos(
        lm_path: &Path,
        lodr_path: Option<&Path>,
        vocab: usize,
        quantize_bits8: bool,
        include_eos: bool,
    ) -> Result<Self, EngineError> {
        arpa::from_arpa_files_with_eos(lm_path, lodr_path, vocab, quantize_bits8, include_eos)
    }

    pub fn save(&self, path: &Path) -> Result<(), EngineError> {
        io::save(self, path)
    }

    /// Save in legacy `ultra.lm-ngram/v1` unpacked format (for backward-compatibility tests).
    pub fn save_v1(&self, path: &Path) -> Result<(), EngineError> {
        io::save_v1(self, path)
    }

    pub fn load(path: &Path) -> Result<Self, EngineError> {
        io::load(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LM: &str = "\
\\data\\
ngram 1=6
ngram 2=5
ngram 3=2

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.3
-1.0\t2\t-0.2
-1.5\t3\t-0.4
-2.0\t4\t-0.1
-2.5\t5\t-0.5

\\2-grams:
-0.5\t<s> 2\t-0.15
-0.8\t2 3\t-0.25
-0.6\t3 4\t-0.35
-0.9\t2 4\t-0.05
-1.2\t4 5

\\3-grams:
-0.2\t<s> 2 3
-0.3\t2 3 4

\\end\\
";

    const SAMPLE_LODR: &str = "\
\\data\\
ngram 1=5
ngram 2=2

\\1-grams:
-3.5\t<unk>
-99.0\t<s>\t-0.2
-1.1\t2\t-0.3
-1.4\t3\t-0.1
-1.8\t4

\\2-grams:
-0.7\t<s> 2
-0.5\t2 3

\\end\\
";

    #[test]
    fn exact_arpa_backoff_and_lodr_parity() {
        let lm = NgramLm::from_arpa_str(SAMPLE_LM, Some(SAMPLE_LODR), 8, false).expect("parse");
        let ln10 = std::f32::consts::LN_10;
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-5;

        let c0 = lm.ctx_state(&[]);
        assert!(approx(lm.lm_logp(&c0, 2), -0.5 * ln10));
        assert!(approx(lm.lm_logp(&c0, 3), (-0.3 - 1.5) * ln10));
        assert!(approx(lm.lm_logp(&c0, 1), (-0.3 - 4.0) * ln10));
        assert!(approx(lm.lodr_logp(&c0, 2), -0.7 * ln10));
        assert!(approx(lm.lodr_logp(&c0, 3), (-0.2 - 1.4) * ln10));

        let c1 = lm.ctx_state(&[2]);
        assert!(approx(lm.lm_logp(&c1, 3), -0.2 * ln10));
        assert!(approx(lm.lm_logp(&c1, 4), (-0.15 - 0.9) * ln10));
        assert!(approx(lm.lm_logp(&c1, 5), (-0.15 - 0.2 - 2.5) * ln10));
        assert!(approx(lm.lodr_logp(&c1, 3), -0.5 * ln10));
        assert!(approx(lm.lodr_logp(&c1, 4), (-0.3 - 1.8) * ln10));

        let c2 = lm.ctx_state(&[2, 3]);
        assert!(approx(lm.lm_logp(&c2, 4), -0.3 * ln10));
        assert!(approx(lm.lm_logp(&c2, 5), (-0.25 - 0.4 - 2.5) * ln10));

        let c3 = lm.ctx_state(&[3, 5]);
        assert!(approx(lm.lm_logp(&c3, 2), (-0.5 - 1.0) * ln10));
        assert!(lm.max_lm_logp() <= 0.0);
    }

    #[test]
    fn binary_save_load_and_tamper_detection() {
        let lm = NgramLm::from_arpa_str(SAMPLE_LM, Some(SAMPLE_LODR), 8, true).expect("parse q8");
        let tmp_v2 = std::env::temp_dir().join(format!("ultra_lm_v2_{}.ulm", std::process::id()));
        let tmp_v1 = std::env::temp_dir().join(format!("ultra_lm_v1_{}.ulm", std::process::id()));
        lm.save(&tmp_v2).expect("save v2");
        lm.save_v1(&tmp_v1).expect("save v1");
        let loaded_v2 = NgramLm::load(&tmp_v2).expect("load v2");
        let loaded_v1 = NgramLm::load(&tmp_v1).expect("load v1");
        for hist in [&[][..], &[2][..], &[2, 3][..], &[4, 5][..]] {
            let c0 = lm.ctx_state(hist);
            let c1 = loaded_v1.ctx_state(hist);
            let c2 = loaded_v2.ctx_state(hist);
            for k in 1..8 {
                assert_eq!(
                    lm.lm_logp(&c0, k).to_bits(),
                    loaded_v2.lm_logp(&c2, k).to_bits()
                );
                assert_eq!(
                    lm.lm_logp(&c0, k).to_bits(),
                    loaded_v1.lm_logp(&c1, k).to_bits()
                );
                assert_eq!(
                    lm.lodr_logp(&c0, k).to_bits(),
                    loaded_v2.lodr_logp(&c2, k).to_bits()
                );
                assert_eq!(
                    lm.lodr_logp(&c0, k).to_bits(),
                    loaded_v1.lodr_logp(&c1, k).to_bits()
                );
            }
        }
        let mut bytes = std::fs::read(&tmp_v2).expect("read");
        bytes[10] ^= 0x55;
        std::fs::write(&tmp_v2, &bytes).expect("corrupt");
        assert!(matches!(
            NgramLm::load(&tmp_v2),
            Err(EngineError::Integrity(_))
        ));
        let _ = std::fs::remove_file(&tmp_v2);
        let _ = std::fs::remove_file(&tmp_v1);
    }
}
