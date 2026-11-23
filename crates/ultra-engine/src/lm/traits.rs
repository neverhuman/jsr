//! Unified zero-overhead streaming token language model interface (`TokenLm`).
//!
//! Implemented by both the bit-packed n-gram trie ([`super::NgramLm`]) and the L2-resident
//! INT8 feed-forward neural LM ([`super::FfnnLm`]) so decoding and byte-ladder benchmarks can
//! score either backend polymorphically or statically without overhead.

use super::NgramLm;

/// Streaming token-level language model returning natural-log next-token probabilities.
pub trait TokenLm: Send + Sync {
    /// Natural-log probability `ln P(token | history)` where `history` is chronological
    /// (oldest to newest) and empty at beginning-of-sentence.
    fn score_next(&self, history: &[u32], token: u32) -> f32;

    /// Fill `out[..self.vocab_size()]` with `ln P(k | history)` for every token `k`.
    ///
    /// # Panics
    /// Panics if `out.len() < self.vocab_size()`.
    fn score_all(&self, history: &[u32], out: &mut [f32]);

    /// Output vocabulary size `V` (token IDs `0..V`).
    fn vocab_size(&self) -> usize;

    /// In-memory weight/table footprint in bytes.
    fn size_bytes(&self) -> usize;
}

impl TokenLm for NgramLm {
    #[inline]
    fn score_next(&self, history: &[u32], token: u32) -> f32 {
        let ctx = self.ctx_state(history);
        self.lm_logp(&ctx, token as usize)
    }

    fn score_all(&self, history: &[u32], out: &mut [f32]) {
        assert!(
            out.len() >= self.vocab,
            "TokenLm::score_all output buffer too small ({} < {})",
            out.len(),
            self.vocab
        );
        let ctx = self.ctx_state(history);
        for (tok, slot) in out.iter_mut().enumerate().take(self.vocab) {
            *slot = self.lm_logp(&ctx, tok);
        }
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    #[inline]
    fn size_bytes(&self) -> usize {
        self.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{FfnnActivation, FfnnLm};

    #[test]
    fn trait_object_scores_ngram_and_ffnn_uniformly() {
        let arpa =
            "\\data\\\nngram 1=3\n\n\\1-grams:\n-3.0\t<unk>\n-99.0\t<s>\n-1.0\t2\n\n\\end\\\n";
        let ngram = NgramLm::from_arpa_str(arpa, None, 4, false).expect("ngram");
        let ffnn = FfnnLm::from_parts(
            4,
            2,
            2,
            2,
            true,
            FfnnActivation::Relu,
            vec![1; 8],
            vec![0.1; 4],
            vec![1; 8],
            vec![0.1; 2],
            vec![0.0; 2],
            Vec::new(),
            Vec::new(),
            vec![0.0; 4],
        )
        .expect("ffnn");
        let models: [&dyn TokenLm; 2] = [&ngram, &ffnn];
        for lm in models {
            let mut row = vec![0.0; lm.vocab_size()];
            lm.score_all(&[2], &mut row);
            assert!(lm.size_bytes() > 0);
            assert_eq!(row[2].to_bits(), lm.score_next(&[2], 2).to_bits());
        }
    }
}
