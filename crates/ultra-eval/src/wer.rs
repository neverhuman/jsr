//! Word error rate (v1.0-B §30.1): Levenshtein word alignment with exact S/D/I counts.

use serde::{Deserialize, Serialize};

use crate::normalise::normalise_words;

/// Edit counts of one alignment or a pool of alignments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditCounts {
    pub ref_words: u64,
    pub hyp_words: u64,
    pub matches: u64,
    pub substitutions: u64,
    pub deletions: u64,
    pub insertions: u64,
}

impl EditCounts {
    /// S + D + I.
    pub fn errors(&self) -> u64 {
        self.substitutions + self.deletions + self.insertions
    }

    /// (S + D + I) / N, or `None` when there are no reference words (undefined).
    pub fn wer(&self) -> Option<f64> {
        (self.ref_words > 0).then(|| self.errors() as f64 / self.ref_words as f64)
    }

    /// Add another pool into this one.
    pub fn add(&mut self, o: &EditCounts) {
        self.ref_words += o.ref_words;
        self.hyp_words += o.hyp_words;
        self.matches += o.matches;
        self.substitutions += o.substitutions;
        self.deletions += o.deletions;
        self.insertions += o.insertions;
    }
}

/// One step of an alignment; indices point into the reference / hypothesis word lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum AlignOp {
    Match { r: usize, h: usize },
    Sub { r: usize, h: usize },
    Del { r: usize },
    Ins { h: usize },
}

/// A minimum-edit alignment and its counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alignment {
    pub ops: Vec<AlignOp>,
    pub counts: EditCounts,
}

impl Alignment {
    /// For each reference word, the hypothesis index it is exactly matched to.
    pub fn ref_to_hyp_matches(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.counts.ref_words as usize];
        for op in &self.ops {
            if let AlignOp::Match { r, h } = *op {
                out[r] = Some(h);
            }
        }
        out
    }

    /// For each hypothesis word, the reference index it is exactly matched to.
    pub fn hyp_to_ref_matches(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.counts.hyp_words as usize];
        for op in &self.ops {
            if let AlignOp::Match { r, h } = *op {
                out[h] = Some(r);
            }
        }
        out
    }
}

const DIAG: u8 = 0;
const UP: u8 = 1; // deletion of a reference word
const LEFT: u8 = 2; // insertion of a hypothesis word

/// Minimum-edit (unit cost) alignment of two word sequences.
///
/// Deterministic tie-breaking, frozen: at every cell prefer the diagonal (match or
/// substitution), then deletion, then insertion. The backtrace therefore aligns a
/// repeated hypothesis word to the latest compatible reference occurrence. Memory is
/// one byte per cell plus two cost rows.
pub fn align<S: AsRef<str>, T: AsRef<str>>(reference: &[S], hypothesis: &[T]) -> Alignment {
    align_with(reference, hypothesis, TieBreak::Latest)
}

/// Tie-breaking rule between equal-cost alignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieBreak {
    /// Prefer diagonal, then deletion, then insertion (the frozen scoring default):
    /// repeated words match the latest compatible reference occurrence.
    Latest,
    /// Prefer deletion, then insertion, then diagonal: repeated words match the
    /// earliest compatible reference occurrence. Used to bound ambiguous appearance
    /// times from the other side (`stream::replay`).
    Earliest,
}

/// [`align`] with an explicit tie-breaking rule. The edit cost is the same under
/// both rules; only the choice among equal-cost alignments differs.
pub fn align_with<S: AsRef<str>, T: AsRef<str>>(
    reference: &[S],
    hypothesis: &[T],
    tie: TieBreak,
) -> Alignment {
    let n = reference.len();
    let m = hypothesis.len();
    let w = m + 1;
    let mut dir = vec![0u8; (n + 1) * w];
    let mut prev: Vec<u32> = (0..=m as u32).collect();
    let mut cur = vec![0u32; w];
    for d in dir.iter_mut().take(w).skip(1) {
        *d = LEFT;
    }
    for i in 1..=n {
        cur[0] = i as u32;
        dir[i * w] = UP;
        let r = reference[i - 1].as_ref();
        for j in 1..=m {
            let sub = prev[j - 1] + u32::from(r != hypothesis[j - 1].as_ref());
            let del = prev[j] + 1;
            let ins = cur[j - 1] + 1;
            let (c, d) = match tie {
                TieBreak::Latest if sub <= del && sub <= ins => (sub, DIAG),
                TieBreak::Latest if del <= ins => (del, UP),
                TieBreak::Latest => (ins, LEFT),
                TieBreak::Earliest if del <= ins && del <= sub => (del, UP),
                TieBreak::Earliest if ins <= sub => (ins, LEFT),
                TieBreak::Earliest => (sub, DIAG),
            };
            cur[j] = c;
            dir[i * w + j] = d;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut ops = Vec::with_capacity(n.max(m));
    let mut counts = EditCounts {
        ref_words: n as u64,
        hyp_words: m as u64,
        ..EditCounts::default()
    };
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        match dir[i * w + j] {
            DIAG if i > 0 && j > 0 => {
                i -= 1;
                j -= 1;
                if reference[i].as_ref() == hypothesis[j].as_ref() {
                    counts.matches += 1;
                    ops.push(AlignOp::Match { r: i, h: j });
                } else {
                    counts.substitutions += 1;
                    ops.push(AlignOp::Sub { r: i, h: j });
                }
            }
            UP if i > 0 => {
                i -= 1;
                counts.deletions += 1;
                ops.push(AlignOp::Del { r: i });
            }
            _ => {
                j -= 1;
                counts.insertions += 1;
                ops.push(AlignOp::Ins { h: j });
            }
        }
    }
    ops.reverse();
    Alignment { ops, counts }
}

/// Normalise both texts with the pinned normaliser, then align.
pub fn align_text(reference: &str, hypothesis: &str) -> Alignment {
    align(&normalise_words(reference), &normalise_words(hypothesis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn w(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    fn counts(r: &str, h: &str) -> (u64, u64, u64) {
        let c = align(&w(r), &w(h)).counts;
        (c.substitutions, c.deletions, c.insertions)
    }

    #[test]
    fn known_answer_fixtures() {
        assert_eq!(counts("a b c", "a b c"), (0, 0, 0));
        assert_eq!(counts("a b c", "a x c"), (1, 0, 0));
        assert_eq!(counts("a b c", "a c"), (0, 1, 0));
        assert_eq!(counts("a b c", "a b b c"), (0, 0, 1));
        assert_eq!(counts("a b c", ""), (0, 3, 0));
        assert_eq!(counts("", "a b"), (0, 0, 2));
        assert_eq!(counts("", ""), (0, 0, 0));
        assert_eq!(
            counts("the cat sat on the mat", "cat sat in the the mat"),
            (1, 1, 1)
        );
        assert_eq!(counts("go go go", "go go"), (0, 1, 0));
    }

    #[test]
    fn reversal_costs_four() {
        let c = align(&w("a b c d"), &w("d c b a")).counts;
        assert_eq!(c.errors(), 4); // LCS is 1 word: all four positions cost one edit
    }

    #[test]
    fn alignment_output_and_tie_break() {
        let a = align(&w("go go"), &w("go"));
        // Prefer diagonal at the end: hypothesis "go" aligns to the later reference "go".
        assert_eq!(
            a.ops,
            vec![AlignOp::Del { r: 0 }, AlignOp::Match { r: 1, h: 0 }]
        );
        let e = align_with(&w("go go"), &w("go"), TieBreak::Earliest);
        assert_eq!(
            e.ops,
            vec![AlignOp::Match { r: 0, h: 0 }, AlignOp::Del { r: 1 }]
        );
        let a = align(&w("a b"), &w("a x b"));
        assert_eq!(
            a.ops,
            vec![
                AlignOp::Match { r: 0, h: 0 },
                AlignOp::Ins { h: 1 },
                AlignOp::Match { r: 1, h: 2 }
            ]
        );
        assert_eq!(a.ref_to_hyp_matches(), vec![Some(0), Some(2)]);
        assert_eq!(a.hyp_to_ref_matches(), vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn normalised_text_alignment() {
        let a = align_text("It's a TEST.", "it is a test");
        assert_eq!(a.counts.errors(), 0);
        assert_eq!(a.counts.ref_words, 4);
    }

    fn words() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(prop::sample::select(vec!["a", "b", "c", "go"]), 0..12)
            .prop_map(|v| v.into_iter().map(String::from).collect())
    }

    /// Reference full-matrix edit distance for cross-checking.
    fn lev(a: &[String], b: &[String]) -> u64 {
        let mut d = vec![vec![0u64; b.len() + 1]; a.len() + 1];
        for (i, row) in d.iter_mut().enumerate() {
            row[0] = i as u64;
        }
        for (j, cell) in d[0].iter_mut().enumerate() {
            *cell = j as u64;
        }
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                let s = d[i - 1][j - 1] + u64::from(a[i - 1] != b[j - 1]);
                d[i][j] = s.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
            }
        }
        d[a.len()][b.len()]
    }

    proptest! {
        #[test]
        fn cost_equals_s_d_i_and_edit_distance(r in words(), h in words()) {
            let a = align(&r, &h);
            let c = a.counts;
            prop_assert_eq!(c.errors(), lev(&r, &h));
            prop_assert_eq!(c.matches + c.substitutions + c.deletions, r.len() as u64);
            prop_assert_eq!(c.matches + c.substitutions + c.insertions, h.len() as u64);
            prop_assert_eq!(a.ops.len() as u64, c.matches + c.errors());
        }

        #[test]
        fn tie_breaks_have_equal_cost(r in words(), h in words()) {
            let a = align_with(&r, &h, TieBreak::Latest).counts;
            let b = align_with(&r, &h, TieBreak::Earliest).counts;
            prop_assert_eq!(a.errors(), b.errors());
        }

        #[test]
        fn identical_is_zero(r in words()) {
            let c = align(&r, &r).counts;
            prop_assert_eq!(c.errors(), 0);
            if !r.is_empty() { prop_assert_eq!(c.wer(), Some(0.0)); }
        }

        #[test]
        fn symmetric_cost_swaps_d_and_i(r in words(), h in words()) {
            let a = align(&r, &h).counts;
            let b = align(&h, &r).counts;
            prop_assert_eq!(a.errors(), b.errors());
            // S+D+I is symmetric; matches+S cover min(len) consistently.
            prop_assert_eq!(a.deletions as i64 - a.insertions as i64,
                            b.insertions as i64 - b.deletions as i64);
        }

        #[test]
        fn ops_are_monotone_and_cover(r in words(), h in words()) {
            let a = align(&r, &h);
            let (mut ri, mut hi) = (0usize, 0usize);
            for op in &a.ops {
                match *op {
                    AlignOp::Match { r: x, h: y } => {
                        prop_assert_eq!((x, y), (ri, hi));
                        prop_assert_eq!(&r[x], &h[y]);
                        ri += 1; hi += 1;
                    }
                    AlignOp::Sub { r: x, h: y } => {
                        prop_assert_eq!((x, y), (ri, hi));
                        prop_assert_ne!(&r[x], &h[y]);
                        ri += 1; hi += 1;
                    }
                    AlignOp::Del { r: x } => { prop_assert_eq!(x, ri); ri += 1; }
                    AlignOp::Ins { h: y } => { prop_assert_eq!(y, hi); hi += 1; }
                }
            }
            prop_assert_eq!((ri, hi), (r.len(), h.len()));
        }
    }
}
