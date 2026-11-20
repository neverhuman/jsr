//! Delay metrics (v1.0-B §10, §12.4, §12.6): conditional stable-word delay quantiles
//! over correctly output words, timing coverage, and CQ(D) = reference words correctly
//! stable by e_i + D / ALL reference words.
//!
//! All times are seconds on one stream clock: reference word ends `e_i` are
//! `ref_end_sample / 16000` and output stable times are
//! `observed_monotonic - stream_start_monotonic`. Negative delays (correct early
//! predictions) are kept, never clamped. A reference word that is deleted,
//! substituted or never stable has no delay and counts as a miss in CQ.

use serde::{Deserialize, Serialize};

use crate::normalise::normalise_words;
use crate::wer::align;

/// Descriptive CQ deadlines of §12.6 plus the EFFICIENT primary (2,000 ms).
pub const CQ_DEADLINES_MS: [u32; 5] = [250, 500, 750, 1500, 2000];

/// One normalised reference word with its human-checked acoustic end time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefWord {
    pub word: String,
    pub end_s: f64,
}

/// Normalise time-stamped reference words. A word that expands into several words
/// (`don't` -> `do not`) gives every piece the original end time; a word that
/// normalises to nothing is dropped.
pub fn normalise_timed<S: AsRef<str>>(words: &[(S, f64)]) -> Vec<RefWord> {
    let mut out = Vec::new();
    for (w, end) in words {
        for n in normalise_words(w.as_ref()) {
            out.push(RefWord {
                word: n,
                end_s: *end,
            });
        }
    }
    out
}

/// Per reference word: `stable_time - end` if the word is exactly matched in the
/// final hypothesis by the frozen minimum-edit alignment, else `None`.
/// `hyp_stable_s[j]` is the stable time of `hyp_words[j]` (see `stream::replay`).
pub fn word_delays<S: AsRef<str>>(
    reference: &[RefWord],
    hyp_words: &[S],
    hyp_stable_s: &[f64],
) -> Vec<Option<f64>> {
    assert_eq!(
        hyp_words.len(),
        hyp_stable_s.len(),
        "one stable time per hypothesis word"
    );
    let ref_words: Vec<&str> = reference.iter().map(|r| r.word.as_str()).collect();
    align(&ref_words, hyp_words)
        .ref_to_hyp_matches()
        .into_iter()
        .zip(reference)
        .map(|(m, r)| m.map(|h| hyp_stable_s[h] - r.end_s))
        .collect()
}

/// Nearest-rank quantile of a sorted slice: element `ceil(q * n) - 1` (q in (0, 1]).
pub fn quantile_nearest_rank(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() || !(q > 0.0 && q <= 1.0) {
        return None;
    }
    let k = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    Some(sorted[k - 1])
}

/// CQ(D): fraction of ALL reference words with a finite delay <= `deadline_s`.
/// `None` only when there are no reference words.
pub fn cq(delays: &[Option<f64>], deadline_s: f64) -> Option<f64> {
    if delays.is_empty() {
        return None;
    }
    let ok = delays
        .iter()
        .filter(|d| d.is_some_and(|d| d <= deadline_s))
        .count();
    Some(ok as f64 / delays.len() as f64)
}

/// Per-word CQ successes (for paired bootstrap of Delta CQ).
pub fn cq_successes(delays: &[Option<f64>], deadline_s: f64) -> u64 {
    delays
        .iter()
        .filter(|d| d.is_some_and(|d| d <= deadline_s))
        .count() as u64
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CqPoint {
    pub deadline_ms: u32,
    pub correct_by_deadline: u64,
    pub cq: Option<f64>,
}

/// Conditional delay, coverage and CQ summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelayReport {
    pub ref_words: u64,
    /// Reference words correctly output and eligible for latency scoring.
    pub matched_words: u64,
    pub coverage: Option<f64>,
    /// Conditional (matched words only) nearest-rank quantiles, seconds.
    pub p50_s: Option<f64>,
    pub p95_s: Option<f64>,
    pub cq: Vec<CqPoint>,
}

/// Summarise per-word delays at the given deadlines.
pub fn delay_report(delays: &[Option<f64>], deadlines_ms: &[u32]) -> DelayReport {
    let mut finite: Vec<f64> = delays.iter().flatten().copied().collect();
    finite.sort_by(f64::total_cmp);
    let n = delays.len() as u64;
    DelayReport {
        ref_words: n,
        matched_words: finite.len() as u64,
        coverage: (n > 0).then(|| finite.len() as f64 / n as f64),
        p50_s: quantile_nearest_rank(&finite, 0.50),
        p95_s: quantile_nearest_rank(&finite, 0.95),
        cq: deadlines_ms
            .iter()
            .map(|&ms| {
                let d = f64::from(ms) / 1000.0;
                CqPoint {
                    deadline_ms: ms,
                    correct_by_deadline: cq_successes(delays, d),
                    cq: cq(delays, d),
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs(ws: &[(&str, f64)]) -> Vec<RefWord> {
        normalise_timed(ws)
    }

    #[test]
    fn deleted_and_substituted_words_are_misses() {
        let r = refs(&[("the", 1.0), ("cat", 1.5), ("sat", 2.0), ("down", 2.5)]);
        let hyp = ["the", "bat", "sat"];
        let stable = [1.2, 1.9, 2.1];
        let d = word_delays(&r, &hyp, &stable);
        assert_eq!(d.len(), 4);
        assert!((d[0].unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(d[1], None);
        assert!((d[2].unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(d[3], None);
        // 2 of 4 correct within 250 ms; the denominator is ALL reference words.
        assert_eq!(cq(&d, 0.25), Some(0.5));
        assert_eq!(cq(&d, 0.15), Some(0.25));
        let rep = delay_report(&d, &CQ_DEADLINES_MS);
        assert_eq!(rep.matched_words, 2);
        assert_eq!(rep.coverage, Some(0.5));
        assert!((rep.p50_s.unwrap() - 0.1).abs() < 1e-12);
        assert!((rep.p95_s.unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(rep.cq[0].correct_by_deadline, 2);
    }

    #[test]
    fn deleting_hard_words_cannot_raise_cq() {
        let r = refs(&[("a", 1.0), ("hard", 2.0), ("b", 3.0)]);
        let all = word_delays(&r, &["a", "hard", "b"], &[1.1, 3.0, 3.1]);
        let deleted = word_delays(&r, &["a", "b"], &[1.1, 3.1]);
        // Conditional p95 improves by deleting the slow word...
        let p_all = delay_report(&all, &[]).p95_s.unwrap();
        let p_del = delay_report(&deleted, &[]).p95_s.unwrap();
        assert!(p_del < p_all);
        // ...but CQ at any deadline does not.
        for ms in [100, 500, 1000, 2000] {
            let dl = f64::from(ms) / 1000.0;
            assert!(cq(&deleted, dl).unwrap() <= cq(&all, dl).unwrap());
        }
    }

    #[test]
    fn negative_delays_are_kept_and_empty_is_undefined() {
        let r = refs(&[("x", 2.0)]);
        let d = word_delays(&r, &["x"], &[1.5]);
        assert_eq!(d, vec![Some(-0.5)]);
        assert_eq!(cq(&d, 0.0), Some(1.0));
        assert_eq!(cq(&[], 0.5), None);
        let rep = delay_report(&[], &[750]);
        assert_eq!(rep.coverage, None);
        assert_eq!(rep.cq[0].cq, None);
    }

    #[test]
    fn contraction_pieces_share_end_time() {
        let r = refs(&[("Don't", 1.0), ("go!", 1.4)]);
        assert_eq!(
            r.iter()
                .map(|w| (w.word.as_str(), w.end_s))
                .collect::<Vec<_>>(),
            vec![("do", 1.0), ("not", 1.0), ("go", 1.4)]
        );
    }

    #[test]
    fn nearest_rank_quantiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(quantile_nearest_rank(&v, 0.5), Some(5.0));
        assert_eq!(quantile_nearest_rank(&v, 0.95), Some(10.0));
        assert_eq!(quantile_nearest_rank(&v, 0.1), Some(1.0));
        assert_eq!(quantile_nearest_rank(&[], 0.5), None);
        assert_eq!(quantile_nearest_rank(&v, 0.0), None);
    }

    proptest::proptest! {
        #[test]
        fn cq_is_monotone_in_deadline(ds in proptest::collection::vec(
            proptest::option::of(-1.0f64..3.0), 1..40), a in 0.0f64..2.0, b in 0.0f64..2.0) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            proptest::prop_assert!(cq(&ds, lo).unwrap() <= cq(&ds, hi).unwrap());
            let cov = delay_report(&ds, &[]).coverage.unwrap();
            proptest::prop_assert!(cq(&ds, 1e9).unwrap() <= cov + 1e-12);
        }
    }
}
