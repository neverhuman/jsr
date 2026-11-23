//! First-text and first-word latency on the capture and monotonic clocks.

use serde::{Deserialize, Serialize};

use crate::delay::{quantile_nearest_rank, RefWord};
use crate::stream::StreamScore;

/// A reference-timed utterance. A missing first word is included in error and
/// deadline denominators and has no conditional stable or commit latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstWordObservation {
    pub id: String,
    pub speaker: String,
    pub first_text_design_s: Option<f64>,
    pub first_correct_stable_s: Option<f64>,
    pub first_correct_commit_s: Option<f64>,
}

impl FirstWordObservation {
    /// Report capture time minus the first reference word's end, and monotonic
    /// publication times minus speech onset. Correctness uses the first normalized
    /// word of each transcript, so a leading insertion can cause a miss.
    pub fn measure(
        id: String,
        speaker: String,
        first_reference: &RefWord,
        speech_onset_s: f64,
        first_text_through_sample: Option<u64>,
        score: &StreamScore,
    ) -> Self {
        let correct = score.words.first() == Some(&first_reference.word);
        Self {
            id,
            speaker,
            first_text_design_s: first_text_through_sample
                .map(|sample| sample as f64 / 16_000.0 - first_reference.end_s),
            first_correct_stable_s: correct.then(|| score.stable_t_s[0] - speech_onset_s),
            first_correct_commit_s: correct.then(|| score.commit_t_s[0] - speech_onset_s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstWordDeadline {
    pub deadline_ms_from_onset: u32,
    pub correct_stable_count: usize,
    pub correct_stable_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstWordReport {
    pub reference_timed_count: usize,
    pub first_text_count: usize,
    pub first_text_coverage: f64,
    pub first_text_design_mean_s: Option<f64>,
    pub first_word_errors: usize,
    pub first_word_error_fraction: f64,
    pub first_correct_stable_p50_s: Option<f64>,
    pub first_correct_stable_p95_s: Option<f64>,
    pub first_correct_commit_p50_s: Option<f64>,
    pub first_correct_commit_p95_s: Option<f64>,
    pub correct_by_deadline: Vec<FirstWordDeadline>,
    pub utterances: Vec<FirstWordObservation>,
}

pub fn first_word_report(utterances: Vec<FirstWordObservation>) -> Option<FirstWordReport> {
    if utterances.is_empty() {
        return None;
    }
    let n = utterances.len();
    let mut first_text_count = 0;
    let mut first_text_sum = 0.0;
    let mut stable = Vec::new();
    let mut commit = Vec::new();
    for row in &utterances {
        if let Some(text_s) = row.first_text_design_s {
            first_text_count += 1;
            first_text_sum += text_s;
        }
        stable.extend(row.first_correct_stable_s);
        commit.extend(row.first_correct_commit_s);
    }
    stable.sort_by(f64::total_cmp);
    commit.sort_by(f64::total_cmp);
    let correct_by_deadline = [300, 500]
        .into_iter()
        .map(|deadline| {
            let count = stable.partition_point(|&s| s <= f64::from(deadline) / 1_000.0);
            FirstWordDeadline {
                deadline_ms_from_onset: deadline,
                correct_stable_count: count,
                correct_stable_fraction: count as f64 / n as f64,
            }
        })
        .collect();
    let first_word_errors = n - stable.len();
    Some(FirstWordReport {
        reference_timed_count: n,
        first_text_count,
        first_text_coverage: first_text_count as f64 / n as f64,
        first_text_design_mean_s: (first_text_count > 0)
            .then(|| first_text_sum / first_text_count as f64),
        first_word_errors,
        first_word_error_fraction: first_word_errors as f64 / n as f64,
        first_correct_stable_p50_s: quantile_nearest_rank(&stable, 0.50),
        first_correct_stable_p95_s: quantile_nearest_rank(&stable, 0.95),
        first_correct_commit_p50_s: quantile_nearest_rank(&commit, 0.50),
        first_correct_commit_p95_s: quantile_nearest_rank(&commit, 0.95),
        correct_by_deadline,
        utterances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_misses_in_coverage_and_uses_distinct_clocks() {
        let row = |id: &str, text, stable, commit| FirstWordObservation {
            id: id.into(),
            speaker: "speaker".into(),
            first_text_design_s: text,
            first_correct_stable_s: stable,
            first_correct_commit_s: commit,
        };
        let report = first_word_report(vec![
            row("first", Some(-0.05), Some(0.2), Some(0.4)),
            row("second", Some(0.31), None, None),
            row("third", None, None, None),
        ])
        .unwrap();
        assert_eq!(report.first_text_design_mean_s, Some(0.13));
        assert_eq!(report.first_text_coverage, 2.0 / 3.0);
        assert_eq!(report.first_word_errors, 2);
        assert_eq!(report.first_correct_stable_p95_s, Some(0.2));
        assert_eq!(report.first_correct_commit_p95_s, Some(0.4));
        assert_eq!(
            report.correct_by_deadline[0].correct_stable_fraction,
            1.0 / 3.0
        );
        assert_eq!(report.correct_by_deadline[1].correct_stable_count, 1);
    }
}
