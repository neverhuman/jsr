//! Finalization metrics (spec v1.0-B §10, §12.7): conditional finalization delay and
//! FC(D) = finalized reference episodes by deadline D / ALL reference episodes.
//!
//! An episode is one reference speech segment (an utterance, or a reference segment of a
//! long stream) with the time its speech ends on the stream clock. It is finalized at the
//! observation time of the successful `Final` that closes the segment owning it. A missing
//! `Final` (incomplete, cancelled, never closed) is a miss in FC and is excluded only from
//! the *conditional* delay percentiles, which are reported together with the coverage.

use serde::{Deserialize, Serialize};

use crate::delay::quantile_nearest_rank;

/// Default deadlines (ms) at which FC is reported.
pub const FC_DEADLINES_MS: [u32; 4] = [800, 1000, 1500, 2000];

/// One reference episode and what the system did with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    /// End of reference speech for the episode (seconds on the stream clock).
    pub ref_end_s: f64,
    /// Observation time of the successful `Final` that closed it, if any.
    pub final_t_s: Option<f64>,
}

impl Episode {
    /// Finalization delay; `None` when there was no successful Final.
    pub fn delay_s(&self) -> Option<f64> {
        self.final_t_s.map(|t| t - self.ref_end_s)
    }
}

/// One successful `Final` as observed by the replay: what it owned and when it arrived.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FinalObs {
    pub segment_id: u64,
    pub owned_start_sample: u64,
    pub owned_end_sample: u64,
    pub t_s: f64,
}

/// A reference speech episode on the stream's sample clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefEpisode {
    pub id: String,
    pub speech_start_sample: u64,
    pub speech_end_sample: u64,
    /// End of reference speech on the stream clock, seconds.
    pub ref_end_s: f64,
}

/// Map reference episodes to the successful Finals that own their speech (spec v1.0-B §12.7).
///
/// An episode is finalized only if its whole speech interval `[start, end)` is covered by
/// the owned intervals of successful Finals. Its finalization time is the latest of those
/// Finals. Any part owned by an incomplete or never-closed segment, or not owned at all,
/// makes it a miss. An empty speech interval is never finalized.
pub fn episodes_from_finals(refs: &[RefEpisode], finals: &[FinalObs]) -> Vec<Episode> {
    let mut sorted: Vec<&FinalObs> = finals.iter().collect();
    sorted.sort_by_key(|f| f.owned_start_sample);
    refs.iter()
        .map(|r| {
            let (mut cursor, mut latest) = (r.speech_start_sample, f64::NEG_INFINITY);
            for f in sorted.iter().filter(|f| {
                f.owned_end_sample > r.speech_start_sample
                    && f.owned_start_sample < r.speech_end_sample
            }) {
                if f.owned_start_sample > cursor {
                    break; // a hole in finalized ownership
                }
                cursor = cursor.max(f.owned_end_sample);
                latest = latest.max(f.t_s);
                if cursor >= r.speech_end_sample {
                    break;
                }
            }
            let covered =
                r.speech_end_sample > r.speech_start_sample && cursor >= r.speech_end_sample;
            Episode {
                id: r.id.clone(),
                ref_end_s: r.ref_end_s,
                final_t_s: covered.then_some(latest),
            }
        })
        .collect()
}

/// FC at one deadline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FcPoint {
    pub deadline_ms: u32,
    /// Finalized within the deadline / all episodes.
    pub fc: Option<f64>,
    pub successes: u64,
}

/// Finalization summary for a set of episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalizationReport {
    pub episodes: u64,
    pub finalized: u64,
    /// finalized / episodes.
    pub coverage: Option<f64>,
    /// Percentiles over finalized episodes only (state them with `coverage`).
    pub conditional_p50_s: Option<f64>,
    pub conditional_p95_s: Option<f64>,
    /// Finals that preceded the end of reference speech (premature closure; still counted).
    pub early_finals: u64,
    pub fc: Vec<FcPoint>,
}

/// Summarise finalization for `episodes` at `deadlines_ms`.
///
/// Non-finite times are treated as a missing Final (a broken observation never helps a
/// system). A Final observed before the reference speech ended has a negative delay; it
/// counts as within every deadline but is also reported in `early_finals`, because closing a
/// segment before the speech ended usually means words were cut.
pub fn finalization_report(episodes: &[Episode], deadlines_ms: &[u32]) -> FinalizationReport {
    let delays: Vec<Option<f64>> = episodes
        .iter()
        .map(|e| {
            e.delay_s()
                .filter(|d| d.is_finite() && e.ref_end_s.is_finite())
        })
        .collect();
    let mut finished: Vec<f64> = delays.iter().flatten().copied().collect();
    finished.sort_by(f64::total_cmp);
    let n = episodes.len() as u64;
    let ratio = |k: u64| (n > 0).then(|| k as f64 / n as f64);
    let fc = deadlines_ms
        .iter()
        .map(|&d| {
            let limit = f64::from(d) / 1000.0;
            let successes = delays.iter().flatten().filter(|&&x| x <= limit).count() as u64;
            FcPoint {
                deadline_ms: d,
                fc: ratio(successes),
                successes,
            }
        })
        .collect();
    FinalizationReport {
        episodes: n,
        finalized: finished.len() as u64,
        coverage: ratio(finished.len() as u64),
        conditional_p50_s: quantile_nearest_rank(&finished, 0.50),
        conditional_p95_s: quantile_nearest_rank(&finished, 0.95),
        early_finals: finished.iter().filter(|&&d| d < 0.0).count() as u64,
        fc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(id: &str, end: f64, fin: Option<f64>) -> Episode {
        Episode {
            id: id.into(),
            ref_end_s: end,
            final_t_s: fin,
        }
    }

    #[test]
    fn missing_finals_count_against_fc_but_not_conditional_delay() {
        let eps = vec![
            ep("a", 1.0, Some(1.4)),
            ep("b", 2.0, Some(3.2)),
            ep("c", 3.0, None),
            ep("d", 4.0, Some(4.9)),
        ];
        let r = finalization_report(&eps, &[500, 1000, 1500]);
        assert_eq!((r.episodes, r.finalized), (4, 3));
        assert_eq!(r.coverage, Some(0.75));
        // delays 0.4, 1.2, 0.9 -> FC(500)=1/4, FC(1000)=2/4, FC(1500)=3/4
        let fc: Vec<_> = r.fc.iter().map(|p| p.fc).collect();
        assert_eq!(fc, vec![Some(0.25), Some(0.5), Some(0.75)]);
        assert!((r.conditional_p95_s.unwrap() - 1.2).abs() < 1e-12);
        assert_eq!(r.early_finals, 0);
    }

    #[test]
    fn early_finals_are_reported_and_nonfinite_is_a_miss() {
        let eps = vec![
            ep("a", 2.0, Some(1.5)),
            ep("b", 1.0, Some(f64::NAN)),
            ep("c", f64::INFINITY, Some(3.0)),
        ];
        let r = finalization_report(&eps, &[1500]);
        assert_eq!((r.finalized, r.early_finals), (1, 1));
        assert_eq!(r.fc[0].successes, 1);
    }

    fn fin(seg: u64, a: u64, b: u64, t: f64) -> FinalObs {
        FinalObs {
            segment_id: seg,
            owned_start_sample: a,
            owned_end_sample: b,
            t_s: t,
        }
    }

    fn rf(id: &str, a: u64, b: u64) -> RefEpisode {
        RefEpisode {
            id: id.into(),
            speech_start_sample: a,
            speech_end_sample: b,
            ref_end_s: b as f64 / 16_000.0,
        }
    }

    #[test]
    fn episode_spanning_two_finalized_segments_takes_the_later_final() {
        let finals = [fin(0, 0, 16_000, 1.3), fin(1, 16_000, 40_000, 2.9)];
        let eps = episodes_from_finals(&[rf("a", 8_000, 24_000), rf("b", 30_000, 39_000)], &finals);
        assert_eq!(eps[0].final_t_s, Some(2.9));
        assert_eq!(eps[1].final_t_s, Some(2.9));
    }

    #[test]
    fn a_hole_or_unowned_tail_is_a_miss() {
        // Segment 1 closed incomplete, so samples 16000..24000 have no successful Final.
        let finals = [fin(0, 0, 16_000, 1.3), fin(2, 24_000, 48_000, 3.5)];
        let eps = episodes_from_finals(
            &[
                rf("hole", 10_000, 30_000),
                rf("tail", 40_000, 60_000),
                rf("empty", 5, 5),
            ],
            &finals,
        );
        assert!(eps.iter().all(|e| e.final_t_s.is_none()));
        let r = finalization_report(&eps, &[1500]);
        assert_eq!(r.fc[0].fc, Some(0.0));
    }

    #[test]
    fn empty_input_is_undefined_not_zero() {
        let r = finalization_report(&[], &FC_DEADLINES_MS);
        assert_eq!(r.coverage, None);
        assert!(r.fc.iter().all(|p| p.fc.is_none()));
        assert_eq!(r.conditional_p50_s, None);
    }
}
