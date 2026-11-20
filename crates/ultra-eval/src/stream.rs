//! Scoring of a recorded transcript event stream (v1.0-B §12.4, §30.2, §33; v1 §11.3).
//!
//! The events are replayed through [`ultra_core::events::Ledger`], which refuses any
//! event that breaks the output contract; a refused stream is an instrument failure,
//! not a score. From the replay this module produces:
//!
//! * `final_text`: immutable text of segments closed by `Final` (Gap-closed and
//!   never-closed segments contribute nothing);
//! * `commit_text`: the whole immutable ledger (`Ledger::committed_text`);
//! * segment-local preview snapshots (immutable prefix + mutable suffix after every
//!   text-bearing event);
//! * preview flicker: displayed words removed or substituted between consecutive
//!   snapshots of a segment (alignment S + D; growth is not flicker), per 100 final
//!   words. It reflects what a client renders under the ledger semantics, including a
//!   `Commit` that clears the suffix;
//! * commit revisions: previously committed normalised words that later changed (e.g.
//!   a commit that glues characters onto the last committed word). Must be 0;
//! * per-word stable times for every normalised word of `commit_text`:
//!   - retrospective (editable preview): the earliest snapshot time after which the
//!     word is matched in every later snapshot of its segment. Matching uses the
//!     frozen alignment of [`crate::wer::align`] between the segment's final words and
//!     each snapshot; a gap in presence restarts the clock. Ambiguous matches of
//!     repeated words are resolved both ways (latest and earliest tie-break) and the
//!     later of the two stable times is kept (v1.0-B §12.6 conservative later time);
//!   - commit: the earliest time after which the word sits, unchanged, in the segment's
//!     immutable text (the actual irreversible event).
//!
//! A segment still open at the end of the stream gets a terminal snapshot of its
//! immutable text at the time of the last event. Cost per segment is
//! O(snapshots x final words x snapshot words).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use ultra_core::events::{Ledger, SegmentStatus, TranscriptEvent};

use crate::normalise::normalise_words;
use crate::wer::{align, align_with, TieBreak};
use crate::EvalError;

/// An event with its observation time in seconds on the stream's monotonic clock
/// (`observed_monotonic - stream_start_monotonic`, v1.0-B §12.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedEvent {
    pub t_s: f64,
    pub event: TranscriptEvent,
}

/// The displayed text of one segment after one text-bearing event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreviewSnapshot {
    pub t_s: f64,
    pub event_sequence: u64,
    pub segment_id: u64,
    pub text: String,
}

/// Result of replaying one stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamScore {
    pub final_text: String,
    pub commit_text: String,
    pub snapshots: Vec<PreviewSnapshot>,
    /// Normalised words of `commit_text`, in order.
    pub words: Vec<String>,
    /// Retrospective stable time of each word in `words`.
    pub stable_t_s: Vec<f64>,
    /// Commit time of each word in `words`.
    pub commit_t_s: Vec<f64>,
    pub flicker_changed_words: u64,
    pub flicker_per_100_words: Option<f64>,
    pub commit_revisions: u64,
    pub incomplete_segments: u64,
    pub unclosed_segments: u64,
    /// Every successful `Final`: segment, owned sample interval, observation time (for FC).
    pub finals: Vec<crate::finalize::FinalObs>,
}

#[derive(Default)]
struct SegTrack {
    display: Vec<(f64, Vec<String>)>,
    committed: Vec<(f64, Vec<String>)>,
}

/// Replay a stream and score it.
pub fn replay(events: &[TimedEvent]) -> Result<StreamScore, EvalError> {
    let mut ledger = Ledger::new();
    let mut segs: BTreeMap<u64, SegTrack> = BTreeMap::new();
    let mut snapshots = Vec::new();
    let mut last_t = f64::NEG_INFINITY;
    let mut commit_revisions = 0u64;
    let mut final_times: Vec<crate::finalize::FinalObs> = Vec::new();
    for te in events {
        let ev = &te.event;
        if !te.t_s.is_finite() || te.t_s < last_t {
            return Err(EvalError::EventTime(ev.event_sequence));
        }
        last_t = te.t_s;
        let applied = ledger.apply(ev).map_err(|source| EvalError::Ledger {
            seq: ev.event_sequence,
            source,
        })?;
        if applied == ultra_core::events::Applied::Duplicate {
            continue; // a retransmission is not a new display observation
        }
        if let ultra_core::events::Body::Final { update, .. } = &ev.body {
            final_times.push(crate::finalize::FinalObs {
                segment_id: update.segment_id,
                owned_start_sample: update.ownership_start_sample,
                owned_end_sample: update.ownership_end_sample,
                t_s: te.t_s,
            });
        }
        if !ev.is_revision() {
            continue;
        }
        let Some(seg_id) = ev.segment_id() else {
            continue;
        };
        let Some(st) = ledger.segment(seg_id) else {
            continue;
        };
        let shown = st.shown();
        let track = segs.entry(seg_id).or_default();
        let committed = normalise_words(&st.immutable);
        if let Some((_, prev)) = track.committed.last() {
            commit_revisions += prev
                .iter()
                .enumerate()
                .filter(|(j, w)| committed.get(*j) != Some(*w))
                .count() as u64;
        }
        track.committed.push((te.t_s, committed));
        track.display.push((te.t_s, normalise_words(&shown)));
        snapshots.push(PreviewSnapshot {
            t_s: te.t_s,
            event_sequence: ev.event_sequence,
            segment_id: seg_id,
            text: shown,
        });
    }

    let mut out = StreamScore {
        final_text: String::new(),
        commit_text: ledger.committed_text(),
        snapshots,
        words: Vec::new(),
        stable_t_s: Vec::new(),
        commit_t_s: Vec::new(),
        flicker_changed_words: 0,
        flicker_per_100_words: None,
        commit_revisions,
        incomplete_segments: 0,
        unclosed_segments: 0,
        finals: final_times,
    };
    let mut finals = Vec::new();
    for (id, track) in &mut segs {
        let Some(st) = ledger.segment(*id) else {
            continue;
        };
        let fin = normalise_words(&st.immutable);
        if st.status == SegmentStatus::Incomplete {
            out.incomplete_segments += 1;
        } else if st.status == SegmentStatus::Final {
            let t = st.immutable.trim();
            if !t.is_empty() {
                finals.push(t.to_owned());
            }
        } else {
            out.unclosed_segments += 1;
            track.display.push((last_t, fin.clone()));
            track.committed.push((last_t, fin.clone()));
        }
        for pair in track.display.windows(2) {
            let c = align(&pair[0].1, &pair[1].1).counts;
            out.flicker_changed_words += c.substitutions + c.deletions;
        }
        let present = |tie| {
            move |f: &[String], snap: &[String]| -> Vec<bool> {
                align_with(f, snap, tie)
                    .ref_to_hyp_matches()
                    .into_iter()
                    .map(|m| m.is_some())
                    .collect()
            }
        };
        let late = stable_times(&fin, &track.display, present(TieBreak::Latest));
        let early = stable_times(&fin, &track.display, present(TieBreak::Earliest));
        out.stable_t_s
            .extend(late.into_iter().zip(early).map(|(a, b)| a.max(b)));
        out.commit_t_s
            .extend(stable_times(&fin, &track.committed, |f, snap| {
                f.iter()
                    .enumerate()
                    .map(|(j, w)| snap.get(j) == Some(w))
                    .collect()
            }));
        out.words.extend(fin);
    }
    out.final_text = finals.join(" ");
    let n = out.words.len();
    out.flicker_per_100_words =
        (n > 0).then(|| 100.0 * out.flicker_changed_words as f64 / n as f64);
    Ok(out)
}

/// Earliest snapshot time after which each final word is present in every later
/// snapshot. `present(final, snapshot)` returns one flag per final word. A word not
/// present in the last snapshot gets the last snapshot's time (cannot happen when the
/// last snapshot equals the final words, which `replay` guarantees).
fn stable_times<F>(fin: &[String], snaps: &[(f64, Vec<String>)], present: F) -> Vec<f64>
where
    F: Fn(&[String], &[String]) -> Vec<bool>,
{
    let last = snaps.last().map_or(0.0, |s| s.0);
    let mut stable = vec![last; fin.len()];
    let mut running = vec![true; fin.len()];
    for (t, snap) in snaps.iter().rev() {
        let p = present(fin, snap);
        for j in 0..fin.len() {
            if running[j] && p[j] {
                stable[j] = *t;
            } else {
                running[j] = false;
            }
        }
    }
    stable
}

/// Builds a timed event stream from simple calls (tests, fake backends). Wraps
/// [`ultra_core::events::Emitter`]; every call advances ownership by `samples_per_call`.
#[derive(Debug, Clone)]
pub struct ScriptedStream {
    emitter: ultra_core::events::Emitter,
    owned: u64,
    samples_per_call: u64,
}

impl Default for ScriptedStream {
    fn default() -> Self {
        Self::new("scripted", 1600)
    }
}

impl ScriptedStream {
    pub fn new(stream_id: &str, samples_per_call: u64) -> Self {
        Self {
            emitter: ultra_core::events::Emitter::new(stream_id, "scripted"),
            owned: 0,
            samples_per_call,
        }
    }

    fn timed(
        &mut self,
        t_s: f64,
        ev: Result<TranscriptEvent, ultra_core::CoreError>,
    ) -> TimedEvent {
        TimedEvent {
            t_s,
            event: ev.expect("scripted event is valid"),
        }
    }

    fn ns(t_s: f64) -> u64 {
        (t_s.max(0.0) * 1e9) as u64
    }

    /// Show `suffix` after the immutable prefix.
    pub fn preview(&mut self, t_s: f64, suffix: &str) -> TimedEvent {
        self.owned += self.samples_per_call;
        let ev = self.emitter.preview(Self::ns(t_s), suffix, self.owned);
        self.timed(t_s, ev)
    }

    /// Show `text` and commit all of it.
    pub fn commit(&mut self, t_s: f64, text: &str) -> TimedEvent {
        self.owned += self.samples_per_call;
        let ev = self
            .emitter
            .commit(Self::ns(t_s), text, text.len(), self.owned);
        self.timed(t_s, ev)
    }

    /// Commit `text` and close the segment (processing complete).
    pub fn finalize(&mut self, t_s: f64, text: &str) -> TimedEvent {
        self.owned += self.samples_per_call;
        let done = ultra_core::events::ProcessingEvidence {
            accepted_through_sample: self.owned,
            required_processed_through_sample: self.owned,
            flushed: true,
        };
        let ev = self.emitter.finalize(Self::ns(t_s), text, self.owned, done);
        self.timed(t_s, ev)
    }

    /// Input discontinuity; closes any open segment as incomplete.
    pub fn gap(&mut self, t_s: f64, range: Option<(u64, u64)>) -> TimedEvent {
        let ev = self.emitter.gap(Self::ns(t_s), range, "scripted gap");
        TimedEvent { t_s, event: ev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_revision_moves_stable_time() {
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.preview(1.0, "the cat"),
            sc.preview(2.0, "the cap"),
            sc.preview(3.0, "the cat sat"),
            sc.finalize(4.0, "the cat sat"),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.final_text, "the cat sat");
        assert_eq!(s.commit_text, "the cat sat");
        assert_eq!(s.words, vec!["the", "cat", "sat"]);
        assert_eq!(s.stable_t_s, vec![1.0, 3.0, 3.0]);
        assert_eq!(s.commit_t_s, vec![4.0, 4.0, 4.0]);
        // cat->cap (1 sub), cap->cat (1 sub); growth is not flicker.
        assert_eq!(s.flicker_changed_words, 2);
        assert!((s.flicker_per_100_words.unwrap() - 200.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.commit_revisions, 0);
        assert_eq!(s.snapshots.len(), 4);
    }

    #[test]
    fn commit_times_and_two_segments() {
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.preview(0.5, "hello"),
            sc.commit(1.0, "hello "),
            sc.preview(1.5, "world"),
            sc.finalize(2.0, "world"),
            sc.preview(3.0, "again"),
            sc.finalize(3.5, "again"),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.commit_text, "hello world again");
        assert_eq!(s.final_text, "hello world again");
        assert_eq!(s.stable_t_s, vec![0.5, 1.5, 3.0]);
        assert_eq!(s.commit_t_s, vec![1.0, 2.0, 3.5]);
        assert_eq!(s.flicker_changed_words, 0);
    }

    #[test]
    fn repeated_words_use_conservative_later_time() {
        // "go" appears, vanishes, then "go go" appears: the first go's earlier
        // appearance does not survive the vanishing.
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.preview(1.0, "go"),
            sc.preview(2.0, ""),
            sc.preview(3.0, "go go"),
            sc.finalize(4.0, "go go"),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.stable_t_s, vec![3.0, 3.0]);
        assert_eq!(s.flicker_changed_words, 1);
    }

    #[test]
    fn moved_word_restarts_clock() {
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.preview(1.0, "b a"),
            sc.preview(2.0, "a b"),
            sc.finalize(3.0, "a b"),
        ];
        let s = replay(&evs).unwrap();
        // "a" is matched in "b a" (alignment keeps one of a/b); the final "b" is not.
        assert_eq!(s.stable_t_s[1], 2.0);
        assert!(s.stable_t_s[0] <= 2.0);
    }

    #[test]
    fn gap_closes_incomplete_and_final_text_excludes_it() {
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.commit(1.0, "kept "),
            sc.preview(1.2, "lost"),
            sc.gap(1.5, Some((10, 20))),
            sc.finalize(2.0, "next"),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.commit_text, "kept next");
        assert_eq!(s.final_text, "next");
        assert_eq!(s.incomplete_segments, 1);
        assert_eq!(s.words, vec!["kept", "next"]);
        assert_eq!(s.flicker_changed_words, 1);
    }

    #[test]
    fn unclosed_segment_and_commit_revision() {
        let mut sc = ScriptedStream::default();
        let evs = vec![
            sc.commit(1.0, "hel"),
            sc.commit(2.0, "lo"),
            sc.preview(3.0, " there"),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.commit_revisions, 1);
        assert_eq!(s.unclosed_segments, 1);
        assert_eq!(s.final_text, "");
        assert_eq!(s.words, vec!["hello"]);
        assert_eq!(s.commit_t_s, vec![2.0]);
        // hel -> hello is one substitution; the suffix vanishing at the terminal
        // snapshot is one deletion.
        assert_eq!(s.flicker_changed_words, 2);
    }

    #[test]
    fn final_times_are_recorded_per_segment_once() {
        let mut sc = ScriptedStream::default();
        let a = sc.preview(0.5, "hi");
        let b = sc.finalize(1.0, "hi");
        let dup = b.clone();
        let c = sc.finalize(2.5, "there");
        let s = replay(&[a, b, dup, c]).unwrap();
        let got: Vec<(u64, f64)> = s.finals.iter().map(|f| (f.segment_id, f.t_s)).collect();
        assert_eq!(got, vec![(0, 1.0), (1, 2.5)]);
        assert_eq!(s.finals[1].owned_start_sample, s.finals[0].owned_end_sample);
    }

    #[test]
    fn delayed_retransmission_is_not_a_new_observation() {
        let mut sc = ScriptedStream::default();
        let a = sc.preview(1.0, "the cap");
        let b = sc.preview(2.0, "the cat");
        let mut again = a.clone();
        again.t_s = 3.0; // the old Preview arrives again, later
        let c = sc.finalize(4.0, "the cat");
        let s = replay(&[a, b, again, c]).unwrap();
        assert_eq!(s.snapshots.len(), 3, "the duplicate adds no snapshot");
        assert_eq!(s.stable_t_s, vec![1.0, 2.0]);
    }

    #[test]
    fn ledger_violation_and_bad_time_are_refused() {
        let mut sc = ScriptedStream::default();
        let first = sc.finalize(1.0, "a");
        let mut late = sc.preview(2.0, "b");
        // Point the preview back at the finalised segment: the ledger must refuse it.
        if let ultra_core::events::Body::Preview(u) = &mut late.event.body {
            u.segment_id = 0;
        }
        assert!(matches!(
            replay(&[first, late]),
            Err(EvalError::Ledger { seq: 1, .. })
        ));
        let mut sc = ScriptedStream::default();
        let a = sc.preview(2.0, "a");
        let mut b = sc.preview(2.5, "b");
        b.t_s = 1.0;
        assert_eq!(replay(&[a, b]), Err(EvalError::EventTime(1)));
    }
}
