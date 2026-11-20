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
//!     each snapshot; a gap in presence restarts the clock (conservative later time);
//!   - commit: the earliest time after which the word sits, unchanged, in the segment's
//!     immutable text (the actual irreversible event).
//!
//! A segment still open at the end of the stream gets a terminal snapshot of its
//! immutable text at the time of the last event. Cost per segment is
//! O(snapshots x final words x snapshot words).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use ultra_core::events::{EventKind, Ledger, TranscriptEvent};

use crate::normalise::normalise_words;
use crate::wer::align;
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
    for te in events {
        let ev = &te.event;
        if !te.t_s.is_finite() || te.t_s < last_t {
            return Err(EvalError::EventTime(ev.event_sequence));
        }
        last_t = te.t_s;
        ledger.apply(ev).map_err(|source| EvalError::Ledger {
            seq: ev.event_sequence,
            source,
        })?;
        let text_bearing = matches!(
            ev.kind,
            EventKind::Preview { .. }
                | EventKind::Commit { .. }
                | EventKind::Final { .. }
                | EventKind::Gap { .. }
        );
        if !text_bearing {
            continue;
        }
        let Some(st) = ledger.segment(ev.segment_id) else {
            continue;
        };
        let shown = format!("{}{}", st.immutable, st.suffix);
        let track = segs.entry(ev.segment_id).or_default();
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
            segment_id: ev.segment_id,
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
    };
    let mut finals = Vec::new();
    for (id, track) in &mut segs {
        let Some(st) = ledger.segment(*id) else {
            continue;
        };
        let fin = normalise_words(&st.immutable);
        if st.incomplete {
            out.incomplete_segments += 1;
        } else if st.closed {
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
        out.stable_t_s
            .extend(stable_times(&fin, &track.display, |f, snap| {
                align(f, snap)
                    .ref_to_hyp_matches()
                    .into_iter()
                    .map(|m| m.is_some())
                    .collect()
            }));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: f64, seq: u64, seg: u64, kind: EventKind) -> TimedEvent {
        TimedEvent {
            t_s: t,
            event: TranscriptEvent {
                stream_id: "s".into(),
                segment_id: seg,
                event_sequence: seq,
                audio_start_sample: 0,
                audio_end_sample: 1,
                kind,
            },
        }
    }
    fn pv(rev: u64, s: &str) -> EventKind {
        EventKind::Preview {
            revision_id: rev,
            replacement_suffix: s.into(),
        }
    }
    fn cm(s: &str) -> EventKind {
        EventKind::Commit { text: s.into() }
    }
    fn fi(s: &str) -> EventKind {
        EventKind::Final { text: s.into() }
    }

    #[test]
    fn preview_revision_moves_stable_time() {
        let evs = vec![
            ev(1.0, 1, 0, pv(1, "the cat")),
            ev(2.0, 2, 0, pv(2, "the cap")),
            ev(3.0, 3, 0, pv(3, "the cat sat")),
            ev(4.0, 4, 0, fi("the cat sat")),
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
        let evs = vec![
            ev(0.5, 1, 0, pv(1, "hello")),
            ev(1.0, 2, 0, cm("hello ")),
            ev(1.5, 3, 0, pv(2, "world")),
            ev(2.0, 4, 0, fi("world")),
            ev(3.0, 5, 1, pv(1, "again")),
            ev(3.5, 6, 1, fi("again")),
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
        let evs = vec![
            ev(1.0, 1, 0, pv(1, "go")),
            ev(2.0, 2, 0, pv(2, "")),
            ev(3.0, 3, 0, pv(3, "go go")),
            ev(4.0, 4, 0, fi("go go")),
        ];
        let s = replay(&evs).unwrap();
        assert_eq!(s.stable_t_s, vec![3.0, 3.0]);
        assert_eq!(s.flicker_changed_words, 1);
    }

    #[test]
    fn moved_word_restarts_clock() {
        let evs = vec![
            ev(1.0, 1, 0, pv(1, "b a")),
            ev(2.0, 2, 0, pv(2, "a b")),
            ev(3.0, 3, 0, fi("a b")),
        ];
        let s = replay(&evs).unwrap();
        // "a" is matched in "b a" (alignment keeps one of a/b); the final "b" is not.
        assert_eq!(s.stable_t_s[1], 2.0);
        assert!(s.stable_t_s[0] <= 2.0);
    }

    #[test]
    fn gap_closes_incomplete_and_final_text_excludes_it() {
        let evs = vec![
            ev(1.0, 1, 0, cm("kept ")),
            ev(1.2, 2, 0, pv(1, "lost")),
            ev(
                1.5,
                3,
                0,
                EventKind::Gap {
                    start_sample: 10,
                    end_sample: 20,
                },
            ),
            ev(2.0, 4, 1, fi("next")),
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
        let evs = vec![
            ev(1.0, 1, 0, cm("hel")),
            ev(2.0, 2, 0, cm("lo")),
            ev(3.0, 3, 0, pv(1, " there")),
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
    fn ledger_violation_and_bad_time_are_refused() {
        let evs = vec![ev(1.0, 1, 0, fi("a")), ev(2.0, 2, 0, pv(1, "b"))];
        assert!(matches!(
            replay(&evs),
            Err(EvalError::Ledger { seq: 2, .. })
        ));
        let evs = vec![ev(2.0, 1, 0, pv(1, "a")), ev(1.0, 2, 0, pv(2, "b"))];
        assert_eq!(replay(&evs), Err(EvalError::EventTime(2)));
    }
}
