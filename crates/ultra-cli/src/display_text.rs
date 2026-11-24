//! Printed transcript for `ultra transcribe` / `ultra stream`.
//!
//! The ledger's committed text is the wire contract and stays untouched. One case needs a
//! presentation-only recovery: the coordinator's `segment_hard_cap` (30 s by default) closes the
//! open segment as `SegmentIncomplete`, which keeps only the immutable prefix, while the inner
//! decoder carries on and the next segment skips exactly the bytes that were already shown. The
//! full transcript is then the capped segment's last displayed snapshot followed by the next
//! segment. Without this, a 40 s utterance printed only its last 10 s.
//!
//! Only hard-cap closures are recovered, and only from the snapshot whose revision the closure
//! names. Cancellation, timeouts, backend failures and `Gap`s keep the ledger's preserved prefix,
//! so text that was never produced is never claimed as final.

use std::collections::BTreeMap;

use ultra_core::events::{Body, Ledger, TranscriptEvent};

/// Closure reason the coordinator uses for its segment hard cap.
pub const HARD_CAP_REASON: &str = "segment_hard_cap";

/// The transcript to print. Equals `ledger.committed_text()` unless a hard-cap closure occurred.
pub fn display_text<'a, I>(ledger: &Ledger, events: I) -> String
where
    I: IntoIterator<Item = &'a TranscriptEvent>,
{
    // segment -> (revision, displayed snapshot) of the last text update seen.
    let mut last: BTreeMap<u64, (u64, &str)> = BTreeMap::new();
    // segment -> printed text; mirrors the ledger except for recovered hard-cap segments.
    let mut parts: BTreeMap<u64, &str> = BTreeMap::new();
    let mut recovered = false;
    for ev in events {
        match &ev.body {
            Body::Preview(u) | Body::Commit(u) => {
                last.insert(u.segment_id, (u.revision, u.snapshot_utf8.as_str()));
                let imm = u
                    .snapshot_utf8
                    .get(..u.immutable_after_utf8_bytes)
                    .unwrap_or("");
                parts.insert(u.segment_id, imm);
            }
            Body::Final { update, .. } => {
                last.insert(update.segment_id, (update.revision, &update.snapshot_utf8));
                parts.insert(update.segment_id, &update.snapshot_utf8);
            }
            Body::SegmentIncomplete { closure, .. } if closure.reason == HARD_CAP_REASON => {
                if let Some((rev, snap)) = last.get(&closure.segment_id) {
                    if *rev == closure.base_revision {
                        parts.insert(closure.segment_id, snap);
                        recovered = true;
                    }
                }
            }
            _ => {}
        }
    }
    if !recovered {
        return ledger.committed_text();
    }
    let kept: Vec<&str> = parts
        .values()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    kept.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultra_core::events::{Emitter, ProcessingEvidence};

    fn done(through: u64) -> ProcessingEvidence {
        ProcessingEvidence {
            accepted_through_sample: through,
            required_processed_through_sample: through,
            flushed: true,
        }
    }

    fn apply_all(evs: &[TranscriptEvent]) -> Ledger {
        let mut l = Ledger::new();
        for e in evs {
            l.apply(e).expect("valid event");
        }
        l
    }

    #[test]
    fn no_cap_matches_ledger_exactly() {
        let mut em = Emitter::new("s", "m");
        let evs = vec![
            em.preview(1, "hello wor", 8_000).unwrap(),
            em.finalize(2, "hello world", 16_000, done(16_000)).unwrap(),
            em.preview(3, "again", 24_000).unwrap(),
        ];
        let l = apply_all(&evs);
        assert_eq!(display_text(&l, &evs), l.committed_text());
        assert_eq!(display_text(&l, &evs), "hello world");
    }

    #[test]
    fn hard_cap_keeps_the_capped_snapshot() {
        let mut em = Emitter::new("s", "m");
        let evs = vec![
            em.preview(1, "the first thirty", 240_000).unwrap(),
            em.preview(2, "the first thirty seconds", 480_000).unwrap(),
            em.incomplete(3, HARD_CAP_REASON, 480_000, 480_000, false)
                .unwrap(),
            em.preview(4, "and the", 560_000).unwrap(),
            em.finalize(5, "and the rest", 640_000, done(640_000))
                .unwrap(),
        ];
        let l = apply_all(&evs);
        assert_eq!(l.committed_text(), "and the rest");
        assert_eq!(
            display_text(&l, &evs),
            "the first thirty seconds and the rest"
        );
    }

    #[test]
    fn repeated_rollovers_over_sixty_seconds() {
        let mut em = Emitter::new("s", "m");
        let evs = vec![
            em.preview(1, "one", 480_000).unwrap(),
            em.incomplete(2, HARD_CAP_REASON, 480_000, 480_000, false)
                .unwrap(),
            em.preview(3, "two", 960_000).unwrap(),
            em.incomplete(4, HARD_CAP_REASON, 960_000, 960_000, false)
                .unwrap(),
            em.finalize(5, "three", 1_040_000, done(1_040_000)).unwrap(),
        ];
        let l = apply_all(&evs);
        assert_eq!(display_text(&l, &evs), "one two three");
    }

    #[test]
    fn preserved_prefix_appears_exactly_once() {
        let mut em = Emitter::new("s", "m");
        let evs = vec![
            em.commit(1, "fixed part tail", 11, 240_000).unwrap(),
            em.preview(2, "tail grows", 480_000).unwrap(),
            em.incomplete(3, HARD_CAP_REASON, 480_000, 480_000, false)
                .unwrap(),
            em.finalize(4, "next", 560_000, done(560_000)).unwrap(),
        ];
        let l = apply_all(&evs);
        assert_eq!(l.committed_text(), "fixed part next");
        assert_eq!(display_text(&l, &evs), "fixed part tail grows next");
    }

    #[test]
    fn other_closures_are_never_recovered() {
        let mut em = Emitter::new("s", "m");
        let evs = vec![
            em.commit(1, "kept lost", 5, 240_000).unwrap(),
            em.gap(2, Some((240_000, 250_000)), "input_gap"),
            em.preview(3, "cancelled words", 300_000).unwrap(),
            em.incomplete(4, "cancelled", 300_000, 200_000, false)
                .unwrap(),
            em.finalize(5, "after", 400_000, done(400_000)).unwrap(),
        ];
        let l = apply_all(&evs);
        assert_eq!(display_text(&l, &evs), l.committed_text());
        assert_eq!(display_text(&l, &evs), "kept after");
    }
}
