use ultra_core::events::*;
use ultra_core::lifecycle::*;

fn ev(seq: u64, seg: u64, kind: EventKind) -> TranscriptEvent {
    TranscriptEvent {
        stream_id: "s".into(),
        segment_id: seg,
        event_sequence: seq,
        audio_start_sample: 0,
        audio_end_sample: 1,
        kind,
    }
}
fn preview(rev: u64, s: &str) -> EventKind {
    EventKind::Preview {
        revision_id: rev,
        replacement_suffix: s.into(),
    }
}
fn commit(s: &str) -> EventKind {
    EventKind::Commit { text: s.into() }
}
fn fin(s: &str) -> EventKind {
    EventKind::Final { text: s.into() }
}

#[test]
fn sequence_must_strictly_increase() {
    let mut l = Ledger::new();
    l.apply(&ev(1, 0, preview(1, "a"))).unwrap();
    assert!(l.apply(&ev(1, 0, preview(2, "b"))).is_err());
}

#[test]
fn preview_touches_only_suffix_and_prefix_only_grows() {
    let mut l = Ledger::new();
    l.apply(&ev(1, 0, commit("hello "))).unwrap();
    l.apply(&ev(2, 0, preview(1, "wor"))).unwrap();
    l.apply(&ev(3, 0, preview(2, "world"))).unwrap();
    let s = l.segment(0).unwrap();
    assert_eq!(s.immutable, "hello ");
    assert_eq!(s.suffix, "world");
    l.apply(&ev(4, 0, fin("world"))).unwrap();
    assert_eq!(l.segment(0).unwrap().immutable, "hello world");
}

#[test]
fn final_closes_exactly_once_and_nothing_after() {
    let mut l = Ledger::new();
    l.apply(&ev(1, 0, fin("a"))).unwrap();
    assert!(l.apply(&ev(2, 0, fin("b"))).is_err());
    assert!(l.apply(&ev(3, 0, commit("c"))).is_err());
    assert!(l.apply(&ev(4, 0, preview(9, "d"))).is_err());
    l.apply(&ev(5, 1, commit("next"))).unwrap();
    assert_eq!(l.committed_text(), "a next");
}

#[test]
fn correction_never_mutates() {
    let mut l = Ledger::new();
    l.apply(&ev(1, 0, fin("their car"))).unwrap();
    l.apply(&ev(
        2,
        0,
        EventKind::CorrectionSuggested {
            byte_start: 0,
            byte_end: 5,
            text: "there".into(),
        },
    ))
    .unwrap();
    assert_eq!(l.segment(0).unwrap().immutable, "their car");
    assert!(l
        .apply(&ev(
            3,
            0,
            EventKind::CorrectionSuggested {
                byte_start: 0,
                byte_end: 99,
                text: "x".into()
            }
        ))
        .is_err());
}

#[test]
fn gap_keeps_prefix_drops_suffix() {
    let mut l = Ledger::new();
    l.apply(&ev(1, 0, commit("kept "))).unwrap();
    l.apply(&ev(2, 0, preview(1, "dropped"))).unwrap();
    l.apply(&ev(
        3,
        0,
        EventKind::Gap {
            start_sample: 10,
            end_sample: 20,
        },
    ))
    .unwrap();
    let s = l.segment(0).unwrap();
    assert_eq!(
        (s.immutable.as_str(), s.suffix.as_str(), s.incomplete),
        ("kept ", "", true)
    );
    assert!(
        l.apply(&ev(4, 0, commit("late"))).is_err(),
        "no delayed worker may emit into a closed segment"
    );
}

#[test]
fn payload_bounded() {
    let mut l = Ledger::new();
    assert!(l
        .apply(&ev(1, 0, commit(&"x".repeat(MAX_PAYLOAD_BYTES + 1))))
        .is_err());
}

#[test]
fn lifecycle_idempotent_and_honest() {
    let mut lc = Lifecycle::default();
    lc.check_push().unwrap();
    assert!(lc.finish_input().unwrap());
    assert!(!lc.finish_input().unwrap());
    assert_eq!(lc.flushes_started(), 1);
    assert!(lc.check_push().is_err());
    lc.complete().unwrap();
    lc.cancel();
    assert_eq!(lc.state(), StreamState::Finished);
    let mut f = Lifecycle::default();
    f.fail();
    assert!(
        f.complete().is_err(),
        "a failure is never reported as completion"
    );
    let mut c = Lifecycle::default();
    c.cancel();
    c.cancel();
    assert_eq!(c.state(), StreamState::Cancelled);
}
