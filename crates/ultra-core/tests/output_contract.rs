//! One test per rule of the output contract (spec v1.0-B §33.2–33.3, §3.5 lifecycle).

use ultra_core::events::*;
use ultra_core::lifecycle::*;
use ultra_core::receipt::sha256_hex;

fn em() -> Emitter {
    Emitter::new("s", "test-model")
}

fn done(through: u64) -> ProcessingEvidence {
    ProcessingEvidence {
        accepted_through_sample: through,
        required_processed_through_sample: through,
        flushed: true,
    }
}

fn apply_all(l: &mut Ledger, evs: &[TranscriptEvent]) {
    for e in evs {
        assert_eq!(l.apply(e).expect("event accepted"), Applied::New, "{e:?}");
    }
}

fn text(ev: &mut TranscriptEvent) -> &mut TextUpdate {
    match &mut ev.body {
        Body::Preview(u) | Body::Commit(u) | Body::Final { update: u, .. } => u,
        _ => panic!("not a text update"),
    }
}

#[test]
fn spec_commit_example_appends_only_the_new_bytes() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(&mut l, &[e.commit(1, "hello ", 6, 100).unwrap()]);
    let c = e.commit(2, "world", 5, 200).unwrap();
    let u = text(&mut c.clone()).clone();
    assert_eq!(
        (
            u.snapshot_utf8.as_str(),
            u.immutable_before_utf8_bytes,
            u.immutable_after_utf8_bytes
        ),
        ("hello world", 6, 11)
    );
    apply_all(&mut l, &[c]);
    assert_eq!(l.segment(0).unwrap().immutable, "hello world");
}

#[test]
fn preview_replaces_suffix_and_never_commits() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(
        &mut l,
        &[
            e.commit(1, "kept ", 5, 10).unwrap(),
            e.preview(2, "wor", 20).unwrap(),
            e.preview(3, "world", 30).unwrap(),
        ],
    );
    let s = l.segment(0).unwrap();
    assert_eq!(
        (s.immutable.as_str(), s.suffix.as_str(), s.revision),
        ("kept ", "world", 3)
    );
    let mut bad = e.preview(4, "x", 40).unwrap();
    text(&mut bad).immutable_after_utf8_bytes += 1;
    assert!(
        l.apply(&bad).is_err(),
        "a Preview that moves the immutable offset is refused"
    );
}

#[test]
fn revision_and_offsets_must_match_receiver_state() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(&mut l, &[e.commit(1, "ab ", 3, 10).unwrap()]);
    let good = e.preview(2, "cd", 20).unwrap();
    let mut wrong_rev = good.clone();
    text(&mut wrong_rev).base_revision = 0;
    assert!(l.apply(&wrong_rev).is_err());
    let mut wrong_before = good.clone();
    text(&mut wrong_before).immutable_before_utf8_bytes = 2;
    text(&mut wrong_before).immutable_after_utf8_bytes = 2;
    assert!(l.apply(&wrong_before).is_err());
    let mut wrong_prefix = good.clone();
    text(&mut wrong_prefix).snapshot_utf8 = "XY cd".into();
    assert!(
        l.apply(&wrong_prefix).is_err(),
        "snapshot[0:before] must equal the immutable prefix"
    );
    l.apply(&good).unwrap();
}

#[test]
fn offsets_must_sit_on_utf8_boundaries() {
    let mut e = em();
    let mut l = Ledger::new();
    let mut c = e.commit(1, "naïve", "na".len(), 10).unwrap();
    text(&mut c).immutable_after_utf8_bytes = 3; // inside the two-byte 'ï'
    assert!(l.apply(&c).is_err());
    assert!(
        e.commit(2, "é", 1, 10).is_err(),
        "the emitter refuses a split character too"
    );
}

#[test]
fn final_closes_once_and_nothing_follows() {
    let mut e = em();
    let mut l = Ledger::new();
    let f = e.finalize(1, "done", 10, done(10)).unwrap();
    apply_all(&mut l, std::slice::from_ref(&f));
    assert_eq!(l.segment(0).unwrap().status, SegmentStatus::Final);
    let mut late = f.clone();
    late.event_sequence = 1;
    assert!(l.apply(&late).is_err(), "no text for a closed segment");
    apply_all(&mut l, &[e.commit(2, "next ", 5, 20).unwrap()]);
    assert_eq!(l.segment(1).unwrap().immutable, "next ");
    assert_eq!(l.committed_text(), "done next");
}

#[test]
fn identical_retransmit_is_a_no_op_but_reuse_with_new_content_is_refused() {
    let mut e = em();
    let mut l = Ledger::new();
    let p = e.preview(1, "a", 10).unwrap();
    apply_all(&mut l, std::slice::from_ref(&p));
    assert_eq!(l.apply(&p).unwrap(), Applied::Duplicate);
    let mut changed = p.clone();
    text(&mut changed).snapshot_utf8 = "b".into();
    assert!(l.apply(&changed).is_err());
}

#[test]
fn missing_predecessor_and_out_of_order_are_refused() {
    let mut e = em();
    let mut l = Ledger::new();
    let a = e.preview(1, "a", 10).unwrap();
    let b = e.preview(2, "ab", 20).unwrap();
    let c = e.preview(3, "abc", 30).unwrap();
    l.apply(&a).unwrap();
    assert!(l.apply(&c).is_err(), "sequence 2 skipped");
    l.apply(&b).unwrap();
    l.apply(&c).unwrap();
}

#[test]
fn gap_preserves_prefix_discards_suffix_and_bumps_epoch() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(
        &mut l,
        &[
            e.commit(1, "kept ", 5, 10).unwrap(),
            e.preview(2, "dropped", 20).unwrap(),
        ],
    );
    let g = e.gap(3, Some((100, 200)), "packet loss");
    apply_all(&mut l, &[g]);
    let s = l.segment(0).unwrap();
    assert_eq!(
        (
            s.immutable.as_str(),
            s.suffix.as_str(),
            s.status,
            s.revision
        ),
        ("kept ", "", SegmentStatus::Incomplete, 3)
    );
    let next = e.preview(4, "new", 300).unwrap();
    assert_eq!(next.state_epoch, 1);
    apply_all(&mut l, &[next]);
}

#[test]
fn closure_must_preserve_the_exact_prefix() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(&mut l, &[e.commit(1, "abc", 3, 10).unwrap()]);
    let mut g = e.gap(2, None, "unknown extent");
    if let Body::Gap {
        closure: Some(c), ..
    } = &mut g.body
    {
        c.prefix_sha256 = sha256_hex(b"abd");
    }
    assert!(l.apply(&g).is_err());
}

#[test]
fn segment_incomplete_is_not_a_final() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(&mut l, &[e.commit(1, "part ", 5, 10).unwrap()]);
    let inc = e
        .incomplete(2, "deadline miss", 16000, 12000, false)
        .unwrap();
    apply_all(&mut l, &[inc]);
    assert_eq!(l.segment(0).unwrap().status, SegmentStatus::Incomplete);
}

#[test]
fn stale_epoch_is_refused() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(
        &mut l,
        &[e.preview(1, "a", 10).unwrap(), e.gap(2, None, "reset")],
    );
    let mut stale = e.preview(3, "b", 20).unwrap();
    stale.state_epoch = 0;
    assert!(l.apply(&stale).is_err());
}

#[test]
fn correction_must_reference_the_final_revision_and_never_mutates() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(&mut l, &[e.finalize(1, "their car", 10, done(10)).unwrap()]);
    let seg = l.segment(0).unwrap().clone();
    let mk = |rev: u64, hash: String, s: usize, t: usize, seq: u64| TranscriptEvent {
        schema: EVENT_SCHEMA.into(),
        stream_id: "s".into(),
        event_sequence: seq,
        state_epoch: 0,
        observed_monotonic_ns: 2,
        model_config_id: "test-model".into(),
        body: Body::CorrectionSuggested {
            segment_id: 0,
            final_revision: rev,
            final_sha256: hash,
            byte_start: s,
            byte_end: t,
            alternative: "there".into(),
        },
    };
    let hash = sha256_hex(seg.immutable.as_bytes());
    assert!(l
        .apply(&mk(seg.revision + 1, hash.clone(), 0, 5, 1))
        .is_err());
    assert!(l.apply(&mk(seg.revision, hash.clone(), 0, 99, 1)).is_err());
    l.apply(&mk(seg.revision, hash, 0, 5, 1)).unwrap();
    assert_eq!(l.segment(0).unwrap().immutable, "their car");
}

#[test]
fn payload_and_schema_bounds() {
    let mut e = em();
    let mut l = Ledger::new();
    let big = e
        .preview(1, &"x".repeat(MAX_PAYLOAD_BYTES + 1), 10)
        .unwrap();
    assert!(l.apply(&big).is_err());
    let mut e2 = em();
    let mut wrong = e2.preview(1, "a", 10).unwrap();
    wrong.schema = "other".into();
    assert!(Ledger::new().apply(&wrong).is_err());
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

#[test]
fn final_requires_complete_processing_evidence() {
    let mut e = em();
    let mut l = Ledger::new();
    let unfinished = ProcessingEvidence {
        accepted_through_sample: 16000,
        required_processed_through_sample: 12000,
        flushed: true,
    };
    assert!(l
        .apply(&e.finalize(1, "a", 16000, unfinished).unwrap())
        .is_err());
    let mut e2 = em();
    let no_flush = ProcessingEvidence {
        flushed: false,
        ..done(16000)
    };
    assert!(Ledger::new()
        .apply(&e2.finalize(1, "a", 16000, no_flush).unwrap())
        .is_err());
    let mut e3 = em();
    let short = done(8000); // claims ownership to 16000 with only 8000 accepted
    assert!(Ledger::new()
        .apply(&e3.finalize(1, "a", 16000, short).unwrap())
        .is_err());
}

#[test]
fn rejected_event_leaves_the_ledger_unchanged() {
    let mut e = em();
    let mut l = Ledger::new();
    let mut bad_first = e.preview(1, "a", 10).unwrap();
    text(&mut bad_first).base_revision = 7;
    assert!(l.apply(&bad_first).is_err());
    // The same stream can still start cleanly: nothing was half-applied.
    let mut e2 = em();
    l.apply(&e2.preview(1, "a", 10).unwrap()).unwrap();
    let good = e2.preview(2, "ab", 20).unwrap();
    let mut other_stream = good.clone();
    other_stream.stream_id = "other".into();
    assert!(l.apply(&other_stream).is_err());
    l.apply(&good).unwrap();
    assert_eq!(l.segment(0).unwrap().suffix, "ab");
}

#[test]
fn ownership_is_contiguous_and_never_regresses() {
    let mut e = em();
    let mut l = Ledger::new();
    apply_all(
        &mut l,
        &[
            e.preview(1, "a", 16000).unwrap(),
            e.finalize(2, "a", 32000, done(32000)).unwrap(),
        ],
    );
    let next = e.preview(3, "b", 40000).unwrap();
    assert_eq!(
        text(&mut next.clone()).ownership_start_sample,
        32000,
        "next segment owns from the previous end"
    );
    let mut overlap = next.clone();
    text(&mut overlap).ownership_start_sample = 1000;
    assert!(l.apply(&overlap).is_err());
    l.apply(&next).unwrap();
    let mut regress = e.preview(4, "bc", 50000).unwrap();
    text(&mut regress).ownership_end_sample = 35000;
    assert!(l.apply(&regress).is_err());
}

#[test]
fn dedup_memory_is_bounded() {
    let mut e = em();
    let mut l = Ledger::new();
    let first = e.preview(0, "x", 1).unwrap();
    l.apply(&first).unwrap();
    for i in 1..(DEDUP_WINDOW as u64 + 10) {
        l.apply(&e.status(i, "tick", "")).unwrap();
    }
    // Too old to be recognised as a duplicate: refused as out of order, never re-applied.
    assert!(l.apply(&first).is_err());
}
