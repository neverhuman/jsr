//! `Receipt::validate_as_evidence` refuses every receipt that cannot back a reported number.

use ultra_core::receipt::*;

fn good() -> Receipt {
    Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: "models/sha256-ab".into(),
        source_commit: "1a09202".into(),
        hardware: "H0".into(),
        target: "x86_64-avx2".into(),
        configured_threads: 1,
        observed_threads: Some(1),
        cpu_only_verified: true,
        audio_seconds: 60.0,
        process_tree_cpu_seconds: Some(4.0),
        peak_rss_bytes: Some(300 << 20),
        events: Some(120),
        wakeups: Some(190),
        cap_hits: Some(0),
        chunk_ms: ChunkMs {
            p50: Some(3.0),
            p95: Some(6.0),
            p99: Some(9.0),
        },
        status_events: vec![],
        gaps: vec![],
        hashes: Hashes {
            events: "1".repeat(64),
            preview: "2".repeat(64),
            commit: "3".repeat(64),
            final_: "4".repeat(64),
        },
        label: HonestyLabel::InCorpus,
        partition: Partition::Dev,
        completion_status: "complete".into(),
    }
}

fn refused(f: impl FnOnce(&mut Receipt)) -> bool {
    let mut r = good();
    f(&mut r);
    r.validate_as_evidence().is_err()
}

#[test]
fn a_complete_measured_receipt_is_accepted() {
    good().validate_as_evidence().unwrap();
    assert_eq!(good().cpu_cost(), Some(4.0 / 60.0));
}

#[test]
fn non_finite_or_negative_measurements_are_refused() {
    assert!(refused(|r| r.process_tree_cpu_seconds = Some(f64::NAN)));
    assert!(refused(|r| r.process_tree_cpu_seconds = Some(-1.0)));
    assert!(refused(|r| r.chunk_ms.p99 = Some(f64::INFINITY)));
    assert!(refused(|r| r.chunk_ms.p50 = Some(-0.5)));
    assert!(refused(|r| r.audio_seconds = f64::INFINITY));
    assert!(refused(|r| r.audio_seconds = 0.0));
}

#[test]
fn percentiles_must_be_ordered() {
    assert!(refused(|r| r.chunk_ms.p95 = Some(20.0)));
}

#[test]
fn zero_or_missing_threads_are_refused() {
    assert!(refused(|r| r.configured_threads = 0));
    assert!(refused(|r| r.observed_threads = Some(0)));
    assert!(refused(|r| r.observed_threads = None));
}

#[test]
fn empty_identities_and_bad_hashes_are_refused() {
    assert!(refused(|r| r.bundle = " ".into()));
    assert!(refused(|r| r.source_commit.clear()));
    assert!(refused(|r| r.hardware.clear()));
    assert!(refused(|r| r.target.clear()));
    assert!(refused(|r| r.hashes.events = "e".into()));
    assert!(refused(|r| r.hashes.final_ = "G".repeat(64)));
}

#[test]
fn null_counts_and_rss_are_refused() {
    assert!(refused(|r| r.events = None));
    assert!(refused(|r| r.wakeups = None));
    assert!(refused(|r| r.cap_hits = None));
    assert!(refused(|r| r.peak_rss_bytes = None));
    assert!(refused(|r| r.peak_rss_bytes = Some(0)));
}

#[test]
fn incomplete_or_unverified_runs_are_refused() {
    assert!(refused(|r| r.completion_status = "schema_example".into()));
    assert!(refused(|r| r.cpu_only_verified = false));
    assert!(refused(|r| r.schema = "ultra.receipt/v0".into()));
    assert!(refused(|r| r.gaps = vec![(10, 10)]));
}
