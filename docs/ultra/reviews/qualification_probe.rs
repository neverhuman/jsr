//! Diagnostic reproductions for the independent EFFICIENT/H0 release review.
//! This prints observed behavior; a successful exit is not release qualification.

use std::sync::atomic::{AtomicBool, Ordering};

use ultra_core::events::{
    Body, Ledger, ProcessingEvidence, TextUpdate, TranscriptEvent, EVENT_SCHEMA,
};
use ultra_core::receipt::{ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA};

static FAILED: AtomicBool = AtomicBool::new(false);

fn event(seq: u64, body: Body) -> TranscriptEvent {
    TranscriptEvent {
        schema: EVENT_SCHEMA.into(),
        stream_id: "review-synthetic".into(),
        event_sequence: seq,
        state_epoch: 0,
        observed_monotonic_ns: 1_000_000,
        model_config_id: "review-synthetic-model".into(),
        body,
    }
}

fn update(revision: u64, snapshot: &str, before: usize, after: usize) -> TextUpdate {
    TextUpdate {
        segment_id: 0,
        base_revision: revision - 1,
        revision,
        snapshot_utf8: snapshot.into(),
        immutable_before_utf8_bytes: before,
        immutable_after_utf8_bytes: after,
        ownership_start_sample: 0,
        ownership_end_sample: 320,
    }
}

fn processing() -> ProcessingEvidence {
    ProcessingEvidence {
        accepted_through_sample: 320,
        required_processed_through_sample: 320,
        flushed: true,
    }
}

fn final_body(u: TextUpdate) -> Body {
    Body::Final {
        update: u,
        processing: processing(),
    }
}

fn receipt() -> Receipt {
    Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: format!("models/sha256-{}", "0".repeat(64)),
        source_commit: "a".repeat(40),
        hardware: "H0".into(),
        target: "avx2".into(),
        configured_threads: 1,
        observed_threads: Some(1),
        cpu_only_verified: true,
        audio_seconds: 1.0,
        process_tree_cpu_seconds: Some(0.1),
        peak_rss_bytes: Some(1024),
        events: Some(1),
        wakeups: Some(1),
        cap_hits: Some(0),
        chunk_ms: ChunkMs {
            p50: Some(1.0),
            p95: Some(2.0),
            p99: Some(3.0),
        },
        status_events: Vec::new(),
        gaps: Vec::new(),
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

fn observed(name: &str, value: bool) {
    println!("{{\"probe\":\"{name}\",\"observed\":{value}}}");
    if !value {
        FAILED.store(true, Ordering::Relaxed);
    }
}

fn main() {
    receipt()
        .validate_as_evidence()
        .expect("synthetic control receipt must pass before mutation probes");
    for (name, kind) in [
        (
            "identical_commit_retry_is_noop",
            Body::Commit(update(1, "a ", 0, 2)),
        ),
        (
            "identical_final_retry_is_noop",
            final_body(update(1, "a", 0, 1)),
        ),
    ] {
        let mut ledger = Ledger::new();
        let ev = event(1, kind);
        ledger.apply(&ev).expect("initial event");
        let before = ledger.committed_text();
        observed(
            name,
            ledger.apply(&ev).is_ok() && ledger.committed_text() == before,
        );
    }

    let mut ledger = Ledger::new();
    ledger
        .apply(&event(1, Body::Commit(update(1, "a ", 0, 2))))
        .expect("commit");
    observed(
        "missing_sequence_is_rejected",
        ledger
            .apply(&event(3, final_body(update(2, "a b", 2, 3))))
            .is_err(),
    );

    let mut ledger = Ledger::new();
    ledger
        .apply(&event(1, final_body(update(1, "é", 0, 2))))
        .expect("UTF-8 final");
    observed(
        "mid_codepoint_correction_is_rejected",
        ledger
            .apply(&event(
                2,
                Body::CorrectionSuggested {
                    segment_id: 0,
                    final_revision: 1,
                    final_sha256: ultra_core::receipt::sha256_hex("é".as_bytes()),
                    byte_start: 1,
                    byte_end: 2,
                    alternative: "x".into(),
                },
            ))
            .is_err(),
    );

    let mut ledger = Ledger::new();
    let mut invalid = update(99, "bad", 0, 3);
    invalid.segment_id = 9;
    let rejected = ledger.apply(&event(1, Body::Commit(invalid))).is_err();
    observed(
        "rejected_first_event_preserves_state",
        rejected
            && ledger
                .apply(&event(1, Body::Commit(update(1, "ok", 0, 2))))
                .is_ok(),
    );

    for (name, p) in [
        (
            "unfinished_final_is_rejected",
            ProcessingEvidence {
                required_processed_through_sample: 160,
                ..processing()
            },
        ),
        (
            "unflushed_final_is_rejected",
            ProcessingEvidence {
                flushed: false,
                ..processing()
            },
        ),
        (
            "impossible_processed_watermark_is_rejected",
            ProcessingEvidence {
                required_processed_through_sample: 640,
                ..processing()
            },
        ),
    ] {
        let mut ledger = Ledger::new();
        observed(
            name,
            ledger
                .apply(&event(
                    1,
                    Body::Final {
                        update: update(1, "ok", 0, 2),
                        processing: p,
                    },
                ))
                .is_err(),
        );
    }

    let preview = event(1, Body::Preview(update(1, "a", 0, 0)));
    let newer = event(2, Body::Preview(update(2, "b", 0, 0)));
    let final_ev = event(3, final_body(update(3, "b", 0, 1)));
    let timed = |t_s, event| ultra_eval::stream::TimedEvent { t_s, event };
    let normal = ultra_eval::stream::replay(&[
        timed(1.0, preview.clone()),
        timed(2.0, newer.clone()),
        timed(4.0, final_ev.clone()),
    ])
    .expect("ordinary scoring control");
    let retry = ultra_eval::stream::replay(&[
        timed(1.0, preview.clone()),
        timed(2.0, newer),
        timed(3.0, preview),
        timed(4.0, final_ev),
    ])
    .expect("scoring with delayed duplicate");
    observed(
        "duplicate_creates_no_scoring_snapshot",
        normal.snapshots == retry.snapshots,
    );

    for (name, invalid) in [
        ("negative_cpu_is_rejected", -1.0),
        ("nan_cpu_is_rejected", f64::NAN),
        ("infinite_cpu_is_rejected", f64::INFINITY),
    ] {
        let mut r = receipt();
        r.process_tree_cpu_seconds = Some(invalid);
        observed(name, r.validate_as_evidence().is_err());
    }
    let mut r = receipt();
    r.audio_seconds = f64::NAN;
    observed("nan_audio_is_rejected", r.validate_as_evidence().is_err());
    let mut r = receipt();
    r.chunk_ms.p95 = Some(-1.0);
    observed(
        "negative_chunk_time_is_rejected",
        r.validate_as_evidence().is_err(),
    );
    let mut r = receipt();
    r.configured_threads = 0;
    r.observed_threads = Some(0);
    observed(
        "zero_threads_are_rejected",
        r.validate_as_evidence().is_err(),
    );
    let mut r = receipt();
    r.bundle.clear();
    r.source_commit.clear();
    r.hardware.clear();
    r.target.clear();
    r.hashes = Hashes::default();
    observed(
        "empty_identities_are_rejected",
        r.validate_as_evidence().is_err(),
    );
    let mut r = receipt();
    r.events = None;
    r.wakeups = None;
    r.cap_hits = None;
    observed(
        "null_counters_are_rejected",
        r.validate_as_evidence().is_err(),
    );
    if FAILED.load(Ordering::Relaxed) {
        std::process::exit(1);
    }
}
