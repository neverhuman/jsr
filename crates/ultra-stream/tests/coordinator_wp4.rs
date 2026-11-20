//! WP4 `StreamCoordinator` verification:
//! 1. Shared injected monotonic `StreamClock` is used for coordinator-generated `Gap`,
//!    `segment_hard_cap`, and `cancel` events (never sample-derived `sample * 1e9 / 16_000`).
//! 2. Deferred-EOF mock stream where `processed_samples == accepted_samples` on `push_pcm` yet
//!    tail text only flushes in `finish_input()` closes as `SegmentIncomplete` (`drained = false`)
//!    on `segment_hard_cap`, never claiming synthetic `Final { flushed: true }` credit.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Body, Emitter, ProcessingEvidence, SegmentStatus, TranscriptEvent};
use ultra_stream::{
    AsrStream, PollResult, PushReport, PushStatus, StreamClock, StreamConfig, StreamCoordinator,
    StreamError, StreamMetrics, TerminalOutcome,
};

struct DeferredTailMock {
    clock: StreamClock,
    emitter: Emitter,
    accepted: u64,
    processed: u64,
    finished: bool,
    cancelled: bool,
    events: VecDeque<TranscriptEvent>,
}

impl DeferredTailMock {
    fn new(stream_id: &str, model_id: &str, clock: StreamClock) -> Self {
        Self {
            clock,
            emitter: Emitter::new(stream_id, model_id),
            accepted: 0,
            processed: 0,
            finished: false,
            cancelled: false,
            events: VecDeque::new(),
        }
    }
}

impl AsrStream for DeferredTailMock {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        self.accepted += pcm.len() as u64;
        // Simulate a backend whose chunk counter reaches `accepted`, while still holding
        // un-flushed right-context / tail decoder words until `finish_input()`.
        self.processed = self.accepted;
        let ev = self
            .emitter
            .preview(self.clock.now_ns(), "draft prefix", self.processed)?;
        self.events.push_back(ev);
        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if let Some(ev) = self.events.pop_front() {
            return Ok(PollResult::Event(Box::new(ev)));
        }
        if self.cancelled {
            return Ok(PollResult::Ended(TerminalOutcome::Cancelled));
        }
        if self.finished {
            return Ok(PollResult::Ended(TerminalOutcome::Finished));
        }
        Ok(PollResult::Pending)
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        if !self.finished {
            self.finished = true;
            let ev = self.emitter.finalize(
                self.clock.now_ns(),
                "draft prefix deferred_tail",
                self.accepted,
                ProcessingEvidence {
                    accepted_through_sample: self.accepted,
                    required_processed_through_sample: self.accepted,
                    flushed: true,
                },
            )?;
            self.events.push_back(ev);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        self.cancelled = true;
        self.events.clear();
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        StreamMetrics {
            accepted_samples: self.accepted,
            processed_samples: self.processed,
            wakeups: 1,
            events_emitted: 0,
            cap_hits: 0,
            refiner_attempts: 0,
            refiner_accepts: 0,
            refiner_rejects: 0,
            refiner_charged_cpu_s: 0.0,
            retained_bytes: 0,
        }
    }
}

fn drain(c: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut v = Vec::new();
    while let PollResult::Event(ev) = c.poll_event().expect("poll") {
        v.push(*ev);
    }
    v
}

#[test]
fn wp4_coordinator_uses_injected_monotonic_clock_and_never_synthesizes_final_on_hard_cap() {
    let tick = Arc::new(AtomicU64::new(7_000_000_000));
    let t_inner = Arc::clone(&tick);
    let inner_clk = StreamClock::custom(move || t_inner.fetch_add(10_000, Ordering::SeqCst));
    let t_coord = Arc::clone(&tick);
    let coord_clk = StreamClock::custom(move || t_coord.fetch_add(10_000, Ordering::SeqCst));

    let cfg = StreamConfig {
        stream_id: "wp4-coord".into(),
        segment_hard_cap_samples: 1_600,
        ..StreamConfig::default()
    };
    let mock = Box::new(DeferredTailMock::new(
        &cfg.stream_id,
        &cfg.model_config_id,
        inner_clk,
    ));
    let mut coord = StreamCoordinator::with_clock(cfg, mock, coord_clk);

    // 1. Push 1,600 samples -> triggers `segment_hard_cap`. Even though `processed == accepted`,
    // deferred EOF work is un-flushed, so coordinator MUST emit `SegmentIncomplete(drained=false)`
    // and NEVER `Final(flushed=true)`, stamped with the 7s+ injected clock (NOT 100_000_000 ns!).
    coord.push_pcm(0, &[1i16; 1_600]).expect("push cap");
    let evs1 = drain(&mut coord);
    assert!(!evs1.iter().any(|e| matches!(e.body, Body::Final { .. })));
    assert!(evs1.iter().any(|e| matches!(
        &e.body,
        Body::SegmentIncomplete {
            closure,
            drained: false,
            ..
        } if closure.reason == "segment_hard_cap" && e.observed_monotonic_ns >= 7_000_000_000
    )));
    assert_eq!(
        coord.ledger().segment(0).expect("seg0").status,
        SegmentStatus::Incomplete
    );

    // 2. Introduce an input gap (`start_sample = 3_200 > expected 1_600`) -> `Gap` uses injected clock.
    coord.push_pcm(3_200, &[1i16; 800]).expect("push gap");
    let evs2 = drain(&mut coord);
    let gap_ev = evs2
        .iter()
        .find(|e| matches!(e.body, Body::Gap { .. }))
        .expect("gap event");
    assert!(gap_ev.observed_monotonic_ns >= 7_000_030_000);

    // 3. Cancel open segment 1 -> `SegmentIncomplete(reason="cancelled")` uses injected clock.
    coord.cancel().expect("cancel");
    let evs3 = drain(&mut coord);
    let cancel_ev = evs3
        .iter()
        .find(|e| matches!(e.body, Body::SegmentIncomplete { .. }))
        .expect("cancel incomplete");
    assert!(cancel_ev.observed_monotonic_ns >= 7_000_050_000);
}
