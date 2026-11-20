//! Independent stream coordinator enforcing lifecycle, ledger invariants, discontinuities,
//! segment caps, causal observable-audio deadlines, refiner compare-and-apply, and replay
//! deduplication (spec v1.0-B §13, §15.2, §18.6, §33.2, §33.3, §38.3).

use std::collections::VecDeque;

use ultra_core::events::{Body, Emitter, Ledger, SegmentStatus, TranscriptEvent};
use ultra_core::lifecycle::{Lifecycle, StreamState};

use crate::api::{
    AsrStream, PollResult, PushReport, PushStatus, StreamClock, StreamConfig, StreamError,
    StreamMetrics, TerminalOutcome,
};
use crate::delivery::slice_from_boundary;
pub use crate::delivery::{DeliveryFilter, RefinerTicket};

/// Coordinator wrapping an inner [`AsrStream`] to guarantee every §33 / §38.3 contract.
pub struct StreamCoordinator {
    pub(crate) cfg: StreamConfig,
    inner: Box<dyn AsrStream>,
    clock: StreamClock,
    pub(crate) lifecycle: Lifecycle,
    pub(crate) emitter: Emitter,
    pub(crate) ledger: Ledger,
    outbox: VecDeque<TranscriptEvent>,
    pub(crate) refiner_epoch: u64,
    pub(crate) current_segment_id: u64,
    inner_consumed_bytes: usize,
    segment_start_sample: u64,
    pub(crate) expected_sample: u64,
    terminal: Option<TerminalOutcome>,
    pub(crate) speech_pause_anchor_ms: Option<u64>,
    pub(crate) extra_metrics: StreamMetrics,
}

impl StreamCoordinator {
    pub fn new(cfg: StreamConfig, inner: Box<dyn AsrStream>) -> Self {
        Self::with_clock(cfg, inner, StreamClock::real())
    }

    pub fn with_clock(cfg: StreamConfig, inner: Box<dyn AsrStream>, clock: StreamClock) -> Self {
        let emitter = Emitter::new(&cfg.stream_id, &cfg.model_config_id);
        Self {
            cfg,
            inner,
            clock,
            lifecycle: Lifecycle::default(),
            emitter,
            ledger: Ledger::new(),
            outbox: VecDeque::new(),
            refiner_epoch: 0,
            current_segment_id: 0,
            inner_consumed_bytes: 0,
            segment_start_sample: 0,
            expected_sample: 0,
            terminal: None,
            speech_pause_anchor_ms: None,
            extra_metrics: StreamMetrics::default(),
        }
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub fn state(&self) -> StreamState {
        self.lifecycle.state()
    }

    pub fn state_epoch(&self) -> u64 {
        self.refiner_epoch
    }

    pub fn current_segment_id(&self) -> u64 {
        self.current_segment_id
    }

    fn has_open_segment(&self) -> bool {
        self.ledger
            .segment(self.current_segment_id)
            .is_some_and(|s| s.status == SegmentStatus::Open)
    }

    pub(crate) fn push_validated(&mut self, ev: TranscriptEvent) -> Result<(), StreamError> {
        self.ledger.apply(&ev)?;
        self.extra_metrics.events_emitted += 1;
        self.outbox.push_back(ev);
        Ok(())
    }

    fn forward_inner_event(&mut self, raw_ev: TranscriptEvent) -> Result<(), StreamError> {
        let now_ns = raw_ev.observed_monotonic_ns;
        match raw_ev.body {
            Body::Preview(u) => {
                let base = self.inner_consumed_bytes + self.emitter.immutable().len();
                let suffix = slice_from_boundary(&u.snapshot_utf8, base);
                let ev = self
                    .emitter
                    .preview(now_ns, suffix, u.ownership_end_sample)?;
                self.push_validated(ev)?;
            }
            Body::Commit(u) => {
                let base = self.inner_consumed_bytes + self.emitter.immutable().len();
                if u.immutable_after_utf8_bytes > base
                    && u.snapshot_utf8.is_char_boundary(base)
                    && u.snapshot_utf8
                        .is_char_boundary(u.immutable_after_utf8_bytes)
                {
                    let suffix = &u.snapshot_utf8[base..];
                    let commit_bytes = u.immutable_after_utf8_bytes - base;
                    let ev = self.emitter.commit(
                        now_ns,
                        suffix,
                        commit_bytes,
                        u.ownership_end_sample,
                    )?;
                    self.push_validated(ev)?;
                }
            }
            Body::Final {
                update: u,
                processing,
            } => {
                let base = self.inner_consumed_bytes + self.emitter.immutable().len();
                let suffix = slice_from_boundary(&u.snapshot_utf8, base);
                let ev =
                    self.emitter
                        .finalize(now_ns, suffix, u.ownership_end_sample, processing)?;
                self.push_validated(ev)?;
                self.current_segment_id += 1;
                self.inner_consumed_bytes = 0;
                self.segment_start_sample = u.ownership_end_sample.max(self.expected_sample);
                self.speech_pause_anchor_ms = None;
                self.refiner_epoch += 1;
            }
            Body::Gap {
                start_sample,
                end_sample,
                gap_reason,
                ..
            } => {
                let was_open = self.has_open_segment();
                let ev = self
                    .emitter
                    .gap(now_ns, start_sample.zip(end_sample), &gap_reason);
                self.push_validated(ev)?;
                if was_open {
                    self.current_segment_id += 1;
                }
                self.inner_consumed_bytes = 0;
                self.refiner_epoch += 1;
            }
            Body::SegmentIncomplete {
                closure,
                accepted_through_sample,
                processed_through_sample,
                drained,
            } => {
                if self.has_open_segment() {
                    let ev = self.emitter.incomplete(
                        now_ns,
                        &closure.reason,
                        accepted_through_sample,
                        processed_through_sample,
                        drained,
                    )?;
                    self.push_validated(ev)?;
                    self.current_segment_id += 1;
                    self.inner_consumed_bytes = 0;
                    self.refiner_epoch += 1;
                }
            }
            Body::Status { code, detail } => {
                let ev = self.emitter.status(now_ns, &code, &detail);
                self.push_validated(ev)?;
            }
            Body::CorrectionSuggested {
                segment_id,
                alternative,
                ..
            } => {
                let detail = format!("seg={segment_id} alt_bytes={}", alternative.len());
                let ev = self.emitter.status(now_ns, "correction_suggested", &detail);
                self.push_validated(ev)?;
            }
        }
        Ok(())
    }

    fn drain_inner_events(&mut self) -> Result<(), StreamError> {
        loop {
            match self.inner.poll_event()? {
                PollResult::Pending => break,
                PollResult::Event(raw_ev) => self.forward_inner_event(*raw_ev)?,
                PollResult::Ended(outcome) => {
                    match &outcome {
                        TerminalOutcome::Finished => self.lifecycle.complete()?,
                        TerminalOutcome::Cancelled => self.lifecycle.cancel(),
                        TerminalOutcome::Failed(_) => self.lifecycle.fail(),
                    }
                    self.terminal = Some(outcome);
                    break;
                }
            }
        }
        Ok(())
    }
}

impl AsrStream for StreamCoordinator {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        self.lifecycle.check_push()?;
        if !pcm.is_empty() && self.speech_pause_anchor_ms.take().is_some() {
            self.refiner_epoch += 1;
        }
        if start_sample < self.expected_sample {
            return Err(StreamError::Protocol(format!(
                "overlapping/duplicate start_sample {start_sample} < expected {}",
                self.expected_sample
            )));
        }

        // True input loss -> emit Gap, closing any open segment incompletely while preserving prefix.
        if start_sample > self.expected_sample {
            let gap_start = self.expected_sample;
            self.expected_sample = start_sample;
            let now_ns = self.clock.now_ns();
            let was_open = self.has_open_segment();
            if let Some(seg) = self
                .ledger
                .segment(self.current_segment_id)
                .filter(|_| was_open)
            {
                self.inner_consumed_bytes += seg.shown().len();
            }
            let ev = self.emitter.gap(
                now_ns,
                Some((gap_start, start_sample)),
                "missing_input_samples",
            );
            self.push_validated(ev)?;
            if was_open {
                self.current_segment_id += 1;
            }
            self.refiner_epoch += 1;
            self.segment_start_sample = start_sample;
        }

        let inner_m = self.inner.metrics();
        let in_flight = self
            .expected_sample
            .saturating_sub(inner_m.processed_samples) as usize;
        let room = self.cfg.max_buffered_samples.saturating_sub(in_flight);
        if room == 0 && !pcm.is_empty() {
            return Ok(PushReport {
                start_sample,
                accepted_samples: 0,
                status: PushStatus::Backpressure {
                    accepted_samples: 0,
                },
            });
        }
        let slice = if pcm.len() > room { &pcm[..room] } else { pcm };
        let rep = self.inner.push_pcm(start_sample, slice)?;
        self.expected_sample += rep.accepted_samples as u64;
        self.extra_metrics.accepted_samples += rep.accepted_samples as u64;
        self.drain_inner_events()?;

        // Hard segment cap (30 s default) prevents unbounded state on 1h+ continuous streams.
        // Because `AsrStream` has no mid-stream segment flush/drain call (`finish_input` ends the
        // entire stream), any open segment that was not already finalized by the inner backend
        // MUST close as `SegmentIncomplete` (`drained = false`), never claiming synthetic `Final`
        // flush credit while right-context or deferred EOF tail decoder work remains un-drained.
        let seg_span = self
            .expected_sample
            .saturating_sub(self.segment_start_sample);
        if seg_span >= self.cfg.segment_hard_cap_samples {
            self.extra_metrics.cap_hits += 1;
            let now_ns = self.clock.now_ns();
            let open_shown_len = self
                .ledger
                .segment(self.current_segment_id)
                .filter(|s| s.status == SegmentStatus::Open)
                .map(|s| s.shown().len());
            if let Some(shown_len) = open_shown_len {
                let im = self.inner.metrics();
                let ev = self.emitter.incomplete(
                    now_ns,
                    "segment_hard_cap",
                    self.expected_sample,
                    im.processed_samples.min(self.expected_sample),
                    false,
                )?;
                self.push_validated(ev)?;
                self.current_segment_id += 1;
                self.inner_consumed_bytes += shown_len;
                self.refiner_epoch += 1;
            }
            let st = self.emitter.status(
                self.clock.now_ns(),
                "segment_hard_cap",
                &format!("span_samples={seg_span}"),
            );
            self.push_validated(st)?;
            self.segment_start_sample = self.expected_sample;
        }

        let status = if rep.accepted_samples < pcm.len() {
            PushStatus::Backpressure {
                accepted_samples: rep.accepted_samples,
            }
        } else {
            rep.status
        };
        Ok(PushReport {
            start_sample,
            accepted_samples: rep.accepted_samples,
            status,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if self.outbox.is_empty() {
            self.drain_inner_events()?;
        }
        if let Some(ev) = self.outbox.pop_front() {
            return Ok(PollResult::Event(Box::new(ev)));
        }
        if let Some(term) = &self.terminal {
            return Ok(PollResult::Ended(term.clone()));
        }
        Ok(PollResult::Pending)
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        let started = self.lifecycle.finish_input()?;
        if started {
            self.inner.finish_input()?;
            self.drain_inner_events()?;
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.lifecycle.state().is_terminal() {
            self.lifecycle.cancel();
            self.refiner_epoch += 1;
            self.inner.cancel()?;
            if self.has_open_segment() {
                let im = self.inner.metrics();
                let acc = self.expected_sample.max(im.accepted_samples);
                let proc = im.processed_samples.min(acc);
                let now_ns = self.clock.now_ns();
                let ev = self
                    .emitter
                    .incomplete(now_ns, "cancelled", acc, proc, false)?;
                self.push_validated(ev)?;
                self.current_segment_id += 1;
            }
            self.terminal = Some(TerminalOutcome::Cancelled);
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let im = self.inner.metrics();
        StreamMetrics {
            accepted_samples: self.extra_metrics.accepted_samples.max(im.accepted_samples),
            processed_samples: im.processed_samples,
            wakeups: im.wakeups,
            events_emitted: self.extra_metrics.events_emitted,
            cap_hits: self.extra_metrics.cap_hits + im.cap_hits,
            refiner_attempts: self.extra_metrics.refiner_attempts,
            refiner_accepts: self.extra_metrics.refiner_accepts,
            refiner_rejects: self.extra_metrics.refiner_rejects,
            refiner_charged_cpu_s: self.extra_metrics.refiner_charged_cpu_s
                + im.refiner_charged_cpu_s,
            retained_bytes: self.extra_metrics.retained_bytes.max(im.retained_bytes),
        }
    }
}
