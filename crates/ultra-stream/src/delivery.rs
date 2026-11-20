//! Ordered-transport duplicate & tampered-replay filter and refiner transaction ticket
//! (spec v1.0-B §18.6, §33.3, §38.3).

use std::collections::BTreeMap;

use ultra_core::events::{SegmentStatus, TranscriptEvent, DEDUP_WINDOW};
use ultra_core::receipt::sha256_hex;

use crate::api::StreamError;
use crate::coordinator::StreamCoordinator;

/// Ticket issued when an optional second-pass refiner starts work on an open segment (§18.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinerTicket {
    pub segment_id: u64,
    pub state_epoch: u64,
    pub base_revision: u64,
    pub immutable_prefix_sha256: String,
    pub immutable_prefix_bytes: usize,
    pub deadline_monotonic_ms: u64,
}

#[derive(Debug, Default)]
struct StreamDedupState {
    seen: BTreeMap<u64, String>,
    next_seq: Option<u64>,
}

/// Ordered-transport duplicate & tampered-replay filter with bounded retention (§33.3, §38.3).
#[derive(Debug, Default)]
pub struct DeliveryFilter {
    streams: BTreeMap<String, StreamDedupState>,
}

impl DeliveryFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `Ok(true)` if the event is new and valid in order, `Ok(false)` if it is an
    /// identical retransmission (`(stream_id, event_sequence, canonical_hash)` match), or
    /// `Err(StreamError::Protocol)` on a reused sequence with different content or missing
    /// predecessor sequence.
    pub fn check_and_record(&mut self, ev: &TranscriptEvent) -> Result<bool, StreamError> {
        let digest = ev
            .canonical_hash()
            .map_err(|e| StreamError::Protocol(e.to_string()))?;
        let st = self.streams.entry(ev.stream_id.clone()).or_default();

        if let Some(prev_hash) = st.seen.get(&ev.event_sequence) {
            if prev_hash == &digest {
                return Ok(false);
            }
            return Err(StreamError::Protocol(format!(
                "reused event_sequence {} on stream `{}` with altered payload",
                ev.event_sequence, ev.stream_id
            )));
        }
        if let Some(next) = st.next_seq {
            if ev.event_sequence < next {
                return Err(StreamError::Protocol(format!(
                    "out-of-order event_sequence {} (expected {})",
                    ev.event_sequence, next
                )));
            }
            if ev.event_sequence > next {
                return Err(StreamError::Protocol(format!(
                    "missing predecessor sequence before {} (expected {})",
                    ev.event_sequence, next
                )));
            }
        }
        let next = ev
            .event_sequence
            .checked_add(1)
            .ok_or_else(|| StreamError::Protocol("event_sequence overflow".into()))?;
        st.next_seq = Some(next);
        st.seen.insert(ev.event_sequence, digest);
        while st.seen.len() > DEDUP_WINDOW {
            st.seen.pop_first();
        }
        Ok(true)
    }
}

pub(crate) fn slice_from_boundary(s: &str, byte_offset: usize) -> &str {
    if byte_offset == 0 {
        s
    } else if byte_offset >= s.len() {
        ""
    } else if s.is_char_boundary(byte_offset) {
        &s[byte_offset..]
    } else {
        ""
    }
}

impl StreamCoordinator {
    /// Observable-clock endpoint detection hook (§18.6, §38.3): records when trailing silence
    /// began (`pause_started_ms`). Hangover elapsed since `pause_started_ms` consumes from
    /// `cfg.final_budget_ms` rather than resetting a fresh refinement budget.
    pub fn mark_observable_pause_start(&mut self, pause_started_ms: u64) {
        if self.speech_pause_anchor_ms.is_none() {
            self.speech_pause_anchor_ms = Some(pause_started_ms);
        }
    }

    /// Remaining causal refinement budget at `now_monotonic_ms` (§38.3 probe 3).
    pub fn remaining_refinement_budget_ms(&self, now_monotonic_ms: u64) -> u64 {
        let Some(anchor) = self.speech_pause_anchor_ms else {
            return self.cfg.final_budget_ms;
        };
        let deadline = anchor.saturating_add(self.cfg.final_budget_ms);
        deadline.saturating_sub(now_monotonic_ms)
    }

    /// Begin a conditional refiner transaction on the currently open segment if causal slack
    /// remains (`now_monotonic_ms` + `qualified_service_ms` <= deadline).
    pub fn try_begin_refiner(
        &mut self,
        now_monotonic_ms: u64,
        qualified_service_ms: u64,
    ) -> Option<RefinerTicket> {
        if self.lifecycle.state().is_terminal() {
            return None;
        }
        let seg = self.ledger.segment(self.current_segment_id)?;
        if seg.status != SegmentStatus::Open {
            return None;
        }
        let slack = self.remaining_refinement_budget_ms(now_monotonic_ms);
        if slack == 0 || qualified_service_ms > slack {
            return None;
        }
        self.extra_metrics.refiner_attempts += 1;
        Some(RefinerTicket {
            segment_id: self.current_segment_id,
            state_epoch: self.refiner_epoch,
            base_revision: seg.revision,
            immutable_prefix_sha256: sha256_hex(seg.immutable.as_bytes()),
            immutable_prefix_bytes: seg.immutable.len(),
            deadline_monotonic_ms: now_monotonic_ms.saturating_add(slack),
        })
    }

    /// Compare-and-apply a second-pass refiner suffix replacement (§18.6, §33.3, §38.3 probe 4).
    /// Always charges `cpu_seconds` and `peak_mem_bytes`, even when rejected.
    pub fn apply_refiner_result(
        &mut self,
        ticket: &RefinerTicket,
        now_monotonic_ms: u64,
        replacement_suffix: String,
        cpu_seconds: f64,
        peak_mem_bytes: u64,
    ) -> Result<bool, StreamError> {
        self.extra_metrics.refiner_charged_cpu_s += cpu_seconds.max(0.0);
        self.extra_metrics.retained_bytes = self.extra_metrics.retained_bytes.max(peak_mem_bytes);
        let now_ns = now_monotonic_ms.saturating_mul(1_000_000);

        let valid = !self.lifecycle.state().is_terminal()
            && now_monotonic_ms <= ticket.deadline_monotonic_ms
            && self.refiner_epoch == ticket.state_epoch
            && self.current_segment_id == ticket.segment_id
            && self.ledger.segment(ticket.segment_id).is_some_and(|seg| {
                seg.status == SegmentStatus::Open
                    && seg.revision == ticket.base_revision
                    && seg.immutable.len() == ticket.immutable_prefix_bytes
                    && sha256_hex(seg.immutable.as_bytes()) == ticket.immutable_prefix_sha256
            });

        if !valid {
            self.extra_metrics.refiner_rejects += 1;
            let detail = format!(
                "seg={} ticket_epoch={} cur_epoch={}",
                ticket.segment_id, ticket.state_epoch, self.refiner_epoch
            );
            let ev = self.emitter.status(now_ns, "refiner_rejected", &detail);
            self.push_validated(ev)?;
            return Ok(false);
        }

        let ev = self
            .emitter
            .preview(now_ns, &replacement_suffix, self.expected_sample)?;
        self.push_validated(ev)?;
        self.extra_metrics.refiner_accepts += 1;
        Ok(true)
    }
}
