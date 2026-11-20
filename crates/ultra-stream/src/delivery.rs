//! Ordered-transport duplicate & tampered-replay filter and refiner transaction ticket
//! (spec v1.0-B §18.6, §33.3, §38.3).

use std::collections::BTreeMap;

use ultra_core::events::{TranscriptEvent, DEDUP_WINDOW};

use crate::api::StreamError;

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
        let digest = ev.canonical_hash();
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
