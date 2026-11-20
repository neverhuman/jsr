//! Output event contract (§3.1) and the ledger that enforces its invariants.

use serde::{Deserialize, Serialize};

use crate::CoreError;

/// Largest text payload of one event, in bytes.
pub const MAX_PAYLOAD_BYTES: usize = 4096;

/// Event kinds of `ultra.transcript-event/v1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum EventKind {
    /// Replace the mutable suffix of the segment.
    Preview {
        revision_id: u64,
        replacement_suffix: String,
    },
    /// Append to the immutable prefix.
    Commit { text: String },
    /// Close the segment; commits `text` (what remains of the suffix).
    Final { text: String },
    /// A later alternative for committed or final text; never mutates the ledger.
    CorrectionSuggested {
        byte_start: usize,
        byte_end: usize,
        text: String,
    },
    /// Lost or discontinuous input samples; closes the segment as incomplete.
    Gap { start_sample: u64, end_sample: u64 },
    /// Mode change, shed step, skipped pass, refused adaptation, endpoint decision.
    Status { code: String, detail: String },
}

/// One emitted event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEvent {
    pub stream_id: String,
    pub segment_id: u64,
    pub event_sequence: u64,
    pub audio_start_sample: u64,
    pub audio_end_sample: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

/// Per-segment state held by the ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegmentState {
    pub immutable: String,
    pub suffix: String,
    pub closed: bool,
    pub incomplete: bool,
    pub last_revision: Option<u64>,
}

/// Applies events and refuses any that would break a §3.1 invariant.
#[derive(Debug, Default)]
pub struct Ledger {
    last_seq: Option<u64>,
    segments: std::collections::BTreeMap<u64, SegmentState>,
    open_segment: Option<u64>,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// State of one segment, if seen.
    pub fn segment(&self, id: u64) -> Option<&SegmentState> {
        self.segments.get(&id)
    }

    /// Immutable text of all segments in order, joined by single spaces.
    pub fn committed_text(&self) -> String {
        let parts: Vec<&str> = self
            .segments
            .values()
            .map(|s| s.immutable.trim())
            .filter(|s| !s.is_empty())
            .collect();
        parts.join(" ")
    }

    /// Apply one event, enforcing every invariant.
    pub fn apply(&mut self, ev: &TranscriptEvent) -> Result<(), CoreError> {
        if let Some(prev) = self.last_seq {
            if ev.event_sequence <= prev {
                return Err(CoreError::Contract("event_sequence must strictly increase"));
            }
        }
        if payload_len(&ev.kind) > MAX_PAYLOAD_BYTES {
            return Err(CoreError::Contract("payload too large"));
        }
        let text_event = matches!(
            ev.kind,
            EventKind::Preview { .. } | EventKind::Commit { .. } | EventKind::Final { .. }
        );
        if text_event {
            match self.open_segment {
                Some(open) if open == ev.segment_id => {}
                Some(open) if ev.segment_id < open => {
                    return Err(CoreError::Contract("text for an earlier segment"));
                }
                _ => {
                    if self.segments.get(&ev.segment_id).is_some_and(|s| s.closed) {
                        return Err(CoreError::Contract("text after Final"));
                    }
                    if let Some(open) = self.open_segment {
                        if self.segments.get(&open).is_some_and(|s| !s.closed) {
                            return Err(CoreError::Contract("new segment while one is open"));
                        }
                    }
                    self.open_segment = Some(ev.segment_id);
                }
            }
        }
        let seg = self.segments.entry(ev.segment_id).or_default();
        match &ev.kind {
            EventKind::Preview {
                revision_id,
                replacement_suffix,
            } => {
                if seg.closed {
                    return Err(CoreError::Contract("Preview after Final"));
                }
                if seg.last_revision.is_some_and(|r| *revision_id <= r) {
                    return Err(CoreError::Contract("revision_id must increase"));
                }
                seg.last_revision = Some(*revision_id);
                seg.suffix.clone_from(replacement_suffix);
            }
            EventKind::Commit { text } => {
                if seg.closed {
                    return Err(CoreError::Contract("Commit after Final"));
                }
                seg.immutable.push_str(text);
                seg.suffix.clear();
            }
            EventKind::Final { text } => {
                if seg.closed {
                    return Err(CoreError::Contract("Final twice"));
                }
                seg.immutable.push_str(text);
                seg.suffix.clear();
                seg.closed = true;
            }
            EventKind::CorrectionSuggested {
                byte_start,
                byte_end,
                ..
            } => {
                if byte_start > byte_end || *byte_end > seg.immutable.len() {
                    return Err(CoreError::Contract("correction outside committed text"));
                }
            }
            EventKind::Gap {
                start_sample,
                end_sample,
            } => {
                if start_sample >= end_sample {
                    return Err(CoreError::Contract("empty gap"));
                }
                seg.suffix.clear();
                seg.closed = true;
                seg.incomplete = true;
            }
            EventKind::Status { .. } => {}
        }
        self.last_seq = Some(ev.event_sequence);
        Ok(())
    }
}

fn payload_len(k: &EventKind) -> usize {
    match k {
        EventKind::Preview {
            replacement_suffix, ..
        } => replacement_suffix.len(),
        EventKind::Commit { text } | EventKind::Final { text } => text.len(),
        EventKind::CorrectionSuggested { text, .. } => text.len(),
        EventKind::Status { code, detail } => code.len() + detail.len(),
        EventKind::Gap { .. } => 0,
    }
}
