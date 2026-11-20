//! Output event contract `ultra.transcript-event.v1` (spec v1.0-B §33.2–33.3; v1.0 §3).
//!
//! [`Ledger`] is the receiver: it applies events and refuses any that would break the
//! contract, validating in the order §33.2 prescribes. [`Emitter`] is its mirror on the
//! producer side: it turns "show this text / commit this much / finish" calls into events the
//! ledger accepts, so backends never hand-assemble revisions or byte offsets.

use serde::{Deserialize, Serialize};

use crate::receipt::sha256_hex;
use crate::CoreError;

mod emitter;
mod ledger;
pub use emitter::Emitter;
pub use ledger::{Applied, Ledger, DEDUP_WINDOW};

/// Schema identifier carried by every event.
pub const EVENT_SCHEMA: &str = "ultra.transcript-event.v1";
/// Largest segment snapshot or alternative, in bytes.
pub const MAX_PAYLOAD_BYTES: usize = 4096;

/// One segment-text update (Preview, Commit or Final). Offsets are segment-local UTF-8 bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextUpdate {
    pub segment_id: u64,
    pub base_revision: u64,
    pub revision: u64,
    /// The whole segment text as now displayed: immutable prefix + mutable suffix.
    pub snapshot_utf8: String,
    pub immutable_before_utf8_bytes: usize,
    pub immutable_after_utf8_bytes: usize,
    pub ownership_start_sample: u64,
    pub ownership_end_sample: u64,
}

/// Watermarks proving a segment's required processing completed (§33.2, §18.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingEvidence {
    /// Input samples accepted for the segment's owned interval.
    pub accepted_through_sample: u64,
    /// Samples whose required processing (encoder + decoder) completed.
    pub required_processed_through_sample: u64,
    /// The end-of-segment flush (right context) ran to completion.
    pub flushed: bool,
}

/// Closure of an open segment without success (Gap or SegmentIncomplete, §33.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Closure {
    pub segment_id: u64,
    pub base_revision: u64,
    pub revision: u64,
    /// Length of the immutable prefix, which is preserved exactly.
    pub preserved_prefix_utf8_bytes: usize,
    /// SHA-256 (hex) of the preserved immutable prefix.
    pub prefix_sha256: String,
    pub reason: String,
}

/// Event payload, tagged by `event_type`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum Body {
    /// Replace the mutable suffix; `after == before`.
    Preview(TextUpdate),
    /// Append `snapshot[before..after]` to the immutable prefix exactly once; `after > before`.
    Commit(TextUpdate),
    /// Append the rest (`after == snapshot.len()`) and close the segment successfully, once.
    /// Requires evidence that all required processing of the owned interval finished (§18.6).
    Final {
        #[serde(flatten)]
        update: TextUpdate,
        #[serde(flatten)]
        processing: ProcessingEvidence,
    },
    /// Required processing did not finish (deadline miss, backend failure, cancel).
    SegmentIncomplete {
        #[serde(flatten)]
        closure: Closure,
        accepted_through_sample: u64,
        processed_through_sample: u64,
        drained: bool,
    },
    /// Missing, duplicated or out-of-order input. Closes the open segment (if any) as incomplete.
    Gap {
        #[serde(flatten)]
        closure: Option<Closure>,
        /// Sample range when known; `None` when the true extent is unknown.
        start_sample: Option<u64>,
        end_sample: Option<u64>,
        gap_reason: String,
    },
    /// A later alternative for a closed segment. Never mutates the ledger.
    CorrectionSuggested {
        segment_id: u64,
        final_revision: u64,
        final_sha256: String,
        byte_start: usize,
        byte_end: usize,
        alternative: String,
    },
    /// Mode change, shed step, skipped pass, endpoint decision.
    Status { code: String, detail: String },
}

/// One emitted event: the common envelope (§33.2) and its body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEvent {
    pub schema: String,
    pub stream_id: String,
    pub event_sequence: u64,
    pub state_epoch: u64,
    pub observed_monotonic_ns: u64,
    pub model_config_id: String,
    #[serde(flatten)]
    pub body: Body,
}

impl TranscriptEvent {
    /// The segment this event refers to, if any.
    pub fn segment_id(&self) -> Option<u64> {
        match &self.body {
            Body::Preview(u) | Body::Commit(u) | Body::Final { update: u, .. } => {
                Some(u.segment_id)
            }
            Body::SegmentIncomplete { closure, .. } => Some(closure.segment_id),
            Body::Gap { closure, .. } => closure.as_ref().map(|c| c.segment_id),
            Body::CorrectionSuggested { segment_id, .. } => Some(*segment_id),
            Body::Status { .. } => None,
        }
    }

    /// Does this event take part in the segment text revision protocol?
    pub fn is_revision(&self) -> bool {
        matches!(
            self.body,
            Body::Preview(_)
                | Body::Commit(_)
                | Body::Final { .. }
                | Body::SegmentIncomplete { .. }
        ) || matches!(
            &self.body,
            Body::Gap {
                closure: Some(_),
                ..
            }
        )
    }

    /// SHA-256 of the canonical (serde) encoding, used for duplicate detection (§33.3).
    pub fn canonical_hash(&self) -> String {
        sha256_hex(&serde_json::to_vec(self).unwrap_or_default())
    }
}

/// How a segment ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentStatus {
    Open,
    Final,
    Incomplete,
}

/// Per-segment state held by the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentState {
    pub revision: u64,
    pub immutable: String,
    pub suffix: String,
    pub status: SegmentStatus,
}

impl SegmentState {
    fn new() -> Self {
        Self {
            revision: 0,
            immutable: String::new(),
            suffix: String::new(),
            status: SegmentStatus::Open,
        }
    }
    /// What a client displays for this segment.
    pub fn shown(&self) -> String {
        format!("{}{}", self.immutable, self.suffix)
    }
}

pub(crate) fn contract(msg: &'static str) -> CoreError {
    CoreError::Contract(msg)
}
