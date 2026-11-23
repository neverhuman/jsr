//! Receiver side of the event contract: [`Ledger`] applies events atomically and refuses any
//! that break §33.2 (validation order) or §33.3 (duplicates, predecessors, epochs).

use std::collections::BTreeMap;

use super::{
    contract, Body, Closure, SegmentState, SegmentStatus, TextUpdate, TranscriptEvent,
    EVENT_SCHEMA, MAX_PAYLOAD_BYTES,
};
use crate::receipt::sha256_hex;
use crate::CoreError;

/// Outcome of a successful apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    New,
    /// Identical retransmission of an already-applied event: a no-op.
    Duplicate,
}

/// How many recent events are remembered for duplicate detection (bounded retention, §33.3).
/// A retransmission older than this window is refused as out of order, never re-applied.
pub const DEDUP_WINDOW: usize = 1024;

/// Receiver-side ledger for one stream (in-process, reliable ordered channel).
/// Every apply is atomic: a refused event leaves the ledger exactly as it was.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    stream_id: Option<String>,
    next_seq: Option<u64>,
    epoch: u64,
    seen: BTreeMap<u64, String>,
    segments: BTreeMap<u64, SegmentState>,
    open: Option<u64>,
    /// Ownership of the open segment starts here; earlier samples belong to closed segments.
    owned_from: u64,
    owned_to: u64,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// State of one segment, if seen.
    pub fn segment(&self, id: u64) -> Option<&SegmentState> {
        self.segments.get(&id)
    }

    /// Immutable text of every segment in order, joined by single spaces.
    pub fn committed_text(&self) -> String {
        let parts: Vec<&str> = self
            .segments
            .values()
            .map(|s| s.immutable.trim())
            .filter(|s| !s.is_empty())
            .collect();
        parts.join(" ")
    }

    /// SHA-256 of the full committed ledger (resynchronisation check, §33.3).
    pub fn committed_hash(&self) -> String {
        sha256_hex(self.committed_text().as_bytes())
    }

    /// Apply one event, enforcing every invariant of §33.2–33.3. Atomic: on error nothing changes.
    pub fn apply(&mut self, ev: &TranscriptEvent) -> Result<Applied, CoreError> {
        if ev.schema != EVENT_SCHEMA {
            return Err(contract("unknown event schema"));
        }
        if self.stream_id.as_ref().is_some_and(|s| *s != ev.stream_id) {
            return Err(contract("event for another stream"));
        }
        let hash = ev.canonical_hash()?;
        if let Some(prev) = self.seen.get(&ev.event_sequence) {
            return if *prev == hash {
                Ok(Applied::Duplicate)
            } else {
                Err(contract("event_sequence reused with different content"))
            };
        }
        if let Some(next) = self.next_seq {
            if ev.event_sequence < next {
                return Err(contract("event_sequence must strictly increase"));
            }
            if ev.event_sequence > next {
                return Err(contract("missing predecessor event"));
            }
        }
        if ev.state_epoch < self.epoch {
            return Err(contract("stale state_epoch"));
        }
        let next_seq = ev
            .event_sequence
            .checked_add(1)
            .ok_or(contract("event_sequence overflow"))?;
        // Stage on a copy without cloning `seen` (untouched by `apply_body`); commit only if every check passes.
        let mut seen = std::mem::take(&mut self.seen);
        let mut staged = self.clone();
        if let Err(e) = staged.apply_body(ev) {
            self.seen = seen;
            return Err(e);
        }
        staged.stream_id = Some(ev.stream_id.clone());
        staged.epoch = ev.state_epoch;
        staged.next_seq = Some(next_seq);
        seen.insert(ev.event_sequence, hash);
        while seen.len() > DEDUP_WINDOW {
            seen.pop_first();
        }
        staged.seen = seen;
        *self = staged;
        Ok(Applied::New)
    }

    fn open_segment(&mut self, id: u64) -> Result<&mut SegmentState, CoreError> {
        if let Some(open) = self.open {
            if open != id {
                return Err(contract("text for a segment other than the open one"));
            }
        } else {
            if self
                .segments
                .keys()
                .next_back()
                .is_some_and(|&last| id <= last)
            {
                return Err(contract(
                    "segment ids must increase; closed segments never reopen",
                ));
            }
            self.segments.insert(id, SegmentState::new());
            self.open = Some(id);
        }
        Ok(self.segments.get_mut(&id).expect("open segment exists"))
    }

    fn apply_body(&mut self, ev: &TranscriptEvent) -> Result<(), CoreError> {
        match &ev.body {
            Body::Preview(u) | Body::Commit(u) | Body::Final { update: u, .. } => {
                let starting = self.open.is_none();
                if starting && u.ownership_start_sample < self.owned_to {
                    return Err(contract("ownership overlaps an earlier segment"));
                }
                if !starting
                    && (u.ownership_start_sample != self.owned_from
                        || u.ownership_end_sample < self.owned_to)
                {
                    return Err(contract(
                        "ownership start changed or ownership end regressed",
                    ));
                }
                if let Body::Final { processing: p, .. } = &ev.body {
                    // ownership_end <= required_processed <= accepted, and the flush ran.
                    if p.required_processed_through_sample > p.accepted_through_sample {
                        return Err(contract("processed watermark beyond accepted input"));
                    }
                    if p.accepted_through_sample < u.ownership_end_sample {
                        return Err(contract("Final claims ownership beyond accepted input"));
                    }
                    if p.required_processed_through_sample < u.ownership_end_sample || !p.flushed {
                        return Err(contract("Final without complete required processing"));
                    }
                }
                let (from, to) = (u.ownership_start_sample, u.ownership_end_sample);
                let seg = self.open_segment(u.segment_id)?;
                check_update(seg, u)?;
                let (b, a, n) = (
                    u.immutable_before_utf8_bytes,
                    u.immutable_after_utf8_bytes,
                    u.snapshot_utf8.len(),
                );
                match &ev.body {
                    Body::Preview(_) if a != b => {
                        return Err(contract("Preview must not commit (after != before)"))
                    }
                    Body::Commit(_) if a <= b => {
                        return Err(contract("Commit must append (after > before)"))
                    }
                    Body::Final { .. } if a != n => {
                        return Err(contract("Final must commit the whole snapshot"))
                    }
                    _ => {}
                }
                seg.immutable.push_str(&u.snapshot_utf8[b..a]);
                seg.suffix = u.snapshot_utf8[a..].to_string();
                seg.revision = u.revision;
                let is_final = matches!(ev.body, Body::Final { .. });
                if is_final {
                    seg.status = SegmentStatus::Final;
                    seg.suffix.clear();
                    self.open = None;
                }
                self.owned_from = from;
                self.owned_to = to;
                Ok(())
            }
            Body::SegmentIncomplete {
                closure,
                accepted_through_sample,
                processed_through_sample,
                ..
            } => {
                if processed_through_sample > accepted_through_sample {
                    return Err(contract("processed watermark beyond accepted watermark"));
                }
                self.close_incomplete(closure)
            }
            Body::Gap {
                closure,
                start_sample,
                end_sample,
                ..
            } => {
                if let (Some(s), Some(e)) = (start_sample, end_sample) {
                    if s >= e {
                        return Err(contract("empty or reversed gap range"));
                    }
                }
                match closure {
                    Some(c) => self.close_incomplete(c),
                    None if self.open.is_some() => Err(contract("Gap must close the open segment")),
                    None => Ok(()),
                }
            }
            Body::CorrectionSuggested {
                segment_id,
                final_revision,
                final_sha256,
                byte_start,
                byte_end,
                alternative,
            } => {
                let seg = self
                    .segments
                    .get(segment_id)
                    .ok_or(contract("correction for an unknown segment"))?;
                if seg.status != SegmentStatus::Final {
                    return Err(contract("correction for a segment that is not finalised"));
                }
                if *final_revision != seg.revision
                    || *final_sha256 != sha256_hex(seg.immutable.as_bytes())
                {
                    return Err(contract("correction does not reference the final revision"));
                }
                if byte_start > byte_end
                    || *byte_end > seg.immutable.len()
                    || !seg.immutable.is_char_boundary(*byte_start)
                    || !seg.immutable.is_char_boundary(*byte_end)
                {
                    return Err(contract(
                        "correction range outside final text or not on UTF-8 boundaries",
                    ));
                }
                if alternative.len() > MAX_PAYLOAD_BYTES {
                    return Err(contract("payload too large"));
                }
                Ok(())
            }
            Body::Status { code, detail } => {
                if code.len() + detail.len() > MAX_PAYLOAD_BYTES {
                    return Err(contract("payload too large"));
                }
                Ok(())
            }
        }
    }

    fn close_incomplete(&mut self, c: &Closure) -> Result<(), CoreError> {
        if self.open != Some(c.segment_id) {
            return Err(contract("incomplete closure of a segment that is not open"));
        }
        let seg = self
            .segments
            .get_mut(&c.segment_id)
            .expect("open segment exists");
        if c.base_revision != seg.revision || Some(c.revision) != c.base_revision.checked_add(1) {
            return Err(contract("revision mismatch"));
        }
        if c.preserved_prefix_utf8_bytes != seg.immutable.len()
            || c.prefix_sha256 != sha256_hex(seg.immutable.as_bytes())
        {
            return Err(contract(
                "closure does not preserve the immutable prefix exactly",
            ));
        }
        seg.suffix.clear();
        seg.revision = c.revision;
        seg.status = SegmentStatus::Incomplete;
        self.open = None;
        Ok(())
    }
}

/// §33.2 checks 1–4 for a text update against the receiver's segment state.
fn check_update(seg: &SegmentState, u: &TextUpdate) -> Result<(), CoreError> {
    if seg.status != SegmentStatus::Open {
        return Err(contract("text after the segment closed"));
    }
    if u.base_revision != seg.revision || Some(u.revision) != u.base_revision.checked_add(1) {
        return Err(contract("revision mismatch"));
    }
    let (b, a, s) = (
        u.immutable_before_utf8_bytes,
        u.immutable_after_utf8_bytes,
        &u.snapshot_utf8,
    );
    if b != seg.immutable.len() {
        return Err(contract(
            "immutable_before differs from the receiver's immutable length",
        ));
    }
    if !(b <= a && a <= s.len()) {
        return Err(contract("offsets out of order or beyond the snapshot"));
    }
    if !s.is_char_boundary(b) || !s.is_char_boundary(a) {
        return Err(contract("offset not on a UTF-8 boundary"));
    }
    if s.as_bytes()[..b] != *seg.immutable.as_bytes() {
        return Err(contract(
            "snapshot does not start with the immutable prefix",
        ));
    }
    if s.len() > MAX_PAYLOAD_BYTES {
        return Err(contract("payload too large"));
    }
    if u.ownership_start_sample > u.ownership_end_sample {
        return Err(contract("ownership interval reversed"));
    }
    Ok(())
}
