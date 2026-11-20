//! Producer side of the event contract: [`Emitter`] turns "show / commit / finish" calls into
//! events that [`super::Ledger`] accepts, so backends never hand-assemble revisions or offsets.

use super::{
    contract, Body, Closure, ProcessingEvidence, TextUpdate, TranscriptEvent, EVENT_SCHEMA,
};
use crate::receipt::sha256_hex;
use crate::CoreError;

/// Producer-side mirror of [`Ledger`]: emits valid events from simple calls.
#[derive(Debug, Clone)]
pub struct Emitter {
    stream_id: String,
    model_config_id: String,
    seq: u64,
    epoch: u64,
    segment: u64,
    revision: u64,
    immutable: String,
    open: bool,
    owned_from: u64,
    /// Where the next segment's ownership starts (end of the previous segment; 0 initially).
    next_owned_start: u64,
    owned_to: u64,
}

impl Emitter {
    pub fn new(stream_id: &str, model_config_id: &str) -> Self {
        Self {
            stream_id: stream_id.into(),
            model_config_id: model_config_id.into(),
            seq: 0,
            epoch: 0,
            segment: 0,
            revision: 0,
            immutable: String::new(),
            open: false,
            owned_from: 0,
            next_owned_start: 0,
            owned_to: 0,
        }
    }

    /// Immutable text of the current (or last) segment.
    pub fn immutable(&self) -> &str {
        &self.immutable
    }

    fn envelope(&mut self, now_ns: u64, body: Body) -> TranscriptEvent {
        let ev = TranscriptEvent {
            schema: EVENT_SCHEMA.into(),
            stream_id: self.stream_id.clone(),
            event_sequence: self.seq,
            state_epoch: self.epoch,
            observed_monotonic_ns: now_ns,
            model_config_id: self.model_config_id.clone(),
            body,
        };
        self.seq += 1;
        ev
    }

    fn ensure_open(&mut self) {
        if !self.open {
            self.open = true;
            self.revision = 0;
            self.immutable.clear();
            self.owned_from = self.next_owned_start;
            self.owned_to = self.owned_from;
        }
    }

    fn update(
        &mut self,
        now_ns: u64,
        display: &str,
        commit_bytes: usize,
        owned_to: u64,
    ) -> Result<TextUpdate, CoreError> {
        let before = self.immutable.len();
        let snapshot = format!("{}{}", self.immutable, display);
        let after = before + commit_bytes;
        if after > snapshot.len() || !snapshot.is_char_boundary(after) {
            return Err(contract(
                "commit length outside the displayed text or not on a UTF-8 boundary",
            ));
        }
        let _ = now_ns;
        self.owned_to = self.owned_to.max(owned_to);
        let u = TextUpdate {
            segment_id: self.segment,
            base_revision: self.revision,
            revision: self.revision + 1,
            snapshot_utf8: snapshot,
            immutable_before_utf8_bytes: before,
            immutable_after_utf8_bytes: after,
            ownership_start_sample: self.owned_from,
            ownership_end_sample: self.owned_to,
        };
        Ok(u)
    }

    fn accept(&mut self, u: &TextUpdate) {
        self.immutable = u.snapshot_utf8[..u.immutable_after_utf8_bytes].to_string();
        self.revision = u.revision;
    }

    /// Show `suffix` after the immutable prefix (editable).
    pub fn preview(
        &mut self,
        now_ns: u64,
        suffix: &str,
        owned_to: u64,
    ) -> Result<TranscriptEvent, CoreError> {
        self.ensure_open();
        let u = self.update(now_ns, suffix, 0, owned_to)?;
        self.accept(&u);
        Ok(self.envelope(now_ns, Body::Preview(u)))
    }

    /// Show `suffix` and make its first `commit_bytes` bytes immutable (`commit_bytes > 0`).
    pub fn commit(
        &mut self,
        now_ns: u64,
        suffix: &str,
        commit_bytes: usize,
        owned_to: u64,
    ) -> Result<TranscriptEvent, CoreError> {
        if commit_bytes == 0 {
            return Err(contract("Commit must append (after > before)"));
        }
        self.ensure_open();
        let u = self.update(now_ns, suffix, commit_bytes, owned_to)?;
        self.accept(&u);
        Ok(self.envelope(now_ns, Body::Commit(u)))
    }

    /// Commit everything in `suffix` and close the segment. `processing` must show that all
    /// required processing of the owned interval finished; the ledger refuses it otherwise.
    pub fn finalize(
        &mut self,
        now_ns: u64,
        suffix: &str,
        owned_to: u64,
        processing: ProcessingEvidence,
    ) -> Result<TranscriptEvent, CoreError> {
        self.ensure_open();
        let update = self.update(now_ns, suffix, suffix.len(), owned_to)?;
        self.accept(&update);
        self.close();
        Ok(self.envelope(now_ns, Body::Final { update, processing }))
    }

    fn close(&mut self) {
        self.open = false;
        self.segment += 1;
        self.next_owned_start = self.owned_to;
    }

    fn closure(&self, reason: &str) -> Closure {
        Closure {
            segment_id: self.segment,
            base_revision: self.revision,
            revision: self.revision + 1,
            preserved_prefix_utf8_bytes: self.immutable.len(),
            prefix_sha256: sha256_hex(self.immutable.as_bytes()),
            reason: reason.into(),
        }
    }

    /// Input discontinuity: close any open segment as incomplete and start a new state epoch.
    pub fn gap(&mut self, now_ns: u64, range: Option<(u64, u64)>, reason: &str) -> TranscriptEvent {
        let closure = self.open.then(|| self.closure(reason));
        if self.open {
            self.close();
        }
        self.epoch += 1;
        let (start_sample, end_sample) = range.map_or((None, None), |(s, e)| (Some(s), Some(e)));
        self.envelope(
            now_ns,
            Body::Gap {
                closure,
                start_sample,
                end_sample,
                gap_reason: reason.into(),
            },
        )
    }

    /// Required processing did not finish for the open segment.
    pub fn incomplete(
        &mut self,
        now_ns: u64,
        reason: &str,
        accepted: u64,
        processed: u64,
        drained: bool,
    ) -> Result<TranscriptEvent, CoreError> {
        if !self.open {
            return Err(contract("incomplete closure of a segment that is not open"));
        }
        let closure = self.closure(reason);
        self.close();
        self.epoch += 1;
        Ok(self.envelope(
            now_ns,
            Body::SegmentIncomplete {
                closure,
                accepted_through_sample: accepted,
                processed_through_sample: processed.min(accepted),
                drained,
            },
        ))
    }

    /// A status event (no text effect).
    pub fn status(&mut self, now_ns: u64, code: &str, detail: &str) -> TranscriptEvent {
        self.envelope(
            now_ns,
            Body::Status {
                code: code.into(),
                detail: detail.into(),
            },
        )
    }
}
