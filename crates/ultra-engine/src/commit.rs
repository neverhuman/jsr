//! H45 dual-pass endpoint commit pass: fast provisional streaming (`c=4`, 160 ms) paired with
//! an endpoint commit re-decode (`c=16`, 640 ms) over bounded buffered PCM using either a single
//! shared `Arc<Model>` or a hybrid `(live_model, final_model)` pair, optional `320 ms` trailing
//! silence pad, real `ultra_core::events::Emitter` ledger events, exact SentencePiece CTC
//! rescoring, and word-level MBR.

mod rescore;
mod spm;
mod spm_proto;
mod types;
mod vocab;

use std::sync::Arc;
use std::time::Instant;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};

use crate::{Engine, EngineError, EngineStream, Workspace};

pub use rescore::{
    combine_ctc_rnnt_scores, ctc_forward_logp, ctc_rescore_text, mbr_select, word_edit_distance,
};
pub use spm::{SpmModelType, SpmTokenizer};
pub use types::{
    pad_tail_silence, reconcile_immutable_suffix, CommitConfig, CommitEvent, CommitResult,
    DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE, DEFAULT_TAIL_PAD_SAMPLES,
};
pub use vocab::CtcVocab;

/// Alias for [`CommitPassSession`] supporting single-model and hybrid `(live, final)` commit.
pub type CommitPassEngine = CommitPassSession;

fn drain_stream(eng: &Engine, s: &mut EngineStream, ws: &mut Workspace) -> Result<(), EngineError> {
    while eng.decode(&mut [s], ws)? > 0 {}
    Ok(())
}

/// Stateful dual-pass streaming session sharing one `Arc<Model>` (or a validated hybrid
/// `(live, final)` pair) across provisional and endpoint commit passes.
#[derive(Debug)]
pub struct CommitPassSession {
    provisional_engine: Engine,
    commit_engine: Engine,
    cfg: CommitConfig,
    stream_id: String,
    model_config_id: String,
    stream: EngineStream,
    buf: types::SegmentPcm,
    segment_open: bool,
    ws: Workspace,
    commit_ws: Option<Workspace>,
    emitter: Emitter,
    committed_prefix: String,
}

impl CommitPassSession {
    pub fn new(engine: Engine, cfg: CommitConfig) -> Result<Self, EngineError> {
        Self::with_stream_id(engine, cfg, "h45-commit", "h45-commit-v1")
    }

    pub fn new_hybrid(
        live_engine: Engine,
        final_engine: Engine,
        cfg: CommitConfig,
    ) -> Result<Self, EngineError> {
        Self::with_stream_id_hybrid(
            live_engine,
            final_engine,
            cfg,
            "h45-commit",
            "h45-commit-v1",
        )
    }

    pub fn with_stream_id(
        engine: Engine,
        cfg: CommitConfig,
        stream_id: &str,
        model_config_id: &str,
    ) -> Result<Self, EngineError> {
        Self::with_stream_id_hybrid(engine.clone(), engine, cfg, stream_id, model_config_id)
    }

    pub fn with_stream_id_hybrid(
        live_engine: Engine,
        final_engine: Engine,
        cfg: CommitConfig,
        stream_id: &str,
        model_config_id: &str,
    ) -> Result<Self, EngineError> {
        cfg.validate()?;
        types::validate_hybrid_vocab(&live_engine.model, &final_engine.model)?;
        let hybrid = !Arc::ptr_eq(&live_engine.model, &final_engine.model);
        let mut commit_engine = final_engine
            .with_chunk(cfg.commit_chunk)?
            .with_beam(cfg.beam_size);
        commit_engine.endpoint_reset_ms = 0;
        commit_engine.endpoint_replay_ms = 0;
        let stream = live_engine.create_stream();
        Ok(Self {
            provisional_engine: live_engine,
            commit_engine,
            cfg,
            stream_id: stream_id.to_owned(),
            model_config_id: model_config_id.to_owned(),
            stream,
            buf: types::SegmentPcm::default(),
            segment_open: false,
            ws: Workspace::default(),
            commit_ws: hybrid.then(Workspace::default),
            emitter: Emitter::new(stream_id, model_config_id),
            committed_prefix: String::new(),
        })
    }

    pub fn config(&self) -> CommitConfig {
        self.cfg
    }
    pub fn provisional_engine(&self) -> &Engine {
        &self.provisional_engine
    }
    pub fn commit_engine(&self) -> &Engine {
        &self.commit_engine
    }
    pub fn is_hybrid(&self) -> bool {
        !Arc::ptr_eq(&self.provisional_engine.model, &self.commit_engine.model)
    }
    pub fn committed_prefix(&self) -> &str {
        &self.committed_prefix
    }
    pub fn active_immutable(&self) -> &str {
        if self.segment_open {
            self.emitter.immutable()
        } else {
            ""
        }
    }
    pub fn provisional_text(&self) -> String {
        self.provisional_engine.text(&self.stream)
    }
    pub fn buffered_samples(&self) -> usize {
        self.buf.pcm.len()
    }
    pub fn pcm_truncated(&self) -> bool {
        self.buf.truncated
    }
    pub fn dropped_pcm_samples(&self) -> u64 {
        self.buf.dropped
    }
    pub fn total_samples(&self) -> u64 {
        self.buf.total_samples
    }
    pub fn segment_sample_range(&self) -> (u64, u64) {
        (self.buf.seg_start, self.buf.total_samples)
    }
    pub fn audio_ms(&self) -> f64 {
        self.buf.seg_samples as f64 * 1000.0 / DEFAULT_SAMPLE_RATE as f64
    }

    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<CommitEvent, EngineError> {
        self.stream.accept_i16(pcm)?;
        self.buf.push(pcm, self.cfg.max_pcm_samples);
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        Ok(CommitEvent::Provisional {
            prefix: self.committed_prefix.clone(),
            text: self.provisional_engine.text(&self.stream),
            audio_ms: self.audio_ms(),
        })
    }

    pub fn preview_event(&mut self, now_ns: u64) -> Result<TranscriptEvent, EngineError> {
        let suffix = reconcile_immutable_suffix(self.active_immutable(), &self.provisional_text());
        let ev = self
            .emitter
            .preview(now_ns, &suffix, self.buf.total_samples)
            .map_err(|e| EngineError::Lifecycle(e.to_string()))?;
        self.segment_open = true;
        Ok(ev)
    }

    pub fn precommit_active_prefix(
        &mut self,
        now_ns: u64,
        commit_bytes: usize,
    ) -> Result<TranscriptEvent, EngineError> {
        let suffix = reconcile_immutable_suffix(self.active_immutable(), &self.provisional_text());
        let ev = self
            .emitter
            .commit(now_ns, &suffix, commit_bytes, self.buf.total_samples)
            .map_err(|e| EngineError::Lifecycle(e.to_string()))?;
        self.segment_open = true;
        Ok(ev)
    }

    pub fn commit_endpoint(&mut self) -> Result<CommitResult, EngineError> {
        Ok(self.commit_inner(0, None)?.0)
    }

    pub fn commit_endpoint_with_ctc(
        &mut self,
        ctc_vocab: &CtcVocab,
        ctc_log_probs: &[f32],
        ctc_frames: usize,
    ) -> Result<CommitResult, EngineError> {
        Ok(self
            .commit_inner(0, Some((ctc_vocab, ctc_log_probs, ctc_frames)))?
            .0)
    }

    pub fn commit_endpoint_event(
        &mut self,
        now_ns: u64,
    ) -> Result<(CommitResult, TranscriptEvent), EngineError> {
        self.commit_inner(now_ns, None)
    }

    pub fn commit_endpoint_event_with_ctc(
        &mut self,
        now_ns: u64,
        ctc_vocab: &CtcVocab,
        ctc_log_probs: &[f32],
        ctc_frames: usize,
    ) -> Result<(CommitResult, TranscriptEvent), EngineError> {
        self.commit_inner(now_ns, Some((ctc_vocab, ctc_log_probs, ctc_frames)))
    }

    fn commit_inner(
        &mut self,
        now_ns: u64,
        ctc: Option<(&CtcVocab, &[f32], usize)>,
    ) -> Result<(CommitResult, TranscriptEvent), EngineError> {
        let (audio_ms, pcm_truncated) = (self.audio_ms(), self.buf.truncated);
        self.stream.input_finished();
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        let provisional = self.provisional_engine.text(&self.stream);
        let t0 = Instant::now();
        let (raw_committed, selected_hyp) = if pcm_truncated {
            (provisional.clone(), 0)
        } else {
            let cws = self.commit_ws.as_mut().unwrap_or(&mut self.ws);
            rescore::run_commit_decode(&self.commit_engine, &self.buf.pcm, self.cfg, cws, ctc)?
        };
        let commit_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let active_imm = self.active_immutable().to_owned();
        let suffix = reconcile_immutable_suffix(&active_imm, &raw_committed);
        let committed = format!("{active_imm}{suffix}");
        let total = self.buf.total_samples;
        let proc = ProcessingEvidence {
            accepted_through_sample: total,
            required_processed_through_sample: total,
            flushed: true,
        };
        let ev = self
            .emitter
            .finalize(now_ns, &suffix, total, proc)
            .map_err(|e| EngineError::Lifecycle(e.to_string()))?;
        self.segment_open = false;
        let revised = committed.trim() != provisional.trim();
        let trimmed = committed.trim();
        if !trimmed.is_empty() {
            if !self.committed_prefix.is_empty() {
                self.committed_prefix.push(' ');
            }
            self.committed_prefix.push_str(trimmed);
        }
        let committed_prefix = self.committed_prefix.clone();
        self.reset_segment();
        Ok((
            CommitResult {
                provisional,
                committed,
                committed_prefix,
                revised,
                pcm_truncated,
                commit_wall_ms,
                audio_ms,
                selected_hyp,
            },
            ev,
        ))
    }

    pub fn reset_segment(&mut self) {
        self.buf.reset_segment();
        self.segment_open = false;
        self.stream = self.provisional_engine.create_stream();
    }

    pub fn reset_all(&mut self) {
        self.committed_prefix.clear();
        self.buf.reset_all();
        self.segment_open = false;
        self.emitter = Emitter::new(&self.stream_id, &self.model_config_id);
        self.stream = self.provisional_engine.create_stream();
    }
}
