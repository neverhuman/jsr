//! H45/H55 dual-pass endpoint commit pass: fast provisional streaming (`c=4`, 160 ms) paired with
//! an endpoint commit re-decode (`c=16`, 640 ms), optional 320 ms tail pad, `Emitter` events,
//! SentencePiece/model-head CTC, word-MBR, and opt-in INT8 Transformer N-best rescoring.

pub(crate) mod ctc_head;
mod live;
mod lm;
pub mod lm_gate;
mod lm_gate_beam;
mod rescore;
mod session_bias;
mod spm;
mod spm_proto;
mod tf;
mod types;
mod vocab;

use std::time::Instant;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};

use crate::{lm::TfWorkspace, DecodeProgress, Engine, EngineError, EngineStream, Workspace};

pub use ctc_head::aggregate_duplicate_hyps;
pub use live::{run_commit_pcm, run_commit_pcm_timed, CommitSnapshot, CommitTiming};
pub use lm::{CommitLmConfig, CommitLmGateConfig};
pub use lm_gate::{CommitTokenLm, InternalLm, LmGateConfig};
pub use rescore::{
    combine_ctc_rnnt_scores, ctc_forward_logp, ctc_rescore_text, mbr_select, mbr_select_with_temp,
    word_edit_distance,
};
pub use session_bias::{SessionLexicalConfig, SessionLexicalMemory};
pub use spm::{SpmModelType, SpmTokenizer};
pub use tf::*;
pub use types::{
    pad_tail_silence, reconcile_immutable_suffix, CommitConfig, CommitEvent, CommitResult,
    DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE, DEFAULT_TAIL_PAD_SAMPLES,
};
pub use vocab::CtcVocab;

pub(super) fn drain_stream(
    eng: &Engine,
    s: &mut EngineStream,
    ws: &mut Workspace,
) -> Result<usize, EngineError> {
    let mut steps = 0;
    while eng.decode(&mut [s], ws)? > 0 {
        steps += 1;
    }
    Ok(steps)
}

#[derive(Debug, Default)]
pub(super) struct SegBuf {
    pub(super) pcm_buf: Vec<i16>,
    pub(super) pcm_truncated: bool,
    pub(super) dropped_pcm_samples: u64,
    pub(super) segment_samples: u64,
    pub(super) total_samples: u64,
    pub(super) segment_start_sample: u64,
    pub(super) segment_open: bool,
}

/// Stateful dual-pass streaming session sharing one loaded `Arc<Model>` across provisional (`c=4`)
/// and endpoint commit (`c=16`) passes with bounded PCM retention and transcript-ledger events.
#[derive(Debug)]
pub struct CommitPassSession {
    pub(super) provisional_engine: Engine,
    pub(crate) commit_engine: Engine,
    pub(super) cfg: CommitConfig,
    stream_id: String,
    model_config_id: String,
    pub(super) stream: EngineStream,
    pub(super) last_live_progress: DecodeProgress,
    pub(super) seg: SegBuf,
    pub(super) ws: Workspace,
    pub(crate) tf: Option<(CommitTfRescoreConfig, TfWorkspace)>,
    pub(super) emitter: Emitter,
    committed_prefix: String,
    model_ctc_vocab: Option<CtcVocab>,
}

impl CommitPassSession {
    pub fn new(engine: Engine, cfg: CommitConfig) -> Result<Self, EngineError> {
        Self::with_stream_id(engine, cfg, "h45-commit", "h45-commit-v1")
    }

    pub fn with_stream_id(
        engine: Engine,
        cfg: CommitConfig,
        stream_id: &str,
        model_config_id: &str,
    ) -> Result<Self, EngineError> {
        cfg.validate()?;
        let mut ce = engine
            .clone()
            .with_chunk(cfg.commit_chunk)?
            .with_max_chunks(engine.max_chunks.max(4))
            .with_beam(cfg.beam_size);
        ce.endpoint_reset_ms = 0;
        ce.endpoint_replay_ms = 0;
        let stream = engine.create_stream();
        let model_ctc_vocab = (cfg.ctc_weight > 0.0)
            .then(|| ctc_head::load_model_ctc_vocab(&engine.model))
            .flatten();
        Ok(Self {
            provisional_engine: engine,
            commit_engine: ce,
            cfg,
            stream_id: stream_id.to_owned(),
            model_config_id: model_config_id.to_owned(),
            stream,
            last_live_progress: DecodeProgress::default(),
            seg: SegBuf::default(),
            ws: Workspace::default(),
            tf: None,
            emitter: Emitter::new(stream_id, model_config_id),
            committed_prefix: String::new(),
            model_ctc_vocab,
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
    pub fn committed_prefix(&self) -> &str {
        &self.committed_prefix
    }
    pub fn active_immutable(&self) -> &str {
        if self.seg.segment_open {
            self.emitter.immutable()
        } else {
            ""
        }
    }
    pub fn provisional_text(&self) -> String {
        self.provisional_engine.text(&self.stream)
    }
    pub fn buffered_samples(&self) -> usize {
        self.seg.pcm_buf.len()
    }
    pub fn pcm_truncated(&self) -> bool {
        self.seg.pcm_truncated
    }
    pub fn dropped_pcm_samples(&self) -> u64 {
        self.seg.dropped_pcm_samples
    }
    pub fn total_samples(&self) -> u64 {
        self.seg.total_samples
    }
    pub fn segment_sample_range(&self) -> (u64, u64) {
        (self.seg.segment_start_sample, self.seg.total_samples)
    }
    pub fn audio_ms(&self) -> f64 {
        self.seg.segment_samples as f64 * 1000.0 / DEFAULT_SAMPLE_RATE as f64
    }
    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<CommitEvent, EngineError> {
        self.accept_i16_live(pcm)?;
        Ok(CommitEvent::Provisional {
            prefix: self.committed_prefix.clone(),
            text: self.provisional_engine.text(&self.stream),
            audio_ms: self.audio_ms(),
        })
    }

    fn emit_open_event(
        &mut self,
        now_ns: u64,
        bytes: Option<usize>,
    ) -> Result<TranscriptEvent, EngineError> {
        let suffix = reconcile_immutable_suffix(self.active_immutable(), &self.provisional_text());
        let t = self.seg.total_samples;
        let ev = match bytes {
            None => self.emitter.preview(now_ns, &suffix, t),
            Some(n) => self.emitter.commit(now_ns, &suffix, n, t),
        }
        .map_err(|e| EngineError::Lifecycle(e.to_string()))?;
        self.seg.segment_open = true;
        Ok(ev)
    }
    pub fn preview_event(&mut self, now_ns: u64) -> Result<TranscriptEvent, EngineError> {
        self.emit_open_event(now_ns, None)
    }
    pub fn precommit_active_prefix(
        &mut self,
        now_ns: u64,
        n: usize,
    ) -> Result<TranscriptEvent, EngineError> {
        self.emit_open_event(now_ns, Some(n))
    }
    pub fn commit_endpoint(&mut self) -> Result<CommitResult, EngineError> {
        Ok(self.commit_inner(0, None)?.0)
    }
    pub fn commit_endpoint_with_ctc(
        &mut self,
        v: &CtcVocab,
        lp: &[f32],
        t: usize,
    ) -> Result<CommitResult, EngineError> {
        Ok(self.commit_inner(0, Some((v, lp, t)))?.0)
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
        v: &CtcVocab,
        lp: &[f32],
        t: usize,
    ) -> Result<(CommitResult, TranscriptEvent), EngineError> {
        self.commit_inner(now_ns, Some((v, lp, t)))
    }

    fn commit_inner(
        &mut self,
        now_ns: u64,
        ctc: Option<(&CtcVocab, &[f32], usize)>,
    ) -> Result<(CommitResult, TranscriptEvent), EngineError> {
        let (audio_ms, pcm_truncated) = (self.audio_ms(), self.seg.pcm_truncated);
        self.stream.input_finished();
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        let provisional = self.provisional_engine.text(&self.stream);
        let t0 = Instant::now();
        let (raw_committed, selected_hyp) = if pcm_truncated {
            (provisional.clone(), 0)
        } else {
            ctc_head::decode_commit_segment(
                &self.commit_engine,
                &self.seg.pcm_buf,
                self.cfg,
                &mut self.ws,
                ctc,
                self.model_ctc_vocab.as_ref(),
                self.tf.as_mut().map(|(c, w)| (&*c, w)),
            )?
        };
        let commit_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let active_imm = self.active_immutable().to_owned();
        let suffix = reconcile_immutable_suffix(&active_imm, &raw_committed);
        let committed = format!("{active_imm}{suffix}");
        let t = self.seg.total_samples;
        let proc = ProcessingEvidence {
            accepted_through_sample: t,
            required_processed_through_sample: t,
            flushed: true,
        };
        let ev = self
            .emitter
            .finalize(now_ns, &suffix, t, proc)
            .map_err(|e| EngineError::Lifecycle(e.to_string()))?;
        self.seg.segment_open = false;
        let (revised, trimmed) = (committed.trim() != provisional.trim(), committed.trim());
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
        let t = self.seg.total_samples;
        self.seg = SegBuf {
            total_samples: t,
            segment_start_sample: t,
            ..SegBuf::default()
        };
        self.stream = self.provisional_engine.create_stream();
        self.last_live_progress = DecodeProgress::default();
    }

    pub fn reset_all(&mut self) {
        self.committed_prefix.clear();
        self.seg.total_samples = 0;
        self.emitter = Emitter::new(&self.stream_id, &self.model_config_id);
        self.reset_segment();
    }
}
