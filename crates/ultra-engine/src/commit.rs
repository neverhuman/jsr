//! H45 dual-pass endpoint commit pass: fast provisional streaming (`c=4`, 160 ms) paired with
//! an endpoint commit re-decode (`c=16`, 640 ms) over buffered PCM using the same `Arc<Model>`
//! weights, optional `320 ms` trailing silence pad, separate-vocab CTC rescoring, and MBR.
//!
//! Correctness invariants:
//! 1. **Separate CTC vocabulary**: CTC rescoring always retokenizes candidate text into the
//!    checkpoint's own [`CtcVocab`]; RNN-T token IDs are never passed into CTC forward scoring.
//! 2. **Impossible CTC paths receive infinite cost**: OOV text, blank targets, or sequences with
//!    `num_frames < targets.len() + adjacent_repeats` return `f32::NEG_INFINITY` log-prob.
//! 3. **Immutable committed prefix**: [`CommitPassSession::committed_prefix`] is append-only;
//!    an endpoint commit may revise only the active uncommitted segment.

mod rescore;

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{Engine, EngineError, EngineStream, Workspace};

pub use rescore::{
    combine_ctc_rnnt_scores, ctc_forward_logp, ctc_rescore_text, mbr_select, word_edit_distance,
    CtcVocab,
};

/// Default audio sample rate in Hz (`16 kHz`).
pub const DEFAULT_SAMPLE_RATE: usize = 16_000;
/// Default trailing silence pad at utterance endpoints (`320 ms` at `16 kHz`).
pub const DEFAULT_TAIL_PAD_SAMPLES: usize = 5_120;

/// Configuration for the H45 endpoint commit pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommitConfig {
    pub commit_chunk: usize,
    pub tail_pad_samples: usize,
    pub beam_size: usize,
    pub ctc_weight: f32,
    pub mbr_enabled: bool,
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self {
            commit_chunk: 16,
            tail_pad_samples: DEFAULT_TAIL_PAD_SAMPLES,
            beam_size: 1,
            ctc_weight: 0.0,
            mbr_enabled: false,
        }
    }
}

impl CommitConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(1..=64).contains(&self.commit_chunk) {
            return Err(EngineError::Shape(format!(
                "commit_chunk {} not in 1..=64",
                self.commit_chunk
            )));
        }
        if !(1..=16).contains(&self.beam_size) {
            return Err(EngineError::Shape(format!(
                "beam_size {} not in 1..=16",
                self.beam_size
            )));
        }
        if !self.ctc_weight.is_finite() || self.ctc_weight < 0.0 {
            return Err(EngineError::Format(format!(
                "invalid ctc_weight {}",
                self.ctc_weight
            )));
        }
        Ok(())
    }
}

/// Streaming transcript event separating fast provisional output from committed endpoint output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommitEvent {
    Provisional {
        prefix: String,
        text: String,
        audio_ms: f64,
    },
    Committed {
        prefix: String,
        provisional: String,
        text: String,
        revised: bool,
        commit_ms: f64,
        audio_ms: f64,
    },
}

/// Output of [`CommitPassSession::commit_endpoint`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitResult {
    pub provisional: String,
    pub committed: String,
    pub committed_prefix: String,
    pub revised: bool,
    pub commit_wall_ms: f64,
    pub audio_ms: f64,
    pub selected_hyp: usize,
}

impl CommitResult {
    pub fn into_event(self) -> CommitEvent {
        CommitEvent::Committed {
            prefix: self.committed_prefix,
            provisional: self.provisional,
            text: self.committed,
            revised: self.revised,
            commit_ms: self.commit_wall_ms,
            audio_ms: self.audio_ms,
        }
    }
}

/// Append `tail_pad_samples` zero samples of trailing silence to `pcm`.
pub fn pad_tail_silence(pcm: &[i16], tail_pad_samples: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(pcm.len() + tail_pad_samples);
    out.extend_from_slice(pcm);
    out.resize(pcm.len() + tail_pad_samples, 0);
    out
}

fn drain_stream(eng: &Engine, s: &mut EngineStream, ws: &mut Workspace) -> Result<(), EngineError> {
    while eng.decode(&mut [s], ws)? > 0 {}
    Ok(())
}

/// Stateful dual-pass streaming session sharing one loaded `Arc<Model>` across provisional (`c=4`)
/// and endpoint commit (`c=16`) passes while preserving an immutable committed prefix.
#[derive(Debug)]
pub struct CommitPassSession {
    provisional_engine: Engine,
    commit_engine: Engine,
    cfg: CommitConfig,
    stream: EngineStream,
    pcm_buf: Vec<i16>,
    ws: Workspace,
    committed_prefix: String,
}

impl CommitPassSession {
    pub fn new(engine: Engine, cfg: CommitConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        let mut commit_engine = engine
            .clone()
            .with_chunk(cfg.commit_chunk)?
            .with_beam(cfg.beam_size);
        commit_engine.endpoint_reset_ms = 0;
        commit_engine.endpoint_replay_ms = 0;
        let stream = engine.create_stream();
        Ok(Self {
            provisional_engine: engine,
            commit_engine,
            cfg,
            stream,
            pcm_buf: Vec::new(),
            ws: Workspace::default(),
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
    pub fn committed_prefix(&self) -> &str {
        &self.committed_prefix
    }
    pub fn provisional_text(&self) -> String {
        self.provisional_engine.text(&self.stream)
    }
    pub fn buffered_samples(&self) -> usize {
        self.pcm_buf.len()
    }
    pub fn audio_ms(&self) -> f64 {
        self.pcm_buf.len() as f64 * 1000.0 / DEFAULT_SAMPLE_RATE as f64
    }

    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<CommitEvent, EngineError> {
        self.stream.accept_i16(pcm)?;
        self.pcm_buf.extend_from_slice(pcm);
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        Ok(CommitEvent::Provisional {
            prefix: self.committed_prefix.clone(),
            text: self.provisional_engine.text(&self.stream),
            audio_ms: self.audio_ms(),
        })
    }

    pub fn commit_endpoint(&mut self) -> Result<CommitResult, EngineError> {
        self.commit_inner(None)
    }

    pub fn commit_endpoint_with_ctc(
        &mut self,
        ctc_vocab: &CtcVocab,
        ctc_log_probs: &[f32],
        ctc_frames: usize,
    ) -> Result<CommitResult, EngineError> {
        self.commit_inner(Some((ctc_vocab, ctc_log_probs, ctc_frames)))
    }

    fn commit_inner(
        &mut self,
        ctc: Option<(&CtcVocab, &[f32], usize)>,
    ) -> Result<CommitResult, EngineError> {
        let audio_ms = self.audio_ms();
        self.stream.input_finished();
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        let provisional = self.provisional_engine.text(&self.stream);
        let t0 = Instant::now();
        let mut cs = self.commit_engine.create_stream();
        cs.accept_i16(&self.pcm_buf)?;
        if self.cfg.tail_pad_samples > 0 {
            cs.accept_i16(&vec![0i16; self.cfg.tail_pad_samples])?;
        }
        cs.input_finished();
        drain_stream(&self.commit_engine, &mut cs, &mut self.ws)?;
        let (committed, selected_hyp) = self.select_commit_hyp(&cs, ctc);
        let commit_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let revised = committed != provisional;
        let trimmed = committed.trim();
        if !trimmed.is_empty() {
            if !self.committed_prefix.is_empty() {
                self.committed_prefix.push(' ');
            }
            self.committed_prefix.push_str(trimmed);
        }
        let committed_prefix = self.committed_prefix.clone();
        self.reset_segment();
        Ok(CommitResult {
            provisional,
            committed,
            committed_prefix,
            revised,
            commit_wall_ms,
            audio_ms,
            selected_hyp,
        })
    }

    fn select_commit_hyp(
        &self,
        stream: &EngineStream,
        ctc: Option<(&CtcVocab, &[f32], usize)>,
    ) -> (String, usize) {
        let Some(beam) = &stream.dec.beam else {
            return (self.commit_engine.text(stream), 0);
        };
        if beam.hyps.len() <= 1 {
            return (self.commit_engine.text(stream), 0);
        }
        let mut texts = Vec::with_capacity(beam.hyps.len());
        let mut cands = Vec::with_capacity(beam.hyps.len());
        for hy in &beam.hyps {
            let ids: Vec<i64> = hy.tokens.iter().map(|&t| i64::from(t)).collect();
            let text = self.commit_engine.model.tokens.decode_ids(&ids);
            let score = ctc.map_or(hy.score, |(vocab, lp, frames)| {
                let ctc_lp = ctc_rescore_text(vocab, lp, frames, &text);
                combine_ctc_rnnt_scores(hy.score, ctc_lp, self.cfg.ctc_weight)
            });
            texts.push(text.clone());
            cands.push((text.split_whitespace().map(str::to_owned).collect(), score));
        }
        let idx = if self.cfg.mbr_enabled {
            mbr_select(&cands)
        } else {
            cands
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.1.total_cmp(&b.1).then_with(|| ib.cmp(ia)))
                .map_or(0, |(i, _)| i)
        };
        (texts.swap_remove(idx), idx)
    }

    /// Clear the active segment buffer and provisional stream without touching `committed_prefix`.
    pub fn reset_segment(&mut self) {
        self.pcm_buf.clear();
        self.stream = self.provisional_engine.create_stream();
    }

    /// Clear both active segment state and the accumulated `committed_prefix`.
    pub fn reset_all(&mut self) {
        self.committed_prefix.clear();
        self.reset_segment();
    }
}
