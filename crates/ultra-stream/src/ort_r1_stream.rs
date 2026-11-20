//! Per-utterance [`OrtR1RawStream`], [`Backend`] impl, and causal [`AsrStream`] wrapper
//! (`OrtR1AsrStream`) for [`OrtTransducerBackend`].

use std::collections::VecDeque;
use std::sync::Arc;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};
use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{chunk_bounds, FeatureRing, RnntStreamState};

use crate::api::{
    AsrStream, Backend, BackendCaps, OutputPolicy, PollResult, PushReport, PushStatus, StreamClock,
    StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};
use crate::ort_r1::{BatchWorkspace, OrtTransducerBackend, StateBuf, R1_SAMPLES_PER_ENC_FRAME};
use crate::sherpa_ffi::verify_cpu_only_process;

/// Low-level per-utterance stream used by [`OrtTransducerBackend::decode_streams`] and SP-A banks.
pub struct OrtR1RawStream {
    pub(crate) engine: Arc<OrtTransducerBackend>,
    pub(crate) mel: TrainerLogMel,
    pub(crate) mel_scratch: Vec<[f32; MEL_BINS]>,
    pub(crate) ring: FeatureRing,
    pub(crate) t_next: usize,
    pub(crate) finished: bool,
    pub(crate) states: Vec<StateBuf>,
    pub rnnt: RnntStreamState,
}

impl OrtR1RawStream {
    pub fn accept_i16(&mut self, pcm: &[i16]) {
        self.mel.push_i16_into(pcm, &mut self.mel_scratch);
        if !self.mel_scratch.is_empty() {
            self.ring.push_rows(&self.mel_scratch);
            self.mel_scratch.clear();
        }
    }

    pub fn input_finished(&mut self) {
        self.finished = true;
    }

    pub fn chunk_bounds(&self) -> Option<(usize, usize)> {
        chunk_bounds(
            self.t_next,
            self.ring.n_feat(),
            self.engine.bundle.streaming.chunk_frames,
            self.finished,
        )
    }

    pub fn is_ready(&self) -> bool {
        self.chunk_bounds().is_some()
    }

    pub fn text(&self) -> String {
        self.engine.bundle.tokens.decode_ids(&self.rnnt.tokens)
    }
}

impl Backend for Arc<OrtTransducerBackend> {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: format!(
                "ort-r1-{}-{}",
                self.bundle.family.as_str(),
                self.bundle.precision.as_str()
            ),
            supports_batch_step: true,
            supports_partial_push: false,
            supports_commit_policy: true,
            chunk_samples: (self.bundle.streaming.chunk_frames) * 640,
            right_context_samples: 0,
            cpu_only_verified: verify_cpu_only_process(),
        }
    }

    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError> {
        Ok(self.create_stream_with_clock(cfg, StreamClock::real()))
    }
}

pub(crate) struct OrtR1AsrStream {
    pub(crate) raw: OrtR1RawStream,
    pub(crate) ws: BatchWorkspace,
    pub(crate) clock: StreamClock,
    pub(crate) emitter: Emitter,
    pub(crate) cfg: StreamConfig,
    pub(crate) accepted_samples: u64,
    pub(crate) processed_samples: u64,
    pub(crate) wakeups: u64,
    pub(crate) events_emitted: u64,
    pub(crate) last_hyp: String,
    pub(crate) finished: bool,
    pub(crate) cancelled: bool,
    pub(crate) events: VecDeque<TranscriptEvent>,
}

impl OrtR1AsrStream {
    fn decode_ready(&mut self) -> Result<(), StreamError> {
        let mut ran = false;
        while self.raw.is_ready() {
            let eng = Arc::clone(&self.raw.engine);
            let (_, w) = eng.decode_streams(&mut [&mut self.raw], &mut self.ws)?;
            self.wakeups += w;
            ran = ran || w > 0;
        }
        if !ran {
            return Ok(());
        }
        self.processed_samples =
            ((self.raw.t_next as u64) * R1_SAMPLES_PER_ENC_FRAME).min(self.accepted_samples);
        let hyp = self.raw.text();
        if hyp != self.last_hyp {
            self.last_hyp.clone_from(&hyp);
            let ts = self.clock.now_ns();
            let owned_to = self.processed_samples;
            match self.cfg.policy {
                OutputPolicy::EditableDictation => {
                    let ev = self.emitter.preview(ts, &hyp, owned_to)?;
                    self.events_emitted += 1;
                    self.events.push_back(ev);
                }
                OutputPolicy::Commit => {
                    let imm_len = self.emitter.immutable().len();
                    if hyp.starts_with(self.emitter.immutable()) && hyp.len() > imm_len {
                        let tail = &hyp[imm_len..];
                        let ev = match tail.rfind(' ').filter(|&sp| sp > 0) {
                            Some(sp) => self.emitter.commit(ts, tail, sp, owned_to)?,
                            None => self.emitter.preview(ts, tail, owned_to)?,
                        };
                        self.events_emitted += 1;
                        self.events.push_back(ev);
                    }
                }
            }
        }
        Ok(())
    }
}

impl AsrStream for OrtR1AsrStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        self.raw.accept_i16(pcm);
        self.accepted_samples += pcm.len() as u64;
        self.decode_ready()?;
        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if let Some(ev) = self.events.pop_front() {
            return Ok(PollResult::Event(Box::new(ev)));
        }
        if self.cancelled {
            return Ok(PollResult::Ended(TerminalOutcome::Cancelled));
        }
        if self.finished {
            return Ok(PollResult::Ended(TerminalOutcome::Finished));
        }
        Ok(PollResult::Pending)
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        if self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "finish_input on a cancelled or failed stream",
            )
            .into());
        }
        if !self.finished {
            self.raw.input_finished();
            while self.raw.is_ready() {
                let eng = Arc::clone(&self.raw.engine);
                let (_, w) = eng.decode_streams(&mut [&mut self.raw], &mut self.ws)?;
                self.wakeups += w;
            }
            let hyp = self.raw.text();
            let imm_len = self.emitter.immutable().len();
            let suffix = if imm_len == 0 {
                hyp.as_str()
            } else if hyp.len() > imm_len && hyp.is_char_boundary(imm_len) {
                &hyp[imm_len..]
            } else {
                ""
            };
            let ts = self.clock.now_ns();
            let ev = self.emitter.finalize(
                ts,
                suffix,
                self.accepted_samples,
                ProcessingEvidence {
                    accepted_through_sample: self.accepted_samples,
                    required_processed_through_sample: self.accepted_samples,
                    flushed: true,
                },
            )?;
            self.finished = true;
            self.processed_samples = self.accepted_samples;
            self.events_emitted += 1;
            self.events.push_back(ev);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.finished {
            self.cancelled = true;
            self.events.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        StreamMetrics {
            accepted_samples: self.accepted_samples,
            processed_samples: self.processed_samples,
            wakeups: self.wakeups,
            events_emitted: self.events_emitted,
            cap_hits: self.raw.rnnt.forced_advances,
            refiner_attempts: 0,
            refiner_accepts: 0,
            refiner_rejects: 0,
            refiner_charged_cpu_s: 0.0,
            retained_bytes: self.raw.ring.retained_bytes() as u64,
        }
    }
}
