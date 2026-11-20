//! Per-utterance [`OrtR1RawStream`], state-output validator, [`Backend`] impl, and causal
//! [`AsrStream`] wrapper (`OrtR1AsrStream`) for [`OrtTransducerBackend`].

use std::collections::VecDeque;
use std::sync::Arc;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};
use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{chunk_bounds, FeatureRing, GpError, R1EncoderStateSpec, RnntStreamState};

use crate::api::{
    AsrStream, Backend, BackendCaps, OutputPolicy, PollResult, PushReport, PushStatus, StreamClock,
    StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};
use crate::ort_r1::{BatchWorkspace, OrtTransducerBackend, StateBuf, R1_SAMPLES_PER_ENC_FRAME};
use crate::sherpa_ffi::verify_cpu_only_process;

/// Validate encoder cache outputs (`k_out`, `v_out`, `kv_len_out`, `conv_out`, and MR `km_out`,
/// `vm_out`, `kvm_len_out`, `convm_out`) before mutating per-stream state.
pub(crate) fn validate_state_outputs(
    specs: &[R1EncoderStateSpec],
    default_left_context: usize,
    outs: &[StateBuf],
) -> Result<(), StreamError> {
    for (si, (spec, buf)) in specs.iter().zip(outs).enumerate() {
        match buf {
            StateBuf::F32(v) => {
                if !v.iter().all(|x| x.is_finite()) {
                    return Err(GpError::Output(format!(
                        "non-finite encoder cache output in {}_out",
                        spec.name
                    ))
                    .into());
                }
            }
            StateBuf::I64(v) => {
                // Match `kv_len` against `k`'s Lc axis (slot 0, dim 3) and `kvm_len` against
                // `km`'s Lc axis (slot 4, dim 3), falling back to `default_left_context`.
                let kv_slot = if si >= 4 { 4 } else { 0 };
                let max_lc = specs
                    .get(kv_slot)
                    .and_then(|ks| ks.per_stream_dims.get(3).copied())
                    .unwrap_or(default_left_context) as i64;
                for &len_val in v {
                    if len_val < 0 || len_val > max_lc {
                        return Err(GpError::Output(format!(
                            "invalid integer cache length {len_val} in {}_out (valid 0..={max_lc})",
                            spec.name
                        ))
                        .into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Low-level per-utterance stream used by [`OrtTransducerBackend::decode_streams`] and SP-A banks.
pub struct OrtR1RawStream {
    pub(crate) engine: Arc<OrtTransducerBackend>,
    pub(crate) mel: TrainerLogMel,
    pub(crate) mel_scratch: Vec<[f32; MEL_BINS]>,
    pub(crate) ring: FeatureRing,
    pub(crate) t_next: usize,
    pub(crate) finished: bool,
    pub(crate) fail_after_chunks: Option<usize>,
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

    /// Complete retained per-stream allocation in bytes: encoder caches (`k`, `v`, `kv_len`,
    /// `conv`, and MR `km`, `vm`, `kvm_len`, `convm`), feature ring, mel scratch, and RNN-T tokens.
    pub fn retained_bytes(&self) -> usize {
        let state_bytes: usize = self.states.iter().map(StateBuf::retained_bytes).sum();
        state_bytes
            + self.ring.retained_bytes()
            + self.mel_scratch.capacity() * MEL_BINS * 4
            + self.rnnt.tokens.capacity() * 8
            + std::mem::size_of::<Self>()
    }
}

impl OrtTransducerBackend {
    pub fn create_stream_with_clock(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
    ) -> Box<dyn AsrStream> {
        self.create_stream_internal(cfg, clock, None)
    }

    /// Constructor injecting a deterministic late decode failure after `ok_chunks` successful
    /// chunks (verifying atomic state rollback and terminal failure handling).
    pub fn create_stream_with_late_failure(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
        ok_chunks: usize,
    ) -> Box<dyn AsrStream> {
        self.create_stream_internal(cfg, clock, Some(ok_chunks))
    }

    fn create_stream_internal(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
        fail_after_chunks: Option<usize>,
    ) -> Box<dyn AsrStream> {
        let model_id = format!(
            "ultra-r1-{}-{}-{}",
            self.bundle.family.as_str(),
            self.bundle.precision.as_str(),
            &self.bundle.manifest_sha256[..12]
        );
        let eff_model_id = if cfg.model_config_id == "default-model-v1" {
            model_id.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        let mut raw = self.create_raw_stream();
        raw.fail_after_chunks = fail_after_chunks;
        Box::new(OrtR1AsrStream {
            raw,
            ws: BatchWorkspace::new(),
            clock,
            emitter: Emitter::new(&cfg.stream_id, eff_model_id),
            cfg: cfg.clone(),
            accepted_samples: 0,
            processed_samples: 0,
            wakeups: 0,
            events_emitted: 0,
            last_hyp: String::new(),
            finished: false,
            cancelled: false,
            failed: None,
            events: VecDeque::new(),
        })
    }
}

impl Backend for Arc<OrtTransducerBackend> {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: format!(
                "ort-r1-{}-{}-{}",
                self.bundle.family.as_str(),
                self.bundle.precision.as_str(),
                &self.bundle.manifest_sha256[..12]
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
        Ok(OrtTransducerBackend::create_stream_with_clock(
            self,
            cfg,
            StreamClock::real(),
        ))
    }

    fn create_stream_with_clock(
        &self,
        cfg: &StreamConfig,
        clock: StreamClock,
    ) -> Result<Box<dyn AsrStream>, StreamError> {
        Ok(OrtTransducerBackend::create_stream_with_clock(
            self, cfg, clock,
        ))
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
    pub(crate) failed: Option<String>,
    pub(crate) events: VecDeque<TranscriptEvent>,
}

impl OrtR1AsrStream {
    fn decode_ready(&mut self) -> Result<(), StreamError> {
        let mut ran = false;
        while self.raw.is_ready() {
            let eng = Arc::clone(&self.raw.engine);
            let (_, w) = match eng.decode_streams(&mut [&mut self.raw], &mut self.ws) {
                Ok(ok) => ok,
                Err(e) => {
                    self.failed = Some(e.to_string());
                    return Err(e);
                }
            };
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
        if self.finished || self.cancelled || self.failed.is_some() {
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
        if let Some(msg) = &self.failed {
            return Ok(PollResult::Ended(TerminalOutcome::Failed(msg.clone())));
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
        if self.cancelled || self.failed.is_some() {
            return Err(ultra_core::CoreError::Lifecycle(
                "finish_input on a cancelled or failed stream",
            )
            .into());
        }
        if !self.finished {
            self.raw.input_finished();
            while self.raw.is_ready() {
                let eng = Arc::clone(&self.raw.engine);
                let (_, w) = match eng.decode_streams(&mut [&mut self.raw], &mut self.ws) {
                    Ok(ok) => ok,
                    Err(e) => {
                        self.failed = Some(e.to_string());
                        return Err(e);
                    }
                };
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
        if !self.finished && self.failed.is_none() {
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
            retained_bytes: (self.raw.retained_bytes() + self.ws.retained_bytes()) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_state_outputs_rejects_nan_inf_and_invalid_kv_lengths() {
        let specs = vec![
            R1EncoderStateSpec {
                name: "k".into(),
                batch_axis: 1,
                per_stream_dims: vec![2, 1, 4, 128, 64],
                per_stream_elems: 2 * 4 * 128 * 64,
                is_int64: false,
            },
            R1EncoderStateSpec {
                name: "kv_len".into(),
                batch_axis: 0,
                per_stream_dims: vec![1],
                per_stream_elems: 1,
                is_int64: true,
            },
        ];
        let ok = vec![StateBuf::F32(vec![0.0, 1.5]), StateBuf::I64(vec![128])];
        assert!(validate_state_outputs(&specs, 128, &ok).is_ok());

        let bad_nan = vec![StateBuf::F32(vec![0.0, f32::NAN]), StateBuf::I64(vec![4])];
        assert!(validate_state_outputs(&specs, 128, &bad_nan).is_err());

        let bad_inf = vec![StateBuf::F32(vec![f32::INFINITY]), StateBuf::I64(vec![4])];
        assert!(validate_state_outputs(&specs, 128, &bad_inf).is_err());

        let bad_neg = vec![StateBuf::F32(vec![0.0]), StateBuf::I64(vec![-1])];
        assert!(validate_state_outputs(&specs, 128, &bad_neg).is_err());

        let bad_over = vec![StateBuf::F32(vec![0.0]), StateBuf::I64(vec![129])];
        assert!(validate_state_outputs(&specs, 128, &bad_over).is_err());
    }
}
