//! Pure-Rust ONNX-free [`Backend`] and causal [`AsrStream`] integration (`E4` / `R-ACCEL-3`)
//! over [`ultra_engine::Engine`] (`ultra.r1-weights/v1` SafeTensors).

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};
use ultra_core::hash::sha256_reader;
use ultra_engine::{Engine, EngineStream, Isa, Model, Precision, Workspace};

use crate::api::{
    AsrStream, Backend, BackendCaps, OutputPolicy, PollResult, PushReport, PushStatus, StreamClock,
    StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};
use crate::ort_r1::R1_SAMPLES_PER_ENC_FRAME;
use crate::sherpa_ffi::verify_cpu_only_process;

/// Immutable pure-Rust `ultra-engine` backend (`Send + Sync`, shared read-only across CCX workers).
pub struct NativeEngineBackend {
    pub engine: Engine,
    pub arch_sha256: String,
    pub int8_source: &'static str,
}

impl NativeEngineBackend {
    /// Load an `ultra.r1-weights/v1` directory (`arch.json` + `weights.safetensors`, plus required
    /// prepared `weights.int8.safetensors` when `precision == Precision::Int8`). Enforces
    /// `threads == 1` per worker and refuses runtime weight quantisation in timed service loads.
    pub fn load(
        weights_dir: &Path,
        precision: Precision,
        threads: i32,
        verify_hashes: bool,
    ) -> Result<Arc<Self>, StreamError> {
        Self::load_inner(weights_dir, precision, threads, verify_hashes, false, 1, 1)
    }

    /// Strict loader with configurable `max_chunks` per engine step (`FREEZE.md §3`: `2` on a
    /// 320 ms density grid). Refuses runtime weight quantisation in timed service loads.
    pub fn load_with_max_chunks(
        weights_dir: &Path,
        precision: Precision,
        threads: i32,
        verify_hashes: bool,
        max_chunks: usize,
    ) -> Result<Arc<Self>, StreamError> {
        Self::load_with_options(
            weights_dir,
            precision,
            threads,
            verify_hashes,
            max_chunks,
            1,
        )
    }

    /// Strict loader with configurable `max_chunks` and bounded RNN-T `beam` width (`1` = greedy,
    /// `2..=16` = beam search). Refuses runtime weight quantisation in timed service loads.
    pub fn load_with_options(
        weights_dir: &Path,
        precision: Precision,
        threads: i32,
        verify_hashes: bool,
        max_chunks: usize,
        beam: usize,
    ) -> Result<Arc<Self>, StreamError> {
        Self::load_inner(
            weights_dir,
            precision,
            threads,
            verify_hashes,
            false,
            max_chunks.clamp(1, 16),
            beam.clamp(1, 16),
        )
    }

    /// Diagnostic-only loader allowing load-time `Model::quantize_int8()` when no prepared
    /// `weights.int8.safetensors` artifact exists yet.
    pub fn load_diagnostic_allow_unprepared_int8(
        weights_dir: &Path,
        precision: Precision,
        threads: i32,
        verify_hashes: bool,
    ) -> Result<Arc<Self>, StreamError> {
        Self::load_inner(weights_dir, precision, threads, verify_hashes, true, 1, 1)
    }

    fn load_inner(
        weights_dir: &Path,
        precision: Precision,
        threads: i32,
        verify_hashes: bool,
        allow_unprepared_int8: bool,
        max_chunks: usize,
        beam: usize,
    ) -> Result<Arc<Self>, StreamError> {
        if threads != 1 {
            return Err(StreamError::Protocol(format!(
                "ultra-engine worker requires threads == 1 (got {threads}); scale via CCX workers"
            )));
        }
        let mut arch_file = File::open(weights_dir.join("arch.json"))
            .map_err(|e| StreamError::BackendUnavailable(format!("arch.json: {e}")))?;
        let arch_sha256 = sha256_reader(&mut arch_file)
            .map_err(|e| StreamError::BackendUnavailable(format!("hash arch.json: {e}")))?;
        let mut model = Model::load(weights_dir, verify_hashes)?;
        let int8_source = match precision {
            Precision::F32 => "n/a",
            Precision::Int8 if model.int8_prepared.is_some() => "prepared",
            Precision::Int8 if allow_unprepared_int8 => {
                model.quantize_int8();
                "diagnostic_quantized_at_load"
            }
            Precision::Int8 => {
                return Err(StreamError::Protocol(format!(
                    "INT8 NativeEngineBackend requires a prepared weights.int8.safetensors directory (got `{}`; run ultra-engine-prepare)",
                    weights_dir.display()
                )));
            }
        };
        let engine = Engine::new(Arc::new(model), Isa::detect(), precision)?
            .with_max_chunks(max_chunks)
            .with_beam(beam);
        Ok(Arc::new(Self {
            engine,
            arch_sha256,
            int8_source,
        }))
    }

    pub fn bundle_id(&self) -> String {
        let family = if self.engine.model.arch.mid > 0 {
            "multirate"
        } else {
            "dense"
        };
        let base = format!(
            "ultra-engine-{family}-{}-{}",
            self.engine.precision.as_str(),
            &self.arch_sha256[..12]
        );
        if self.engine.beam > 1 {
            format!("{base}-beam{}", self.engine.beam)
        } else {
            base
        }
    }

    pub fn create_raw_stream(&self) -> EngineStream {
        self.engine.create_stream()
    }

    pub fn create_stream_with_clock(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
    ) -> Box<dyn AsrStream> {
        let model_id = self.bundle_id();
        let eff_model_id = if cfg.model_config_id == "default-model-v1" {
            model_id.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        let raw = self.engine.create_stream();
        Box::new(NativeEngineAsrStream {
            backend: Arc::clone(self),
            raw,
            ws: Workspace::default(),
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

impl Backend for Arc<NativeEngineBackend> {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: self.bundle_id(),
            supports_batch_step: true,
            supports_partial_push: false,
            supports_commit_policy: true,
            chunk_samples: self.engine.model.arch.chunk * 640,
            right_context_samples: 0,
            cpu_only_verified: verify_cpu_only_process(),
        }
    }

    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError> {
        Ok(NativeEngineBackend::create_stream_with_clock(
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
        Ok(NativeEngineBackend::create_stream_with_clock(
            self, cfg, clock,
        ))
    }
}

struct NativeEngineAsrStream {
    backend: Arc<NativeEngineBackend>,
    raw: EngineStream,
    ws: Workspace,
    clock: StreamClock,
    emitter: Emitter,
    cfg: StreamConfig,
    accepted_samples: u64,
    processed_samples: u64,
    wakeups: u64,
    events_emitted: u64,
    last_hyp: String,
    finished: bool,
    cancelled: bool,
    failed: Option<String>,
    events: VecDeque<TranscriptEvent>,
}

impl NativeEngineAsrStream {
    /// Lower-bound per-stream retained state bytes: covers both 25 Hz and S_MR 12.5 Hz encoder
    /// caches (`EncState::retained_bytes`), decoder state, hypothesis string, and event queue.
    /// Excludes `EngineStream`'s private frontend/feature-ring buffers and shared `Workspace`;
    /// aggregate process-tree RSS/PSS from `ultra-measure` remains the authoritative memory gate.
    fn retained_stream_bytes(&self) -> u64 {
        let enc_bytes = self.raw.enc.retained_bytes();
        let dec_bytes = self.raw.dec.tokens.capacity() * std::mem::size_of::<u32>()
            + self.raw.dec.dec.capacity() * std::mem::size_of::<f32>();
        let stream_overhead = self.last_hyp.capacity()
            + self.events.capacity() * std::mem::size_of::<TranscriptEvent>()
            + std::mem::size_of::<Self>();
        (enc_bytes + dec_bytes + stream_overhead) as u64
    }

    fn decode_ready(&mut self) -> Result<(), StreamError> {
        let mut ran = false;
        while self.backend.engine.is_ready(&self.raw) {
            let n_adv = match self
                .backend
                .engine
                .decode(&mut [&mut self.raw], &mut self.ws)
            {
                Ok(k) => k,
                Err(e) => {
                    self.failed = Some(e.to_string());
                    return Err(e.into());
                }
            };
            if n_adv > 0 {
                self.wakeups += 1;
                ran = true;
            } else {
                break;
            }
        }
        if !ran {
            return Ok(());
        }
        self.processed_samples =
            ((self.raw.enc.t as u64) * R1_SAMPLES_PER_ENC_FRAME).min(self.accepted_samples);
        let hyp = self.backend.engine.text(&self.raw);
        let changed = hyp != self.last_hyp;
        let (suffix, commit) = match self.cfg.policy {
            OutputPolicy::EditableDictation => (hyp.as_str(), None),
            OutputPolicy::Commit => {
                let suffix = self.hypothesis_suffix(&hyp)?;
                let shared = self.backend.engine.shared_beam_text(&self.raw);
                let committable = shared.as_deref().unwrap_or(&hyp);
                let stable = self.hypothesis_suffix(committable)?;
                (suffix, stable.rfind(' ').filter(|&sp| sp > 0))
            }
        };
        if changed || commit.is_some() {
            let ts = self.clock.now_ns();
            let owned_to = self.processed_samples;
            let ev = match commit {
                Some(bytes) => self.emitter.commit(ts, suffix, bytes, owned_to)?,
                None => self.emitter.preview(ts, suffix, owned_to)?,
            };
            self.last_hyp = hyp;
            self.events_emitted += 1;
            self.events.push_back(ev);
        }
        Ok(())
    }

    fn hypothesis_suffix<'a>(&mut self, hyp: &'a str) -> Result<&'a str, StreamError> {
        hyp.strip_prefix(self.emitter.immutable()).ok_or_else(|| {
            let message = "native hypothesis revised the committed prefix";
            self.failed = Some(message.into());
            StreamError::Runtime(message.into())
        })
    }
}

impl AsrStream for NativeEngineAsrStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled || self.failed.is_some() {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        if let Err(e) = self.raw.accept_i16(pcm) {
            self.failed = Some(e.to_string());
            return Err(e.into());
        }
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
            while self.backend.engine.is_ready(&self.raw) {
                if let Err(e) = self
                    .backend
                    .engine
                    .decode(&mut [&mut self.raw], &mut self.ws)
                {
                    self.failed = Some(e.to_string());
                    return Err(e.into());
                }
                self.wakeups += 1;
            }
            let hyp = self.backend.engine.text(&self.raw);
            let suffix = self.hypothesis_suffix(&hyp)?;
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
            cap_hits: self.raw.forced_advances(),
            refiner_attempts: 0,
            refiner_accepts: 0,
            refiner_rejects: 0,
            refiner_charged_cpu_s: 0.0,
            retained_bytes: self.retained_stream_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ultra_engine::model::{Arch, Linear};
    use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
    use ultra_gp::TokenTable;

    use super::*;

    fn tiny_commit_stream() -> NativeEngineAsrStream {
        let (vocab, pred_emb, joiner) = (4, 4, 8);
        let arch = Arch {
            n_mels: 80,
            d: 4,
            n_blocks: 0,
            lower: 0,
            mid: 0,
            fin: 0,
            heads: 1,
            dh: 4,
            ffn: 16,
            kernel: 3,
            stem_ch: 4,
            f1: 39,
            f2: 19,
            chunk: 4,
            left: 16,
            post_block_norm: false,
            vocab,
            pred_emb,
            joiner,
            blank: 0,
            max_symbols: 1,
            ctx_init: vec![0, 0],
            log_floor: 1e-6,
            checkpoint_sha256: String::new(),
            weights_sha256: String::new(),
            qat_stem_int8: false,
        };
        let tokens = Arc::new(
            TokenTable::from_str_checked(
                "<blk> 0\n\u{2581}hello 1\n\u{2581}world 2\n\u{2581}halo 3\n",
                vocab,
            )
            .expect("tokens"),
        );
        let cmvn = GlobalCmvn::new(&[0.0; 80], &[1.0; 80]).expect("cmvn");
        let mel_cfg = Arc::new(
            TrainerMelConfig::from_fbank_le_bytes(&vec![0u8; 257 * 80 * 4], 1e-6, cmvn)
                .expect("mel_cfg"),
        );
        let dummy = Linear::from_fp32(&[0.0], 1, 1, None);
        let model = Arc::new(Model {
            arch,
            root: PathBuf::new(),
            c1_w: Vec::new(),
            c1_b: Vec::new(),
            c2: dummy.clone(),
            stem_out: dummy.clone(),
            blocks: Vec::new(),
            final_norm: None,
            pair: None,
            restore: None,
            enc_proj: dummy,
            emb: vec![0.0f32; vocab * pred_emb],
            pred_proj: Linear::from_fp32(
                &vec![0.0f32; joiner * 2 * pred_emb],
                joiner,
                2 * pred_emb,
                Some(&vec![0.0f32; joiner]),
            ),
            pred_cache: None,
            joiner_out: Linear::from_fp32(
                &vec![0.0f32; vocab * joiner],
                vocab,
                joiner,
                Some(&vec![0.0f32; vocab]),
            ),
            blank_cert_f32: None,
            blank_cert_i8: None,
            rope_inv: Vec::new(),
            mel_cfg,
            tokens,
            int8_prepared: None,
        });
        let engine = Engine::new(model, Isa::Scalar, Precision::F32).expect("tiny engine");
        let backend = Arc::new(NativeEngineBackend {
            engine,
            arch_sha256: String::new(),
            int8_source: "n/a",
        });
        let cfg = StreamConfig {
            stream_id: "prefix-divergence".into(),
            policy: OutputPolicy::Commit,
            ..StreamConfig::default()
        };
        let raw = backend.engine.create_stream();
        NativeEngineAsrStream {
            backend,
            raw,
            ws: Workspace::default(),
            clock: StreamClock::custom(|| 1),
            emitter: Emitter::new(&cfg.stream_id, "tiny"),
            cfg,
            accepted_samples: 0,
            processed_samples: 0,
            wakeups: 0,
            events_emitted: 0,
            last_hyp: String::new(),
            finished: false,
            cancelled: false,
            failed: None,
            events: VecDeque::new(),
        }
    }

    #[test]
    fn commit_policy_fails_loudly_on_prefix_divergence_and_finish() {
        let mut s = tiny_commit_stream();
        let _ev = s.emitter.commit(1, "hello world", 6, 320).expect("commit");
        assert_eq!(s.emitter.immutable(), "hello ");
        assert_eq!(s.hypothesis_suffix("hello world").expect("suffix"), "world");

        let err = s
            .hypothesis_suffix("halo world")
            .expect_err("committed prefix revision must error");
        assert!(matches!(err, StreamError::Runtime(_)));
        assert!(s.failed.is_some());
        assert!(matches!(
            s.poll_event().expect("poll terminal"),
            PollResult::Ended(TerminalOutcome::Failed(_))
        ));

        // Finalize path must also return a terminal error rather than an empty final suffix.
        let mut s2 = tiny_commit_stream();
        let _ev = s2.emitter.commit(1, "hello world", 6, 320).expect("commit");
        s2.raw.dec.tokens = vec![3, 2]; // decodes to "halo world"
        let err = s2
            .finish_input()
            .expect_err("finish_input must error on committed prefix divergence");
        assert!(matches!(err, StreamError::Runtime(_)));
        assert!(s2.failed.is_some());
        assert!(matches!(
            s2.poll_event().expect("poll terminal"),
            PollResult::Ended(TerminalOutcome::Failed(_))
        ));
    }
}
