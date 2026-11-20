//! Offline contrast recogniser wrapper (`Moonshine v2`, `Moonshine v1`, `Whisper`) over `SherpaLib`.

use std::collections::VecDeque;
use std::ffi::{c_char, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};

use crate::api::{
    AsrStream, Backend, BackendCaps, PollResult, PushReport, PushStatus, StreamConfig, StreamError,
    StreamMetrics, TerminalOutcome,
};
use crate::sherpa_c_types::{
    SherpaOnnxOfflineRecognizer, SherpaOnnxOfflineRecognizerConfig, SherpaOnnxOfflineStream,
};
use crate::sherpa_ffi::{cstr_or_empty, path_to_cstring, verify_cpu_only_process, SherpaLib};

/// Supported offline contrast model families in `libsherpa-onnx-c-api.so` (P05).
#[derive(Debug, Clone)]
pub enum OfflineModelKind {
    MoonshineV2 {
        encoder: PathBuf,
        merged_decoder: PathBuf,
        tokens: PathBuf,
    },
    MoonshineV1 {
        preprocessor: PathBuf,
        encoder: PathBuf,
        uncached_decoder: PathBuf,
        cached_decoder: PathBuf,
        tokens: PathBuf,
    },
    Whisper {
        encoder: PathBuf,
        decoder: PathBuf,
        tokens: PathBuf,
    },
}

#[derive(Debug, Clone)]
pub struct SherpaOfflineConfig {
    pub candidate_id: String,
    pub kind: OfflineModelKind,
    pub num_threads: i32,
}

/// Shared immutable Offline Recogniser handle (`Send + Sync` via `usize` handle address).
pub struct SherpaOfflineBackend {
    lib: Arc<SherpaLib>,
    rec_addr: usize,
    cfg: SherpaOfflineConfig,
}

impl Drop for SherpaOfflineBackend {
    fn drop(&mut self) {
        if self.rec_addr != 0 {
            let ptr = self.rec_addr as *const SherpaOnnxOfflineRecognizer;
            // SAFETY: Created by `create_offline_rec` and destroyed once when `Arc` drops.
            unsafe { (self.lib.destroy_offline_rec)(ptr) };
        }
    }
}

impl SherpaOfflineBackend {
    pub fn new(lib: Arc<SherpaLib>, cfg: SherpaOfflineConfig) -> Result<Arc<Self>, StreamError> {
        let mut c_cfg = SherpaOnnxOfflineRecognizerConfig::default();
        c_cfg.feat_config.sample_rate = 16_000;
        c_cfg.feat_config.feature_dim = 80;
        let c_prov = CString::new("cpu").expect("static");
        let c_meth = CString::new("greedy_search").expect("static");
        let c_lang = CString::new("en").expect("static");
        let c_task = CString::new("transcribe").expect("static");
        c_cfg.model_config.provider = c_prov.as_ptr();
        c_cfg.model_config.num_threads = cfg.num_threads.max(1);
        c_cfg.decoding_method = c_meth.as_ptr();

        let mut owned_cstrings: Vec<CString> = Vec::new();
        let mut push_path = |p: &Path| -> Result<*const c_char, StreamError> {
            if !p.exists() {
                return Err(StreamError::BackendUnavailable(format!(
                    "missing model file `{}`",
                    p.display()
                )));
            }
            let cs = path_to_cstring(p)?;
            let ptr = cs.as_ptr();
            owned_cstrings.push(cs);
            Ok(ptr)
        };

        match &cfg.kind {
            OfflineModelKind::MoonshineV2 {
                encoder,
                merged_decoder,
                tokens,
            } => {
                c_cfg.model_config.moonshine.encoder = push_path(encoder)?;
                c_cfg.model_config.moonshine.merged_decoder = push_path(merged_decoder)?;
                c_cfg.model_config.tokens = push_path(tokens)?;
            }
            OfflineModelKind::MoonshineV1 {
                preprocessor,
                encoder,
                uncached_decoder,
                cached_decoder,
                tokens,
            } => {
                c_cfg.model_config.moonshine.preprocessor = push_path(preprocessor)?;
                c_cfg.model_config.moonshine.encoder = push_path(encoder)?;
                c_cfg.model_config.moonshine.uncached_decoder = push_path(uncached_decoder)?;
                c_cfg.model_config.moonshine.cached_decoder = push_path(cached_decoder)?;
                c_cfg.model_config.tokens = push_path(tokens)?;
            }
            OfflineModelKind::Whisper {
                encoder,
                decoder,
                tokens,
            } => {
                c_cfg.model_config.whisper.encoder = push_path(encoder)?;
                c_cfg.model_config.whisper.decoder = push_path(decoder)?;
                c_cfg.model_config.whisper.language = c_lang.as_ptr();
                c_cfg.model_config.whisper.task = c_task.as_ptr();
                c_cfg.model_config.whisper.tail_paddings = -1;
                c_cfg.model_config.tokens = push_path(tokens)?;
            }
        }

        // SAFETY: All referenced CStrings in `owned_cstrings` outlive this call.
        let rec = unsafe { (lib.create_offline_rec)(&c_cfg) };
        drop(owned_cstrings);
        if rec.is_null() {
            return Err(StreamError::BackendUnavailable(format!(
                "SherpaOnnxCreateOfflineRecognizer returned NULL for `{}`",
                cfg.candidate_id
            )));
        }
        Ok(Arc::new(Self {
            lib,
            rec_addr: rec as usize,
            cfg,
        }))
    }

    fn rec_ptr(&self) -> *const SherpaOnnxOfflineRecognizer {
        self.rec_addr as *const SherpaOnnxOfflineRecognizer
    }

    pub fn candidate_id(&self) -> &str {
        &self.cfg.candidate_id
    }

    pub fn transcribe_batch(&self, utterances: &[&[i16]]) -> Result<Vec<String>, StreamError> {
        if utterances.is_empty() {
            return Ok(Vec::new());
        }
        let mut streams: Vec<*const SherpaOnnxOfflineStream> = Vec::with_capacity(utterances.len());
        let mut scratch = Vec::new();
        for pcm in utterances {
            // SAFETY: `self.rec_ptr()` is valid.
            let st = unsafe { (self.lib.create_offline_stream)(self.rec_ptr()) };
            if st.is_null() {
                for prev in streams {
                    // SAFETY: Previously created stream pointers are valid.
                    unsafe { (self.lib.destroy_offline_stream)(prev) };
                }
                return Err(StreamError::Runtime(
                    "SherpaOnnxCreateOfflineStream returned NULL".into(),
                ));
            }
            scratch.clear();
            scratch.extend(pcm.iter().map(|&s| f32::from(s) / 32768.0));
            // SAFETY: Stream and sample buffer are valid.
            unsafe {
                (self.lib.accept_offline)(st, 16_000, scratch.as_ptr(), scratch.len() as i32);
            }
            streams.push(st);
        }

        if streams.len() == 1 {
            // SAFETY: Valid recognizer and stream.
            unsafe { (self.lib.decode_offline)(self.rec_ptr(), streams[0]) };
        } else {
            // SAFETY: Valid recognizer and array of streams.
            unsafe {
                (self.lib.decode_multi_offline)(
                    self.rec_ptr(),
                    streams.as_ptr(),
                    streams.len() as i32,
                );
            }
        }

        let mut out = Vec::with_capacity(streams.len());
        for st in streams {
            // SAFETY: Decode completed; result copied and destroyed before freeing stream.
            let text = unsafe {
                let res = (self.lib.get_offline_res)(st);
                let s = if res.is_null() {
                    String::new()
                } else {
                    let v = cstr_or_empty((*res).text);
                    (self.lib.destroy_offline_res)(res);
                    v
                };
                (self.lib.destroy_offline_stream)(st);
                s
            };
            out.push(text.trim().to_owned());
        }
        Ok(out)
    }
}

impl Backend for Arc<SherpaOfflineBackend> {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: self.cfg.candidate_id.clone(),
            supports_batch_step: true,
            supports_partial_push: false,
            supports_commit_policy: false,
            chunk_samples: 16_000 * 2,
            right_context_samples: 0,
            cpu_only_verified: verify_cpu_only_process(),
        }
    }

    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError> {
        let eff_model_id = if cfg.model_config_id == "default-model-v1" {
            self.cfg.candidate_id.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        Ok(Box::new(OfflineWindowStream {
            engine: Arc::clone(self),
            emitter: Emitter::new(&cfg.stream_id, eff_model_id),
            pcm_buf: Vec::new(),
            accepted_samples: 0,
            wakeups: 0,
            events_emitted: 0,
            finished: false,
            cancelled: false,
            events: VecDeque::new(),
        }))
    }
}

struct OfflineWindowStream {
    engine: Arc<SherpaOfflineBackend>,
    emitter: Emitter,
    pcm_buf: Vec<i16>,
    accepted_samples: u64,
    wakeups: u64,
    events_emitted: u64,
    finished: bool,
    cancelled: bool,
    events: VecDeque<TranscriptEvent>,
}

impl AsrStream for OfflineWindowStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        self.pcm_buf.extend_from_slice(pcm);
        self.accepted_samples += pcm.len() as u64;
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
            self.finished = true;
            let text = if self.pcm_buf.is_empty() {
                String::new()
            } else {
                self.wakeups += 1;
                let v = self.engine.transcribe_batch(&[&self.pcm_buf])?;
                v.into_iter()
                    .next()
                    .ok_or_else(|| StreamError::Runtime("empty offline batch output".into()))?
            };
            let ts = (self.accepted_samples * 1_000_000_000) / 16_000;
            let ev = self.emitter.finalize(
                ts,
                &text,
                self.accepted_samples,
                ProcessingEvidence {
                    accepted_through_sample: self.accepted_samples,
                    required_processed_through_sample: self.accepted_samples,
                    flushed: true,
                },
            )?;
            self.events_emitted += 1;
            self.events.push_back(ev);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.finished {
            self.cancelled = true;
            self.events.clear();
            self.pcm_buf.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        StreamMetrics {
            accepted_samples: self.accepted_samples,
            processed_samples: if self.finished {
                self.accepted_samples
            } else {
                0
            },
            wakeups: self.wakeups,
            events_emitted: self.events_emitted,
            cap_hits: 0,
            refiner_attempts: 0,
            refiner_accepts: 0,
            refiner_rejects: 0,
            refiner_charged_cpu_s: 0.0,
            retained_bytes: (self.pcm_buf.len() * 2) as u64,
        }
    }
}
