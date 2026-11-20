//! Online Streaming Zipformer transducer wrapper over `SherpaLib` (`single` and `batch`).

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{AsrStream, Backend, BackendCaps, StreamClock, StreamConfig, StreamError};
use crate::sherpa_c_types::{
    SherpaOnnxOnlineRecognizer, SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineStream,
};
use crate::sherpa_ffi::{cstr_or_empty, path_to_cstring, verify_cpu_only_process, SherpaLib};

mod checked_result;
mod stream;
use checked_result::copy_result_and_free;
use stream::{RealZipformerDriver, ZipformerAsrStream};

/// Configuration for the Online Streaming Zipformer transducer (`int8` or `fp32`).
#[derive(Debug, Clone)]
pub struct SherpaZipformerConfig {
    pub model_dir: PathBuf,
    pub precision: String,
    pub num_threads: i32,
    pub tail_pad_s: f64,
}

impl Default for SherpaZipformerConfig {
    fn default() -> Self {
        let data = match std::env::var("JSR_DATA") {
            Ok(v) => v,
            Err(_) => "/mnt/xbabe3-external/jsr".into(),
        };
        Self {
            model_dir: PathBuf::from(data)
                .join("models")
                .join("zipformer-en-2023-06-26"),
            precision: "int8".into(),
            num_threads: 1,
            tail_pad_s: 0.8,
        }
    }
}

/// Shared immutable Online Streaming Zipformer engine handle.
pub struct SherpaZipformerBackend {
    lib: Arc<SherpaLib>,
    rec_addr: usize,
    cfg: SherpaZipformerConfig,
}

impl Drop for SherpaZipformerBackend {
    fn drop(&mut self) {
        if self.rec_addr != 0 {
            let ptr = self.rec_addr as *const SherpaOnnxOnlineRecognizer;
            // SAFETY: Created by `create_online_rec` and destroyed once when `Arc` drops.
            unsafe { (self.lib.destroy_online_rec)(ptr) };
        }
    }
}

impl SherpaZipformerBackend {
    pub fn new(lib: Arc<SherpaLib>, cfg: SherpaZipformerConfig) -> Result<Arc<Self>, StreamError> {
        let suffix = match cfg.precision.as_str() {
            "int8" => ".int8.onnx",
            "fp32" => ".onnx",
            other => {
                return Err(StreamError::BackendUnavailable(format!(
                    "unsupported precision `{other}`"
                )));
            }
        };
        let stem = "epoch-99-avg-1-chunk-16-left-128";
        let enc = cfg.model_dir.join(format!("encoder-{stem}{suffix}"));
        let dec = cfg.model_dir.join(format!("decoder-{stem}{suffix}"));
        let joi = cfg.model_dir.join(format!("joiner-{stem}{suffix}"));
        let tok = cfg.model_dir.join("tokens.txt");
        for p in [&enc, &dec, &joi, &tok] {
            if !p.exists() {
                return Err(StreamError::BackendUnavailable(format!(
                    "missing model file `{}`",
                    p.display()
                )));
            }
        }
        let c_enc = path_to_cstring(&enc)?;
        let c_dec = path_to_cstring(&dec)?;
        let c_joi = path_to_cstring(&joi)?;
        let c_tok = path_to_cstring(&tok)?;
        let c_prov = CString::new("cpu").expect("static");
        let c_meth = CString::new("greedy_search").expect("static");

        let mut c_cfg = SherpaOnnxOnlineRecognizerConfig::default();
        c_cfg.feat_config.sample_rate = 16_000;
        c_cfg.feat_config.feature_dim = 80;
        c_cfg.model_config.transducer.encoder = c_enc.as_ptr();
        c_cfg.model_config.transducer.decoder = c_dec.as_ptr();
        c_cfg.model_config.transducer.joiner = c_joi.as_ptr();
        c_cfg.model_config.tokens = c_tok.as_ptr();
        c_cfg.model_config.num_threads = cfg.num_threads.max(1);
        c_cfg.model_config.provider = c_prov.as_ptr();
        c_cfg.decoding_method = c_meth.as_ptr();

        // SAFETY: All CStrings live through `create_online_rec`, which copies strings internally.
        let rec = unsafe { (lib.create_online_rec)(&c_cfg) };
        if rec.is_null() {
            return Err(StreamError::BackendUnavailable(
                "SherpaOnnxCreateOnlineRecognizer returned NULL".into(),
            ));
        }
        Ok(Arc::new(Self {
            lib,
            rec_addr: rec as usize,
            cfg,
        }))
    }

    fn rec_ptr(&self) -> *const SherpaOnnxOnlineRecognizer {
        self.rec_addr as *const SherpaOnnxOnlineRecognizer
    }

    pub fn config(&self) -> &SherpaZipformerConfig {
        &self.cfg
    }

    pub fn create_raw_stream(self: &Arc<Self>) -> Result<RawOnlineStream, StreamError> {
        // SAFETY: `self.rec_ptr()` is valid for the lifetime of `Arc<Self>`.
        let ptr = unsafe { (self.lib.create_online_stream)(self.rec_ptr()) };
        if ptr.is_null() {
            return Err(StreamError::Runtime(
                "SherpaOnnxCreateOnlineStream returned NULL".into(),
            ));
        }
        Ok(RawOnlineStream {
            owner: Arc::clone(self),
            stream_addr: ptr as usize,
        })
    }

    pub fn decode_one(&self, s: &RawOnlineStream) {
        // SAFETY: Both recognizer and stream handles are valid and live.
        unsafe { (self.lib.decode_online)(self.rec_ptr(), s.as_ptr()) };
    }

    pub fn decode_batch_addrs(&self, addrs: &[usize]) {
        if addrs.is_empty() {
            return;
        }
        let ptrs: Vec<*const SherpaOnnxOnlineStream> = addrs
            .iter()
            .map(|&a| a as *const SherpaOnnxOnlineStream)
            .collect();
        // SAFETY: Every address in `addrs` belongs to a live `RawOnlineStream` on `self`.
        unsafe {
            (self.lib.decode_multi_online)(self.rec_ptr(), ptrs.as_ptr(), ptrs.len() as i32);
        }
    }

    pub fn create_stream_with_clock(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
    ) -> Result<Box<dyn AsrStream>, StreamError> {
        let raw = self.create_raw_stream()?;
        let model_id = format!("sherpa-onnx-zipformer-{}", self.cfg.precision);
        let eff_model_id = if cfg.model_config_id == "default-model-v1" {
            model_id.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        let tail_pad_samples = (16_000.0 * self.cfg.tail_pad_s) as usize;
        Ok(Box::new(ZipformerAsrStream::new(
            RealZipformerDriver {
                engine: Arc::clone(self),
                raw,
                scratch: Vec::with_capacity(1024),
            },
            clock,
            tail_pad_samples,
            eff_model_id,
            cfg.clone(),
        )))
    }
}

/// RAII wrapper around `SherpaOnnxOnlineStream`.
pub struct RawOnlineStream {
    owner: Arc<SherpaZipformerBackend>,
    stream_addr: usize,
}

impl Drop for RawOnlineStream {
    fn drop(&mut self) {
        if self.stream_addr != 0 {
            let ptr = self.stream_addr as *const SherpaOnnxOnlineStream;
            // SAFETY: Created by `create_online_stream`, destroyed once.
            unsafe { (self.owner.lib.destroy_online_stream)(ptr) };
        }
    }
}

impl RawOnlineStream {
    pub fn addr(&self) -> usize {
        self.stream_addr
    }

    pub(crate) fn as_ptr(&self) -> *const SherpaOnnxOnlineStream {
        self.stream_addr as *const SherpaOnnxOnlineStream
    }

    pub fn accept_f32(&mut self, samples_unit: &[f32]) {
        if samples_unit.is_empty() {
            return;
        }
        // SAFETY: Valid stream handle and contiguous `f32` buffer.
        unsafe {
            (self.owner.lib.accept_online)(
                self.as_ptr(),
                16_000,
                samples_unit.as_ptr(),
                samples_unit.len() as i32,
            );
        }
    }

    pub fn accept_i16(&mut self, pcm: &[i16], scratch: &mut Vec<f32>) {
        scratch.clear();
        scratch.extend(pcm.iter().map(|&s| f32::from(s) / 32768.0));
        self.accept_f32(scratch);
    }

    pub fn input_finished(&mut self) {
        // SAFETY: Valid stream handle.
        unsafe { (self.owner.lib.input_finished)(self.as_ptr()) };
    }

    pub fn is_ready(&self) -> bool {
        // SAFETY: Valid recognizer and stream handles.
        unsafe { (self.owner.lib.is_ready)(self.owner.rec_ptr(), self.as_ptr()) != 0 }
    }

    pub fn text(&self) -> String {
        // SAFETY: Snapshot returned by `get_online_res` is copied and immediately freed.
        unsafe {
            let res = (self.owner.lib.get_online_res)(self.owner.rec_ptr(), self.as_ptr());
            if res.is_null() {
                return String::new();
            }
            let s = cstr_or_empty((*res).text);
            (self.owner.lib.destroy_online_res)(res);
            s
        }
    }

    /// Copy a checked native result. Missing output and invalid UTF-8 are failures,
    /// distinct from a successful empty hypothesis (e.g. silence).
    pub fn try_text(&self) -> Result<String, StreamError> {
        // SAFETY: stream and library are live; the helper copies and frees once.
        unsafe {
            copy_result_and_free(
                (self.owner.lib.get_online_res)(self.owner.rec_ptr(), self.as_ptr()),
                self.owner.lib.destroy_online_res,
            )
        }
    }
}

impl Backend for Arc<SherpaZipformerBackend> {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: format!("sherpa-onnx-zipformer-{}", self.cfg.precision),
            supports_batch_step: true,
            supports_partial_push: false,
            supports_commit_policy: true,
            chunk_samples: 16_000 * 320 / 1000,
            right_context_samples: 0,
            cpu_only_verified: verify_cpu_only_process(),
        }
    }

    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError> {
        self.create_stream_with_clock(cfg, StreamClock::real())
    }
}
