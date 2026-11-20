//! Online `#[repr(C)]` struct definitions matching `sherpa-onnx/c-api/c-api.h` byte for byte.

use std::ffi::c_char;
use std::ptr::null;

pub use crate::sherpa_c_offline::*;

#[repr(C)]
pub struct SherpaOnnxOnlineRecognizer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct SherpaOnnxOnlineStream {
    _private: [u8; 0],
}
#[repr(C)]
pub struct SherpaOnnxOfflineRecognizer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct SherpaOnnxOfflineStream {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SherpaOnnxFeatureConfig {
    pub sample_rate: i32,
    pub feature_dim: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOnlineTransducerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub joiner: *const c_char,
}
impl Default for SherpaOnnxOnlineTransducerModelConfig {
    fn default() -> Self {
        Self {
            encoder: null(),
            decoder: null(),
            joiner: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOnlineParaformerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
}
impl Default for SherpaOnnxOnlineParaformerModelConfig {
    fn default() -> Self {
        Self {
            encoder: null(),
            decoder: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxSingleModelPath {
    pub model: *const c_char,
}
impl Default for SherpaOnnxSingleModelPath {
    fn default() -> Self {
        Self { model: null() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOnlineModelConfig {
    pub transducer: SherpaOnnxOnlineTransducerModelConfig,
    pub paraformer: SherpaOnnxOnlineParaformerModelConfig,
    pub zipformer2_ctc: SherpaOnnxSingleModelPath,
    pub tokens: *const c_char,
    pub num_threads: i32,
    pub provider: *const c_char,
    pub debug: i32,
    pub model_type: *const c_char,
    pub modeling_unit: *const c_char,
    pub bpe_vocab: *const c_char,
    pub tokens_buf: *const c_char,
    pub tokens_buf_size: i32,
    pub nemo_ctc: SherpaOnnxSingleModelPath,
    pub t_one_ctc: SherpaOnnxSingleModelPath,
}
impl Default for SherpaOnnxOnlineModelConfig {
    fn default() -> Self {
        Self {
            transducer: Default::default(),
            paraformer: Default::default(),
            zipformer2_ctc: Default::default(),
            tokens: null(),
            num_threads: 0,
            provider: null(),
            debug: 0,
            model_type: null(),
            modeling_unit: null(),
            bpe_vocab: null(),
            tokens_buf: null(),
            tokens_buf_size: 0,
            nemo_ctc: Default::default(),
            t_one_ctc: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxCtcFstDecoderConfig {
    pub graph: *const c_char,
    pub max_active: i32,
}
impl Default for SherpaOnnxCtcFstDecoderConfig {
    fn default() -> Self {
        Self {
            graph: null(),
            max_active: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxHomophoneReplacerConfig {
    pub dict_dir: *const c_char,
    pub lexicon: *const c_char,
    pub rule_fsts: *const c_char,
}
impl Default for SherpaOnnxHomophoneReplacerConfig {
    fn default() -> Self {
        Self {
            dict_dir: null(),
            lexicon: null(),
            rule_fsts: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOnlineRecognizerConfig {
    pub feat_config: SherpaOnnxFeatureConfig,
    pub model_config: SherpaOnnxOnlineModelConfig,
    pub decoding_method: *const c_char,
    pub max_active_paths: i32,
    pub enable_endpoint: i32,
    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,
    pub hotwords_file: *const c_char,
    pub hotwords_score: f32,
    pub ctc_fst_decoder_config: SherpaOnnxCtcFstDecoderConfig,
    pub rule_fsts: *const c_char,
    pub rule_fars: *const c_char,
    pub blank_penalty: f32,
    pub hotwords_buf: *const c_char,
    pub hotwords_buf_size: i32,
    pub hr: SherpaOnnxHomophoneReplacerConfig,
}
impl Default for SherpaOnnxOnlineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: Default::default(),
            model_config: Default::default(),
            decoding_method: null(),
            max_active_paths: 0,
            enable_endpoint: 0,
            rule1_min_trailing_silence: 0.0,
            rule2_min_trailing_silence: 0.0,
            rule3_min_utterance_length: 0.0,
            hotwords_file: null(),
            hotwords_score: 0.0,
            ctc_fst_decoder_config: Default::default(),
            rule_fsts: null(),
            rule_fars: null(),
            blank_penalty: 0.0,
            hotwords_buf: null(),
            hotwords_buf_size: 0,
            hr: Default::default(),
        }
    }
}

#[repr(C)]
pub struct SherpaOnnxOnlineRecognizerResult {
    pub text: *const c_char,
    pub tokens: *const c_char,
    pub tokens_arr: *const *const c_char,
    pub timestamps: *mut f32,
    pub count: i32,
    pub json: *const c_char,
}
