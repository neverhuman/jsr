//! Offline `#[repr(C)]` struct definitions matching `sherpa-onnx/c-api/c-api.h` byte for byte.

use std::ffi::c_char;
use std::ptr::null;

use crate::sherpa_c_types::{
    SherpaOnnxCtcFstDecoderConfig, SherpaOnnxFeatureConfig, SherpaOnnxHomophoneReplacerConfig,
    SherpaOnnxOnlineParaformerModelConfig, SherpaOnnxOnlineTransducerModelConfig,
    SherpaOnnxSingleModelPath,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineWhisperModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub language: *const c_char,
    pub task: *const c_char,
    pub tail_paddings: i32,
    pub enable_token_timestamps: i32,
    pub enable_segment_timestamps: i32,
}
impl Default for SherpaOnnxOfflineWhisperModelConfig {
    fn default() -> Self {
        Self {
            encoder: null(),
            decoder: null(),
            language: null(),
            task: null(),
            tail_paddings: 0,
            enable_token_timestamps: 0,
            enable_segment_timestamps: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineSenseVoiceModelConfig {
    pub model: *const c_char,
    pub language: *const c_char,
    pub use_itn: i32,
}
impl Default for SherpaOnnxOfflineSenseVoiceModelConfig {
    fn default() -> Self {
        Self {
            model: null(),
            language: null(),
            use_itn: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineMoonshineModelConfig {
    pub preprocessor: *const c_char,
    pub encoder: *const c_char,
    pub uncached_decoder: *const c_char,
    pub cached_decoder: *const c_char,
    pub merged_decoder: *const c_char,
}
impl Default for SherpaOnnxOfflineMoonshineModelConfig {
    fn default() -> Self {
        Self {
            preprocessor: null(),
            encoder: null(),
            uncached_decoder: null(),
            cached_decoder: null(),
            merged_decoder: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineCanaryModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub src_lang: *const c_char,
    pub tgt_lang: *const c_char,
    pub use_pnc: i32,
}
impl Default for SherpaOnnxOfflineCanaryModelConfig {
    fn default() -> Self {
        Self {
            encoder: null(),
            decoder: null(),
            src_lang: null(),
            tgt_lang: null(),
            use_pnc: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineFunASRNanoModelConfig {
    pub encoder_adaptor: *const c_char,
    pub llm: *const c_char,
    pub embedding: *const c_char,
    pub tokenizer: *const c_char,
    pub system_prompt: *const c_char,
    pub user_prompt: *const c_char,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub language: *const c_char,
    pub itn: i32,
    pub hotwords: *const c_char,
}
impl Default for SherpaOnnxOfflineFunASRNanoModelConfig {
    fn default() -> Self {
        Self {
            encoder_adaptor: null(),
            llm: null(),
            embedding: null(),
            tokenizer: null(),
            system_prompt: null(),
            user_prompt: null(),
            max_new_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            seed: 0,
            language: null(),
            itn: 0,
            hotwords: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineQwen3ASRModelConfig {
    pub conv_frontend: *const c_char,
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub tokenizer: *const c_char,
    pub max_total_len: i32,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub hotwords: *const c_char,
}
impl Default for SherpaOnnxOfflineQwen3ASRModelConfig {
    fn default() -> Self {
        Self {
            conv_frontend: null(),
            encoder: null(),
            decoder: null(),
            tokenizer: null(),
            max_total_len: 0,
            max_new_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            seed: 0,
            hotwords: null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineCohereTranscribeModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub language: *const c_char,
    pub use_punct: i32,
    pub use_itn: i32,
}
impl Default for SherpaOnnxOfflineCohereTranscribeModelConfig {
    fn default() -> Self {
        Self {
            encoder: null(),
            decoder: null(),
            language: null(),
            use_punct: 0,
            use_itn: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineLMConfig {
    pub model: *const c_char,
    pub scale: f32,
}
impl Default for SherpaOnnxOfflineLMConfig {
    fn default() -> Self {
        Self {
            model: null(),
            scale: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineModelConfig {
    pub transducer: SherpaOnnxOnlineTransducerModelConfig,
    pub paraformer: SherpaOnnxSingleModelPath,
    pub nemo_ctc: SherpaOnnxSingleModelPath,
    pub whisper: SherpaOnnxOfflineWhisperModelConfig,
    pub tdnn: SherpaOnnxSingleModelPath,
    pub tokens: *const c_char,
    pub num_threads: i32,
    pub debug: i32,
    pub provider: *const c_char,
    pub model_type: *const c_char,
    pub modeling_unit: *const c_char,
    pub bpe_vocab: *const c_char,
    pub telespeech_ctc: *const c_char,
    pub sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig,
    pub moonshine: SherpaOnnxOfflineMoonshineModelConfig,
    pub fire_red_asr: SherpaOnnxOnlineParaformerModelConfig,
    pub dolphin: SherpaOnnxSingleModelPath,
    pub zipformer_ctc: SherpaOnnxSingleModelPath,
    pub canary: SherpaOnnxOfflineCanaryModelConfig,
    pub wenet_ctc: SherpaOnnxSingleModelPath,
    pub omnilingual: SherpaOnnxSingleModelPath,
    pub medasr: SherpaOnnxSingleModelPath,
    pub funasr_nano: SherpaOnnxOfflineFunASRNanoModelConfig,
    pub fire_red_asr_ctc: SherpaOnnxSingleModelPath,
    pub qwen3_asr: SherpaOnnxOfflineQwen3ASRModelConfig,
    pub cohere_transcribe: SherpaOnnxOfflineCohereTranscribeModelConfig,
}
impl Default for SherpaOnnxOfflineModelConfig {
    fn default() -> Self {
        Self {
            transducer: Default::default(),
            paraformer: Default::default(),
            nemo_ctc: Default::default(),
            whisper: Default::default(),
            tdnn: Default::default(),
            tokens: null(),
            num_threads: 0,
            debug: 0,
            provider: null(),
            model_type: null(),
            modeling_unit: null(),
            bpe_vocab: null(),
            telespeech_ctc: null(),
            sense_voice: Default::default(),
            moonshine: Default::default(),
            fire_red_asr: Default::default(),
            dolphin: Default::default(),
            zipformer_ctc: Default::default(),
            canary: Default::default(),
            wenet_ctc: Default::default(),
            omnilingual: Default::default(),
            medasr: Default::default(),
            funasr_nano: Default::default(),
            fire_red_asr_ctc: Default::default(),
            qwen3_asr: Default::default(),
            cohere_transcribe: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SherpaOnnxOfflineRecognizerConfig {
    pub feat_config: SherpaOnnxFeatureConfig,
    pub model_config: SherpaOnnxOfflineModelConfig,
    pub lm_config: SherpaOnnxOfflineLMConfig,
    pub decoding_method: *const c_char,
    pub max_active_paths: i32,
    pub hotwords_file: *const c_char,
    pub hotwords_score: f32,
    pub rule_fsts: *const c_char,
    pub rule_fars: *const c_char,
    pub blank_penalty: f32,
    pub hr: SherpaOnnxHomophoneReplacerConfig,
    pub ctc_fst_decoder_config: SherpaOnnxCtcFstDecoderConfig,
}
impl Default for SherpaOnnxOfflineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: Default::default(),
            model_config: Default::default(),
            lm_config: Default::default(),
            decoding_method: null(),
            max_active_paths: 0,
            hotwords_file: null(),
            hotwords_score: 0.0,
            rule_fsts: null(),
            rule_fars: null(),
            blank_penalty: 0.0,
            hr: Default::default(),
            ctc_fst_decoder_config: Default::default(),
        }
    }
}

#[repr(C)]
pub struct SherpaOnnxOfflineRecognizerResult {
    pub text: *const c_char,
    pub timestamps: *mut f32,
    pub count: i32,
    pub tokens: *const c_char,
    pub tokens_arr: *const *const c_char,
    pub json: *const c_char,
    pub lang: *const c_char,
    pub emotion: *const c_char,
    pub event: *const c_char,
    pub durations: *mut f32,
    pub ys_log_probs: *mut f32,
    pub segment_timestamps: *const f32,
    pub segment_durations: *const f32,
    pub segment_texts: *const c_char,
    pub segment_texts_arr: *const *const c_char,
    pub segment_count: i32,
}
