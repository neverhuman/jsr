//! ULTRA audio frontend (spec v1.0-B §14.1; ULTRA v1 §4; CADENCE v3 integer PCEN).
//!
//! - [`wav`]: 16 kHz mono PCM16 WAV / raw `.s16le` reader & writer, channel policy, and
//!   stateful anti-aliased resampler (`48 kHz`, `44.1 kHz`, ... -> `16 kHz` mono) with
//!   explicit algorithmic delay tracking.
//! - [`npz`]: zero-Python reader for uncompressed `.npz` corpus shards (such as
//!   `/mnt/xbabe3-external/jsr/cache/librispeech_pcm/*.npz`), strictly separating
//!   **reference-blind audio access** (`pcm.npy` + `offsets.npy` only) from **scorer-only
//!   reference access** (`uids.npy`, `speakers.npy`, `texts.npy`).
//! - [`mel`]: streaming 25 ms / 10 ms 80-bin float log-mel reference extractor, invariant
//!   across arbitrary packet boundaries (`10 ms`, `20 ms`, `37 ms`, random splits).
//! - [`cochlea`]: integer `isqrt`-PCEN Cochlea arm producing A7 (`0..=127`) features via
//!   `ultra_core::numeric::{isqrt_u64, requant, clipped_relu}`.

pub mod cochlea;
pub mod level;
pub mod mel;
pub mod npz;
pub mod portable_math;
pub mod trainer_mel;
mod trainer_mel_avx2;
pub mod wav;

pub use cochlea::{CochleaConfig, StreamingCochlea};
pub use level::{rms_gain, rms_normalize, rms_normalize_in_place, RmsNormStats};
pub use mel::{MelConfig, StreamingLogMel, MEL_BINS};
pub use npz::{load_npz_audio_only, load_npz_references, AudioUtterance, RefUtterance};
pub use trainer_mel::{GlobalCmvn, TrainerLogMel, TrainerMelConfig, SPEC_BINS};
pub use wav::{
    read_audio_16k_mono, read_wav_pcm, write_wav_pcm16, Resampler16k, WavData, SAMPLE_RATE_16K,
};

/// Errors from the audio frontend and corpus readers.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrontendError {
    #[error("unsupported or malformed audio format: {0}")]
    Format(String),
    #[error("unsupported sample rate {0} Hz")]
    SampleRate(u32),
    #[error("unsupported channel count {0}")]
    Channels(u16),
    #[error("npz archive error: {0}")]
    Npz(String),
    #[error(transparent)]
    Core(#[from] ultra_core::CoreError),
    #[error("io error: {0}")]
    Io(String),
}

impl FrontendError {
    /// Structured repair guidance for agents and operators (`docs/errors.md`).
    pub fn repair(&self) -> ultra_core::repair::RepairHint {
        match self {
            Self::Core(e) => e.repair(),
            Self::Format(_) | Self::SampleRate(_) | Self::Channels(_) => {
                ultra_core::repair::RepairHint {
                    purpose: "enforce 16 kHz mono PCM16 and finite trainer mel/CMVN parameters",
                    reason: "audio header, sample rate, channel count, or mel filterbank/CMVN failed validation",
                    common_fixes: &[
                        "pass 16 kHz mono PCM16 audio (or resample explicitly via Resampler16k)",
                        "load mel_fbank.f32 (257x80 le f32) and finite 80-bin CMVN stats from manifest.json",
                    ],
                    docs_url: "docs/errors.md#frontend",
                    repair_hint: "cargo test -p ultra-frontend",
                }
            }
            Self::Npz(_) | Self::Io(_) => ultra_core::repair::RepairHint {
                purpose: "read uncompressed NPZ audio shards and bundle mel matrices safely",
                reason: "expected .npz or mel_fbank.f32 file was missing, truncated, or compressed",
                common_fixes: &[
                    "verify $JSR_DATA/cache/librispeech_pcm/*.npz and bundle files match SHA-256",
                ],
                docs_url: "docs/errors.md#frontend",
                repair_hint: "cargo test -p ultra-frontend",
            },
        }
    }
}

impl From<std::io::Error> for FrontendError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
