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
pub mod mel;
pub mod npz;
pub mod wav;

pub use cochlea::{CochleaConfig, StreamingCochlea};
pub use mel::{MelConfig, StreamingLogMel, MEL_BINS};
pub use npz::{load_npz_audio_only, load_npz_references, AudioUtterance, RefUtterance};
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

impl From<std::io::Error> for FrontendError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
