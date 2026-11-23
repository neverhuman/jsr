//! Typed error contract for `ultra-alwayson`.

use thiserror::Error;
use ultra_engine::EngineError;

/// Errors returned by ring-buffer operations, configuration validation, or always-on decoding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AlwaysOnError {
    #[error("invalid always-on configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "ring range [{start}..{end}) outside live window [{base}..{ring_end}) (refusing overwritten/future read)"
    )]
    RingOutOfRange {
        start: u64,
        end: u64,
        base: u64,
        ring_end: u64,
    },
    #[error("session is in Recovery state; call recover() before pushing audio")]
    NeedsRecovery,
    #[error("session already finished; create a new session for further audio")]
    AlreadyFinished,
    #[error(
        "backend fault after {consumed_samples} consumed samples (capture_end={capture_end}): {message}"
    )]
    BackendFault {
        /// Samples in the failing `push_pcm` call already forwarded before the fault.
        consumed_samples: usize,
        /// Exact absolute capture cursor reached at the end of the failing call.
        capture_end: u64,
        /// Underlying backend failure message.
        message: String,
    },
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<EngineError> for AlwaysOnError {
    fn from(err: EngineError) -> Self {
        Self::Backend(err.to_string())
    }
}
