//! ULTRA evaluation lane: the pinned English text normaliser, WER with exact S/D/I
//! alignment, event-stream scoring (final / commit / preview views, flicker, stable
//! times), delay metrics including CQ(D), the paired cluster bootstrap with Holm
//! correction, Linux process-tree resource accounting and a paced PCM replay primitive.
//!
//! Spec anchors: v1.0-B §10, §12.4, §12.6, §12.7, §30, §33, §38.1; v1 §11.3, §11.6, §11.7.

pub mod bootstrap;
pub mod corpus;
pub mod delay;
pub mod finalize;
pub mod normalise;
pub mod pace;
pub mod procfs;
pub mod score;
pub mod stream;
pub mod wer;

/// Errors from the evaluation lane. Every refusal names what it refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EvalError {
    #[error("macro weights invalid: {0}")]
    Weights(String),
    #[error("dataset {0} has no reference words; its WER is undefined")]
    UndefinedWer(String),
    #[error("event stream refused by the ledger at event_sequence {seq}: {source}")]
    Ledger {
        seq: u64,
        source: ultra_core::CoreError,
    },
    #[error("event times must be finite and non-decreasing (event_sequence {0})")]
    EventTime(u64),
    #[error("input mismatch: {0}")]
    Input(String),
    #[error("bootstrap: {0}")]
    Bootstrap(String),
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for EvalError {
    fn from(e: std::io::Error) -> Self {
        EvalError::Io(e.to_string())
    }
}
