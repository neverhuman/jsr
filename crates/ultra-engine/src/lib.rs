//! ULTRA pure-Rust fused CPU inference engine for the R1 streaming Conformer transducer
//! (`docs/ultra/ENGINE_PLAN.md`).
//!
//! The engine reads an `ultra.r1-weights/v1` directory (`weights.safetensors` + `arch.json`,
//! written by `python/ultra/tools/r1_to_safetensors.py`) and runs the trained streaming
//! arithmetic of `python/ultra/train/infer.py::ChunkEncoder` chunk by chunk for a batch of
//! streams, followed by frame-synchronous greedy RNN-T decoding with the same symbol cap.
//!
//! Layout:
//! - [`model`]: weights, architecture and the [`model::Model`] loader.
//! - [`encoder`]: the chunk forward (stem, Conformer blocks with RoPE attention over a
//!   left-context KV ring, causal depthwise conv with a state ring, norms).
//! - [`decoder`]: stateless predictor, joiner and batched greedy search.
//! - [`engine`]: the streaming API (`Engine`, `EngineStream`, per-worker `Workspace`).
//! - [`quant`]: per-output-channel int8 weights and per-row dynamic activation quantisation.
//! - [`kernels`]: scalar references and runtime-dispatched AVX2/FMA kernels; `unsafe` lives
//!   only in `kernels::x86_avx2`.

pub mod agc;
pub mod agc_stream;
pub mod beam;
pub mod commit;
pub mod decode_config;
pub mod decoder;
pub mod encoder;
pub mod engine;
pub mod fscore;
pub mod kernels;
pub mod lm;
pub mod model;
pub mod panel;
pub mod peek;
pub mod pool;
pub mod prof;
pub mod quant;
pub mod run_info;
pub mod safetensors;
pub mod sealed;
pub mod workload;

pub use agc::{WarmAgc, WarmAgcConfig};
pub use agc_stream::AgcStream;
pub use decode_config::EffectiveDecodeConfig;
pub use engine::{DecodeProgress, Engine, EngineStream, Precision, Workspace};
pub use kernels::Isa;
pub use lm::{
    FfnnActivation, FfnnLm, LmConfig, NgramLm, TfConfig, TfDirection, TfRescorer, TfScore,
    TfWorkspace, TokenLm,
};
pub use model::{Arch, Model};
pub use peek::{PeekConfig, PeekEvent, PeekSession, PeekStats};

/// Errors from loading an engine weight directory or running a chunk.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    #[error("weight directory format: {0}")]
    Format(String),
    #[error("integrity failure: {0}")]
    Integrity(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("unsupported model or request: {0}")]
    Unsupported(String),
    #[error("stream lifecycle violation: {0}")]
    Lifecycle(String),
    #[error("malformed or non-finite engine output: {0}")]
    Output(String),
    #[error(transparent)]
    Frontend(#[from] ultra_frontend::FrontendError),
    #[error(transparent)]
    Gp(#[from] ultra_gp::GpError),
    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl EngineError {
    /// Structured repair guidance for agents and operators (`docs/errors.md`).
    pub fn repair(&self) -> ultra_core::repair::RepairHint {
        use ultra_core::repair::RepairHint;
        match self {
            Self::Frontend(e) => e.repair(),
            Self::Gp(e) => e.repair(),
            Self::Output(_) => RepairHint {
                purpose: "refuse NaN/Inf encoder or joiner values instead of emitting a silent blank transcript",
                reason: "a chunk produced a non-finite activation or logit",
                common_fixes: &[
                    "re-run the per-layer parity test against the torch reference to find the first diverging layer",
                    "check the weights.safetensors hash recorded in arch.json",
                ],
                docs_url: "docs/errors.md#bundle",
                repair_hint: "cargo test -p ultra-engine",
            },
            Self::Lifecycle(_) => RepairHint {
                purpose: "never accept audio after input_finished or reuse a stream whose chunk step failed",
                reason: "the caller fed a finished stream or retried a terminal (failed) stream",
                common_fixes: &[
                    "create a new stream for new audio",
                    "drop streams reported failed and restart their utterance from the beginning",
                ],
                docs_url: "docs/errors.md#lifecycle",
                repair_hint: "cargo test -p ultra-engine",
            },
            Self::Unsupported(_) => RepairHint {
                purpose: "run only architectures and precisions the engine implements",
                reason: "arch.json names a family, activation or precision the engine does not support",
                common_fixes: &["use a dense-family R1 checkpoint converted by r1_to_safetensors.py"],
                docs_url: "docs/errors.md#bundle",
                repair_hint: "cargo test -p ultra-engine",
            },
            Self::Format(_) | Self::Integrity(_) | Self::Shape(_) | Self::Io(_) => RepairHint {
                purpose: "load only a complete, hash-verified ultra.r1-weights/v1 directory",
                reason: "arch.json, weights.safetensors or a copied bundle file failed a schema, shape or SHA-256 check",
                common_fixes: &[
                    "re-run python/ultra/tools/r1_to_safetensors.py convert into a new directory",
                    "compare arch.json files.*.sha256 with sha256sum of the files",
                ],
                docs_url: "docs/errors.md#bundle",
                repair_hint: "cargo test -p ultra-engine",
            },
        }
    }
}
