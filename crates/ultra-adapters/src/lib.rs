//! Reference-blind external recognizer process-tree adapters (`B4`).

pub mod nemotron;

pub use nemotron::{run_nemotron_batch, NemotronChunkMode, NemotronConfig, NemotronRunResult};

/// Errors from external recognizer process-tree adapters.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdapterError {
    #[error("adapter configuration or binary/model unavailable: {0}")]
    Unavailable(String),
    #[error("process execution or timeout failure: {0}")]
    Process(String),
    #[error("output completeness or integrity violation: {0}")]
    Integrity(String),
    #[error("CPU-only execution check failed: {0}")]
    GpuDetected(String),
    #[error(transparent)]
    Frontend(#[from] ultra_frontend::FrontendError),
    #[error(transparent)]
    Scoring(#[from] ultra_eval::EvalError),
    #[error(transparent)]
    Core(#[from] ultra_core::CoreError),
    #[error("io error: {0}")]
    Io(String),
}

impl AdapterError {
    /// Structured repair guidance (`docs/errors.md`).
    pub fn repair(&self) -> ultra_core::repair::RepairHint {
        match self {
            Self::Core(e) => e.repair(),
            Self::Frontend(e) => e.repair(),
            Self::Unavailable(_) => ultra_core::repair::RepairHint {
                purpose: "locate pinned NeMo-Speech.cpp CPU binary and Q8_0 GGUF artifact",
                reason: "nemo-speech binary or model artifact was missing or SHA-256 mismatched",
                common_fixes: &[
                    "verify /mnt/xbabe3-external/jsr/ext/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech exists",
                    "check docs/ultra/candidates/nemotron-0.6b-q8.yaml artifact paths and hashes",
                ],
                docs_url: "docs/errors.md#adapter",
                repair_hint: "cargo test -p ultra-adapters",
            },
            Self::Process(_) | Self::Integrity(_) | Self::GpuDetected(_) | Self::Scoring(_) | Self::Io(_) => {
                ultra_core::repair::RepairHint {
                    purpose: "enforce reference-blind, CPU-only, exact-output, orphan-free subprocess execution",
                    reason: "child exited non-zero, timed out, mapped GPU libraries, left orphans, or omitted/duplicated .txt outputs",
                    common_fixes: &[
                        "pass --device cpu and --asr.backend.threads 1 to nemo-speech",
                        "ensure every staged u%05d.wav produces exactly one u%05d.txt",
                    ],
                    docs_url: "docs/errors.md#adapter",
                    repair_hint: "cargo test -p ultra-adapters",
                }
            }
        }
    }
}

impl From<std::io::Error> for AdapterError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
