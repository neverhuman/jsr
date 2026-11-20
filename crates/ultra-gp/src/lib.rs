//! ULTRA general-purpose R1/R3 transducer bundle loader, strict `ultra.r1-export/v1`
//! validator, SentencePiece (`U+2581`) detokenizer, and cached greedy RNN-T state.

pub mod bundle;
pub mod rnnt;

pub use bundle::{
    ModelFamily, R1Bundle, R1DecoderSpec, R1EncoderStateSpec, R1Precision, R1StreamingSpec,
    R1_EXPORT_SCHEMA, R1_EXPORT_SCHEMA_V1_1,
};
pub use rnnt::{
    argmax_f32, checked_argmax_f32, chunk_bounds, FeatureRing, RnntStreamState, TokenTable,
};

/// Errors produced when loading, validating, or stepping an `ultra.r1-export/v1` transducer bundle.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpError {
    #[error("invalid bundle manifest or layout: {0}")]
    Manifest(String),
    #[error("file integrity failure: {0}")]
    Integrity(String),
    #[error("unsupported or unknown precision '{0}' (expected 'fp32' or 'int8')")]
    Precision(String),
    #[error("token table invalid: {0}")]
    Tokens(String),
    #[error("malformed or non-finite model output: {0}")]
    Output(String),
    #[error(transparent)]
    Frontend(#[from] ultra_frontend::FrontendError),
    #[error(transparent)]
    Core(#[from] ultra_core::CoreError),
    #[error("io error: {0}")]
    Io(String),
}

impl GpError {
    /// Structured repair guidance for agents and operators (`docs/errors.md`).
    pub fn repair(&self) -> ultra_core::repair::RepairHint {
        match self {
            Self::Core(e) => e.repair(),
            Self::Frontend(e) => e.repair(),
            Self::Precision(_) => ultra_core::repair::RepairHint {
                purpose: "refuse silent precision fallback when loading ONNX transducer bundles",
                reason: "caller passed a precision string other than 'fp32' or 'int8'",
                common_fixes: &[
                    "pass explicit 'int8' or 'fp32' precision matching manifest.files",
                ],
                docs_url: "docs/errors.md#precision",
                repair_hint: "cargo test -p ultra-gp",
            },
            Self::Output(_) => ultra_core::repair::RepairHint {
                purpose: "reject NaN, Inf, or empty encoder/decoder/joiner tensors instead of emitting blank transcripts",
                reason: "ONNX Runtime session produced a non-finite or empty output slice",
                common_fixes: &[
                    "inspect exported checkpoint weights and dynamic quantization scales for overflow or NaN",
                ],
                docs_url: "docs/errors.md#bundle",
                repair_hint: "cargo test -p ultra-gp",
            },
            Self::Integrity(_) | Self::Manifest(_) | Self::Tokens(_) | Self::Io(_) => {
                ultra_core::repair::RepairHint {
                    purpose: "validate ultra.r1-export/v1 schema, SHA-256 digests, state shapes, and complete token IDs",
                    reason: "bundle manifest, ONNX artifact digest, state spec, or tokens.txt failed strict checks",
                    common_fixes: &[
                        "re-export the R1 checkpoint with python -m train.export and verify manifest.json hashes",
                        "ensure files.mel_fbank, encoder.{fp32,int8}.onnx, decoder.onnx, joiner.{fp32,int8}.onnx, and tokens.txt exist",
                    ],
                    docs_url: "docs/errors.md#bundle",
                    repair_hint: "cargo test -p ultra-gp",
                }
            }
        }
    }
}

impl From<std::io::Error> for GpError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
