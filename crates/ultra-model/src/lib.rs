//! ULTRA model, candidate, and hardware manifest contracts (spec v1.0-B §11.2, §34).
//!
//! Enforces strict pre-run validation:
//! - Candidate manifests (`ultra.candidate.v1` / `ultra.candidate/v1`) refuse unresolved
//!   `REQUIRED` or `UNPINNED` placeholders, non-CPU backends, and mismatched SHA-256 digests.
//! - Hardware inventories (`ultra.hardware/v1`) bind ISA capabilities (`avx2`, `fma`, ...)
//!   so bundles requiring unavailable instructions (e.g. `avx512vnni` on Zen 2 H0) are refused
//!   cleanly at load rather than crashing with `SIGILL`.
//! - Model bundles (`ultra.bundle/v1`) validate schema compatibility, tensor shapes, packed vs
//!   expanded weight byte counts, path-traversal safety, and SHA-256 checksums at load/verify.

pub mod bundle;
pub mod candidate;
pub mod hardware;
pub mod yaml_min;

pub use bundle::{
    verify_bundle_dir, BundleEntry, BundleFrontend, BundleLimits, BundleManifest, VerifiedBundle,
    BUNDLE_SCHEMA,
};
pub use candidate::{
    parse_candidate_docs, CandidateDoc, CandidateModel, CandidateResources, CandidateRuntime,
    CandidateStream,
};
pub use hardware::{HardwareInventory, MachineRecord, HARDWARE_SCHEMA};

/// Errors from candidate, hardware, and bundle validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    #[error("wrong schema `{found}` (expected `{expected}`)")]
    WrongSchema {
        expected: &'static str,
        found: String,
    },
    #[error("unresolved or unpinned field `{field}`: `{value}` (spec §11.2)")]
    Unpinned { field: String, value: String },
    #[error("non-CPU backend `{0}` refused by CPU-only contract (spec §0.2 rule 1)")]
    NonCpuBackend(String),
    #[error("invalid sha256 hex in `{field}`: `{value}`")]
    BadSha256 { field: String, value: String },
    #[error("unsafe bundle entry path `{0}` refused (spec §34.2 path traversal)")]
    UnsafePath(String),
    #[error("missing required artifact `{0}`")]
    MissingArtifact(String),
    #[error("checksum mismatch for `{path}`: expected {expected}, got {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("byte length mismatch for `{path}`: expected {expected}, got {actual}")]
    LengthMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    #[error("required ISA `{isa}` is not supported by hardware `{machine}`")]
    UnsupportedIsa { isa: String, machine: String },
    #[error("invalid bundle metadata: {0}")]
    InvalidBundle(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Check whether `s` is a valid lower-case 64-char SHA-256 hex digest.
pub fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check whether a string looks like an unresolved specification template value.
pub fn is_unresolved_placeholder(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return true;
    }
    let up = trimmed.to_ascii_uppercase();
    up.starts_with("REQUIRED")
        || up.contains("UNPINNED")
        || up == "TBD"
        || up == "UNRESOLVED"
        || up == "EXAMPLE_SCHEMA_ONLY"
}
