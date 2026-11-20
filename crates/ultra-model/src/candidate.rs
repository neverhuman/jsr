//! Candidate configuration lock (`ultra.candidate.v1`, spec v1.0-B §11.2).
//!
//! Compatible with both `docs/ultra/candidates/*.yaml` (e.g. `nemotron-0.6b-q8.yaml`)
//! and JSON candidate configs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use ultra_core::receipt::sha256_hex;

use crate::yaml_min::parse_yaml_or_json_docs;
use crate::{is_unresolved_placeholder, is_valid_sha256_hex, ModelError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateModel {
    pub repository: String,
    pub revision: String,
    #[serde(default)]
    pub artifact: Option<String>,
    #[serde(default)]
    pub artifact_bytes: Option<u64>,
    #[serde(default)]
    pub artifact_sha256: Option<String>,
    #[serde(default)]
    pub artifacts: BTreeMap<String, String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    #[serde(default)]
    pub config_sha256: Option<String>,
    #[serde(default)]
    pub local_path: Option<String>,
    #[serde(default)]
    pub honesty_label: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRuntime {
    pub repository: String,
    pub commit: String,
    pub backend: String,
    pub quantization: String,
    pub runtime_options_sha256: String,
    #[serde(default)]
    pub compiler_and_flags: Option<String>,
    #[serde(default)]
    pub build_path: Option<String>,
    #[serde(default)]
    pub engine_family: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateStream {
    pub requested_chunk_ms: u32,
    pub actual_right_context_ms: Value,
    pub left_context_policy: String,
    pub revision_policy: String,
    pub endpoint_policy_id: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateResources {
    pub intra_op_threads: u32,
    pub inter_op_threads: u32,
    pub affinity: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateLicenses {
    pub code_id: String,
    pub weights_id: String,
    pub distribution_status: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One `ultra.candidate.v1` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDoc {
    pub schema: String,
    pub candidate_id: String,
    pub model: CandidateModel,
    pub runtime: CandidateRuntime,
    pub stream: CandidateStream,
    pub resources: CandidateResources,
    pub licenses: CandidateLicenses,
}

impl CandidateDoc {
    /// Validate that every mandatory field is resolved (no `REQUIRED*` / `UNPINNED`
    /// templates), CPU-only execution is enforced, and SHA-256 digests are well-formed.
    /// When `verify_files_on_disk` is true and `model.local_path` is set, also checks
    /// the local artifact file(s) byte size and SHA-256 on disk.
    pub fn validate(&self, verify_files_on_disk: bool) -> Result<(), ModelError> {
        if self.schema != "ultra.candidate.v1" && self.schema != "ultra.candidate/v1" {
            return Err(ModelError::WrongSchema {
                expected: "ultra.candidate.v1",
                found: self.schema.clone(),
            });
        }

        let required_strings = [
            ("candidate_id", self.candidate_id.as_str()),
            ("model.repository", self.model.repository.as_str()),
            ("model.revision", self.model.revision.as_str()),
            ("runtime.repository", self.runtime.repository.as_str()),
            ("runtime.commit", self.runtime.commit.as_str()),
            ("runtime.backend", self.runtime.backend.as_str()),
            ("runtime.quantization", self.runtime.quantization.as_str()),
            (
                "runtime.runtime_options_sha256",
                self.runtime.runtime_options_sha256.as_str(),
            ),
            (
                "stream.left_context_policy",
                self.stream.left_context_policy.as_str(),
            ),
            (
                "stream.revision_policy",
                self.stream.revision_policy.as_str(),
            ),
            (
                "stream.endpoint_policy_id",
                self.stream.endpoint_policy_id.as_str(),
            ),
            ("resources.affinity", self.resources.affinity.as_str()),
            ("licenses.code_id", self.licenses.code_id.as_str()),
            ("licenses.weights_id", self.licenses.weights_id.as_str()),
            (
                "licenses.distribution_status",
                self.licenses.distribution_status.as_str(),
            ),
        ];
        for (field, val) in required_strings {
            if is_unresolved_placeholder(val) {
                return Err(ModelError::Unpinned {
                    field: field.into(),
                    value: val.into(),
                });
            }
        }
        if let Value::String(s) = &self.stream.actual_right_context_ms {
            if is_unresolved_placeholder(s) {
                return Err(ModelError::Unpinned {
                    field: "stream.actual_right_context_ms".into(),
                    value: s.clone(),
                });
            }
        } else if !self.stream.actual_right_context_ms.is_number() {
            return Err(ModelError::Unpinned {
                field: "stream.actual_right_context_ms".into(),
                value: self.stream.actual_right_context_ms.to_string(),
            });
        }

        if !self.runtime.backend.eq_ignore_ascii_case("cpu") {
            return Err(ModelError::NonCpuBackend(self.runtime.backend.clone()));
        }

        if !is_valid_sha256_hex(&self.runtime.runtime_options_sha256) {
            return Err(ModelError::BadSha256 {
                field: "runtime.runtime_options_sha256".into(),
                value: self.runtime.runtime_options_sha256.clone(),
            });
        }

        if self.model.artifact_sha256.is_none() && self.model.artifacts.is_empty() {
            return Err(ModelError::Unpinned {
                field: "model.artifact_sha256".into(),
                value: "missing".into(),
            });
        }

        if let Some(h) = &self.model.artifact_sha256 {
            if !is_valid_sha256_hex(h) {
                return Err(ModelError::BadSha256 {
                    field: "model.artifact_sha256".into(),
                    value: h.clone(),
                });
            }
        }
        for (name, h) in &self.model.artifacts {
            if !is_valid_sha256_hex(h) {
                return Err(ModelError::BadSha256 {
                    field: format!("model.artifacts.{name}"),
                    value: h.clone(),
                });
            }
        }
        if let Some(th) = &self.model.tokenizer_sha256 {
            if !is_valid_sha256_hex(th) {
                return Err(ModelError::BadSha256 {
                    field: "model.tokenizer_sha256".into(),
                    value: th.clone(),
                });
            }
        }

        if verify_files_on_disk {
            if let Some(base) = &self.model.local_path {
                let dir = PathBuf::from(base);
                if let (Some(art), Some(expect_sha)) =
                    (&self.model.artifact, &self.model.artifact_sha256)
                {
                    verify_one_file(&dir.join(art), expect_sha, self.model.artifact_bytes)?;
                }
                for (rel, expect_sha) in &self.model.artifacts {
                    verify_one_file(&dir.join(rel), expect_sha, None)?;
                }
                if let (Some(tok), Some(expect_sha)) =
                    (&self.model.tokenizer, &self.model.tokenizer_sha256)
                {
                    verify_one_file(&dir.join(tok), expect_sha, None)?;
                }
            }
        }
        Ok(())
    }
}

fn verify_one_file(
    path: &Path,
    expect_sha: &str,
    expect_bytes: Option<u64>,
) -> Result<(), ModelError> {
    if !path.exists() {
        return Err(ModelError::MissingArtifact(path.display().to_string()));
    }
    let bytes = std::fs::read(path)?;
    if let Some(eb) = expect_bytes {
        if bytes.len() as u64 != eb {
            return Err(ModelError::LengthMismatch {
                path: path.display().to_string(),
                expected: eb,
                actual: bytes.len() as u64,
            });
        }
    }
    let got = sha256_hex(&bytes);
    if got != expect_sha {
        return Err(ModelError::HashMismatch {
            path: path.display().to_string(),
            expected: expect_sha.to_owned(),
            actual: got,
        });
    }
    Ok(())
}

/// Parse one or more `CandidateDoc` documents from a YAML or JSON text.
pub fn parse_candidate_docs(text: &str) -> Result<Vec<CandidateDoc>, ModelError> {
    let vals = parse_yaml_or_json_docs(text)?;
    if vals.is_empty() {
        return Err(ModelError::Parse("empty candidate configuration".into()));
    }
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        let doc: CandidateDoc =
            serde_json::from_value(v).map_err(|e| ModelError::Parse(e.to_string()))?;
        out.push(doc);
    }
    Ok(out)
}
