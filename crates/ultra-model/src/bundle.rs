//! Model bundle loader and validator (`ultra.bundle/v1`, spec v1.0-B §34).
//!
//! Performs expensive SHA-256 verification once at load/install time (never per audio chunk),
//! refuses path traversal (`..`, absolute paths, backslashes, or symlinks escaping the bundle
//! root), checks required ISA flags against `MachineRecord`, and records both packed disk bytes
//! and expanded resident bytes.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use ultra_core::receipt::sha256_hex;

use crate::hardware::MachineRecord;
use crate::{is_unresolved_placeholder, is_valid_sha256_hex, ModelError};

pub const BUNDLE_SCHEMA: &str = "ultra.bundle/v1";

/// One content-addressed file inside a model bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Frontend parameters bound to the bundle (§34.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleFrontend {
    pub sample_rate_hz: u32,
    pub feature_dim: usize,
    pub kind: String,
}

/// Stream and memory limits bound to the bundle (§34.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleLimits {
    pub max_concurrent_streams: usize,
    pub max_service_rss_bytes: u64,
    pub supported_chunk_ms: Vec<u32>,
}

/// Top-level `manifest.json` inside a bundle directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub schema: String,
    pub bundle_id: String,
    pub candidate_id: String,
    pub precision: String,
    pub required_isa: Vec<String>,
    pub packed_weight_bytes: u64,
    pub expanded_weight_bytes: u64,
    pub frontend: BundleFrontend,
    pub limits: BundleLimits,
    pub normalizer_id: String,
    pub license_code: String,
    pub license_weights: String,
    pub provenance_summary: String,
    pub entries: Vec<BundleEntry>,
}

/// Receipt of a successfully verified model bundle on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedBundle {
    pub root: PathBuf,
    pub manifest: BundleManifest,
    pub manifest_sha256: String,
    pub total_verified_bytes: u64,
}

/// Refuse `..`, absolute roots, Windows prefixes, backslashes, or empty paths (§34.2).
pub fn check_safe_relative_path(rel: &str) -> Result<&Path, ModelError> {
    if rel.trim().is_empty() || rel.contains('\\') || rel.contains('\0') {
        return Err(ModelError::UnsafePath(rel.to_owned()));
    }
    let p = Path::new(rel);
    if p.is_absolute() {
        return Err(ModelError::UnsafePath(rel.to_owned()));
    }
    let mut saw_normal = false;
    for comp in p.components() {
        match comp {
            Component::Normal(_) => saw_normal = true,
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ModelError::UnsafePath(rel.to_owned()));
            }
        }
    }
    if !saw_normal {
        return Err(ModelError::UnsafePath(rel.to_owned()));
    }
    Ok(p)
}

impl BundleManifest {
    /// Structural validation of `manifest.json` before touching payload files.
    pub fn validate_structure(&self) -> Result<(), ModelError> {
        if self.schema != BUNDLE_SCHEMA {
            return Err(ModelError::WrongSchema {
                expected: BUNDLE_SCHEMA,
                found: self.schema.clone(),
            });
        }
        for (field, val) in [
            ("bundle_id", self.bundle_id.as_str()),
            ("candidate_id", self.candidate_id.as_str()),
            ("precision", self.precision.as_str()),
            ("normalizer_id", self.normalizer_id.as_str()),
            ("license_code", self.license_code.as_str()),
            ("license_weights", self.license_weights.as_str()),
            ("provenance_summary", self.provenance_summary.as_str()),
        ] {
            if is_unresolved_placeholder(val) {
                return Err(ModelError::Unpinned {
                    field: field.into(),
                    value: val.into(),
                });
            }
        }
        if self.frontend.sample_rate_hz != 16_000 || self.frontend.feature_dim == 0 {
            return Err(ModelError::InvalidBundle(format!(
                "unsupported frontend sample_rate_hz={} feature_dim={}",
                self.frontend.sample_rate_hz, self.frontend.feature_dim
            )));
        }
        if self.packed_weight_bytes == 0 || self.expanded_weight_bytes < self.packed_weight_bytes {
            return Err(ModelError::InvalidBundle(format!(
                "invalid weight sizes packed={} expanded={}",
                self.packed_weight_bytes, self.expanded_weight_bytes
            )));
        }
        if self.limits.max_concurrent_streams == 0
            || self.limits.max_service_rss_bytes == 0
            || self.limits.supported_chunk_ms.is_empty()
        {
            return Err(ModelError::InvalidBundle("invalid bundle limits".into()));
        }
        if self.entries.is_empty() {
            return Err(ModelError::InvalidBundle("bundle has no entries".into()));
        }
        for e in &self.entries {
            check_safe_relative_path(&e.path)?;
            if !is_valid_sha256_hex(&e.sha256) {
                return Err(ModelError::BadSha256 {
                    field: format!("entries[{}].sha256", e.path),
                    value: e.sha256.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Verify `manifest.json`, optional hardware ISA compatibility, and every listed payload's
/// byte length and SHA-256 digest inside `bundle_dir`.
pub fn verify_bundle_dir(
    bundle_dir: &Path,
    hardware: Option<&MachineRecord>,
) -> Result<VerifiedBundle, ModelError> {
    let manifest_path = bundle_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ModelError::MissingArtifact(
            manifest_path.display().to_string(),
        ));
    }
    let canon_root = std::fs::canonicalize(bundle_dir)?;
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest: BundleManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|e| ModelError::Parse(e.to_string()))?;
    manifest.validate_structure()?;

    if let Some(hw) = hardware {
        hw.supports_all_isa(&manifest.required_isa)?;
    }

    let mut total_verified_bytes = 0u64;
    for entry in &manifest.entries {
        let rel = check_safe_relative_path(&entry.path)?;
        let full = bundle_dir.join(rel);
        if !full.exists() {
            return Err(ModelError::MissingArtifact(entry.path.clone()));
        }
        let canon_entry = std::fs::canonicalize(&full)?;
        if !canon_entry.starts_with(&canon_root) {
            return Err(ModelError::UnsafePath(entry.path.clone()));
        }
        let data = std::fs::read(&canon_entry)?;
        if data.len() as u64 != entry.bytes {
            return Err(ModelError::LengthMismatch {
                path: entry.path.clone(),
                expected: entry.bytes,
                actual: data.len() as u64,
            });
        }
        let got_sha = sha256_hex(&data);
        if got_sha != entry.sha256 {
            return Err(ModelError::HashMismatch {
                path: entry.path.clone(),
                expected: entry.sha256.clone(),
                actual: got_sha,
            });
        }
        total_verified_bytes = total_verified_bytes.saturating_add(entry.bytes);
    }

    Ok(VerifiedBundle {
        root: canon_root,
        manifest,
        manifest_sha256,
        total_verified_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::parse_candidate_docs;
    use crate::hardware::HardwareInventory;

    fn sample_hw() -> MachineRecord {
        let inv = HardwareInventory::from_json_str(
            r#"{
              "schema": "ultra.hardware/v1",
              "machines": [{
                "id": "H0",
                "host": "xbabe3",
                "cpu": "AMD Ryzen Threadripper PRO 3995WX 64-Cores",
                "cores": 64,
                "threads_per_core": 2,
                "isa": ["avx2", "fma"],
                "ram_gib": 62.6,
                "os": "Linux",
                "governor": "schedutil",
                "status": "provisional"
              }]
            }"#,
        )
        .expect("valid hardware");
        inv.machines.into_iter().next().expect("one machine")
    }

    #[test]
    fn refuses_path_traversal_and_bad_isa_and_corrupt_hash() {
        assert!(check_safe_relative_path("../secret").is_err());
        assert!(check_safe_relative_path("/etc/passwd").is_err());
        assert!(check_safe_relative_path("model/..\\x").is_err());
        assert!(check_safe_relative_path("model/encoder.onnx").is_ok());

        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).expect("mkdir");
        let payload = b"fake-int8-weights-payload";
        std::fs::write(model_dir.join("encoder.int8.onnx"), payload).expect("write");

        let mut manifest = BundleManifest {
            schema: BUNDLE_SCHEMA.into(),
            bundle_id: "r0-zipformer-int8".into(),
            candidate_id: "sherpa-onnx-streaming-zipformer-en-int8".into(),
            precision: "int8".into(),
            required_isa: vec!["avx2".into()],
            packed_weight_bytes: payload.len() as u64,
            expanded_weight_bytes: payload.len() as u64,
            frontend: BundleFrontend {
                sample_rate_hz: 16_000,
                feature_dim: 80,
                kind: "logmel80".into(),
            },
            limits: BundleLimits {
                max_concurrent_streams: 128,
                max_service_rss_bytes: 1_500_000_000,
                supported_chunk_ms: vec![320],
            },
            normalizer_id: "ultra.en-norm.v1".into(),
            license_code: "Apache-2.0".into(),
            license_weights: "Apache-2.0".into(),
            provenance_summary: "LibriSpeech 960h only".into(),
            entries: vec![BundleEntry {
                path: "model/encoder.int8.onnx".into(),
                bytes: payload.len() as u64,
                sha256: sha256_hex(payload),
            }],
        };
        std::fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).expect("json"),
        )
        .expect("write manifest");

        let hw = sample_hw();
        let verified = verify_bundle_dir(tmp.path(), Some(&hw)).expect("bundle verifies on H0");
        assert_eq!(verified.total_verified_bytes, payload.len() as u64);

        // Require AVX-512 -> refused on H0.
        manifest.required_isa = vec!["avx2".into(), "avx512vnni".into()];
        std::fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).expect("json"),
        )
        .expect("write manifest");
        assert!(matches!(
            verify_bundle_dir(tmp.path(), Some(&hw)),
            Err(ModelError::UnsupportedIsa { .. })
        ));

        // Corrupt payload -> refused by SHA-256 mismatch.
        manifest.required_isa = vec!["avx2".into()];
        std::fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).expect("json"),
        )
        .expect("write manifest");
        std::fs::write(
            model_dir.join("encoder.int8.onnx"),
            b"corrupted-int8-weights!!!",
        )
        .expect("corrupt");
        assert!(matches!(
            verify_bundle_dir(tmp.path(), Some(&hw)),
            Err(ModelError::HashMismatch { .. })
        ));
    }

    #[test]
    fn parses_multi_doc_yaml_candidate_and_refuses_required_placeholders() {
        let yaml = r#"
---
schema: ultra.candidate.v1
candidate_id: zipformer_en_int8_320
model:
  repository: csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
  revision: 672fbf1b30579d6585301139bb363f42a0ad4a24
  artifact: encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
  artifact_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  training_data: >-
    LibriSpeech 960 h only,
    exported to ONNX int8.
runtime:
  repository: https://github.com/k2-fsa/sherpa-onnx
  commit: 2.0.2
  backend: cpu
  quantization: int8
  runtime_options_sha256: fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
stream:
  requested_chunk_ms: 320
  actual_right_context_ms: 0
  left_context_policy: cached_128_frames
  revision_policy: editable_until_final
  endpoint_policy_id: none.single_final_at_eos
resources:
  intra_op_threads: 1
  inter_op_threads: 1
  affinity: pinned_one_physical_core
licenses:
  code_id: Apache-2.0
  weights_id: Apache-2.0
  distribution_status: internal_evaluation_only
---
schema: ultra.candidate.v1
candidate_id: unpinned_template
model:
  repository: nvidia/nemotron
  revision: REQUIRED_RESOLVED_IMMUTABLE_REVISION
  artifact_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
runtime:
  repository: https://github.com/NVIDIA/NeMo-Speech.cpp
  commit: abcdef1
  backend: cpu
  quantization: Q8_0
  runtime_options_sha256: fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
stream:
  requested_chunk_ms: 160
  actual_right_context_ms: 80
  left_context_policy: cache
  revision_policy: editable_until_final
  endpoint_policy_id: none
resources:
  intra_op_threads: 1
  inter_op_threads: 1
  affinity: taskset
licenses:
  code_id: Apache-2.0
  weights_id: Open
  distribution_status: internal_evaluation_only
"#;
        let docs = parse_candidate_docs(yaml).expect("two yaml docs");
        assert_eq!(docs.len(), 2);
        docs[0].validate(false).expect("first doc is pinned");
        assert!(matches!(
            docs[1].validate(false),
            Err(ModelError::Unpinned { .. })
        ));
    }
}
