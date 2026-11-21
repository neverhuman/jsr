//! Speaker-stratified DEV selection panel loader (`ultra.dev-panel/v1`, `FREEZE.md §1` & `B5`).
//!
//! Validates `schema == "ultra.dev-panel/v1"`, checks the split's `cache_sha256` against the
//! actual NPZ file SHA-256, and maps the declared `uids` to NPZ indices without exposing
//! reference transcripts to inference.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use ultra_frontend::npz::load_npz_references;

use crate::suite_manifest::sha256_file;
use crate::CliError;

pub const DEV_PANEL_SCHEMA: &str = "ultra.dev-panel/v1";

#[derive(Debug, Deserialize)]
struct PanelManifest {
    schema: String,
    manifest_sha256: String,
    splits: Vec<PanelSplit>,
}

#[derive(Debug, Deserialize)]
struct PanelSplit {
    split: String,
    cache_sha256: String,
    uids: Vec<String>,
}

/// Verified panel selection bound to a specific split NPZ file.
#[derive(Debug, Clone)]
pub struct ResolvedPanel {
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub split: String,
    pub indices: Vec<usize>,
}

/// Load and validate `ultra.dev-panel/v1` for `split` against `npz_path`, returning the exact
/// ordered NPZ utterance indices matching the panel's `uids`.
pub fn load_panel_for_split(
    panel_path: &Path,
    split: &str,
    npz_path: &Path,
    npz_sha256: &str,
) -> Result<ResolvedPanel, CliError> {
    let raw = std::fs::read_to_string(panel_path).map_err(|e| {
        CliError::Config(format!(
            "cannot read panel manifest `{}`: {e}",
            panel_path.display()
        ))
    })?;
    let manifest: PanelManifest = serde_json::from_str(&raw).map_err(|e| {
        CliError::Config(format!(
            "invalid panel manifest JSON `{}`: {e}",
            panel_path.display()
        ))
    })?;
    if manifest.schema != DEV_PANEL_SCHEMA {
        return Err(CliError::Config(format!(
            "unsupported panel schema `{}` (expected `{DEV_PANEL_SCHEMA}`)",
            manifest.schema
        )));
    }
    if manifest.manifest_sha256.len() != 64 {
        return Err(CliError::Config(format!(
            "panel manifest_sha256 `{}` is not a 64-hex SHA-256",
            manifest.manifest_sha256
        )));
    }
    let split_entry = manifest
        .splits
        .iter()
        .find(|s| s.split == split)
        .ok_or_else(|| {
            CliError::Config(format!(
                "panel `{}` has no entry for split `{split}`",
                panel_path.display()
            ))
        })?;
    if !split_entry.cache_sha256.eq_ignore_ascii_case(npz_sha256) {
        return Err(CliError::Config(format!(
            "panel cache_sha256 mismatch for `{split}`: manifest expects {}, npz `{}` has {npz_sha256}",
            split_entry.cache_sha256,
            npz_path.display()
        )));
    }
    if split_entry.uids.is_empty() {
        return Err(CliError::Config(format!(
            "panel `{}` split `{split}` has zero uids",
            panel_path.display()
        )));
    }
    let refs = load_npz_references(npz_path, 0, None)
        .map_err(|e| CliError::Unavailable(format!("read split uids: {e}")))?;
    let mut uid_to_idx: HashMap<&str, usize> = HashMap::with_capacity(refs.len());
    for (idx, r) in refs.iter().enumerate() {
        uid_to_idx.insert(r.uid.as_str(), idx);
    }
    let mut indices = Vec::with_capacity(split_entry.uids.len());
    for uid in &split_entry.uids {
        let &idx = uid_to_idx.get(uid.as_str()).ok_or_else(|| {
            CliError::Config(format!(
                "panel uid `{uid}` not found in `{}`",
                npz_path.display()
            ))
        })?;
        indices.push(idx);
    }
    let _file_sha = sha256_file(panel_path)?;
    Ok(ResolvedPanel {
        manifest_path: panel_path.to_string_lossy().into_owned(),
        manifest_sha256: manifest.manifest_sha256,
        split: split.to_owned(),
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_loader_rejects_bad_schema_and_mismatched_cache_hash() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = tmp.path().join("bad.json");
        std::fs::write(
            &bad,
            r#"{"schema":"wrong/v0","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","splits":[]}"#,
        )
        .expect("write");
        assert!(load_panel_for_split(&bad, "dev-clean", &bad, "deadbeef").is_err());

        let mismatch = tmp.path().join("mismatch.json");
        std::fs::write(
            &mismatch,
            r#"{"schema":"ultra.dev-panel/v1","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","splits":[{"split":"dev-clean","cache_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","uids":["u0"]}]}"#,
        )
        .expect("write");
        assert!(load_panel_for_split(&mismatch, "dev-clean", &mismatch, "deadbeef").is_err());
    }
}
