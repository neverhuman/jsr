//! Speaker-stratified DEV selection panel loader (`ultra.dev-panel/v1`, `FREEZE.md §1` / `B5` / `N5`).
//!
//! Delegates manifest parsing and canonical `manifest_sha256` recomputation to
//! [`ultra_engine::panel::Panel::read`], refuses non-`dev-*` splits (preventing unlogged
//! PUBLIC-TEST reads via `--panel`), verifies `cache_sha256` against the actual NPZ SHA-256,
//! and maps the declared `uids` to ordered NPZ indices without exposing reference transcripts.

use std::collections::HashMap;
use std::path::Path;

use ultra_engine::panel::Panel;
use ultra_frontend::npz::load_npz_references;

use crate::CliError;

pub const DEV_PANEL_SCHEMA: &str = ultra_engine::panel::PANEL_SCHEMA;

/// Verified panel selection bound to a specific split NPZ file.
#[derive(Debug, Clone)]
pub struct ResolvedPanel {
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub split: String,
    pub indices: Vec<usize>,
}

/// Load and cryptographically verify `ultra.dev-panel/v1` for `split` against `npz_path`.
pub fn load_panel_for_split(
    panel_path: &Path,
    split: &str,
    npz_path: &Path,
    npz_sha256: &str,
) -> Result<ResolvedPanel, CliError> {
    if !split.starts_with("dev-") {
        return Err(CliError::Config(format!(
            "--panel only permits dev-* selection splits (refused `{split}`; re-review B5/S6)"
        )));
    }
    let panel = Panel::read(panel_path).map_err(|e| {
        CliError::Config(format!(
            "panel manifest verification failed for `{}`: {e}",
            panel_path.display()
        ))
    })?;
    for s in &panel.splits {
        if !s.split.starts_with("dev-") {
            return Err(CliError::Config(format!(
                "panel `{}` contains non-dev split `{}` (refused; re-review B5/S6)",
                panel_path.display(),
                s.split
            )));
        }
    }
    let split_entry = panel
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
    Ok(ResolvedPanel {
        manifest_path: panel_path.to_string_lossy().into_owned(),
        manifest_sha256: panel.manifest_sha256,
        split: split.to_owned(),
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultra_core::receipt::sha256_hex;
    use ultra_engine::panel::py_dumps_indent1;

    fn make_valid_panel_json(split: &str, cache_sha: &str, uids: &[&str]) -> String {
        let mut spk = serde_json::Map::new();
        for (i, u) in uids.iter().enumerate() {
            spk.insert((*u).to_owned(), serde_json::json!(100 + i as i64));
        }
        let mut body = serde_json::json!({
            "schema": DEV_PANEL_SCHEMA,
            "splits": [{
                "split": split,
                "cache_sha256": cache_sha,
                "utterances": uids.len(),
                "uids": uids,
                "speaker_of": spk
            }]
        });
        let sha = sha256_hex(py_dumps_indent1(&body).expect("dump").as_bytes());
        body.as_object_mut()
            .expect("obj")
            .insert("manifest_sha256".into(), serde_json::json!(sha));
        serde_json::to_string_pretty(&body).expect("json")
    }

    #[test]
    fn panel_loader_verifies_manifest_hash_and_rejects_tamper_or_test_split() {
        let tmp = tempfile::tempdir().expect("tempdir");

        // 1. Tampered UID with old manifest_sha256 must be rejected (B5/N5).
        let tampered = tmp.path().join("tampered.json");
        let valid_str = make_valid_panel_json("dev-clean", &"b".repeat(64), &["u0"]);
        std::fs::write(&tampered, valid_str.replace("\"u0\"", "\"u1\"")).expect("write");
        let err = load_panel_for_split(&tampered, "dev-clean", &tampered, &"b".repeat(64))
            .expect_err("tampered manifest must fail");
        assert!(err.to_string().contains("manifest_sha256"));

        // 2. Valid manifest hash for test-clean must be rejected (B5/S6).
        let test_panel = tmp.path().join("test_panel.json");
        std::fs::write(
            &test_panel,
            make_valid_panel_json("test-clean", &"c".repeat(64), &["u0"]),
        )
        .expect("write");
        assert!(
            load_panel_for_split(&test_panel, "test-clean", &test_panel, &"c".repeat(64)).is_err()
        );

        // 3. Valid manifest hash with mismatched cache_sha256 must be rejected.
        let mismatch = tmp.path().join("mismatch.json");
        std::fs::write(
            &mismatch,
            make_valid_panel_json("dev-clean", &"b".repeat(64), &["u0"]),
        )
        .expect("write");
        let err = load_panel_for_split(&mismatch, "dev-clean", &mismatch, &"d".repeat(64))
            .expect_err("cache mismatch must fail");
        assert!(err.to_string().contains("cache_sha256 mismatch"));
    }
}
