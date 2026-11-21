//! Sealed `PUBLIC-TEST` (`test-*`) read guard and `docs/ultra/LOOKS.jsonl` ledger (`FREEZE.md §4.2` / `S6`).
//!
//! Enforced for EVERY `Partition::PublicTest` invocation regardless of `--qualified` vs
//! exploratory mode: requires `ULTRA_SEALED_READ=<look-id>`, refuses a second read of the same
//! `(candidate, split, freeze_tag)` already recorded in `LOOKS.jsonl`, and appends the new read
//! record before inference begins.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use ultra_core::receipt::Partition;

use crate::CliError;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LookRecord {
    kind: String,
    look_id: String,
    time_unix_s: u64,
    who: String,
    board: String,
    split: String,
    candidate: String,
    freeze_tag: String,
    build_commit: String,
    completion_status: String,
}

fn default_looks_path() -> PathBuf {
    if let Ok(p) = std::env::var("ULTRA_LOOKS_PATH") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("docs/ultra/LOOKS.jsonl")
}

/// Guard and log every `PUBLIC-TEST` read for `candidates` on `split`.
/// No-op for `Partition::Dev` and `Partition::PrivateOod`.
pub fn guard_and_log_public_test_reads(
    partition: Partition,
    split: &str,
    candidates: &[&str],
    freeze_tag: &str,
    build_commit: &str,
    completion_status: &str,
) -> Result<(), CliError> {
    if partition != Partition::PublicTest {
        return Ok(());
    }
    let look_id = std::env::var("ULTRA_SEALED_READ")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            CliError::Config(format!(
                "PUBLIC-TEST split `{split}` requires ULTRA_SEALED_READ=<look-id> in EVERY mode (FREEZE.md §4.2 / S6)"
            ))
        })?;
    let looks_path = default_looks_path();
    guard_and_log_at_path(
        &looks_path,
        &look_id,
        split,
        candidates,
        freeze_tag,
        build_commit,
        completion_status,
    )
}

pub fn guard_and_log_at_path(
    looks_path: &Path,
    look_id: &str,
    split: &str,
    candidates: &[&str],
    freeze_tag: &str,
    build_commit: &str,
    completion_status: &str,
) -> Result<(), CliError> {
    let existing = if looks_path.exists() {
        std::fs::read_to_string(looks_path)?
    } else {
        String::new()
    };
    for cand in candidates {
        for line in existing.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if v.get("historical_reconstruction").and_then(|x| x.as_bool()) == Some(true) {
                continue;
            }
            let same_kind = v.get("kind").and_then(|x| x.as_str()) == Some("public-test-read");
            let same_split = v.get("split").and_then(|x| x.as_str()) == Some(split);
            let same_cand = v.get("candidate").and_then(|x| x.as_str()) == Some(*cand);
            let same_tag = v.get("freeze_tag").and_then(|x| x.as_str()) == Some(freeze_tag);
            if same_kind && same_split && same_cand && same_tag {
                return Err(CliError::Config(format!(
                    "refusing second PUBLIC-TEST read for candidate `{cand}` on `{split}` under `{freeze_tag}` (already logged in `{}`; FREEZE.md §4.2 / S6)",
                    looks_path.display()
                )));
            }
        }
    }

    if let Some(parent) = looks_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut fh = OpenOptions::new()
        .create(true)
        .append(true)
        .open(looks_path)?;
    let now_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    for cand in candidates {
        let rec = LookRecord {
            kind: "public-test-read".into(),
            look_id: look_id.to_owned(),
            time_unix_s: now_s,
            who: "ultra-bench-suite".into(),
            board: "PUBLIC-TEST".into(),
            split: split.to_owned(),
            candidate: (*cand).to_owned(),
            freeze_tag: freeze_tag.to_owned(),
            build_commit: build_commit.to_owned(),
            completion_status: completion_status.to_owned(),
        };
        let line = serde_json::to_string(&rec).map_err(|e| CliError::Internal(e.to_string()))?;
        writeln!(fh, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_test_guard_appends_looks_and_refuses_second_read() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let looks = tmp.path().join("LOOKS.jsonl");
        // Historical reconstruction line must NOT block the first post-freeze read.
        std::fs::write(
            &looks,
            "{\"kind\":\"public-test-read\",\"historical_reconstruction\":true,\"split\":\"test-clean\",\"candidate\":\"r0\",\"freeze_tag\":\"bench-freeze-v1\"}\n",
        )
        .expect("seed");

        guard_and_log_at_path(
            &looks,
            "look-001",
            "test-clean",
            &["r0", "r1-engine"],
            "bench-freeze-v1",
            "abc123",
            "qualified",
        )
        .expect("first post-freeze read passes and appends");

        let content = std::fs::read_to_string(&looks).expect("read");
        assert_eq!(content.lines().count(), 3);
        assert!(content.contains("look-001"));

        // Second read for the same (candidate, split, freeze_tag) MUST fail closed.
        let err = guard_and_log_at_path(
            &looks,
            "look-002",
            "test-clean",
            &["r0"],
            "bench-freeze-v1",
            "abc123",
            "qualified",
        )
        .expect_err("second read must fail");
        assert!(err.to_string().contains("second PUBLIC-TEST read"));
    }
}
