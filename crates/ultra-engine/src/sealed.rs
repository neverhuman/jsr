//! Sealed PUBLIC-TEST guard for `ultra-engine-eval` (FREEZE.md §4.2 / review S6), mirroring
//! `ultra.ood` and `ultra-cli` `suite_looks`.
//!
//! A `test-*` split (an npz named `test-*.npz`, or a panel split `test-*`) is read only when
//! all of these hold:
//! - an explicit `--sealed-read <look-id>` flag is passed;
//! - the git tag `bench-freeze-v1` exists in the repository, and its commit is an ancestor of
//!   the commit the binary was built from, which must be a clean build;
//! - `docs/ultra/LOOKS.jsonl` has no earlier (non-reconstructed) `public-test-read` of the same
//!   candidate and split under that tag.
//!
//! The read is appended to `LOOKS.jsonl` under an exclusive file lock before any audio is
//! decoded. DEV splits are never affected.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

pub const FREEZE_TAG: &str = "bench-freeze-v1";

/// Whether a split name is a sealed PUBLIC-TEST split.
pub fn is_public_test(split: &str) -> bool {
    split.starts_with("test")
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

/// The repository containing `start` (`git rev-parse --show-toplevel`).
pub fn repo_root(start: &Path) -> Option<PathBuf> {
    git(start, &["rev-parse", "--show-toplevel"]).map(PathBuf::from)
}

/// Everything the guard needs, so tests can point it at a scratch repository.
pub struct SealedRead<'a> {
    pub repo: &'a Path,
    pub looks: &'a Path,
    pub look_id: Option<&'a str>,
    /// Identity of what is evaluated (checkpoint sha plus int8 artifact sha, if any).
    pub candidate: &'a str,
    pub build_commit: &'a str,
    pub build_dirty: &'a str,
}

/// Refuse or permit-and-log a read of `splits`. Returns the LOOKS records written (empty
/// when no split is sealed).
pub fn guard(splits: &[String], r: &SealedRead<'_>) -> Result<Vec<Value>, String> {
    let sealed: Vec<&String> = splits.iter().filter(|s| is_public_test(s)).collect();
    if sealed.is_empty() {
        return Ok(Vec::new());
    }
    let refuse = |why: String| {
        Err(format!(
            "PUBLIC-TEST split(s) {sealed:?} are sealed (FREEZE.md §4.2): {why}"
        ))
    };
    let Some(look) = r.look_id.map(str::trim).filter(|s| !s.is_empty()) else {
        return refuse("pass --sealed-read <look-id>".into());
    };
    let Some(tag_commit) = git(
        r.repo,
        &[
            "rev-parse",
            "-q",
            "--verify",
            &format!("refs/tags/{FREEZE_TAG}^{{commit}}"),
        ],
    ) else {
        return refuse(format!(
            "git tag {FREEZE_TAG} does not exist in {}",
            r.repo.display()
        ));
    };
    if r.build_dirty != "false" {
        return refuse(format!(
            "the binary is not a clean build (dirty = {})",
            r.build_dirty
        ));
    }
    let ancestor = Command::new("git")
        .arg("-C")
        .arg(r.repo)
        .args(["merge-base", "--is-ancestor", &tag_commit, r.build_commit])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ancestor {
        return refuse(format!(
            "{FREEZE_TAG} ({}) is not an ancestor of the build commit {}",
            &tag_commit[..tag_commit.len().min(12)],
            r.build_commit
        ));
    }
    if let Some(parent) = r.looks.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(r.looks)
        .map_err(|e| format!("{}: {e}", r.looks.display()))?;
    f.lock()
        .map_err(|e| format!("lock {}: {e}", r.looks.display()))?;
    let existing = std::fs::read_to_string(r.looks).map_err(|e| e.to_string())?;
    for split in &sealed {
        let prior = existing
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["historical_reconstruction"] != true)
            .any(|v| {
                v["kind"] == "public-test-read"
                    && v["split"] == split.as_str()
                    && v["candidate"] == r.candidate
                    && v["freeze_tag"] == FREEZE_TAG
            });
        if prior {
            return refuse(format!(
                "{split} was already read for candidate {} under {FREEZE_TAG}; one read per set",
                r.candidate
            ));
        }
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut out = Vec::new();
    for split in sealed {
        let rec = json!({"kind": "public-test-read", "look_id": look, "time_unix_s": ts,
            "who": "ultra-engine-eval", "board": "PUBLIC-TEST", "split": split,
            "candidate": r.candidate, "freeze_tag": FREEZE_TAG, "freeze_tag_commit": tag_commit,
            "build_commit": r.build_commit, "completion_status": "started"});
        writeln!(f, "{rec}").map_err(|e| e.to_string())?;
        out.push(rec);
    }
    f.flush().map_err(|e| e.to_string())?;
    Ok(out)
}
