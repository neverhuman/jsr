//! Sealed `PUBLIC-TEST` (`test-*`) read guard and `docs/ultra/LOOKS.jsonl` ledger (`FREEZE.md §4.2` / `S6`).
//!
//! Enforced for EVERY `Partition::PublicTest` invocation regardless of `--qualified` vs
//! exploratory mode:
//! - requires an explicit `--sealed-read <look-id>` flag (or `ULTRA_SEALED_READ=<look-id>`);
//! - requires git tag `bench-freeze-v1` to exist and be an ancestor of a clean build commit;
//! - anchors `LOOKS.jsonl` to `<git_toplevel>/docs/ultra/LOOKS.jsonl` with NO env override;
//! - acquires an exclusive file lock (`File::lock`), keys the one-read rule on the **resolved
//!   canonical loaded candidate identity** (never raw CLI aliases), and appends before inference;
//! - writes matching `public-test-read-aborted` records if a run fails after logging.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    raw_candidate: String,
    freeze_tag: String,
    freeze_tag_commit: String,
    build_commit: String,
    completion_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    abort_reason: Option<String>,
}

/// Active sealed-read receipt returned after logging, so downstream failure can emit an
/// `aborted` audit record (`S6` item 4).
#[derive(Debug, Clone)]
pub struct LoggedPublicTestRead {
    pub looks_path: PathBuf,
    pub look_id: String,
    pub board: String,
    pub kind: String,
    pub split: String,
    pub freeze_tag: String,
    pub freeze_tag_commit: String,
    pub build_commit: String,
    pub completion_status: String,
    pub candidates: Vec<(String, String)>, // (canonical_id, raw_arg)
}

fn git_in(repo: &Path, args: &[&str]) -> Option<String> {
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

pub fn git_repo_root() -> Result<PathBuf, CliError> {
    let cwd = std::env::current_dir()?;
    git_in(&cwd, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .ok_or_else(|| {
            CliError::Config(
                "cannot locate git repository root to anchor docs/ultra/LOOKS.jsonl (S6)".into(),
            )
        })
}

/// Guard and log every sealed (`Partition::PublicTest` and `Partition::DomainSealed` OOD) read for
/// `candidates` (`(canonical_id, raw_arg)`) on `split`, BEFORE any audio/NPZ bytes are opened.
/// No-op for `Partition::Dev` and `Partition::Train`.
#[allow(clippy::too_many_arguments)]
pub fn guard_and_log_public_test_reads(
    partition: Partition,
    split: &str,
    candidates: &[(String, String)],
    freeze_tag: &str,
    build_commit: &str,
    build_dirty: bool,
    completion_status: &str,
    sealed_read_flag: Option<&str>,
) -> Result<Option<LoggedPublicTestRead>, CliError> {
    if !matches!(partition, Partition::PublicTest | Partition::DomainSealed) {
        return Ok(None);
    }
    let board = if partition == Partition::PublicTest {
        "PUBLIC-TEST"
    } else {
        "PRIVATE-OOD"
    };
    let look_id = sealed_read_flag
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            std::env::var("ULTRA_SEALED_READ")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        })
        .ok_or_else(|| {
            CliError::Config(format!(
                "{board} split `{split}` requires --sealed-read <look-id> (or ULTRA_SEALED_READ) in EVERY mode (FREEZE.md §4.2 / S6)"
            ))
        })?;
    let repo_root = git_repo_root()?;
    let logged = guard_and_log_in_repo_for_partition(
        &repo_root,
        partition,
        &look_id,
        split,
        candidates,
        freeze_tag,
        build_commit,
        build_dirty,
        completion_status,
    )?;
    Ok(Some(logged))
}

/// Core S6 guard against an explicit `repo_root` (defaults to `Partition::PublicTest`).
#[allow(clippy::too_many_arguments)]
pub fn guard_and_log_in_repo(
    repo_root: &Path,
    look_id: &str,
    split: &str,
    candidates: &[(String, String)],
    freeze_tag: &str,
    build_commit: &str,
    build_dirty: bool,
    completion_status: &str,
) -> Result<LoggedPublicTestRead, CliError> {
    guard_and_log_in_repo_for_partition(
        repo_root,
        Partition::PublicTest,
        look_id,
        split,
        candidates,
        freeze_tag,
        build_commit,
        build_dirty,
        completion_status,
    )
}

/// Core S6 guard supporting both `Partition::PublicTest` and `Partition::DomainSealed` (OOD).
#[allow(clippy::too_many_arguments)]
pub fn guard_and_log_in_repo_for_partition(
    repo_root: &Path,
    partition: Partition,
    look_id: &str,
    split: &str,
    candidates: &[(String, String)],
    freeze_tag: &str,
    build_commit: &str,
    build_dirty: bool,
    completion_status: &str,
) -> Result<LoggedPublicTestRead, CliError> {
    let (board, kind) = if partition == Partition::DomainSealed {
        ("PRIVATE-OOD", "sealed-ood-read")
    } else {
        ("PUBLIC-TEST", "public-test-read")
    };
    let look_id = look_id.trim();
    if look_id.is_empty() {
        return Err(CliError::Config(format!(
            "{board} sealed read requires non-empty look_id (S6)"
        )));
    }
    if build_dirty || build_commit.is_empty() || build_commit == "unknown" {
        return Err(CliError::Config(format!(
            "{board} split `{split}` requires a clean committed binary (commit={build_commit}, dirty={build_dirty}; FREEZE.md §4.2 / S6)"
        )));
    }
    let tag_commit = git_in(
        repo_root,
        &[
            "rev-parse",
            "-q",
            "--verify",
            &format!("refs/tags/{freeze_tag}^{{commit}}"),
        ],
    )
    .ok_or_else(|| {
        CliError::Config(format!(
            "{board} split `{split}` is sealed until git tag `{freeze_tag}` exists in `{}` (FREEZE.md §4.2 / S6)",
            repo_root.display()
        ))
    })?;
    let is_ancestor = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["merge-base", "--is-ancestor", &tag_commit, build_commit])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !is_ancestor {
        return Err(CliError::Config(format!(
            "{board} split `{split}` refused: `{freeze_tag}` ({}) is not an ancestor of build_commit `{build_commit}` (S6)",
            &tag_commit[..tag_commit.len().min(12)]
        )));
    }

    let looks_path = repo_root.join("docs/ultra/LOOKS.jsonl");
    if let Some(parent) = looks_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut fh = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&looks_path)?;
    fh.lock()
        .map_err(|e| CliError::Resource(format!("lock {}: {e}", looks_path.display())))?;

    let existing = std::fs::read_to_string(&looks_path)?;
    for (canon, raw) in candidates {
        for line in existing.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if v.get("historical_reconstruction").and_then(|x| x.as_bool()) == Some(true) {
                continue;
            }
            let same_kind = v.get("kind").and_then(|x| x.as_str()) == Some(kind);
            let same_split = v.get("split").and_then(|x| x.as_str()) == Some(split);
            let same_cand = v.get("candidate").and_then(|x| x.as_str()) == Some(canon.as_str());
            let same_tag = v.get("freeze_tag").and_then(|x| x.as_str()) == Some(freeze_tag);
            if same_kind && same_split && same_cand && same_tag {
                return Err(CliError::Config(format!(
                    "refusing second {board} read for canonical candidate `{canon}` (arg `{raw}`) on `{split}` under `{freeze_tag}` (already logged in `{}`; FREEZE.md §4.2 / S6)",
                    looks_path.display()
                )));
            }
        }
    }

    let now_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    for (canon, raw) in candidates {
        let rec = LookRecord {
            kind: kind.into(),
            look_id: look_id.to_owned(),
            time_unix_s: now_s,
            who: "ultra-bench-suite".into(),
            board: board.into(),
            split: split.to_owned(),
            candidate: canon.clone(),
            raw_candidate: raw.clone(),
            freeze_tag: freeze_tag.to_owned(),
            freeze_tag_commit: tag_commit.clone(),
            build_commit: build_commit.to_owned(),
            completion_status: completion_status.to_owned(),
            abort_reason: None,
        };
        let line = serde_json::to_string(&rec).map_err(|e| CliError::Internal(e.to_string()))?;
        writeln!(fh, "{line}")?;
    }
    fh.flush()?;
    Ok(LoggedPublicTestRead {
        looks_path,
        look_id: look_id.to_owned(),
        board: board.into(),
        kind: kind.into(),
        split: split.to_owned(),
        freeze_tag: freeze_tag.to_owned(),
        freeze_tag_commit: tag_commit,
        build_commit: build_commit.to_owned(),
        completion_status: completion_status.to_owned(),
        candidates: candidates.to_vec(),
    })
}

/// Append a matching `*-aborted` audit entry if a logged sealed run fails downstream (`S6`).
pub fn record_aborted_public_test_reads(logged: &LoggedPublicTestRead, reason: &str) {
    let Ok(mut fh) = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&logged.looks_path)
    else {
        return;
    };
    let _ = fh.lock();
    let now_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let abort_kind = format!("{}-aborted", logged.kind);
    for (canon, raw) in &logged.candidates {
        let rec = LookRecord {
            kind: abort_kind.clone(),
            look_id: logged.look_id.clone(),
            time_unix_s: now_s,
            who: "ultra-bench-suite".into(),
            board: logged.board.clone(),
            split: logged.split.clone(),
            candidate: canon.clone(),
            raw_candidate: raw.clone(),
            freeze_tag: logged.freeze_tag.clone(),
            freeze_tag_commit: logged.freeze_tag_commit.clone(),
            build_commit: logged.build_commit.clone(),
            completion_status: logged.completion_status.clone(),
            abort_reason: Some(reason.to_owned()),
        };
        if let Ok(line) = serde_json::to_string(&rec) {
            let _ = writeln!(fh, "{line}");
        }
    }
    let _ = fh.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_repo() -> (tempfile::TempDir, String) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let run = |args: &[&str]| {
            let st = Command::new("git")
                .arg("-C")
                .arg(tmp.path())
                .args(args)
                .status()
                .expect("git");
            assert!(st.success(), "git {args:?} failed");
        };
        run(&["init", "-q"]);
        std::fs::write(tmp.path().join("README"), "x").expect("write");
        run(&["add", "README"]);
        run(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "init",
        ]);
        let commit = git_in(tmp.path(), &["rev-parse", "HEAD"]).expect("head");
        (tmp, commit)
    }

    #[test]
    fn public_test_guard_enforces_tag_clean_build_canonical_id_and_abort_log() {
        let (tmp, commit) = scratch_repo();
        let cands = vec![("nemotron-560:gguf=deadbeef".into(), "nemotron-560".into())];

        // S6(1): refused before bench-freeze-v1 tag exists, even in exploratory mode.
        assert!(guard_and_log_in_repo(
            tmp.path(),
            "look-001",
            "test-clean",
            &cands,
            "bench-freeze-v1",
            &commit,
            false,
            "exploratory",
        )
        .is_err());

        // Create freeze tag.
        assert!(Command::new("git")
            .arg("-C")
            .arg(tmp.path())
            .args(["tag", "bench-freeze-v1"])
            .status()
            .expect("tag")
            .success());

        // Dirty build still refused.
        assert!(guard_and_log_in_repo(
            tmp.path(),
            "look-001",
            "test-clean",
            &cands,
            "bench-freeze-v1",
            &commit,
            true,
            "qualified",
        )
        .is_err());

        // Historical reconstruction line must NOT block the first post-freeze read.
        let looks = tmp.path().join("docs/ultra/LOOKS.jsonl");
        std::fs::create_dir_all(looks.parent().expect("parent")).expect("mkdir");
        std::fs::write(
            &looks,
            "{\"kind\":\"public-test-read\",\"historical_reconstruction\":true,\"split\":\"test-clean\",\"candidate\":\"nemotron-560:gguf=deadbeef\",\"freeze_tag\":\"bench-freeze-v1\"}\n",
        )
        .expect("seed");

        let logged = guard_and_log_in_repo(
            tmp.path(),
            "look-001",
            "test-clean",
            &cands,
            "bench-freeze-v1",
            &commit,
            false,
            "qualified",
        )
        .expect("first post-freeze read passes");

        // S6(3): different CLI alias resolving to the same canonical_id MUST be refused.
        let alias = vec![(
            "nemotron-560:gguf=deadbeef".into(),
            "nemotron_en_q8_live_560".into(),
        )];
        let err = guard_and_log_in_repo(
            tmp.path(),
            "look-002",
            "test-clean",
            &alias,
            "bench-freeze-v1",
            &commit,
            false,
            "qualified",
        )
        .expect_err("alias second read must fail");
        assert!(err.to_string().contains("second PUBLIC-TEST read"));

        // S6(4): abort audit record appends cleanly.
        record_aborted_public_test_reads(&logged, "simulated downstream error");
        let text = std::fs::read_to_string(&looks).expect("read");
        assert!(text.contains("public-test-read-aborted"));
        assert!(text.contains("simulated downstream error"));

        // S6(OOD): Partition::DomainSealed (AMI / VoxPopuli) enforces the exact same tag/one-read guard.
        let ood_logged = guard_and_log_in_repo_for_partition(
            tmp.path(),
            Partition::DomainSealed,
            "look-ood-1",
            "ami-ihm",
            &cands,
            "bench-freeze-v1",
            &commit,
            false,
            "qualified",
        )
        .expect("first OOD read passes");
        assert_eq!(ood_logged.board, "PRIVATE-OOD");
        assert_eq!(ood_logged.kind, "sealed-ood-read");
        assert!(guard_and_log_in_repo_for_partition(
            tmp.path(),
            Partition::DomainSealed,
            "look-ood-2",
            "ami-ihm",
            &alias,
            "bench-freeze-v1",
            &commit,
            false,
            "qualified",
        )
        .is_err());
    }
}
