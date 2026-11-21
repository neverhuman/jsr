//! PUBLIC-TEST (test-*) guard: refused without the flag, without the freeze tag, from a dirty
//! build, or on a second read; permitted and logged once otherwise; DEV splits untouched.

use std::path::{Path, PathBuf};
use std::process::Command;

use ultra_engine::sealed::{guard, SealedRead, FREEZE_TAG};

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

/// A scratch repository with one commit; returns (dir, commit).
fn scratch() -> Option<(PathBuf, String)> {
    let d =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("sealed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).ok()?;
    git(&d, &["init", "-q"])?;
    std::fs::write(d.join("f"), "x").ok()?;
    git(&d, &["add", "f"])?;
    git(
        &d,
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "c",
        ],
    )?;
    let c = git(&d, &["rev-parse", "HEAD"])?;
    Some((d, c))
}

#[test]
fn public_test_reads_are_sealed_and_logged_once() {
    let Some((repo, commit)) = scratch() else {
        eprintln!("SKIPPED: no git");
        return;
    };
    let looks = repo.join("docs/ultra/LOOKS.jsonl");
    let test = vec!["test-clean".to_owned()];
    let r = |look: Option<&'static str>, dirty: &'static str| SealedRead {
        repo: &repo,
        looks: &looks,
        look_id: look,
        candidate: "ckpt+int8",
        build_commit: &commit,
        build_dirty: dirty,
    };
    // DEV is never touched, even without a tag or flag.
    assert!(guard(&["dev-clean".into()], &r(None, "true"))
        .unwrap_or_default()
        .is_empty());
    assert!(guard(&test, &r(None, "false")).is_err(), "no flag");
    assert!(
        guard(&test, &r(Some("L1"), "false")).is_err(),
        "no freeze tag"
    );
    git(&repo, &["tag", FREEZE_TAG]).expect("tag");
    assert!(guard(&test, &r(Some("L1"), "true")).is_err(), "dirty build");
    let recs = guard(&test, &r(Some("L1"), "false")).expect("permitted read");
    assert_eq!(recs.len(), 1);
    let text = std::fs::read_to_string(&looks).expect("LOOKS written");
    assert!(text.contains("\"public-test-read\"") && text.contains("\"L1\""));
    assert!(
        guard(&test, &r(Some("L2"), "false")).is_err(),
        "second read refused"
    );
    std::fs::remove_dir_all(&repo).ok();
}
