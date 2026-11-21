//! Verifies re-review `N3`: `ultra-cli`'s `build.rs` bakes in the live git HEAD commit and
//! workspace-wide dirtiness (`git status --porcelain --untracked-files=no`), covering edits
//! across every workspace crate (`ultra-stream`, `ultra-frontend`, `ultra-engine`, etc.).

use std::process::Command;

fn git(args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .output()
        .expect("git must be runnable in repo checkout");
    assert!(out.status.success(), "git {args:?} failed");
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

#[test]
fn embedded_build_commit_and_workspace_dirty_flag_match_live_git() {
    let embedded_commit = env!("ULTRA_BUILD_GIT_COMMIT");
    let embedded_dirty = env!("ULTRA_BUILD_GIT_DIRTY");

    let live_commit = git(&["rev-parse", "HEAD"]);
    let live_dirty = if git(&["status", "--porcelain", "--untracked-files=no"]).is_empty() {
        "false"
    } else {
        "true"
    };

    assert_eq!(embedded_commit, live_commit, "embedded commit drifted");
    assert_eq!(
        embedded_dirty, live_dirty,
        "embedded workspace dirty flag drifted from live git status"
    );
}
