//! The git identity baked in by build.rs matches the live tree: the commit is HEAD and the
//! dirty flag equals `git status --porcelain --untracked-files=no` over the whole repository
//! (build.rs reruns on any workspace crate change, so an edit outside ultra-engine is seen).

use std::process::Command;

use ultra_engine::run_info::{GIT_COMMIT, GIT_DIRTY};

fn git(args: &[&str]) -> Option<String> {
    let o = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

#[test]
fn recorded_commit_and_dirty_flag_match_the_tree() {
    let (Some(head), Some(status)) = (
        git(&["rev-parse", "HEAD"]),
        git(&["status", "--porcelain", "--untracked-files=no"]),
    ) else {
        eprintln!("SKIPPED: no git");
        return;
    };
    assert_eq!(GIT_COMMIT, head);
    let live = if status.is_empty() { "false" } else { "true" };
    assert_eq!(GIT_DIRTY, live, "dirty flag stale; live status:\n{status}");
}
