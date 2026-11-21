//! Embeds compile-time git commit (`ULTRA_BUILD_GIT_COMMIT`) and worktree dirtiness
//! (`ULTRA_BUILD_GIT_DIRTY`) while watching resolved git refs (`HEAD`, symbolic branch ref,
//! `packed-refs`, and `Cargo.lock`) so attached-branch commits always trigger rebuilds.

use std::path::PathBuf;
use std::process::Command;

fn git_str(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn main() {
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=src");
    if let Some(git_dir) = git_str(&["rev-parse", "--git-dir"]).map(PathBuf::from) {
        let head_path = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head_path.display());
        let common_dir = git_str(&["rev-parse", "--git-common-dir"])
            .map(PathBuf::from)
            .unwrap_or_else(|| git_dir.clone());
        if let Ok(head_txt) = std::fs::read_to_string(&head_path) {
            if let Some(sym_ref) = head_txt.trim().strip_prefix("ref: ") {
                println!(
                    "cargo:rerun-if-changed={}",
                    common_dir.join(sym_ref.trim()).display()
                );
            }
        }
        println!(
            "cargo:rerun-if-changed={}",
            common_dir.join("packed-refs").display()
        );
    }

    let commit = git_str(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git_str(&["status", "--porcelain", "--untracked-files=no"]).is_some();
    println!("cargo:rustc-env=ULTRA_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=ULTRA_BUILD_GIT_DIRTY={dirty}");
}
