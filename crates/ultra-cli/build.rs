//! Embeds compile-time git commit (`ULTRA_BUILD_GIT_COMMIT`) and workspace-wide worktree
//! dirtiness (`ULTRA_BUILD_GIT_DIRTY`, re-review `N3`).
//!
//! Mirrors `crates/ultra-engine/build.rs`: watches `.git/{HEAD,logs/HEAD,index,packed-refs}`,
//! workspace manifests, and EVERY workspace crate directory under `crates/*` so uncommitted edits
//! in `ultra-stream`, `ultra-frontend`, `ultra-engine`, etc. always trigger a rebuild and update
//! `ULTRA_BUILD_GIT_DIRTY`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git_str(args: &[&str]) -> Option<String> {
    let o = Command::new("git").args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

fn main() {
    let commit = git_str(&["rev-parse", "HEAD"])
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let dirty = match git_str(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) if s.is_empty() => "false",
        Some(_) => "true",
        None => "unknown",
    };
    println!("cargo:rustc-env=ULTRA_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=ULTRA_BUILD_GIT_DIRTY={dirty}");

    if let Some(gd) = git_str(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from) {
        for f in ["HEAD", "logs/HEAD", "index"] {
            let p = gd.join(f);
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
        let common_dir = git_str(&["rev-parse", "--git-common-dir"])
            .map(PathBuf::from)
            .unwrap_or_else(|| gd.clone());
        let packed = common_dir.join("packed-refs");
        if packed.exists() {
            println!("cargo:rerun-if-changed={}", packed.display());
        }
    }

    // Watch EVERY workspace crate directory (not just ultra-cli/src) and workspace manifests.
    let crates_dir = Path::new("..");
    if let Ok(rd) = std::fs::read_dir(crates_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.join("Cargo.toml").is_file() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    for f in [
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../../rust-toolchain.toml",
    ] {
        if Path::new(f).exists() {
            println!("cargo:rerun-if-changed={f}");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
