//! Records the git commit, and whether any tracked file of the repository differs from it, in
//! the binaries, so every eval/density output names the source it was built from.
//!
//! The dirty flag is `git status --porcelain --untracked-files=no` over the whole repository
//! at build time. Cargo reruns this script when HEAD, the index or the reflog move, and when
//! anything changes in any workspace crate (not only this one: an uncommitted edit in
//! ultra-kernels must not produce a binary that claims to be clean) or the workspace
//! manifests. `tests/build_identity.rs` checks the recorded flag against the live tree.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let o = Command::new("git").args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) if s.is_empty() => "false",
        Some(_) => "true",
        None => "unknown",
    };
    println!("cargo:rustc-env=ULTRA_ENGINE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=ULTRA_ENGINE_GIT_DIRTY={dirty}");
    if let Some(gd) = git(&["rev-parse", "--absolute-git-dir"]) {
        for f in ["HEAD", "logs/HEAD", "index"] {
            println!("cargo:rerun-if-changed={gd}/{f}");
        }
    }
    // Every workspace crate (a directory path makes cargo watch every file below it), and the
    // workspace manifests.
    let crates = Path::new("..");
    if let Ok(rd) = std::fs::read_dir(crates) {
        for e in rd.flatten() {
            let p = e.path();
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
