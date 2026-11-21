//! Embeds the compile-time git commit (`ULTRA_BUILD_GIT_COMMIT`) so `SuiteJobManifest`
//! distinguishes retained-binary build source from runtime launch checkout (`HEAD`).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    let commit = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| String::from("unknown"));
    println!("cargo:rustc-env=ULTRA_BUILD_GIT_COMMIT={commit}");
}
