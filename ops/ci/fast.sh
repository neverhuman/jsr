#!/usr/bin/env bash
# CI fast lane: the Rust part of `just fast` (fmt, clippy -D warnings, tests).
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane fast "bash ops/ci/fast.sh"
ci_step "cargo fmt --check"
cargo fmt --all -- --check
ci_step "cargo clippy -D warnings"
cargo clippy --workspace --all-targets -- -D warnings
ci_step "cargo test"
cargo test --workspace -q
