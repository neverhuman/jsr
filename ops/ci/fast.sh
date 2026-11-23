#!/usr/bin/env bash
# CI fast lane: the Rust part of `just fast` (fmt, clippy -D warnings, tests), plus, on GitHub Actions only,
# the torch-free pytest lane (CI has no .venv; locally `just pytest` in quality-gates.sh covers pytest).
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane fast "bash ops/ci/fast.sh"
ci_step "cargo fmt --check"
cargo fmt --all -- --check
ci_step "cargo clippy -D warnings"
cargo clippy --workspace --all-targets -- -D warnings
ci_step "aarch64 cross-check"
if ! rustup target list --installed | grep -qx "aarch64-unknown-linux-gnu"; then
  rustup target add aarch64-unknown-linux-gnu
fi
cargo check --target aarch64-unknown-linux-gnu -p ultra-kernels -p ultra-engine
ci_step "cargo test"
cargo test --workspace -q
# Called from here rather than as its own workflow step: the repo token used by the agents cannot
# push changes to .github/workflows/ (no `workflow` scope).
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  bash ops/ci/pytest-light.sh
fi
