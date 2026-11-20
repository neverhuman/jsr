#!/usr/bin/env bash
# CI fast lane; identical to `just fast` minus the local-only data-drive target dir.
set -euo pipefail
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
if [ -n "${CI:-}" ]; then export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"; fi
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -q
