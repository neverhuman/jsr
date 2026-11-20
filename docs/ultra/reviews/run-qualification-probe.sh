#!/usr/bin/env bash
# Standalone reviewer diagnostics. Exit 0 covers these probes, not qualification.
set -Eeuo pipefail

if [[ "$(rtk git rev-parse --show-toplevel)" != "$PWD" ]]; then
  echo "Run from the active repository root." >&2
  exit 2
fi

rtk proxy env CARGO_BUILD_JOBS=8 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo build -p ultra-core -p ultra-eval --lib
review_target=$(rtk proxy cargo metadata --no-deps --format-version 1 | jq -er .target_directory)
review_binary="$review_target/debug/codex_qualification_probe"
rtk proxy rustc --edition=2021 docs/ultra/reviews/qualification_probe.rs \
  --extern "ultra_core=$review_target/debug/libultra_core.rlib" \
  --extern "ultra_eval=$review_target/debug/libultra_eval.rlib" \
  -L "dependency=$review_target/debug/deps" -o "$review_binary"
rtk proxy "$review_binary"
