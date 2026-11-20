#!/usr/bin/env bash
# Shared helpers for the ops/ci lanes. Source it; do not execute it.
# The tool pins below are the versions the lanes were last proven with. CI
# installs exactly these; scripts/ci-doctor.sh reports drift on a local host.
set -euo pipefail

export JSR_GITLEAKS_VERSION="8.21.2"
export JSR_CARGO_AUDIT_VERSION="0.22.1"
export JSR_ACTIONLINT_VERSION="1.7.12"
export JSR_SHELLCHECK_VERSION="0.9.0"
export JSR_PIP_AUDIT_VERSION="2.9.0"

# HOST_RULES build settings, unless the caller already chose.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}"

# Hosted CI has no data drive, so build into the checkout there.
if [ -n "${CI:-}" ]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
fi

ci_repo_root() {
  git rev-parse --show-toplevel
}

ci_step() {
  printf '==> %s\n' "$*"
}

ci_fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

# ci_lane <name> <rerun>: on any failure, print which lane failed, the exact
# command that reproduces it, and where the proof routing lives.
ci_lane() {
  CI_LANE_NAME="$1"
  CI_LANE_RERUN="$2"
  trap 'ci_lane_failed $?' ERR
}

ci_lane_failed() {
  printf 'FAIL lane=%s rc=%s\nrerun: %s\nrouting: agent/test-map.json; error catalogue: docs/errors.md\n' \
    "$CI_LANE_NAME" "$1" "$CI_LANE_RERUN" >&2
}

# ci_require_tool <command> [hint]: fail with an install hint when a tool is absent.
ci_require_tool() {
  local tool="$1" hint="${2:-see scripts/ci-doctor.sh}"
  command -v "$tool" >/dev/null 2>&1 || ci_fail "$tool not found on PATH ($hint)"
}
