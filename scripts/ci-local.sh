#!/usr/bin/env bash
# Run every CI lane locally, in CI order (same script as `just required`).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec bash ops/ci/quality-gates.sh
