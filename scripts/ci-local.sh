#!/usr/bin/env bash
# Run every CI lane locally, in CI order.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
bash ops/ci/fast.sh
just security
just score
