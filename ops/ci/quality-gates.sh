#!/usr/bin/env bash
# Every gate lane in CI order. `just required` and scripts/ci-local.sh run this;
# ops/git-hooks/pre-push runs it before a push.
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane quality-gates "just required"
bash ops/ci/fast.sh
ci_step "pytest"
just pytest
bash ops/ci/contract-drift.sh
bash tools/security-lane.sh
ci_step "jankurai audit"
just score
