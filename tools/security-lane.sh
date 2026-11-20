#!/usr/bin/env bash
# Security lane: secrets in the working tree, RustSec advisories, workflow and shell lint.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=ops/ci/lib.sh
. ops/ci/lib.sh
ci_lane security "just security"
for t in gitleaks cargo-audit actionlint shellcheck; do ci_require_tool "$t"; done
ci_step "gitleaks (working tree)"
gitleaks detect --no-banner --redact --source . --no-git
ci_step "cargo audit"
cargo audit --deny warnings --ignore yanked
ci_step "actionlint"
actionlint .github/workflows/*.yml
ci_step "shellcheck"
{ git ls-files '*.sh'; git ls-files ops/git-hooks tools/jankurai-hooks; } | sort -u | xargs -r shellcheck -x -S warning
echo "security lane: pass"
