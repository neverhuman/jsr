#!/usr/bin/env bash
# Security lane: secrets in the working tree, RustSec and PyPI advisories, workflow and shell lint.
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
ci_step "pip-audit (python/ultra environment)"
# Audit what is installed in the trainer venv, not a fresh resolve: that is the
# code that actually runs. Without a venv there is nothing installed to audit.
site_packages=$(compgen -G ".venv/lib/python3*/site-packages" | head -n 1 || true)
if [ -n "$site_packages" ]; then
  ci_require_tool uv "https://docs.astral.sh/uv/"
  uv tool run --from "pip-audit==$JSR_PIP_AUDIT_VERSION" pip-audit --path "$site_packages" --progress-spinner off
else
  echo "pip-audit: no .venv in this checkout (just venv); Python dependencies not audited here"
fi
ci_step "actionlint"
actionlint .github/workflows/*.yml
ci_step "shellcheck"
{ git ls-files '*.sh'; git ls-files ops/git-hooks tools/jankurai-hooks; } | sort -u | xargs -r shellcheck -x -S warning
echo "security lane: pass"
