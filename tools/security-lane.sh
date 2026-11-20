#!/usr/bin/env bash
# Security lane: secrets in the working tree, RustSec advisories, shell lint.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
gitleaks detect --no-banner --redact --source . --no-git
cargo audit --deny warnings --ignore yanked
git ls-files '*.sh' | xargs -r shellcheck -S warning
echo "security lane: pass"
