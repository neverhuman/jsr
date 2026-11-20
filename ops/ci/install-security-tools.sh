#!/usr/bin/env bash
# Install the security-lane and SBOM tools at the versions pinned in ops/ci/lib.sh.
# Used by hosted CI. Go modules are verified against sum.golang.org and
# cargo install --locked uses the crate's published lockfile.
set -euo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
ci_require_tool go "hosted runners ship Go"
ci_require_tool cargo
ci_step "gitleaks v$JSR_GITLEAKS_VERSION"
go install "github.com/zricethezav/gitleaks/v8@v$JSR_GITLEAKS_VERSION"
ci_step "actionlint v$JSR_ACTIONLINT_VERSION"
go install "github.com/rhysd/actionlint/cmd/actionlint@v$JSR_ACTIONLINT_VERSION"
ci_step "cargo-audit $JSR_CARGO_AUDIT_VERSION"
cargo install --locked cargo-audit --version "$JSR_CARGO_AUDIT_VERSION"
ci_step "cargo-cyclonedx $JSR_CARGO_CYCLONEDX_VERSION"
cargo install --locked cargo-cyclonedx --version "$JSR_CARGO_CYCLONEDX_VERSION"
ci_require_tool shellcheck "apt-get install shellcheck"
