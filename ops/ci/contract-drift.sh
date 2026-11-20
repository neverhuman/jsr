#!/usr/bin/env bash
# Contract drift lane: the ledger contract tests and the JSON-schema check of
# serialised ultra-core values against contracts/events/*.schema.json.
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane contract-drift "just contract-drift"
ci_step "output contract"
cargo test -p ultra-core --test output_contract -q
ci_step "schema contract"
cargo test -p ultra-core --test schema_contract -q
