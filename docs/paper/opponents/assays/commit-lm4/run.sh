#!/usr/bin/env bash
# bench-open 14m, variant commit-lm4: see ../commit-pass/run.sh (staged beside this folder).
#   run.sh ROUND CPU OUT
set -euo pipefail
COMMIT_VARIANT=commit-lm4 exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/commit-pass/run.sh" "$@"
