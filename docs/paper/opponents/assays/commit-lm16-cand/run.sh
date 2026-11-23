#!/usr/bin/env bash
# bench-open 14p, the CANDIDATE timing hold (lead 13:00:49Z): the release decoder exactly as in 14m's commit-lm16 (c4 greedy
# stream + c16 beam 16 + M5.ulm / LODR 0.80 / 0.60, 320 ms tail pad, 40 s cap; ../commit-pass/run.sh) on the release candidate's
# INT8 export, staged at models/candidate-rc/int8 (its weights.int8 hash is in OUT/inputs.sha256).
#   run.sh ROUND CPU OUT
set -euo pipefail
opp=${PAPER_OPPONENTS_ROOT:-"$HOME/jsr-data/opponents"}
COMMIT_VARIANT=commit-lm16 COMMIT_MODEL=${COMMIT_MODEL:-"$opp/models/candidate-rc/int8"} \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/commit-pass/run.sh" "$@"
