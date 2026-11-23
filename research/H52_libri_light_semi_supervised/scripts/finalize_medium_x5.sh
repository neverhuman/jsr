#!/usr/bin/env bash
# H52: finalise MEDIUM on xbabe5 once R0 labels (raw/librilight.medium/r0src/, from h52-labeler) and student
# hyps (raw/librilight.medium/student_src/, relayed) cover every row. Strict by default: the finaliser refuses
# rows without an R0 label; pass "--allow-unlabeled" / "--allow-missing-student" only as an explicit decision.
# Disk: raw 220 GB exists; buckets + final need ~400 GB more at peak (x5 had 506 GB free at 08:22Z).
set -euo pipefail
L=$HOME/jsr-data/librilight; cd $L
EXTRA=${*:---allow-missing-student --label-policy shards/raw/librilight.medium/LABEL_POLICY.json}
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise medium start (extra: ${EXTRA:-none})"
nice -n 19 ionice -c3 $L/venv/bin/python -I bin/ll_finalize.py --raw shards/raw/librilight.medium --out shards/r0_v11 \
  --subset medium --rows-per-shard 2000 --guard-runs "~/jsr-data/runs/train/*/train.jsonl" $EXTRA
( cd shards/r0_v11/librilight.medium && sha256sum -c --quiet SHA256SUMS && echo "SHA256SUMS verified: $(wc -l < SHA256SUMS) shards" )
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise medium done"
