#!/usr/bin/env bash
# trainer-lane: when RC1 (H53_RC1_s7919) reaches its total_steps and stops, hash its end checkpoint from the protected dir
# (keep_ckpts keeps the latest 2) and post the path + sha256 for eval-lane (candidate = EMA unless raw beats it by >= 0.05 pp on
# the selection half, Amendment 7). Run on H0, detached, right after rc1_launch.sh.
set -u
D=/mnt/xbabe3-external/jsr/trainer-lane/s2; ST=$D/state; RUN=H53_RC1_s7919; R=/home/ubuntu/jsr-data/runs/train/$RUN
TS=$(sed -n 's/.*RC1_STEPS=\([0-9]*\).*/\1/p' $ST/rc1_env.txt)
N=$(printf 'ckpt-%08d' "$TS"); P=/home/ubuntu/jsr-data/protected/$RUN/$N.pt
until ssh -n -o BatchMode=yes xbabe5 "[ -e $P ] && [ -e ${P%.pt}.sha256 ]" 2>/dev/null; do sleep 30; done
H=$(ssh -n -o BatchMode=yes xbabe5 "sha256sum $P | cut -c1-64"); M=$(ssh -n -o BatchMode=yes xbabe5 "cut -c1-64 ${P%.pt}.sha256")
L=$(ssh -n -o BatchMode=yes xbabe5 "grep '\"kind\": \"train\"' $R/train.jsonl | tail -1 | cut -c1-200")
$D/tools/post.sh "RC1 ENDPOINT READY (auto): $N, sha256 ${H:0:16}... @eval-lane @claude-lead @bench-open" <<B
- **Path:** xbabe5:$P (hard link of the run's final checkpoint; marker $( [ "$H" = "$M" ] && echo matches || echo "MISMATCH $M" )). sha256 **$H**.
- The payload carries both the raw weights (model) and the EMA (ema, tau 0.25 D). Per Amendment 7 both are scored on the selection half (dev_split_v2, chunk 4, canonical); the EMA is the candidate unless raw is better there by at least 0.05 pp.
- Last train record: $L
B
