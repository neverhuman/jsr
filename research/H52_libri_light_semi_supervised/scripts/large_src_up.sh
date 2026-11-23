#!/usr/bin/env bash
# H52 large tier A FROM SOURCE on one host: N R0 servers (v1.1 only, frame-KD on) + M feature fronts that fetch tier-A
# FLAC members from the official source by byte range, sha256-check them and re-cut them with h52's own ll_segment /
# ll_pass2 (the versions pass 2 runs on x6), so ids / t0 / t1 / PCM equal x6's shards. Prints one pids JSON per process.
# usage (on the host): large_src_up.sh <base> <n servers> <front workers> <n fronts>
set -euo pipefail
B=$1; NS=${2:-2}; FW=${3:-16}; NF=${4:-1}
test "$(sha256sum $B/kdcode/r0_gpu.py | cut -c1-64)" = 7474f0835051f13a695346fbb0defdebdc13d364f5301e42f661b1b3cedb2168 || { echo "kdcode not v1.1"; exit 3; }
test "$(sha256sum $B/h52bin-large/ll_pass2.py | cut -c1-64)" = f7699a787bc13ea83197e5a7a7767640a99fd98f366f0aa1c63d9fb9d5ab5d3e || { echo "ll_pass2 mismatch"; exit 3; }
test "$(sha256sum $B/h52bin-large/ll_segment.py | cut -c1-64)" = 917e28c35add6fa22538a56ad946e1274c08bf7c410e54003ff4f5f6276bc0ed || { echo "ll_segment mismatch"; exit 3; }
mkdir -p $B/L-src $B/L-stage $B/L-feats $B/L-out
export H52LAB_CODE=$B/code-next H52LAB_H52BIN=$B/h52bin-large
for i in $(seq 1 $NS); do
  case $i in 1) mf=1000000; mg=8;; 2) mf=800000; mg=6;; *) mf=600000; mg=5;; esac
  bash $B/code-next/launch_worker.sh $B Lsrc-cons$i gpu 0 0.88 0.80 -- serve --feats $B/L-feats --out $B/L-out \
    --max-frames $mf --mem-gb $mg --torch-frac 0.08 --frames 1 --beam 8
done
for i in $(seq 1 $NF); do
  bash $B/code-next/launch_worker.sh $B Lsrc-prod$i front -1 0.88 0.80 -- serve --inbox $B/L-src --stage $B/L-stage \
    --workers $FW --ahead 4 --max-unshipped 3
done
