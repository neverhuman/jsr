#!/usr/bin/env bash
# A1: DEV WER curve of D2 on eval-only snapshots model-<step>.pt every 10k steps (CPU, cores 12-15).
set -uo pipefail
D2=/mnt/xbabe3-external/jsr/runs/train/D2_S_960h
REPO=/home/ubuntu/jsr/.agent/worktrees/trainer
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
OUT=$D2/dev_curve.jsonl
cd "$REPO/python/ultra"
while kill -0 "$(cat $D2/PID)" 2>/dev/null || ls $D2/model-*.sha256 >/dev/null 2>&1; do
  for m in $(ls $D2/model-*.sha256 2>/dev/null | sort); do
    step=$((10#$(basename "$m" .sha256 | sed 's/model-//')))
    (( step % 10000 == 0 )) || continue
    grep -q "\"step\": $step," "$OUT" 2>/dev/null && continue
    line=""
    for split in dev-clean dev-other; do
      r=$(CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu taskset -c 12-15 nice -n 10 \
          "$REPO/.venv/bin/python" -m train.evaluate --run-dir "$D2" --checkpoint "${m%.sha256}.pt" \
          --split $split --limit 400 --threads 4 2>/dev/null | grep '^{')
      echo "$r" >> "$OUT"
      line="$line $(echo "$r" | python3 -c 'import sys,json; r=json.loads(sys.stdin.read()); print(f"{r[\"split\"]} {r[\"wer\"]}% (S {r[\"sub\"]} D {r[\"del\"]} I {r[\"ins\"]}; CTC {r[\"ctc_wer\"]}%)")');"
    done
    printf '\n## %s | trainer-lane | A1 D2 DEV curve (auto): model-%08d.pt\n- 400 utts each, greedy, chunk-masked (== streaming):%s\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$step" "$line" >> "$CHAT"
  done
  kill -0 "$(cat $D2/PID)" 2>/dev/null || break
  sleep 300
done
