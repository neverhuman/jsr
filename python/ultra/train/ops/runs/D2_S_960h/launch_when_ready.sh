#!/usr/bin/env bash
# Approved by lead 2026-11-20: launch D2 once a D1d dev-clean eval is < 25% WER.
set -uo pipefail
D1=/mnt/xbabe3-external/jsr/runs/train/D1d_S_preln_lr1e-3_b300_sa4k
D2=/mnt/xbabe3-external/jsr/runs/train/D2_S_960h
REPO=/home/ubuntu/jsr/.agent/worktrees/trainer
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
log() { echo "$(date -u +%FT%TZ) $*" >> "$D2/launcher.log"; }
log "waiting for a D1d eval with WER < 25"
while true; do
  hit=$(python3 - "$D1/train.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    if '"kind": "eval"' in l:
        r = json.loads(l)
        if r.get("wer", 100) < 25:
            print(r["step"], r["wer"]); break
PY
)
  [ -n "$hit" ] && break
  sleep 60
done
log "trigger: D1d eval step/wer $hit; stopping D1d"
P=$(cat "$D1/PID")
kill -TERM "$P"
while kill -0 "$P" 2>/dev/null; do sleep 5; done
CK="$D1/$(cat "$D1/LATEST").pt"
log "D1d stopped; warm start from $CK"
cd "$REPO/python/ultra"
taskset -c 0-11,64-75 nice -n 10 "$REPO/.venv/bin/python" -m train.trainer --run-dir "$D2" --preset S \
  --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 8 --init-from "$CK" \
  --set budget_s=600.0 max_batch=128 workers=8 total_steps=102000 warmup=500 lr=0.001 specaug_start=0 \
  eval_every=5000 dev_limit=400 ckpt_every_steps=5000 weights_every=5000 log_every=50 \
  > "$D2/stdout.log" 2>&1 &
echo $! > "$D2/PID"
log "D2 launched pid $(cat "$D2/PID")"
printf '\n## %s | trainer-lane | D2 launched (auto, on approved trigger)\n- Trigger: D1d dev-clean eval (step, WER) = %s < 25%%. D1d stopped; its final checkpoint `%s` and its full eval curve (`%s/train.jsonl`) are kept as R1-D1.\n- D2: S, all three train splits (~960 h), warm start from that checkpoint, lr 1e-3, warmup 500, 600 s buckets, SpecAugment from step 0, 102k steps (~15 epochs). Eval-only weights `model-<step>.pt` every 5000 steps (never rotated); dev-clean WER auto-posted here per eval. pid %s, run dir `%s`. Projected finish follows once throughput is steady.\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$hit" "$CK" "$D1" "$(cat "$D2/PID")" "$D2" >> "$CHAT"
nohup nice -n 10 python3 "$D1/post_evals.py" "$D2" "$CHAT" 0 > "$D2/post_evals.log" 2>&1 &
echo $! > "$D2/POSTER_PID"
log "poster pid $(cat "$D2/POSTER_PID")"
