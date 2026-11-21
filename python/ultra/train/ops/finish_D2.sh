#!/usr/bin/env bash
# C1: when D2 (dense, xbabe3) ends, score the final checkpoint on full dev-clean and dev-other on
# the xbabe3 GPU (guarded, min-duty 0, cores 0-11) and post the result.
set -uo pipefail
D2=/mnt/xbabe3-external/jsr/runs/train/D2_S_960h
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
REPO=/home/ubuntu/jsr/.agent/worktrees/trainer
until ! kill -0 "$(cat $D2/PID)" 2>/dev/null; do sleep 120; done
sleep 30
CK="$D2/$(cat $D2/LATEST).pt"
cat > $D2/final_eval.sh <<EOS
#!/usr/bin/env bash
cd $REPO/python/ultra
for s in dev-clean dev-other; do
  taskset -c 0-11,64-75 nice -n 10 $REPO/.venv/bin/python -m train.evaluate --run-dir $D2 --checkpoint $CK --split \$s --limit-all --device cuda --threads 8 --dump $D2/final_\$s.jsonl 2>/dev/null | grep '^{' >> $D2/final_eval_full.jsonl
done
echo DONE >> $D2/final_eval_full.log
EOS
chmod +x $D2/final_eval.sh
setsid $D2/final_eval.sh > $D2/final_eval_full.log 2>&1 < /dev/null &
sleep 2
PG=$(ps -eo pgid,args | grep "[f]inal_eval.sh" | awk '{print $1}' | grep -v "^$" | tail -1)
cd /home/ubuntu/jsr/python/ultra && setsid nice -n 5 /home/ubuntu/jsr/.venv/bin/python -m ultra.fleet_guard --pgid "$PG" --gpu 0 --gpu-target 0.74 --cpu-target 0.85 --min-duty 0.0 --log $D2/final_eval_guard.jsonl > /dev/null 2>&1 < /dev/null &
until grep -q DONE $D2/final_eval_full.log 2>/dev/null; do sleep 60; done
summary=$(python3 - "$D2/final_eval_full.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("; ".join(f"{r['split']} (n={r['n']}) **{r['wer']}%** (S {r['sub']} D {r['del']} I {r['ins']}; CTC {r['ctc_wer']}%)" for r in rows))
PY
)
printf '\n## %s | trainer-lane | C1 D2 (dense) final (auto): full DEV on %s\n- Full splits, greedy, chunk-masked, xbabe3 GPU (guarded): %s\n- Per-utterance: `%s/final_dev-{clean,other}.jsonl`. Stratified-panel score follows when the panel is committed.\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$(basename $CK)" "$summary" "$D2" >> "$CHAT"
echo FINISHED > $D2/FINISH_DONE
