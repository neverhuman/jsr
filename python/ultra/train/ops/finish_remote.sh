#!/usr/bin/env bash
# finish_remote.sh <host> <remote_run_dir> <remote_jsr_data> <H0_copy_dir> <label> <final_step>
# Waits for the remote trainer to exit, scores the final checkpoint on full dev-clean and dev-other
# on that host's GPU (guarded, min-duty 0), copies model/ckpt/config/train.jsonl + sha256 to H0,
# verifies the hashes and posts the result to AGENT_CHAT.
set -uo pipefail
host=$1 R=$2 JD=$3 H=$4 label=$5 step=$6
S=$(printf '%08d' "$step")
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
until ! ssh -n "$host" "kill -0 \$(cat $R/PID)" 2>/dev/null; do sleep 120; done
sleep 30
ssh -n "$host" "cat > $R/final_eval.sh <<EOS
#!/usr/bin/env bash
cd ~/jsr/python/ultra
export JSR_DATA=$JD
for s in dev-clean dev-other; do
  taskset -c 0-15,64-79 nice -n 10 ~/jsr/.venv/bin/python -m train.evaluate --run-dir $R --checkpoint $R/ckpt-$S.pt --split \\\$s --limit-all --device cuda --threads 8 --dump $R/final_\\\$s.jsonl 2>/dev/null | grep '^{' >> $R/final_eval_full.jsonl
done
echo DONE >> $R/final_eval_full.log
EOS
chmod +x $R/final_eval.sh; setsid $R/final_eval.sh > $R/final_eval_full.log 2>&1 < /dev/null & P=\$!; sleep 2; PG=\$(ps -o pgid= -p \$P | tr -d ' '); cd ~/jsr/python/ultra; setsid nice -n 5 ~/jsr/.venv/bin/python -m ultra.fleet_guard --pgid \$PG --gpu 0 --gpu-target 0.74 --cpu-target 0.85 --min-duty 0.0 --log $R/final_eval_guard.jsonl > /dev/null 2>&1 < /dev/null &"
until ssh -n "$host" "grep -q DONE $R/final_eval_full.log" 2>/dev/null; do sleep 60; done
mkdir -p "$H"
ionice -c3 nice -n 10 rsync -rt --include="model-$S.*" --include="ckpt-$S.*" --include='config.json' --include='train.jsonl' --include='final_eval_full.jsonl' --include='final_dev-*.jsonl' --exclude='*' "$host:$R/" "$H/"
cd "$H" && for s in model-$S ckpt-$S; do [ "$(sha256sum $s.pt | cut -d' ' -f1)" = "$(cat $s.sha256)" ] && echo "OK $s" || echo "BAD $s"; done > sha_check.txt
summary=$(python3 - "$H/final_eval_full.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("; ".join(f"{r['split']} (n={r['n']}) **{r['wer']}%** (S {r['sub']} D {r['del']} I {r['ins']}; CTC {r['ctc_wer']}%)" for r in rows))
PY
)
printf '\n## %s | trainer-lane | %s final (auto): full DEV on ckpt-%s; hashed copy on H0\n- Full splits, greedy, chunk-masked, %s GPU (guarded): %s\n- Copy: `%s/` (model-%s, ckpt-%s, config.json, train.jsonl, per-utterance final_dev-*.jsonl); sha256: %s\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$label" "$S" "$host" "$summary" "$H" "$S" "$S" "$(tr '\n' ' ' < $H/sha_check.txt)" >> "$CHAT"
echo FINISHED > "$H/FINISH_DONE"
