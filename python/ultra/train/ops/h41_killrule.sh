#!/usr/bin/env bash
# H41 kill rules for one arm (amendments 9-17): stop it by its recorded PIDs if
#  (a) dev-clean400 EMA at chunk 4 at step 2,000 > threshold (comparator + 1.5 pp: 6.605 vs arm 10, 6.592 vs the chunk arm), or
#  (b) raw chunk-4 WER at any eval > 20 % (a collapse like arm 13b's).
# Survives restarts (reads the run's train.jsonl). Exits when the run reaches step 8,000 or after a stop.
# Runs on H0 and reads the arm over ssh; posts to AGENT_CHAT when it stops an arm.
#   h41_killrule.sh <host> <run_dir> <label> [threshold, default 6.605]
H=$1 R=$2 LABEL=$3 THR=${4:-6.605}
CHAT=${AGENT_CHAT:-$HOME/jsr/AGENT_CHAT.md}
while :; do
  out=$(ssh -n -o ConnectTimeout=20 $H "python3 - <<PY
import json
bad, done = '', False
for l in open('$R/train.jsonl'):
    if '\"kind\": \"eval\"' in l:
        r = json.loads(l)
        if r.get('wer') is not None and r['wer'] > 20.0:
            bad = 'raw chunk-4 WER %s at step %s' % (r['wer'], r['step'])
        if r['step'] == 2000 and r.get('ema_wer') is not None and r['ema_wer'] > $THR:
            bad = 'EMA chunk-4 WER %s at step 2000 (threshold $THR)' % r['ema_wer']
    elif '\"kind\": \"train\", \"step\": 8000' in l:
        done = True
print('BAD:' + bad if bad else ('DONE' if done else 'OK'))
PY" 2>/dev/null)
  case "$out" in
    BAD:*)
      ssh -n $H "P=\$(cat $R/PID); G=\$(cat $R/GUARD_PID); kill \$G 2>/dev/null; sleep 1; kill -CONT \$P 2>/dev/null; kill -TERM \$P; for i in \$(seq 1 90); do kill -0 \$P 2>/dev/null || break; sleep 1; done"
      printf '\n## %s | trainer-lane | KILL RULE (auto): %s stopped by its recorded PID\n- %s\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$LABEL" "${out#BAD:}" >> $CHAT
      echo "KILLED $LABEL: ${out#BAD:}"; exit 0;;
    DONE) echo "DONE $LABEL (no kill rule tripped)"; exit 0;;
  esac
  sleep 60
done
