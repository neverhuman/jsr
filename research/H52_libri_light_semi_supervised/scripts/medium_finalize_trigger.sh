#!/usr/bin/env bash
# H52 (H0): fire medium's finaliser on x5 at 100% policy coverage (post first), then post its results.
set -u
B=/mnt/xbabe3-external/jsr/librilight; EXPECT=${1:-960}
V11=7474f0835051f13a695346fbb0defdebdc13d364f5301e42f661b1b3cedb2168
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
last_n=-1; same=0
while :; do
  n=$(timeout 120 ssh -o BatchMode=yes xbabe5 "cd ~/jsr-data/librilight/shards/raw/librilight.medium/r0src && grep -l '\"code_sha256\": \"$V11\"' *.labeled 2>/dev/null | wc -l")
  log "accepted v1.1 units: ${n:-?} / $EXPECT"
  if [ -n "$n" ] && [ "$n" = "$last_n" ]; then same=$((same+1)); else same=0; fi; last_n=${n:--1}
  if [ -n "$n" ] && { [ "$n" -ge "$EXPECT" ] || { [ "$same" -ge 3 ] && [ "$n" -ge $((EXPECT * 9 / 10)) ]; }; }; then
    cov=$(timeout 1200 ssh -o BatchMode=yes xbabe5 "cd ~/jsr-data/librilight && nice -n 19 ionice -c3 venv/bin/python -I bin/coverage_medium.py shards/raw/librilight.medium shards/raw/librilight.medium/LABEL_POLICY.json")
    log "coverage: $cov"
    frac=$(echo "$cov" | python3 -c "import json,sys; print(json.load(sys.stdin)['r0_frac'])" 2>/dev/null)
    if [ "$frac" = "1.0" ]; then
      printf -- '- **Starting medium finaliser now on x5** (coverage %s). `finalize_medium_x5.sh` = `--allow-missing-student --label-policy LABEL_POLICY.json` (v1.1 only), `ionice -c3` + nice 19, pausing while the trunk data_wait > 5%%; ~400 GB peak extra disk. @fleet-sched @claude-lead @trainer-lane\n' "$cov" > $B/logs/body_medfin_start.md
      $B/bin/post.sh "H52 medium finaliser STARTING on x5 (R0 coverage 100% under the v1.1 policy)" $B/logs/body_medfin_start.md
      timeout 60 ssh -o BatchMode=yes xbabe5 "cd ~/jsr-data/librilight && setsid nohup bash bin/finalize_medium_x5.sh > logs/finalize_medium.log 2>&1 < /dev/null &"
      break
    fi
  fi
  sleep 120
done
posted_guard=0
until timeout 60 ssh -o BatchMode=yes xbabe5 "grep -q 'finalise medium done\|Traceback\|Error' ~/jsr-data/librilight/logs/finalize_medium.log"; do
  g=$(timeout 60 ssh -o BatchMode=yes xbabe5 "grep 'GUARD PAUSE\|GUARD RESUME\|GUARD v2' ~/jsr-data/librilight/logs/finalize_medium.log" 2>/dev/null)
  n=$(printf '%s\n' "$g" | grep -c GUARD)
  if [ "$n" -gt "$posted_guard" ]; then
    printf '%s\n' "$g" | tail -n $((n - posted_guard)) | sed 's/^/- /' > $B/logs/body_medfin_guard.md
    $B/bin/post.sh "H52 medium finaliser guard event (x5, lead's scoped exception: rate -5% / wait 8%)" $B/logs/body_medfin_guard.md
    posted_guard=$n
  fi
  sleep 120
done
timeout 60 scp -q xbabe5:jsr-data/librilight/shards/r0_v11/librilight.medium/FILTER_REPORT.json $B/research/H52_libri_light_semi_supervised/results/medium_FILTER_REPORT.json 2>/dev/null
timeout 60 scp -q xbabe5:jsr-data/librilight/shards/r0_v11/librilight.medium/SHA256SUMS $B/research/H52_libri_light_semi_supervised/results/medium_final_SHA256SUMS 2>/dev/null
tail=$(timeout 60 ssh -o BatchMode=yes xbabe5 "grep -E 'SHA256SUMS verified|finalise medium done|Traceback|Error' ~/jsr-data/librilight/logs/finalize_medium.log | tail -3")
/home/ubuntu/jsr/.venv/bin/python -I - "$tail" > $B/logs/body_medfin_done.md <<'PYEOF'
import json, sys, hashlib, os
d = '/mnt/xbabe3-external/jsr/librilight/research/H52_libri_light_semi_supervised/results/'
print('@claude-lead @trainer-lane @fleet-sched @eval-lane')
try:
    r = json.load(open(d + 'medium_FILTER_REPORT.json'))
    print(f"- **MEDIUM final shards** (semi-supervised (R0 labels)): **kept {r['rows_kept']:,} / {r['rows_in']:,} rows ({r['kept_fraction_rows']:.1%}) = {r['hours_kept']:,} / {r['hours_in']:,} h**; {r['output_shards']} shards at `x5:~/jsr-data/librilight/shards/r0_v11/librilight.medium/`.")
    print(f"- Drops: {r['drop_counts']}; S1 threshold {r['S1_lp_threshold']:.4f}; checks {r['checks']}.")
    sp = d + 'medium_final_SHA256SUMS'
    if os.path.exists(sp):
        print(f"- Manifest SHA256SUMS sha256 `{hashlib.sha256(open(sp, 'rb').read()).hexdigest()}`.")
except Exception as ex:
    print(f"- finaliser did not produce a report: {ex}")
print(f"- Log tail: {sys.argv[1]}")
PYEOF
$B/bin/post.sh "H52 RESULT medium finaliser done (kept hours, drops, SHA256SUMS)" $B/logs/body_medfin_done.md
