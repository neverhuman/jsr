#!/usr/bin/env bash
# H53 launch preflight (read-only): prints READY / NOT READY for the trunk's launch gates G1-G8 (README A6).
# Runs from H0; touches nothing (git queries, sha256 of local files, nvidia-smi / ls over ssh, file reads).
#
#   h53_preflight.sh <hosts e.g. xbabe5,xbabe9> <gpus per host e.g. "0,1;0">
# Env: PIN_COMMIT, PIN_TREE_SHA256, INIT (path on the hosts), INIT_SHA256,
#      H50_RECEIPTS (space-separated h50_measure JSON paths on H0; each must say PASS on the tuned arm),
#      DP_DECISION (file on H0 holding "delay_penalty=<v>" + the chat timestamp of amendment 21's verdict),
#      RUN_NAME (must appear in fleet-sched's queue.json as a reservation), CA_PREREG (optional, for CA arms).
set -uo pipefail
HOSTS=${1:?hosts}; GPUSETS=${2:?gpu sets}
REPO=/home/ubuntu/jsr; Q=/mnt/xbabe3-external/jsr/fleet-sched/queue.json
ok=0; bad=0
pass() { echo "  PASS  $*"; ok=$((ok + 1)); }
fail() { echo "  FAIL  $*"; bad=$((bad + 1)); }
ssh_() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=6 "$@"; }

echo "G1 code: H44 merged and pinned"
if [ -n "${PIN_COMMIT:-}" ] && git -C "$REPO" branch -r --contains "$PIN_COMMIT" 2>/dev/null | grep -q 'origin/main'; then
  pass "PIN_COMMIT $PIN_COMMIT is on origin/main"
else fail "PIN_COMMIT unset or not on origin/main"; fi
if [ -n "${PIN_COMMIT:-}" ] && git -C "$REPO" cat-file -e "$PIN_COMMIT:python/ultra/train/ddp_loss.py" 2>/dev/null; then
  pass "per-objective DDP denominators (train/ddp_loss.py) present"
else fail "train/ddp_loss.py (H44) missing at PIN_COMMIT"; fi
if [ -n "${PIN_COMMIT:-}" ] && git -C "$REPO" show "$PIN_COMMIT:python/ultra/train/recipe_opts.py" 2>/dev/null \
     | grep -q encoder_blocks; then
  pass "patch 0001 (encoder_blocks) present (needed only for growth arms)"
else echo "  INFO  patch 0001 not in PIN_COMMIT (needed only for P1 / capacity branch)"; fi

echo "G2 transport receipts (h50-fleet): G2a single node = SHM-memcpy exact-sum / FP32 parity; G2b multi-node = + ladder T2X / T4 / T6"
if [ -n "${H50_RECEIPTS:-}" ]; then
  for f in $H50_RECEIPTS; do
    if [ -f "$f" ] && python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if 'PASS' in json.dumps(d.get('verdict', d)) else 1)" "$f"; then
      pass "$f says PASS"; else fail "$f missing or not PASS"; fi
  done
else fail "H50_RECEIPTS unset (ladder T2L / T2X [/ T4 / T6] on the tuned arm + exact-sum / FP32 parity)"; fi

echo "G3 trunk delay_penalty decided (amendment 21)"
if [ -f "${DP_DECISION:-/nonexistent}" ] && grep -qE '^delay_penalty=(0|0\.0|0\.005)$' "$DP_DECISION"; then
  pass "$(grep -E '^delay_penalty=' "$DP_DECISION")"; else fail "DP_DECISION file with delay_penalty=0|0.005 missing"; fi

IFS=',' read -r -a H <<< "$HOSTS"; IFS=';' read -r -a G <<< "$GPUSETS"
ref=""
for i in "${!H[@]}"; do
  h=${H[$i]}; g=${G[$i]:-0}
  echo "host $h (GPUs $g)"
  # G4 init digest on that host
  got=$(ssh_ "$h" "sha256sum '${INIT:-/nonexistent}' 2>/dev/null | cut -c1-64")
  [ -n "${INIT_SHA256:-}" ] && [ "$got" = "$INIT_SHA256" ] && pass "G4 init digest" || fail "G4 init digest ($got)"
  # G1 tree hash on that host (the lane worktree trainer-lane pins)
  th=$(ssh_ "$h" "cd ~/jsr/.agent/worktrees/trainer 2>/dev/null && git ls-files -co --exclude-standard python/ultra | grep -E '\.py\$' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64")
  [ -n "${PIN_TREE_SHA256:-}" ] && [ "$th" = "$PIN_TREE_SHA256" ] && pass "G1 tree hash" || fail "G1 tree hash ($th)"
  # G5 data manifest: names + sizes of the 960 h parquet shards, identical on every host
  man=$(ssh_ "$h" "cd ~/jsr-data/corpora/librispeech/hf_parquet/all 2>/dev/null && find train.clean.100 train.clean.360 train.other.500 -name '*.parquet' -printf '%p %s\n' | LC_ALL=C sort | sha256sum | cut -c1-16")
  if [ -z "$ref" ]; then ref=$man; fi
  [ -n "$man" ] && [ "$man" = "$ref" ] && pass "G5 960 h manifest $man" || fail "G5 manifest $man != $ref"
  # G6 GPUs empty enough (foreign memory < 15%) - read only
  for gg in ${g//,/ }; do
    pct=$(ssh_ "$h" "nvidia-smi -i $gg --query-gpu=memory.used,memory.total --format=csv,noheader,nounits" \
          | awk -F', *' '{printf "%d", 100 * $1 / $2}')
    [ -n "$pct" ] && [ "$pct" -lt 15 ] && pass "G6 GPU $gg ${pct}% used" || fail "G6 GPU $gg ${pct:-?}% used"
  done
  # G7 run dirs on the host's own disk (cloud hosts), never H0's root
  df=$(ssh_ "$h" "df -P ~/jsr-data | tail -1 | awk '{print \$4}'")
  [ "${df:-0}" -gt 104857600 ] && pass "G7 >100 GB free for run dirs" || fail "G7 free space ${df:-?} KB"
done

echo "G6 fleet-sched reservation"
if [ -f "$Q" ] && grep -q "${RUN_NAME:-__none__}" "$Q"; then pass "$RUN_NAME reserved in queue.json"
else fail "RUN_NAME not in $Q"; fi
echo "G8 cooldown-arm pre-registrations (CA runs only)"
if [ -z "${CA_PREREG:-}" ]; then echo "  INFO  no CA arm requested"
elif [ -f "$CA_PREREG" ]; then pass "$CA_PREREG"; else fail "CA pre-registration $CA_PREREG missing"; fi

echo; if [ "$bad" -eq 0 ]; then echo "READY ($ok checks)"; exit 0; else echo "NOT READY ($bad failing, $ok passing)"; exit 1; fi
