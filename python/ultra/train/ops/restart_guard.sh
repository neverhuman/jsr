#!/usr/bin/env bash
# Restart only the fleet guard of a running trainer with a new GPU target (Linux hosts).
#   restart_guard.sh <run_dir> <gpu_target> [cpu_target=0.85] [gpu_index=0]
# Kills every fleet_guard for the run's PGID (the guard SIGCONTs the group on exit), checks the
# trainer is not left stopped, starts one guard with the new target and records its real PID.
set -uo pipefail
R=$1 GT=$2 CT=${3:-0.85} GI=${4:-0}
PG=$(cat "$R/PGID") P=$(cat "$R/PID")
ps -p "$P" -o args= | grep -q train.trainer || { echo "no trainer at $P"; exit 1; }
guards() {  # python processes running this run's guard; never a shell whose command line quotes it
  local p
  for p in $(pgrep -x 'python[0-9.]*'); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "-m ultra.fleet_guard --pgid $PG " && echo "$p"
  done
}
for g in $(guards); do kill -TERM "$g"; done
for _ in $(seq 1 15); do [ -z "$(guards)" ] && break; sleep 1; done
[ -n "$(guards)" ] && { echo "old guard still alive"; exit 1; }
case "$(ps -o stat= -p "$P")" in T*) kill -CONT -- "-$PG"; echo "trainer was stopped; SIGCONT sent";; esac
# this script's own checkout (so a lane's fleet_guard fix applies); the guard is stdlib-only
cd "$(dirname "${BASH_SOURCE[0]}")/../../" || exit 1
setsid nice -n 5 "${GUARD_PY:-python3}" -m ultra.fleet_guard --pgid "$PG" --gpu "$GI" \
  --gpu-target "$GT" --cpu-target "$CT" --min-duty 0.0 --log "$R/fleet_guard.jsonl" \
  >> "$R/guard.log" 2>&1 < /dev/null &
sleep 2
G=$(guards | head -1)
echo "$G" > "$R/GUARD_PID"
echo "guard $G target $GT; trainer $(ps -o stat= -p "$P")"
