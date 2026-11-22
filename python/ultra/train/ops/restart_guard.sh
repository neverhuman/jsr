#!/usr/bin/env bash
# Restart only the fleet guard of a running trainer with a new GPU target (Linux hosts).
#   restart_guard.sh <run_dir> <gpu_target> [cpu_target=0.85]
# Kills every fleet_guard for the run's PGID (the guard SIGCONTs the group on exit), checks the
# trainer is not left stopped, starts one guard with the new target and records its real PID.
set -uo pipefail
R=$1 GT=$2 CT=${3:-0.85}
PG=$(cat "$R/PGID") P=$(cat "$R/PID")
ps -p "$P" -o args= | grep -q train.trainer || { echo "no trainer at $P"; exit 1; }
for g in $(pgrep -f "ultra.fleet_guard --pgid $PG "); do kill -TERM "$g"; done
for _ in $(seq 1 15); do pgrep -f "ultra.fleet_guard --pgid $PG " > /dev/null || break; sleep 1; done
pgrep -f "ultra.fleet_guard --pgid $PG " > /dev/null && { echo "old guard still alive"; exit 1; }
case "$(ps -o stat= -p "$P")" in T*) kill -CONT -- "-$PG"; echo "trainer was stopped; SIGCONT sent";; esac
cd "$HOME/jsr/python/ultra" || exit 1
setsid nice -n 5 "$HOME/jsr/.venv/bin/python" -m ultra.fleet_guard --pgid "$PG" --gpu 0 \
  --gpu-target "$GT" --cpu-target "$CT" --min-duty 0.0 --log "$R/fleet_guard.jsonl" \
  >> "$R/guard.log" 2>&1 < /dev/null &
sleep 2
G=$(pgrep -f "ultra.fleet_guard --pgid $PG " | head -1)
echo "$G" > "$R/GUARD_PID"
echo "guard $G target $GT; trainer $(ps -o stat= -p "$P")"
