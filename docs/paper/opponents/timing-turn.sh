#!/usr/bin/env bash
# Take one entry of the host's timing order and run its repetitions, one arm on the whole host.
#
#   OPP_ROOT=… OPP_MACHINE=zen2 bash timing-turn.sh ENTRY AGENT MINUTES < LINES
#
# Protocol (xbabe0:~/jsr-data/opponents/TIMING.order, lead 2026-11-22T21:28Z): entry N may start
# only when every earlier entry is in TIMING.done as DONE or SKIP; the lock is the directory
# TIMING.lock (mkdir is atomic) with an `owner` file; a hold lasts at most MINUTES; afterwards
# the lock is removed and `DONE N <UTC> <agent>` is appended. LINES are
# `SERVICE_CPUS OBSERVER_CPU ARM BUDGET ROUND [WORKLOAD [TRACE]]`, run in order. Every finished
# repetition gets a `timing.lock` file with the owner line, which marks it as a quiet-host row.
set -uo pipefail
entry=$1 agent=$2 minutes=$3
here=$(cd "$(dirname "$0")" && pwd)
root=${OPP_ROOT:?set OPP_ROOT}
lock="$root/TIMING.lock"
log="$root/bench-cross/logs/timing-$entry.log"
lines=$(cat)

earlier_done() {
    local number
    for ((number = 1; number < entry; number++)); do
        if ! grep -Eq "^(DONE|SKIP) $number( |\$)" "$root/TIMING.done" 2>/dev/null; then
            return 1
        fi
    done
}
until earlier_done && mkdir "$lock" 2>/dev/null; do
    sleep "${OPP_TIMING_POLL:-20}"
done
owner="$agent $entry pid=$$ start=$(date -u +%FT%TZ) until=$(date -u -d "+$minutes min" +%FT%TZ)"
printf '%s\n' "$owner" > "$lock/owner"
release() {
    rm -f "$lock/owner"
    rmdir "$lock" 2>/dev/null
}
trap release EXIT
printf '%s lock taken: %s\n' "$(date -u +%FT%TZ)" "$owner" >> "$log"
deadline=$(( $(date +%s) + minutes * 60 ))
while read -r service observer arm budget round workload trace; do
    if [[ -z $service || $service == \#* ]]; then
        continue
    fi
    left=$(( deadline - $(date +%s) ))
    if (( left < 60 )); then
        printf '%s hold exhausted before %s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$budget" "$round" >> "$log"
        break
    fi
    status=0
    OPP_SERVICE_CPUS=$service OPP_OBSERVER_CPU=$observer OPP_WORKLOAD=${workload:-panel730} OPP_TRACE=${trace:-0} \
        timeout "$left" bash "$here/run-opponent.sh" "$arm" "$budget" "$round" >> "$log" 2>&1 < /dev/null || status=$?
    suffix=""
    if [[ ${workload:-panel730} != panel730 ]]; then
        suffix="-$workload"
    fi
    run="$root/bench-cross/receipts/${OPP_MACHINE:?set OPP_MACHINE}/$arm-$budget$suffix/round-$round"
    if [[ $status == 0 && -f "$run/finished" ]]; then
        printf '%s\n' "$owner" > "$run/timing.lock"
    fi
    printf '%s %s %s %s %s status %s\n' "$(date -u +%FT%TZ)" "$arm" "$budget" "$round" "${workload:-panel730}" "$status" >> "$log"
done <<< "$lines"
release
trap - EXIT
printf 'DONE %s %s %s\n' "$entry" "$(date -u +%FT%TZ)" "$agent" >> "$root/TIMING.done"
printf '%s entry %s done\n' "$(date -u +%FT%TZ)" "$entry" >> "$log"
