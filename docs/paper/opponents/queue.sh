#!/usr/bin/env bash
# Run a list of repetitions one after another on one set of pinned CPUs.
#
#   OPP_ROOT=… OPP_MACHINE=… OPP_SERVICE_CPUS=… OPP_OBSERVER_CPU=… bash queue.sh NAME < LIST
#
# Each LIST line is `ARM BUDGET ROUND [WORKLOAD [TRACE]]`. A failed repetition is logged and
# the queue moves on; its directory keeps the evidence and has no `finished` file.
set -uo pipefail
name=$1
here=$(cd "$(dirname "$0")" && pwd)
log="${OPP_ROOT:?set OPP_ROOT}/bench-cross/logs/queue-$name.log"
while read -r arm budget round workload trace; do
    if [[ -z $arm || $arm == \#* ]]; then
        continue
    fi
    printf '%s start %s %s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$budget" "$round" "${workload:-panel730}" >> "$log"
    OPP_WORKLOAD=${workload:-panel730} OPP_TRACE=${trace:-0} \
        bash "$here/run-opponent.sh" "$arm" "$budget" "$round" >> "$log" 2>&1 < /dev/null
    printf '%s end %s %s %s status %s\n' "$(date -u +%FT%TZ)" "$arm" "$budget" "$round" "$?" >> "$log"
done
printf '%s queue %s done\n' "$(date -u +%FT%TZ)" "$name" >> "$log"
