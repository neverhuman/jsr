#!/usr/bin/env bash
# Run a benchmark plan: phases one after another, the CPU sets inside a phase side by side.
#
#   OPP_ROOT=… OPP_MACHINE=… bash plan.sh NAME < PLAN
#
# PLAN lines are `SERVICE_CPUS OBSERVER_CPU ARM BUDGET ROUND [WORKLOAD [TRACE]]`; a line `--`
# ends a phase. Inside a phase, lines with the same SERVICE_CPUS run in order and different
# CPU sets run concurrently, so a phase is written to keep each arm alone on its L3 group
# and the number of busy benchmark cores within the host's limit. The next phase starts
# when every repetition of the current one has ended. SERVICE_CPUS `-` (macOS) means no
# affinity; such lines always run in order.
set -uo pipefail
name=$1
here=$(cd "$(dirname "$0")" && pwd)
root=${OPP_ROOT:?set OPP_ROOT}
log="$root/bench-cross/logs/plan-$name.log"
tmp=$(mktemp -d "$root/bench-cross/logs/plan-$name.XXXXXX")
phase=0
flush() {
    local list pids=()
    for list in "$tmp/phase-$phase".*; do
        if [[ ! -f $list || $list == *.cpus ]]; then
            continue
        fi
        IFS=' ' read -r service observer < "$list.cpus"
        (
            if [[ $service != - ]]; then
                export OPP_SERVICE_CPUS=$service OPP_OBSERVER_CPU=$observer
            fi
            bash "$here/queue.sh" "$name-p$phase-${list##*.}" < "$list"
        ) &
        pids+=("$!")
    done
    if (( ${#pids[@]} )); then
        wait "${pids[@]}"
        printf '%s phase %s done\n' "$(date -u +%FT%TZ)" "$phase" >> "$log"
    fi
    phase=$((phase + 1))
}
while read -r service observer rest; do
    if [[ -z $service || $service == \#* ]]; then
        continue
    fi
    if [[ $service == -- ]]; then
        flush
        continue
    fi
    key=$(printf '%s' "$service" | tr ',' '_')
    printf '%s\n' "$rest" >> "$tmp/phase-$phase.$key"
    printf '%s %s\n' "$service" "$observer" > "$tmp/phase-$phase.$key.cpus"
done
flush
printf '%s plan %s done\n' "$(date -u +%FT%TZ)" "$name" >> "$log"
