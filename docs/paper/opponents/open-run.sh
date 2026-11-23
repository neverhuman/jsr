#!/usr/bin/env bash
# One guarded repetition of an open opponent arm on the benchmark host (bench-open).
# Run from the repository root (or a deployed copy that keeps the repository paths):
#
#   [OPEN_PANEL=full|p73] bash docs/paper/opponents/open-run.sh ARM ROUND SERVICE_CPUS OBSERVER_CPU
#
# ARM            an arm of open-arms.sh, e.g. vosk-small, whispercpp-stream-base-w8
# ROUND          repetition number (1, 2, 3)
# SERVICE_CPUS   one CPU for the one-core budget, eight for the eight-core budget ("8-15")
# OBSERVER_CPU   CPU of the meter, the scorer and this script, outside the service CCX
# OPEN_PANEL     full = the 730-utterance panel, timed and scored in the same run (default);
#                p73  = every tenth utterance (cost only: slow systems take their WER from one
#                       full-panel pass elsewhere, see open_summary.py)
#
# A full-panel one-core repetition goes through docs/paper/run-cold.sh unchanged. Every other
# repetition follows the same steps here, with a CPU list and the chosen panel. Around either,
# this script
#   - waits while $PAPER_OPPONENTS_ROOT/QUIET.lock (`energy until=<UTC>`) is in force, and shows
#     that it is waiting with a QUIET.waiting.bench-open.<pid> file beside the lock,
#   - waits (at most OPEN_GATE_WAIT_S) until the host is quiet: no more than OPEN_OTHER_CORES
#     other cores busy, since the memory bus is shared and a busy host slows a memory-bound arm,
#   - records per-CPU busy time and the service CPUs' clock frequency for the repetition,
#   - sets aside a repetition that was not alone on its CCX or on the host (the caller repeats),
#   - records the model size and the timing window, and summarises, hashes and compresses the
#     partial-result trace.
# Receipts: $OPEN_RUNS/{1c,8c}[-p73]/cold/round-ROUND/ARM/. The host-wide TIMING.lock is taken by
# the caller (open_turn.py); this script does not touch it.
set -euo pipefail

# Receipts name the machine by role, never by address (the repository is public). A cloud VM's hostname
# is its public IP (a-b-c-d), so the role comes from JSR_HOST_ROLE or ~/.config/jsr/host-role (untracked).
host_role() {
    local role=${JSR_HOST_ROLE:-}
    if [[ -z $role && -r $HOME/.config/jsr/host-role ]]; then
        role=$(head -n 1 "$HOME/.config/jsr/host-role")
    fi
    if [[ -z $role ]]; then
        role=$(hostname)
    fi
    if printf '%s\n' "$role" | grep -Eq '[0-9]{1,3}[.-][0-9]{1,3}[.-][0-9]{1,3}[.-][0-9]{1,3}'; then
        printf 'host name looks like an address; set JSR_HOST_ROLE or ~/.config/jsr/host-role\n' >&2
        role=unlabelled-host
    fi
    printf '%s\n' "$role"
}

if [[ $# != 4 ]]; then
    printf 'usage: bash docs/paper/opponents/open-run.sh ARM ROUND SERVICE_CPUS OBSERVER_CPU\n' >&2
    exit 2
fi
arm=$1 round=$2 service=$3 observer=$4
panel=${OPEN_PANEL:-full}
data=${JSR_DATA:-"$HOME/jsr-data"}
opp=${PAPER_OPPONENTS_ROOT:-"$data/opponents"}
runs=${OPEN_RUNS:-"$opp/bench-open/runs"}
measure=${PAPER_MEASURE:-"$opp/bench-open/bin/ultra-measure"}
scorer=${PAPER_SCORER:-"$opp/bench-open/bin/ultra-score"}
others=${OPEN_OTHER_CORES:-2}
gate_wait=${OPEN_GATE_WAIT_S:-120}
tool="$PWD/docs/paper/opponents/open_receipt.py"
taskset -cp "$observer" $$ > /dev/null

read -r -a service_list <<< "$(python3 "$tool" cpus "$service")"
group=8c
if [[ ${#service_list[@]} == 1 ]]; then
    group=1c
fi
case "$panel" in
    full) panel_root=${OPEN_PANEL_ROOT:-"$opp/panel"} ;;
    p73)
        panel_root=${OPEN_P73_ROOT:-"$opp/bench-cross/workloads/panel73"}
        group="$group-p73"
        ;;
    *)
        printf 'unknown panel %s\n' "$panel" >&2
        exit 2
        ;;
esac
root="$runs/$group"
run="$root/cold/round-$round/$arm"
wav="$root/panel/wav"
mkdir -p "$root/cold"
if [[ ! -e $root/panel ]]; then
    ln -s "$panel_root" "$root/panel"
fi
if [[ -f $run/finished && -f $run/host.cpu.json ]]; then
    exit 0
fi

waiting="$opp/QUIET.waiting.bench-open.$$"
trap 'rm -f "$waiting"' EXIT

note() {
    printf '%s %s round %s on %s (%s): %s\n' "$(date -u +%FT%TZ)" "$arm" "$round" "$service" "$panel" "$*"
}

# A quiet window is in force while the lock exists and its time has not passed by two minutes
# (an unreadable lock counts for 15 minutes after it was written).
quiet_in_force() {
    local lock="$opp/QUIET.lock" until end now
    [[ -f $lock ]] || return 1
    until=$(sed -n '1s/.*until=\([^ ]*\).*/\1/p' "$lock" 2> /dev/null || true)
    now=$(date -u +%s)
    if [[ -n $until ]] && end=$(date -u -d "$until" +%s 2> /dev/null); then
        (( now <= end + 120 ))
    else
        (( now - $(stat -c %Y "$lock" 2> /dev/null || echo "$now") <= 900 ))
    fi
}

wait_quiet() {
    local waited
    while :; do
        waited=0
        while quiet_in_force; do
            : > "$waiting"  # tells the lock's owner that a run is queued behind it
            waited=1
            sleep 10
        done
        if (( waited )); then
            sleep 20  # back-to-back windows replace the lock: make sure it stays away
            if quiet_in_force; then
                continue
            fi
        fi
        rm -f "$waiting"
        return
    done
}

# Wait for the quiet window to end and, for a bounded time, for the host to be quiet. The
# gate's last answer is kept in the receipt either way; the guard judges the whole repetition.
wait_turn() {
    local limit=$(( ${#service_list[@]} + others )) deadline=$(( $(date +%s) + gate_wait ))
    while :; do
        wait_quiet
        if python3 "$tool" gate --need "${#service_list[@]}" --limit "$limit" > "$side/host.gate.json"; then
            return
        fi
        if (( $(date +%s) >= deadline )); then
            note "host still busy after ${gate_wait} s: $(cat "$side/host.gate.json")"
            return
        fi
        sleep 10
    done
}

sample_frequency() {
    local cpu
    while :; do
        printf '%s' "$(date -u +%s)"
        for cpu in "${service_list[@]}"; do
            printf '\t%s' "$(cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq")"
        done
        printf '\t%s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
        sleep 10
    done
}

# The steps of run-cold.sh's run_arm with a CPU list in place of one CPU and any staged panel.
# Only the full panel is scored here.
run_listed() {
    local -a cmd files thread_env=()
    # shellcheck source=docs/paper/opponents/open-arms.sh
    source "$PWD/docs/paper/opponents/open-arms.sh"
    open_arm "$arm"
    if [[ ${#service_list[@]} == 1 ]]; then
        thread_env=(OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1)
    fi
    python3 "$tool" verify-panel "$root/panel"
    mkdir -p "${run%/*}"
    mkdir "$run"
    printf '%q ' "${cmd[@]}" > "$run/command"
    printf '\n' >> "$run/command"
    sha256sum "$measure" "${cmd[0]}" "${files[@]}" > "$run/config.sha256"
    ldd "${cmd[0]}" | awk '
        $2 == "=>" && $3 ~ /^\// {print $3}
        $1 ~ /^\// {print $1}
    ' | sort -u | xargs -r sha256sum > "$run/runtime-libraries.sha256"
    sha256sum "$root/panel/audio.json" > "$run/workload.sha256"
    host_role > "$run/host"
    cat /proc/loadavg "/sys/devices/system/cpu/cpu${service_list[0]}/cpufreq/scaling_governor" \
        /sys/devices/system/cpu/cpufreq/boost > "$run/host.before"
    printf '%s\n' "$round" > "$run/round"
    date -u +%FT%TZ > "$run/started"
    env "${thread_env[@]}" taskset -c "$observer" \
        "$measure" --interval-ms 200 --service-cpus "$service" --observer-cpus "$observer" \
        -- "${cmd[@]}" > "$run/measurement.json" 2> "$run/service.stderr"
    cat /proc/loadavg > "$run/host.after"
    if [[ $panel == full ]]; then
        python3 "$tool" score "$root/panel" "$run" "$scorer" "$service"
    fi
    date -u +%FT%TZ > "$run/finished"
}

if [[ -d $run ]]; then
    mv -- "$run" "$run.incomplete.$(date -u +%Y%m%dT%H%M%S)"
fi
side=$(mktemp -d "$root/.side.XXXXXX")
wait_turn
python3 "$tool" snapshot > "$side/host.cpu.before.json"
sample_frequency > "$side/host.freq.tsv" &
sampler=$!
# Not in an `||` list: errexit must stay in force inside the repetition.
set +e
if [[ $group == 1c ]]; then
    env JSR_DATA="$data" PAPER_OPPONENTS_ROOT="$opp" PAPER_RUN_ROOT="$root" PAPER_MEASURE="$measure" \
        PAPER_SCORER="$scorer" PAPER_SERVICE_CPU="$service" PAPER_OBSERVER_CPU="$observer" \
        bash docs/paper/run-cold.sh "$round" "$arm"
else
    (set -e; run_listed)
fi
status=$?
set -e
kill "$sampler" 2> /dev/null || true
wait "$sampler" 2> /dev/null || true
python3 "$tool" snapshot > "$side/host.cpu.after.json"
if [[ $status != 0 || ! -f $run/finished ]]; then
    note "failed with status $status; evidence kept in $side"
    exit 1
fi
verdict=0
python3 "$tool" guard "$side/host.cpu.before.json" "$side/host.cpu.after.json" \
    --service-cpus "$service" --other-limit "$others" > "$side/host.cpu.json" || verdict=$?
mv "$side/host.gate.json" "$side/host.freq.tsv" "$side/host.cpu.json" "$run/"
rm -r "$side"
printf '%s\n' "$arm" > "$run/arm"
printf '%s\n' "$panel" > "$run/panel"
if [[ -f $PWD/harness.commit ]]; then
    cp "$PWD/harness.commit" "$run/harness.commit"  # the deployed lane commit, when the deployment records it
fi
python3 "$tool" audio-seconds "$root/panel" > "$run/audio.seconds"
(
    # shellcheck source=docs/paper/opponents/open-arms.sh
    source "$PWD/docs/paper/opponents/open-arms.sh"
    open_arm "$arm"
    find "${artifacts[@]}" -type f -printf '%s\n' | awk '{bytes += $1} END {print bytes}' > "$run/artifact.bytes"
)
if [[ -f $run/trace.jsonl ]]; then
    python3 "$tool" trace-summary "$run/trace.jsonl" > "$run/trace.summary.json"
    (cd "$run" && sha256sum trace.jsonl > trace.sha256 && gzip trace.jsonl && sha256sum trace.jsonl.gz >> trace.sha256)
fi
if [[ $verdict == 0 ]]; then
    note "finished clean"
    exit 0
fi
note "not alone: $(python3 -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["reasons"]))' "$run/host.cpu.json")"
mv -- "$run" "$run.contaminated.$(date -u +%Y%m%dT%H%M%S)"
exit 3
