#!/usr/bin/env bash
# One repetition of one opponent arm under the cold protocol (docs/paper/cold-benchmark.md),
# on any of the three benchmark machines. Scoring happens afterwards (collect.py).
#
#   bash docs/paper/opponents/run-opponent.sh ARM BUDGET ROUND
#
# ARM     one configure() case below
# BUDGET  t1 = one pinned core, one thread
#         t8 = eight pinned cores, eight threads inside one recogniser (parallel: intra_op)
#         w8 = eight pinned cores, eight single-thread recognisers side by side (parallel: workers)
# ROUND   repetition number
#
# Environment:
#   OPP_ROOT          opponents data root on this machine (models, runtimes, bench-cross/)
#   OPP_MACHINE       zen2 | xeon8358 | m3ultra
#   OPP_SERVICE_CPUS  Linux: pinned CPUs, one for t1, eight for t8 and w8 (comma list)
#   OPP_OBSERVER_CPU  Linux: CPU of the meter, outside the service CPUs' L3 and SMT domain
#   OPP_WORKLOAD      panel730 (default) | panel73 | lag100 | smoke10 (see workload.py)
#   OPP_TRACE         1 = streaming arms with a trace option also write the partials trace
#
# macOS has no CPU affinity and no procfs: the thread count is the only bound and the meter
# is bsd_time_measure.py (`/usr/bin/time -l`), whose differences are listed in that file.
set -euo pipefail

if [[ $# != 3 ]]; then
    printf 'usage: bash docs/paper/opponents/run-opponent.sh ARM t1|t8|w8 ROUND\n' >&2
    exit 2
fi
arm=$1 budget=$2 round=$3
here=$(cd "$(dirname "$0")" && pwd)
O=${OPP_ROOT:?set OPP_ROOT}
machine=${OPP_MACHINE:?set OPP_MACHINE}
workload=${OPP_WORKLOAD:-panel730}
bin="$O/bench-cross/bin"
work="$O/bench-cross/workloads/$workload"
os=$(uname -s)
python=${OPP_PYTHON:-python3}

case "$budget" in
    t1) threads=1 parallel=intra_op cores=1 ;;
    t8) threads=8 parallel=intra_op cores=8 ;;
    w8) threads=1 parallel=workers cores=8 ;;
    *) printf 'unknown budget %s\n' "$budget" >&2; exit 2 ;;
esac

sum() {
    if [[ $os == Darwin ]]; then shasum -a 256 "$@"; else sha256sum "$@"; fi
}

# Sets cmd, files, libs, envs, kind, single, class. Tokens in cmd: {wav} staged WAV directory,
# {out} output path, {trace} trace path, {threads} thread count, {engine} npz view of the same
# audio for ultra-engine-eval (engine_cache.py).
configure() {
    local gguf nemo venv
    libs=() envs=() kind=jsonl single=0 class=whole_utterance
    case "$1" in
        parakeet-unified-offline|parakeet-unified-stream160)
            gguf="$O/parakeet-unified/model-gguf/parakeet-unified-en-0.6b-Q8_0.gguf"
            cmd=("$bin/transcribe-cold" offline "$gguf" "{wav}" "{out}" "{threads}")
            if [[ $1 == *stream160 ]]; then
                class=streaming
                cmd=("$bin/transcribe-cold" stream:5600:80:80 "$gguf" "{wav}" "{out}" "{threads}" "{trace}")
            fi
            files=("$gguf")
            ;;
        nemotron-labs-unified-offline|nemotron-labs-unified-stream160)
            nemo="$O/nemotron-labs-unified/model-nemo/nemotron-labs-unified-asr-0.6b.nemo"
            venv="$O/nemotron-labs-unified/runtime/venv"
            cmd=("$venv/bin/python" "$here/nemo_unified.py" offline "$nemo" "{wav}" "{out}" "{threads}" --lang en)
            if [[ $1 == *stream160 ]]; then
                class=streaming
                cmd=("$venv/bin/python" "$here/nemo_unified.py" stream:5.6:0.08:0.08 "$nemo" "{wav}" "{out}"
                    "{threads}" --lang en --trace "{trace}")
            fi
            files=("$nemo" "$here/nemo_unified.py" "$O/nemotron-labs-unified/runtime/venv.freeze.txt")
            libs=("$venv"/lib/python3.*/site-packages/torch/lib/libtorch_cpu.*)
            ;;
        parakeet-redux)
            venv="$O/parakeet-redux/venv"
            cmd=("$venv/bin/python" "$here/photon_redux.py" "{wav}" "{out}" "{threads}")
            files=("$here/photon_redux.py" "$O/parakeet-redux/venv.freeze.txt" "$O/parakeet-redux/weights.sha256")
            libs=("$venv"/lib/python3.*/site-packages/kestrel_kernels/kestrel_cpu.kstlc
                "$venv"/lib/python3.*/site-packages/kestrel_kernels/_native_loader.*
                "$venv"/lib/python3.*/site-packages/torch/lib/libtorch_cpu.*)
            envs=(HF_HOME="$O/parakeet-redux/hf-home" HF_HUB_OFFLINE=1)
            ;;
        jsr-d2e-int8)
            class=streaming single=1
            cmd=("$bin/ultra-cold-wav" r1-engine-int8 "$O/jsr-d2e-16457-int8/model" "{wav}" "{out}")
            files=("$O/jsr-d2e-16457-int8/model/arch.json" "$O/jsr-d2e-16457-int8/model/weights.int8.safetensors")
            ;;
        zipformer-int8)
            class=streaming single=1
            cmd=("$bin/ultra-cold-wav" zipformer-int8 "$O/zipformer-int8/bundle" "{wav}" "{out}")
            files=("$O/zipformer-int8/bundle/manifest.json")
            libs=("$O"/zipformer-int8/sherpa/lib/*)
            envs=(ULTRA_SHERPA_LIB="$(printf '%s\n' "$O"/zipformer-int8/sherpa/lib/libsherpa-onnx-c-api.* | head -1)")
            ;;
        jsr-d2e-int8-enginetool)
            # Same engine and weights through the engine's own tool: the only form of our engine
            # that builds on macOS; also run on Linux to measure the harness difference.
            class=streaming single=1 kind=engine
            cmd=("$bin/ultra-engine-eval" --weights "$O/jsr-d2e-16457-int8/model" --panel "{engine}/panel.json"
                --cache-dir "{engine}" --precision int8 --streams 1 --packet-ms 20 --qualifying --out "{out}")
            files=("$O/jsr-d2e-16457-int8/model/arch.json" "$O/jsr-d2e-16457-int8/model/weights.int8.safetensors")
            ;;
        zipformer-int8-sherpapy)
            # Same sherpa-onnx library and bundle through its Python binding (see sherpa_zipformer.py).
            class=streaming single=1
            venv="$O/zipformer-int8/sherpa-venv"
            cmd=("$venv/bin/python" "$here/sherpa_zipformer.py" "$O/zipformer-int8/bundle" "{wav}" "{out}"
                --trace "{trace}")
            files=("$O/zipformer-int8/bundle/manifest.json" "$here/sherpa_zipformer.py"
                "$O/zipformer-int8/sherpa-venv.freeze.txt")
            libs=("$venv"/lib/python3.*/site-packages/sherpa_onnx/lib/*)
            ;;
        moonshine-tiny)
            class=streaming single=1 kind=dir
            cmd=("$bin/moonshine-stream" tiny "$O/moonshine-tiny/model" "{wav}" "{out}")
            files=("$O/moonshine-tiny/model/"*)
            ;;
        *)
            printf 'unknown opponent arm %s\n' "$1" >&2
            exit 2
            ;;
    esac
}

configure "$arm"
if [[ $single == 1 && $budget == t8 ]]; then
    printf '%s is single-threaded per stream: use w8\n' "$arm" >&2
    exit 2
fi
service=${OPP_SERVICE_CPUS:-}
observer=${OPP_OBSERVER_CPU:-}
if [[ $os == Linux ]]; then
    count=$(tr ',' '\n' <<< "$service" | grep -c .)
    if [[ $count != "$cores" || -z $observer ]]; then
        printf 'budget %s needs %s service CPUs and an observer CPU\n' "$budget" "$cores" >&2
        exit 2
    fi
fi

suffix=""
if [[ $workload != panel730 ]]; then
    suffix="-$workload"
fi
run="$O/bench-cross/receipts/$machine/$arm-$budget$suffix/round-$round"
if [[ -f "$run/finished" ]]; then
    exit 0
fi

# The energy agent's quiet windows on xbabe0: do not start a timed run inside one.
lock="$O/QUIET.lock"
while [[ -f $lock ]]; do
    until_utc=$(sed -n 's/.*until=\([^ ]*\).*/\1/p' "$lock" | head -1)
    left=$(( $(date -u -d "$until_utc" +%s 2>/dev/null || echo 0) - $(date -u +%s) ))
    if (( left < -120 )); then
        printf 'stale QUIET.lock (until=%s) ignored\n' "$until_utc" >&2
        break
    fi
    sleep 20
done

mkdir -p "${run%/*}"
if [[ -d $run && ${OPP_RETRY_INCOMPLETE:-0} == 1 ]]; then
    mv -- "$run" "$run.incomplete.$(date -u +%Y%m%dT%H%M%S)"
fi
mkdir "$run"

trace=""
if [[ $class == streaming && ${OPP_TRACE:-0} == 1 && " ${cmd[*]} " == *" {trace} "* ]]; then
    trace=1
fi
case "$kind" in
    dir) out="$run/outputs" ;;
    engine) out="$run/engine-eval" ;;
    *) out="$run/predictions.jsonl" ;;
esac
argv=()
for token in "${cmd[@]}"; do
    if [[ $token == "{trace}" && -z $trace ]]; then
        # No trace requested: drop the token, and a `--trace` flag in front of it.
        if [[ ${argv[${#argv[@]}-1]} == --trace ]]; then
            unset "argv[${#argv[@]}-1]"
        fi
        continue
    fi
    token=${token//\{threads\}/$threads}
    if [[ $parallel != workers ]]; then
        token=${token//\{wav\}/$work/wav}
        token=${token//\{out\}/$out}
        token=${token//\{trace\}/$run/trace.jsonl}
        token=${token//\{engine\}/$work/engine}
    fi
    argv+=("$token")
done
if [[ $parallel == workers ]]; then
    if [[ $os == Linux ]]; then
        argv=("$python" "$here/workers.py" "$service" "$work/shards-8" "$run" -- "${argv[@]}")
    else
        argv=("$python" "$here/workers.py" --time-l none:8 "$work/shards-8" "$run" -- "${argv[@]}")
    fi
fi
# Moonshine sets the SDK's one-thread ORT option in the adapter before load.
if [[ $arm != moonshine-* ]]; then
    envs+=(OMP_NUM_THREADS="$threads" OMP_THREAD_LIMIT="$threads" MKL_NUM_THREADS="$threads")
fi

printf '%q ' "${argv[@]}" > "$run/command"
printf '\n' >> "$run/command"
printf '%s\n' ${envs[@]+"${envs[@]}"} > "$run/environment"
if [[ $os == Linux ]]; then
    meter="$bin/ultra-measure"
    ldd "${cmd[0]}" | awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' \
        | sort -u | xargs -r sha256sum > "$run/runtime-libraries.sha256"
    cat /proc/loadavg "/sys/devices/system/cpu/cpu${service%%,*}/cpufreq/scaling_governor" \
        /sys/devices/system/cpu/cpufreq/boost > "$run/host.before" 2>/dev/null || true
    grep '^cpu' /proc/stat > "$run/cpustat.before"
else
    meter="$here/bsd_time_measure.py"
    otool -L "${cmd[0]}" | awk 'NR > 1 && $1 ~ /^\// {print $1}' | sort -u \
        | while read -r lib; do if [[ -f $lib ]]; then sum "$lib"; fi; done > "$run/runtime-libraries.sha256"
    { sysctl -n vm.loadavg; pmset -g therm | grep -i -E 'cpu_speed_limit|scheduler_limit' || true; } > "$run/host.before"
fi
sum "$meter" "${cmd[0]}" "${files[@]}" "$here/run-opponent.sh" > "$run/config.sha256"
if (( ${#libs[@]} )); then
    sum "${libs[@]}" >> "$run/runtime-libraries.sha256"
fi
sum "$work/audio.json" "$O/panel/audio.json" > "$run/workload.sha256"
hostname > "$run/host"
printf '%s\n' "$round" > "$run/round"
cat > "$run/budget.json" <<EOF
{"arm": "$arm", "budget": "$budget", "class": "$class", "machine": "$machine", "threads": $cores,
 "parallel": "$parallel", "service_cpus": "$service", "observer_cpu": "$observer", "workload": "$workload",
 "trace": ${trace:-0}}
EOF
date -u +%FT%TZ > "$run/started"
status=0
sampler=""
if [[ $os == Linux && -r /sys/devices/system/cpu/cpu${service%%,*}/cpufreq/scaling_cur_freq ]]; then
    # Clock of the service CPUs every two seconds, read from the meter's CPU.
    taskset -c "$observer" bash -c '
        IFS=, read -r -a cpus <<< "$1"
        while :; do
            line=$(date +%s)
            for cpu in "${cpus[@]}"; do
                read -r khz < "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq"
                line+=" $khz"
            done
            printf "%s\n" "$line"
            sleep 2
        done' _ "$service" > "$run/freq.tsv" &
    sampler=$!
fi
if [[ $os == Linux ]]; then
    env ${envs[@]+"${envs[@]}"} taskset -c "$observer" \
        "$meter" --interval-ms 200 --service-cpus "$service" --observer-cpus "$observer" \
        -- "${argv[@]}" > "$run/measurement.json" 2> "$run/service.stderr" || status=$?
    cat /proc/loadavg > "$run/host.after"
    grep '^cpu' /proc/stat > "$run/cpustat.after"
else
    extra=()
    if [[ $parallel == workers ]]; then
        extra=(--workers "$run")
    fi
    env ${envs[@]+"${envs[@]}"} "$python" "$meter" "$run/service.stderr" ${extra[@]+"${extra[@]}"} \
        -- "${argv[@]}" > "$run/measurement.json" || status=$?
    sysctl -n vm.loadavg > "$run/host.after"
fi
if [[ -n $sampler ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
fi
if [[ $status != 0 ]]; then
    printf 'arm %s %s round %s failed with status %s\n' "$arm" "$budget" "$round" "$status" >&2
    exit "$status"
fi
if [[ $kind == engine ]]; then
    "$python" "$here/engine_cache.py" --rows "$O/panel" "$run"
elif [[ $parallel == workers ]]; then
    if [[ $kind == dir ]]; then
        mkdir "$run/outputs"
        for part in "$run"/worker-*.out; do
            mv "$part"/*.txt "$run/outputs/"
            rmdir "$part"
        done
    else
        cat "$run"/worker-*.out > "$run/predictions.jsonl"
        rm "$run"/worker-*.out
    fi
fi
if [[ $parallel == workers ]]; then
    if [[ -n $trace ]]; then
        cat "$run"/worker-*.trace.jsonl > "$run/trace.jsonl"
        rm "$run"/worker-*.trace.jsonl
    fi
fi
if [[ -f "$run/trace.jsonl" ]]; then
    sum "$run/trace.jsonl" > "$run/trace.sha256"
    wc -l < "$run/trace.jsonl" > "$run/trace.lines"
fi
date -u +%FT%TZ > "$run/finished"
