#!/usr/bin/env bash
# Run from the repository root after staging the pinned DEV audio and model bundles.
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

root=${PAPER_RUN_ROOT:-"$PWD/target/paper-evidence"}
sdk=${PAPER_SDK_ROOT:-"$PWD/target/paper-smoke"}
data=${JSR_DATA:-/mnt/xbabe3-external/jsr}
measure=${PAPER_MEASURE:-"$root/runtime/ultra-measure"}
cold_wav=${PAPER_COLD_WAV:-"$root/runtime/ultra-cold-wav"}
scorer=${PAPER_SCORER:-"$root/runtime/ultra-score"}
audio_tool="$PWD/python/ultra/tools/paper_dev_audio.py"
service_cpu=${PAPER_SERVICE_CPU:-46}
observer_cpu=${PAPER_OBSERVER_CPU:-127}
wav="$root/panel/wav"
nemo="$data/ext/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech"
nemo_model="$data/models/nemotron-0.6b/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
nemo35="$sdk/nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech"
nemo35_model="$sdk/nemo35-verified.gguf"
order=(moonshine-tiny nemotron-160 zipformer-int8 dense-int8 moonshine-small nemotron-560 mr-int8 dense-ort dense-fp32)

if [[ ${PAPER_NEMOTRON_35:-1} == 1 ]]; then
    order+=(nemotron-35)
fi

configure() {
    local size context
    case "$1" in
        moonshine-*)
            size=${1#moonshine-}
            cmd=("$sdk/moonshine-stream" "$size" "$sdk/moonshine-$size" "$wav" "$run/outputs")
            files=("$sdk/moonshine-$size/"*)
            ;;
        nemotron-160|nemotron-560)
            context=1
            if [[ $1 == nemotron-560 ]]; then
                context=6
            fi
            cmd=("$nemo" --json bench asr "$wav" --model "$nemo_model" --device cpu --mode stream
                --asr.backend.threads 1 --asr.streaming.rnnt_right_context "$context"
                --concurrency 1 --repetitions 1 --warmup 0 --save "$run/outputs")
            files=("$nemo_model" "$data/models/nemotron-0.6b/tokenizer.json")
            ;;
        nemotron-35)
            cmd=("$nemo35" --json bench asr "$wav" --model "$nemo35_model" --device cpu --mode stream
                --language en-US --chunk-ms 80 --asr.backend.threads 1 --asr.streaming.rnnt_right_context 0
                --concurrency 1 --repetitions 1 --warmup 0 --save "$run/outputs")
            files=("$nemo35_model")
            ;;
        zipformer-int8)
            local model="$data/bundles/r0-zipformer-int8"
            cmd=("$cold_wav" zipformer-int8 "$model" "$wav" "$run/predictions.jsonl")
            files=("$model/manifest.json")
            ;;
        dense-int8|dense-fp32)
            local model="$root/dense-model/prepared"
            if [[ $1 == dense-fp32 ]]; then
                model="$root/dense-model/weights"
            fi
            cmd=("$cold_wav" "r1-engine-${1#dense-}" "$model" "$wav" "$run/predictions.jsonl")
            files=("$model/arch.json")
            ;;
        mr-int8)
            local model="$data/runs/engine/d2mr-50000/weights-int8prep"
            cmd=("$cold_wav" r1-engine-int8 "$model" "$wav" "$run/predictions.jsonl")
            files=("$model/arch.json")
            ;;
        dense-ort)
            local model="$root/dense-model/ort-fp32"
            cmd=("$cold_wav" r1-ort-fp32 "$model" "$wav" "$run/predictions.jsonl")
            files=("$model/manifest.json")
            ;;
        vosk-large|vosk-small|whispercpp-stream-base|whispercpp-stream-tiny|parakeet-v3-onnxasr-int8|jsr-h22ctl-79800-ema-int8|jsr-a1-30000-ema-int8|jsr-a1-49000-ema-int8|jsr-f1-60000-ema-int8|jsr-d2e-16457-int8|r0-zipformer-int8|jsr-rc1-ema-int8)
            # Open-source opponents and their same-session anchors (docs/paper/opponents/open-arms.sh).
            # shellcheck source=docs/paper/opponents/open-arms.sh
            source "$PWD/docs/paper/opponents/open-arms.sh"
            open_arm "$1"
            ;;
        *)
            printf 'unknown cold comparison arm %s\n' "$1" >&2
            exit 2
            ;;
    esac
}

run_arm() {
    local round=$1 arm=$2 run="$root/cold/round-$1/$2"
    local -a cmd files thread_env
    if [[ -f "$run/finished" ]]; then
        if [[ ! -f "$run/score.json" ]]; then
            python3 "$audio_tool" score "$root/panel" "$run" "$scorer" "$service_cpu"
        fi
        return
    fi
    configure "$arm"
    # Moonshine sets the SDK's one-thread ORT option in the adapter before load.
    thread_env=()
    if [[ $arm != moonshine-* ]]; then
        thread_env=(OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1)
    fi
    mkdir -p "${run%/*}"
    if [[ -d "$run" && ${PAPER_RETRY_INCOMPLETE:-0} == 1 ]]; then
        mv -- "$run" "$run.incomplete.$(date -u +%Y%m%dT%H%M%S)"
    fi
    mkdir "$run"
    if [[ $arm == nemotron-* ]]; then
        mkdir "$run/outputs"
    fi
    printf '%q ' "${cmd[@]}" > "$run/command"
    printf '\n' >> "$run/command"
    sha256sum "$measure" "${cmd[0]}" "${files[@]}" > "$run/config.sha256"
    ldd "${cmd[0]}" | awk '
        $2 == "=>" && $3 ~ /^\// {print $3}
        $1 ~ /^\// {print $1}
    ' | sort -u | xargs -r sha256sum > "$run/runtime-libraries.sha256"
    sha256sum "$root/panel/audio.json" > "$run/workload.sha256"
    host_role > "$run/host"
    cat /proc/loadavg "/sys/devices/system/cpu/cpu$service_cpu/cpufreq/scaling_governor" \
        /sys/devices/system/cpu/cpufreq/boost > "$run/host.before"
    printf '%s\n' "$round" > "$run/round"
    date -u +%FT%TZ > "$run/started"
    env "${thread_env[@]}" taskset -c "$observer_cpu" \
        "$measure" --interval-ms 200 --service-cpus "$service_cpu" --observer-cpus "$observer_cpu" \
        -- "${cmd[@]}" > "$run/measurement.json" 2> "$run/service.stderr"
    cat /proc/loadavg > "$run/host.after"
    python3 "$audio_tool" score "$root/panel" "$run" "$scorer" "$service_cpu"
    date -u +%FT%TZ > "$run/finished"
}

python3 "$audio_tool" verify "$root/panel"

if [[ $# == 2 ]]; then
    run_arm "$1" "$2"
elif [[ $# == 0 ]]; then
    for round in 1 2 3; do
        for index in "${!order[@]}"; do
            arm=${order[(index + round - 1) % ${#order[@]}]}
            run_arm "$round" "$arm"
        done
    done
else
    printf 'usage: bash docs/paper/run-cold.sh [ROUND ARM]\n' >&2
    exit 2
fi
