#!/usr/bin/env bash
# Energy rows (H39): watts per live stream with python/ultra/tools/energy_bench.py.
# Run on the benchmark host from the staging directory (default ~/jsr-data/opponents/energy), which
# holds energy_bench.py, bin/, lib/, ext/, models/ and audio/ as listed in research/H39_energy/README.md.
#
#   run-energy.sh quiet MINUTES ARM [ARGS...]    hold QUIET.lock around one arm (see `quiet` below)
#   run-energy.sh idle                           idle against idle on eight cores
#   run-energy.sh spin WORKERS REPS              busy loop on WORKERS cores
#   run-energy.sh spin-stair REPS                busy loop stepping 1, 2, 4, 8 cores inside one run
#   run-energy.sh engine NAME WORKERS REPS SET   our engine, WORKERS single-thread processes
#   run-energy.sh engine-stair REPS              our engine stepping 1, 2, 4, 8 workers inside one run
#   run-energy.sh zipformer NAME WORKERS REPS SET
#   run-energy.sh paced NAME MODEL_DIR STREAMS REPS      live streams paced at 1x on one core
#   run-energy.sh nemotron NAME WORKERS REPS SET RIGHT_CONTEXT
#   run-energy.sh nemotron35 NAME WORKERS REPS SET
#   run-energy.sh moonshine NAME WORKERS REPS SET SIZE
#   R7 matched campaign (one pinned core, one repetition; SET is panel73 or panel730; see research/H39_energy):
#   run-energy.sh r7-live-lm NAME SET            Isosyll live +LM, RC2 160 ms, LM c4 v2 (3bd85e9e binaries)
#   run-energy.sh r7-final NAME SET              Isosyll final: commit pass, M5 LM, beam 16 (3bd85e9e)
#   run-energy.sh r7-zipformer NAME SET          Zipformer INT8, sherpa streaming, through the 3bd85e9e cold binary
#   run-energy.sh vosk NAME SET small|large      Vosk 0.3.45, 20 ms packets
#   run-energy.sh whispercpp NAME SET tiny|base  whisper.cpp 1.8.2 through pywhispercpp, 3 s chunks
#   run-energy.sh onnxasr NAME SET               Parakeet TDT v3 INT8, onnx-asr, whole utterance
#   run-energy.sh sherpa-offline NAME SET        Parakeet TDT v3 INT8, sherpa-onnx OfflineRecognizer, whole utterance
#   (Nemotron 560 ms / 160 ms: `nemotron NAME 1 1 SET 6|1`; Nemotron 3.5: `nemotron35 NAME 1 1 SET`;
#    Moonshine: `moonshine NAME 1 1 SET tiny|small`.)
#   ENERGY_IDLE_SECONDS=5 shortens the idle windows for a dry run; the default is the protocol's 60 s.
set -euo pipefail

root=${ENERGY_ROOT:-$HOME/jsr-data/opponents/energy}
lock=${ENERGY_LOCK:-$HOME/jsr-data/opponents/QUIET.lock}
first=${ENERGY_FIRST_CPU:-16}        # first of up to eight physical cores; SMT siblings stay idle
observer=${ENERGY_OBSERVER_CPU:-31}  # sampler and this script
opp=${ENERGY_OPPONENTS:-$(dirname "$root")}  # staging root of the open-source opponents (venvs, models, harness)
r6=$root/v31_3bd85e9e                        # the frozen 3bd85e9e binaries, RC2 model and LMs of the R6 timing
engine_bin=${ENERGY_ENGINE_BIN:-bin/ultra-cold-wav}
cd "$root"
export ULTRA_SHERPA_LIB=$root/lib
mkdir -p runs/series runs/out

threads=(--env OMP_NUM_THREADS=1 --env OMP_THREAD_LIMIT=1 --env MKL_NUM_THREADS=1)

cpus() { echo "$first-$((first + $1 - 1))"; }

# measure NAME WORKERS REPS [energy_bench options...] -- COMMAND...
measure() {
    local name=$1 workers=$2 reps=$3
    shift 3
    mkdir -p "runs/$(dirname "$name")" "runs/out/$name"
    python3 energy_bench.py run --arm "$(basename "$name")" --cpus "$(cpus "$workers")" --workers "$workers" \
        --reps "$reps" --observer-cpu "$observer" --idle-seconds "${ENERGY_IDLE_SECONDS:-60}" --log-dir "runs/out/$name" \
        --out "runs/$name.json" --series-out "runs/series/$(basename "$name").json" "$@"
}

audio() { printf '%s\n' --wav-dir "audio/$1" --audio-manifest "audio/$1.audio.json"; }

# quiet MINUTES ARM...: hold the lock (renewed every minute, never more than 15 minutes ahead, never
# past MINUTES), wait for runs in flight, run the arm, release.
renew_pid=
quiet_cleanup() {
    if [[ -n $renew_pid ]]; then kill "$renew_pid" 2>/dev/null || true; fi
    rm -f "$lock"
}
quiet() {
    local end=$(($(date +%s) + $1 * 60)) waited=0 busy
    shift
    write_lock() {
        local until=$(($(date +%s) + 900))
        ((until > end)) && until=$end
        printf 'energy until=%s\n' "$(date -u -d "@$until" +%FT%TZ)" > "$lock.tmp.$1"
        mv "$lock.tmp.$1" "$lock"
    }
    if [[ -e $lock ]] && ! grep -q '^energy ' "$lock"; then
        echo "QUIET.lock is held by someone else: $(cat "$lock")" >&2
        return 4
    fi
    write_lock main
    (while sleep 60; do write_lock renew; done) &
    renew_pid=$!
    # the PID lives in script scope (a local would be gone when the EXIT trap fires, and `set -u` would abort)
    trap quiet_cleanup EXIT
    while busy=$(top -b -n 2 -d 3 -w 200 | awk '/^top -/ {n++} n == 2 && $1 ~ /^[0-9]+$/ && $9 + 0 >= 25 {print $1 ":" $12}' | tr '\n' ' ') && [[ -n $busy ]]; do
        echo "$(date -u +%T) runs in flight, waiting: $busy"
        sleep 20
        waited=$((waited + 23))
        if ((waited > 900)); then
            echo "still busy after 15 minutes: $busy" >&2
            return 3
        fi
    done
    "$@"
}

arm=${1:?usage: run-energy.sh ARM [ARGS...] (see the header)}
shift
case "$arm" in
    quiet)
        quiet "$1" "$0" "${@:2}"
        ;;
    idle)
        measure v1/idle-vs-idle 8 5 --workers 1 --multi-threaded -- sleep 60
        ;;
    spin)
        measure "v2/spin-$1" "$1" "$2" -- python3 energy_bench.py spin --seconds 30
        ;;
    spin-stair)
        # Workers start at 0, 20, 40, 40, 60, 60, 60, 60 s and all stop at 80 s.
        measure stair/spin-stair 8 "$1" --multi-threaded -- bash -c \
            'd=(0 20 40 40 60 60 60 60); s=(80 60 40 40 20 20 20 20); sleep "${d[{worker}]}"; exec python3 energy_bench.py spin --seconds "${s[{worker}]}"'
        ;;
    engine)
        mapfile -t set < <(audio "$4")
        measure "$1" "$2" "$3" "${threads[@]}" "${set[@]}" --hash models/d2e-16457-int8 \
            -- "$engine_bin" r1-engine-int8 models/d2e-16457-int8 "audio/$4" "runs/out/$1/rep{rep}-w{worker}.jsonl"
        ;;
    engine-stair)
        # Workers start at 0, 30, 60, 60, 90, 90, 90, 90 s; each decodes the whole 730-utterance panel.
        mapfile -t set < <(audio panel730)
        measure stair/engine-stair 8 "$1" "${threads[@]}" "${set[@]}" --multi-threaded --hash models/d2e-16457-int8 \
            -- bash -c 'd=(0 30 60 60 90 90 90 90); sleep "${d[{worker}]}"; exec "$0" r1-engine-int8 models/d2e-16457-int8 audio/panel730 "runs/out/stair/engine-stair/rep{rep}-w{worker}.jsonl"' "$engine_bin"
        ;;
    zipformer)
        mapfile -t set < <(audio "$4")
        measure "$1" "$2" "$3" "${threads[@]}" "${set[@]}" --hash models/r0-zipformer-int8 --hash lib \
            -- "$engine_bin" zipformer-int8 models/r0-zipformer-int8 "audio/$4" "runs/out/$1/rep{rep}-w{worker}.jsonl"
        ;;
    paced)
        # STREAMS processes of `ultra stream --pace 1.0`, all on the first CPU; 12 s cut at both ends.
        mkdir -p "runs/$(dirname "$1")" "runs/out/$1"
        python3 energy_bench.py run --arm "$(basename "$1")" --cpus "$first" --workers "$3" --reps "$4" \
            --observer-cpu "$observer" --idle-seconds 60 "${threads[@]}" --audio-rate 1.0 --trim-seconds 12 \
            --hash "$2" --hash audio/paced120.wav --hash lib --log-dir "runs/out/$1" \
            --out "runs/$1.json" --series-out "runs/series/$(basename "$1").json" \
            -- bin/ultra stream --candidate "$root/$2" --input audio/paced120.wav --pace 1.0 \
            --receipt "runs/out/$1/rep{rep}-w{worker}.receipt.json"
        ;;
    nemotron)
        mapfile -t set < <(audio "$4")
        model=models/nemotron-0.6b/nemotron-speech-streaming-en-0.6b.q8_0.gguf
        measure "$1" "$2" "$3" "${threads[@]}" "${set[@]}" --hash "$model" --hash models/nemotron-0.6b/tokenizer.json \
            --hash ext/nemo-speech-cpu-asr/bin \
            -- bash -c 'mkdir -p "$1" && exec ext/nemo-speech-cpu-asr/bin/nemo-speech --json bench asr "$2" --model "$4" --device cpu --mode stream --asr.backend.threads 1 --asr.streaming.rnnt_right_context "$3" --concurrency 1 --repetitions 1 --warmup 0 --save "$1"' \
            nemotron "runs/out/$1/rep{rep}-w{worker}" "audio/$4" "$5" "$model"
        ;;
    nemotron35)
        mapfile -t set < <(audio "$4")
        measure "$1" "$2" "$3" "${threads[@]}" "${set[@]}" --hash models/nemo35-verified.gguf \
            --hash ext/nemo-speech-0.3.0-linux-x86_64-cpu \
            -- bash -c 'mkdir -p "$1" && exec ext/nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech --json bench asr "$2" --model models/nemo35-verified.gguf --device cpu --mode stream --language en-US --chunk-ms 80 --asr.backend.threads 1 --asr.streaming.rnnt_right_context 0 --concurrency 1 --repetitions 1 --warmup 0 --save "$1"' \
            nemotron35 "runs/out/$1/rep{rep}-w{worker}" "audio/$4"
        ;;
    moonshine)
        # The SDK adapter sets one ONNX Runtime thread itself; no thread environment.
        mapfile -t set < <(audio "$4")
        measure "$1" "$2" "$3" --env "LD_LIBRARY_PATH=$root/ext/moonshine-sdk/lib" "${set[@]}" \
            --hash "models/moonshine-$5" --hash ext/moonshine-sdk/lib \
            -- bin/moonshine-stream "$5" "models/moonshine-$5" "audio/$4" "runs/out/$1/rep{rep}-w{worker}"
        ;;
    r7-live-lm)
        # R6's live_c4_lm arm, byte for byte (docs/paper/data/raw/release_v31/timing_interleaved).
        mapfile -t set < <(audio "$2")
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$r6/model_rc2" --hash "$r6/lms/c4_p0_0_10_40.v2.ulm" \
            -- "$r6/bin/ultra-cold-wav" --chunk 4 --beam 4 --beam-max-symbols 1 --beam-margin 0.0 \
            --lm "$r6/lms/c4_p0_0_10_40.v2.ulm" --lm-weight 0.70 --lodr-weight 0.50 --blank-penalty 0.0 \
            r1-engine-int8 "$r6/model_rc2" "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl"
        ;;
    r7-final)
        mapfile -t set < <(audio "$2")
        m5=$opp/bench-open/assays/l4-kit/M5.ulm
        manifest=runs/r7/manifest-$2.jsonl
        mkdir -p runs/r7
        python3 - "$root" "$2" "$manifest" <<'PY'
import json, sys
root, name, out = sys.argv[1:]
rows = json.load(open(f"{root}/audio/{name}.audio.json"))
with open(out, "w") as w:
    for row in rows:
        w.write(json.dumps({"uid": row["id"], "wav": f"{root}/audio/{name}/{row['id']}.wav"}) + "\n")
PY
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$r6/model_rc2" --hash "$m5" \
            -- "$r6/bin/ultra-engine-commit" --model "$r6/model_rc2" --manifest "$root/$manifest" \
            --commit-chunk 16 --commit-beam 16 --max-sym-per-frame 1 --commit-lm "$m5" --lm-scale 0.80 \
            --lodr-scale 0.60 --token-bonus 0.0 --tail-pad-ms 320 --threads 1 \
            --out "runs/out/$1/rep{rep}-w{worker}.commit.jsonl"
        ;;
    r7-zipformer)
        mapfile -t set < <(audio "$2")
        measure "$1" 1 1 "${threads[@]}" --env "ULTRA_SHERPA_LIB=$root/lib" "${set[@]}" \
            --hash models/r0-zipformer-int8 --hash lib \
            -- "$r6/bin/ultra-cold-wav" zipformer-int8 models/r0-zipformer-int8 "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl"
        ;;
    vosk)
        mapfile -t set < <(audio "$2")
        model=$opp/vosk/models/vosk-model-en-us-0.22
        [[ $3 == small ]] && model=$opp/vosk/models/vosk-model-small-en-us-0.15
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$model.SHA256SUMS" --hash "$opp/vosk/venv.freeze.txt" \
            -- "$opp/vosk/venv/bin/python" -I "$opp/bench-open/harness/docs/paper/opponents/vosk_stream.py" \
            "$model" "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl" "runs/out/$1/rep{rep}-w{worker}.trace.jsonl"
        ;;
    whispercpp)
        mapfile -t set < <(audio "$2")
        model=$opp/whispercpp-stream/models/ggml-$3.en.bin
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$model" --hash "$opp/whispercpp-stream/venv.freeze.txt" \
            -- "$opp/whispercpp-stream/venv/bin/python" -I "$opp/bench-open/harness/docs/paper/opponents/whispercpp_stream.py" \
            "$model" "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl" "runs/out/$1/rep{rep}-w{worker}.trace.jsonl"
        ;;
    onnxasr)
        mapfile -t set < <(audio "$2")
        model=$opp/parakeet-onnx-asr/models/parakeet-tdt-0.6b-v3-onnx
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$model/encoder-model.int8.onnx" \
            --hash "$model/decoder_joint-model.int8.onnx" --hash "$opp/parakeet-onnx-asr/venv.freeze.txt" \
            -- "$opp/parakeet-onnx-asr/venv/bin/python" -I "$opp/bench-open/harness/docs/paper/opponents/onnxasr_whole.py" \
            "$model" "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl" "runs/out/$1/rep{rep}-w{worker}.trace.jsonl"
        ;;
    sherpa-offline)
        # The runtime-lane runner of the H0/H1 Parakeet TDT v3 row (copy: docs/paper/opponents/sherpa_parakeet_offline.py).
        mapfile -t set < <(audio "$2")
        model=$opp/runtime-lane/parakeet-tdt-v3-sherpa
        measure "$1" 1 1 "${threads[@]}" "${set[@]}" --hash "$model/encoder.int8.onnx" --hash "$opp/zipformer-int8/sherpa-venv.freeze.txt" \
            -- "$opp/zipformer-int8/sherpa-venv/bin/python" "$opp/runtime-lane/run_parakeet_tdt_v3.py" \
            "$model" "audio/$2" "runs/out/$1/rep{rep}-w{worker}.jsonl" 1
        ;;
    *)
        echo "unknown arm $arm" >&2
        exit 2
        ;;
esac
