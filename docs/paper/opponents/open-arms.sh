#!/usr/bin/env bash
# Arm definitions for the open-source opponents measured by bench-open.
# Sourced by docs/paper/run-cold.sh (one-thread arms) and by open-run.sh (every arm).
#
#   open_arm ARM   fills the caller's `cmd` and `files` arrays (run-cold.sh's contract) and
#                  `artifacts`, the model files or directory the row's model size counts;
#                  reads the caller's $wav, $run, $data.
#
# ARM is a system, optionally followed by its eight-core mode:
#   vosk-large  vosk-small              Vosk 0.3.45, streaming, 20 ms packets
#   whispercpp-stream-base|-tiny        whisper.cpp through pywhispercpp, Picovoice's 3 s chunk engine
#   parakeet-v3-onnxasr-int8            Parakeet TDT 0.6B v3, onnx-asr INT8, whole utterance
#   jsr-h22ctl-79800-ema-int8           our engine, H22ctl 79800 EMA INT8 (same-session anchor)
#   jsr-d2e-16457-int8                  our engine, D2e 16457 INT8 (earlier cross-session anchor)
#   r0-zipformer-int8                   R0 Zipformer INT8 through sherpa-onnx (same-session anchor)
#   <system>-t8                         eight intra-op threads in one recogniser
#   <system>-w8                         eight single-thread recognisers side by side
# Models, virtualenvs and the anchors live under $PAPER_OPPONENTS_ROOT (default $data/opponents).
# shellcheck disable=SC2034,SC2154  # sourced: cmd, files, artifacts go back to the caller; wav, run, data come from it

open_arm() {
    local opp=${PAPER_OPPONENTS_ROOT:-"$data/opponents"} here="$PWD/docs/paper/opponents"
    local system=$1 model venv runner
    local -a mode=()
    case "$system" in
        *-t8) mode=(--threads 8); system=${system%-t8} ;;
        *-w8) mode=(--workers 8); system=${system%-w8} ;;
    esac
    case "$system" in
        vosk-large|vosk-small)
            model="$opp/vosk/models/vosk-model-en-us-0.22"
            if [[ $system == vosk-small ]]; then
                model="$opp/vosk/models/vosk-model-small-en-us-0.15"
            fi
            if [[ ${mode[0]:-} == --threads ]]; then
                printf 'Vosk has no intra-op thread knob: %s\n' "$1" >&2
                exit 2
            fi
            venv="$opp/vosk/venv"
            runner="$here/vosk_stream.py"
            files=("$model.SHA256SUMS" "$venv"/lib/python3*/site-packages/vosk/libvosk.so "$opp/vosk/venv.freeze.txt")
            artifacts=("$model")
            ;;
        whispercpp-stream-base|whispercpp-stream-tiny)
            model="$opp/whispercpp-stream/models/ggml-${system#whispercpp-stream-}.en.bin"
            venv="$opp/whispercpp-stream/venv"
            runner="$here/whispercpp_stream.py"
            files=("$model" "$venv"/lib/python3*/site-packages/_pywhispercpp.*.so
                "$venv"/lib/python3*/site-packages/pywhispercpp.libs/* "$opp/whispercpp-stream/venv.freeze.txt")
            artifacts=("$model")
            ;;
        parakeet-v3-onnxasr-int8)
            model="$opp/parakeet-onnx-asr/models/parakeet-tdt-0.6b-v3-onnx"
            venv="$opp/parakeet-onnx-asr/venv"
            runner="$here/onnxasr_whole.py"
            artifacts=("$model/encoder-model.int8.onnx" "$model/decoder_joint-model.int8.onnx" "$model/vocab.txt"
                "$model/config.json")
            files=("${artifacts[@]}" "$venv"/lib/python3*/site-packages/onnxruntime/capi/*.so*
                "$opp/parakeet-onnx-asr/venv.freeze.txt")
            ;;
        jsr-h22ctl-79800-ema-int8|jsr-d2e-16457-int8|r0-zipformer-int8)
            # Same-session anchors through ultra-cold-wav: our engine (H22ctl-79800 EMA INT8 with the
            # primary row's binary; D2e-16457 as the earlier cross-session anchor) and the R0 Zipformer.
            local anchor="$opp/bench-open/anchor" binary
            local -a engine
            binary="$anchor/ultra-cold-wav"
            case "$system" in
                jsr-h22ctl-79800-ema-int8)
                    model="$opp/models/H22ctl-79800-ema/int8"
                    binary="$anchor/ultra-cold-wav-457a4db7"  # copy of the primary row's binary
                    ;;
                jsr-d2e-16457-int8) model="$anchor/d2e-16457-int8" ;;
                r0-zipformer-int8) model="$anchor/r0-zipformer-int8" ;;
            esac
            if [[ $system == r0-zipformer-int8 ]]; then
                engine=("$(command -v env)" "ULTRA_SHERPA_LIB=$anchor/sherpa-lib/libsherpa-onnx-c-api.so"
                    "$binary" zipformer-int8 "$model")
                artifacts=("$model/model")
                files=("$model/manifest.json" "$model"/model/* "$anchor"/sherpa-lib/*)
            else
                engine=("$binary" r1-engine-int8 "$model")
                # INT8 mode opens the INT8 weights only; the FP32 file is pinned but not counted.
                artifacts=("$model/arch.json" "$model/weights.int8.safetensors" "$model/bpe.model"
                    "$model/tokens.txt" "$model/mel_fbank.f32")
                files=("${artifacts[@]}" "$model/weights.safetensors")
            fi
            files+=("$binary")
            if [[ ${mode[0]:-} == --workers ]]; then
                cmd=("$(command -v python3)" -I "$here/open_workers.py" "${mode[1]}" "$wav" "$run/predictions.jsonl"
                    -- "${engine[@]}")
                files+=("$here/open_workers.py" "$here/open_common.py")
            elif [[ ${#mode[@]} != 0 ]]; then
                printf 'ultra-cold-wav is single-threaded per stream: %s\n' "$1" >&2
                exit 2
            else
                cmd=("${engine[@]}" "$wav" "$run/predictions.jsonl")
            fi
            return
            ;;
        *)
            printf 'unknown open opponent arm %s\n' "$1" >&2
            exit 2
            ;;
    esac
    cmd=("$venv/bin/python" -I "$runner" "$model" "$wav" "$run/predictions.jsonl" "$run/trace.jsonl" "${mode[@]}")
    files+=("$runner" "$here/open_common.py")
}
