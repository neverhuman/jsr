#!/usr/bin/env bash
# Isolated Python environments for the opponent runtimes, pinned and frozen.
# usage: setup-python.sh OPPONENTS_ROOT nemo|redux|sherpa
# Uses the hash-verified uv binary in OPPONENTS_ROOT/bench-cross/bin (see setup-uv.sh) and
# keeps its cache and interpreters under OPPONENTS_ROOT. Nothing is installed system-wide.
set -euo pipefail
O=$1; which=$2
uv="$O/bench-cross/bin/uv"
export UV_CACHE_DIR="$O/bench-cross/uv-cache" UV_PYTHON_INSTALL_DIR="$O/bench-cross/uv-python"
nemo_commit=c02239ab43822ea837b21fb5c0c29e45fbd063fd
case "$which" in
    nemo)
        # NeMo for the unified RNN-T models: the commit of NVIDIA-NeMo/Speech main that carries
        # the prompt-conditioned unified model (the "3.1.0" line the model card names).
        dir="$O/nemotron-labs-unified/runtime"; mkdir -p "$dir"; cd "$dir"
        "$uv" venv --python 3.12 venv
        if [[ $(uname -s) == Linux ]]; then
            "$uv" pip install --python venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            "$uv" pip install --python venv/bin/python torch torchaudio
        fi
        "$uv" pip install --python venv/bin/python \
            "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@$nemo_commit"
        "$uv" pip freeze --python venv/bin/python > venv.freeze.txt
        ;;
    redux)
        # Photon (kestrel) through the model card's own install line.
        dir="$O/parakeet-redux"; mkdir -p "$dir"; cd "$dir"
        "$uv" venv --python 3.12 venv
        "$uv" pip install --python venv/bin/python "moondream>=2.4.0"
        "$uv" pip freeze --python venv/bin/python > venv.freeze.txt
        ;;
    sherpa)
        # sherpa-onnx 2.0.2: the shared library behind `ultra-cold-wav zipformer-int8`, and
        # the Python binding sherpa_zipformer.py uses.
        dir="$O/zipformer-int8"; mkdir -p "$dir/sherpa/lib"; cd "$dir"
        "$uv" venv --python 3.12 sherpa-venv
        "$uv" pip install --python sherpa-venv/bin/python sherpa-onnx==2.0.2 numpy
        "$uv" pip freeze --python sherpa-venv/bin/python > sherpa-venv.freeze.txt
        cp -a sherpa-venv/lib/python3.12/site-packages/sherpa_onnx/lib/. sherpa/lib/
        ;;
    *)
        printf 'unknown environment %s\n' "$which" >&2
        exit 2
        ;;
esac
