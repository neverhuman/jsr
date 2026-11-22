#!/usr/bin/env bash
# Set up a new GPU node for training (run ON the node, from the lane/trainer worktree):
# uv + a Python 3.12 venv with torch 2.7.0+cu128, k2 (cuda12.8) and the trainer's deps, then the
# given LibriSpeech parquet splits downloaded directly from openslr/librispeech_asr (HF) and
# sha256-checked against MANIFEST.sha256 (which the caller copies to <data>/corpora/librispeech/hf_parquet/).
#   node_setup.sh <jsr_data> <hf_split_dir...>     e.g. train.clean.100 train.clean.360 train.other.500
set -uo pipefail
DATA=$1; shift
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
mkdir -p "$DATA/venvs" "$DATA/logs" "$DATA/corpora/librispeech/hf_parquet" "$DATA/runs/train"
P=$DATA/venvs/train/bin/python
if ! "$P" -c "import torch, k2" 2> /dev/null; then
  uv venv -q --python 3.12 "$DATA/venvs/train"
  uv pip install -q --python "$P" torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$P" numpy==1.26.4 sentencepiece==0.2.3 pyarrow==17.0.0 soundfile scipy pytest huggingface_hub hf_transfer onnxruntime==1.20.1 kaldi-native-fbank
  uv pip install -q --python "$P" --no-deps "k2==1.24.4.dev20250714+cuda12.8.torch2.7.0" -f https://k2-fsa.github.io/k2/cuda.html
fi
"$P" -c "import torch, k2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
cd "$DATA/corpora/librispeech/hf_parquet" || exit 1
inc=(); for s in "$@"; do inc+=(--include "all/$s/*"); done
HF_HUB_ENABLE_HF_TRANSFER=1 "$DATA/venvs/train/bin/hf" download openslr/librispeech_asr --repo-type dataset \
  "${inc[@]}" --max-workers 16 --local-dir . > "$DATA/logs/hf_download.log" 2>&1
ok=1
for s in "$@"; do grep "all/$s/" MANIFEST.sha256 | sha256sum -c --quiet || ok=0; done
[ $ok = 1 ] && echo SHA_OK || echo SHA_FAIL
