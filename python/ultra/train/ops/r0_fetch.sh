#!/usr/bin/env bash
# Download R0 (sherpa-onnx streaming zipformer-en-2023-06-26, pinned revision) onto this node and
# check it against the sha256 list the caller copies to <jsr_data>/models/zipformer-en-2023-06-26/SHA256SUMS.
#   r0_fetch.sh <jsr_data>
set -uo pipefail
M=$1/models/zipformer-en-2023-06-26
mkdir -p "$M" && cd "$M" || exit 1
HF_HUB_ENABLE_HF_TRANSFER=1 "$1/venvs/train/bin/hf" download csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --revision 672fbf1b30579d6585301139bb363f42a0ad4a24 --include "bpe.model" --include "tokens.txt" \
  --include "*chunk-16-left-128*.onnx" --local-dir . > /dev/null 2>&1
sha256sum -c --quiet SHA256SUMS && echo R0_SHA_OK || echo R0_SHA_FAIL
