#!/usr/bin/env bash
# Stage evaluation inputs on H0's internal NVMe so CPU evals never read the USB exFAT data drive
# that D2 trains from: dev caches, tokenizers and the given D2 snapshots, copied once with
# ionice -c3 and sha256-checked against the source.
#   stage_h0_cache.sh <step...>     (D2 snapshot steps, e.g. 65000 60000 55000)
set -uo pipefail
SRC=/mnt/xbabe3-external/jsr
DST=/home/ubuntu/jsr-local-cache
mkdir -p "$DST/cache/librispeech_pcm" "$DST/tokenizers" "$DST/runs/train/D2_S_960h"
cp_check() {  # <src> <dst_dir>
  local f=$1 d=$2 b
  b=$(basename "$f")
  if [ ! -f "$d/$b" ] || [ "$(stat -c %s "$f")" != "$(stat -c %s "$d/$b")" ]; then
    ionice -c3 cp "$f" "$d/$b.tmp" && mv "$d/$b.tmp" "$d/$b"
  fi
}
for f in "$SRC"/cache/librispeech_pcm/dev-clean.npz "$SRC"/cache/librispeech_pcm/dev-other.npz; do
  cp_check "$f" "$DST/cache/librispeech_pcm"
  [ "$(sha256sum "$f" | cut -c1-64)" = "$(sha256sum "$DST/cache/librispeech_pcm/$(basename "$f")" | cut -c1-64)" ] \
    || { echo "sha mismatch $f"; exit 1; }
done
for f in "$SRC"/tokenizers/librispeech_train_bpe{512,1024,4096}.model; do cp_check "$f" "$DST/tokenizers"; done
cp_check "$SRC/runs/train/D2_S_960h/config.json" "$DST/runs/train/D2_S_960h"
for s in "$@"; do
  m=$(printf 'model-%08d' "$s")
  cp_check "$SRC/runs/train/D2_S_960h/$m.sha256" "$DST/runs/train/D2_S_960h"
  cp_check "$SRC/runs/train/D2_S_960h/$m.pt" "$DST/runs/train/D2_S_960h"
  [ "$(sha256sum "$DST/runs/train/D2_S_960h/$m.pt" | cut -c1-64)" = "$(cut -c1-64 "$DST/runs/train/D2_S_960h/$m.sha256")" ] \
    || { echo "sha mismatch $m"; rm -f "$DST/runs/train/D2_S_960h/$m.pt"; exit 1; }
  echo "staged $m"
done
du -sh "$DST"
