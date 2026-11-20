#!/usr/bin/env bash
# Fetch LibriSpeech (CC BY 4.0, OpenSLR SLR12) into the USB data root; resumable; verifies md5 from OpenSLR.
set -euo pipefail
DEST=${JSR_DATA:-/mnt/xbabe3-external/jsr}/corpora/librispeech/flac_tgz
BASE=${LS_MIRROR:-https://www.openslr.org/resources/12}
PARTS=${*:-"dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500"}
mkdir -p "$DEST"; cd "$DEST"
curl -fsSL "$BASE/md5sum.txt" -o md5sum.txt
for p in $PARTS; do
  f="$p.tar.gz"
  [ -f "$f.ok" ] && { echo "have $f"; continue; }
  wget -q -c "$BASE/$f" -O "$f"
  grep " $f\$" md5sum.txt | md5sum -c - && touch "$f.ok"
done
echo FETCH_DONE
