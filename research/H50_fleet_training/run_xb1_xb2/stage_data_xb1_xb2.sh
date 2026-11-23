#!/usr/bin/env bash
# Stage pinned code (31ce5a6) + local NVMe shards on xbabe1 & xbabe2 AFTER medium labeling ends (~10:51Z).
# Usage: stage_data_xb1_xb2.sh [--dry-run] [--small-src HOST:PATH]
set -euo pipefail
DRY_RUN=0
PIN="31ce5a63e1c8703531c7a8a8563da83904632a1c"
SMALL_HOST="xbabe5"
SMALL_PATH="~/jsr-data/librilight/shards/r0_v11/librilight.small"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --small-host) SMALL_HOST="$2"; shift 2 ;;
    --small-path) SMALL_PATH="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

echo "=== Step 1: Create clean detached worktree ~/jsr-d0d1 at pinned main ($PIN) on xbabe1 & xbabe2 ==="
for h in xbabe1 xbabe2; do
  run_cmd "ssh $h 'git -C ~/jsr fetch origin main && if [ -d ~/jsr-d0d1/.git ] || [ -f ~/jsr-d0d1/.git ]; then git -C ~/jsr-d0d1 checkout --detach $PIN; else git -C ~/jsr worktree add --detach ~/jsr-d0d1 $PIN; fi && git -C ~/jsr-d0d1 rev-parse HEAD'"
done

echo "=== Step 2: Sync LibriSpeech 960h (57G) + seed + bpe from xbabe2 -> xbabe1 (local NVMe, nice 19) ==="
run_cmd "ssh xbabe1 'mkdir -p ~/jsr-data/corpora/librispeech/hf_parquet ~/jsr-data/seeds/H41 ~/jsr-data/models/zipformer-en-2023-06-26'"
run_cmd "ssh xbabe2 'nice -n 19 ionice -c 3 rsync -a --bwlimit=80000 ~/jsr-data/corpora/librispeech/hf_parquet/ xbabe1:~/jsr-data/corpora/librispeech/hf_parquet/'"
run_cmd "ssh xbabe2 'nice -n 19 ionice -c 3 rsync -a ~/jsr-data/seeds/H41/ xbabe1:~/jsr-data/seeds/H41/ && rsync -a ~/jsr-data/models/zipformer-en-2023-06-26/ xbabe1:~/jsr-data/models/zipformer-en-2023-06-26/'"

echo "=== Step 3: Verify seed & LibriSpeech 960h shard count on both hosts ==="
for h in xbabe1 xbabe2; do
  run_cmd "ssh $h 'sha256sum ~/jsr-data/seeds/H41/model-00060000.pt && du -sh ~/jsr-data/corpora/librispeech/hf_parquet && df -h ~'"
done
