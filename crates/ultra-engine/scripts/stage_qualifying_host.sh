#!/usr/bin/env bash
# Stage the qualifying host (xbabe2) for the G6 window. PREPARED, not run until the lead calls it.
#
# usage (from H0):
#   stage_qualifying_host.sh <freeze-commit> <host> <artifact-dir>... [-- <cache>...]
#   e.g. stage_qualifying_host.sh <sha> xbabe2 \
#          /mnt/xbabe3-external/jsr/runs/engine/d2mr-50000/weights-int8prep \
#          /mnt/xbabe3-external/jsr/runs/engine/d2mr-50000/weights \
#          -- dev-clean dev-other
#
# What it does, failing closed at every step:
#  1. On <host>: a clean checkout of <freeze-commit> under ~/jsr-engine (a plain clone,
#     never /tmp); refuses a dirty tree; builds ultra-engine and ultra-cli release
#     binaries (CARGO_TARGET_DIR=~/jsr-engine/target, <= 12 jobs, nice 10).
#  2. Records the sha256 of every built binary.
#  3. Copies each artifact directory to ~/jsr-engine-data/artifacts/<parent>-<name>/ and
#     verifies every file's sha256 against the H0 original (the engine's loader re-checks the
#     arch.json file hashes at every load).
#  4. Copies the named caches (librispeech_pcm/<split>.npz) and verifies them: DEV splits
#     against docs/ultra/data/dev_selection_panel.json cache_sha256, every split against H0.
#     test-* caches are copied only when listed explicitly (sealed reads are gated by the
#     engine/bench guards, not by this script).
#  5. Writes ~/jsr-engine-data/staging-<commit12>.json with host, commit, binary and file
#     hashes, and copies it back to H0 next to this run's log.
set -euo pipefail

[ $# -ge 3 ] || { sed -n 2,26p "$0"; exit 2; }
COMMIT=$1 HOST=$2; shift 2
ARTS=(); CACHES=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; CACHES=("$@"); break; fi
  ARTS+=("$1"); shift
done
REPO_URL=$(git -C "$(dirname "$0")" remote get-url origin)
H0_CACHE=/mnt/xbabe3-external/jsr/cache/librispeech_pcm
PANEL_JSON=$(git -C "$(dirname "$0")" show "$COMMIT:docs/ultra/data/dev_selection_panel.json")
C12=${COMMIT:0:12}
LOG=/mnt/xbabe3-external/jsr/runs/engine/staging-$HOST-$C12-$(date -u +%Y%m%dT%H%M%SZ)
[ -e "$LOG" ] && { echo "exists $LOG"; exit 1; }
mkdir -p "$LOG"
exec > >(tee "$LOG/stage.log") 2>&1
echo "staging $HOST at $COMMIT ($(date -u +%FT%TZ))"

# 1-2. Clean build on the host.
ssh "$HOST" bash -s "$REPO_URL" "$COMMIT" <<'REMOTE'
set -euo pipefail
URL=$1 COMMIT=$2
[ -d ~/jsr-engine/.git ] || git clone -q "$URL" ~/jsr-engine
cd ~/jsr-engine
git fetch -q origin
git checkout -q --detach "$COMMIT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty checkout"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$COMMIT^{commit}")" ] || { echo "wrong HEAD"; exit 3; }
export CARGO_TARGET_DIR=$HOME/jsr-engine/target CARGO_BUILD_JOBS=12 CARGO_INCREMENTAL=0
nice -n 10 cargo build --locked --release -p ultra-engine -p ultra-cli --bins 2>&1 | tail -1
mkdir -p ~/jsr-engine-data/bin-${COMMIT:0:12}
for b in ultra-engine-eval ultra-engine-density ultra-engine-prepare ultra-engine-gate ultra; do
  [ -x "$CARGO_TARGET_DIR/release/$b" ] && cp "$CARGO_TARGET_DIR/release/$b" ~/jsr-engine-data/bin-${COMMIT:0:12}/
done
(cd ~/jsr-engine-data/bin-${COMMIT:0:12} && sha256sum * > SHA256SUMS && cat SHA256SUMS)
REMOTE

# 3. Artifacts, verified file by file against H0.
for A in "${ARTS[@]}"; do
  NAME="$(basename "$(dirname "$A")")-$(basename "$A")"
  (cd "$A" && sha256sum -- *) > "$LOG/$NAME.SHA256SUMS"
  ssh "$HOST" "mkdir -p ~/jsr-engine-data/artifacts && [ ! -e ~/jsr-engine-data/artifacts/$NAME ]" \
    || { echo "artifact $NAME already staged on $HOST (never overwrite)"; exit 4; }
  tar -C "$(dirname "$A")" -cf - "$(basename "$A")" \
    | ssh "$HOST" "mkdir -p ~/jsr-engine-data/artifacts/.in && tar -C ~/jsr-engine-data/artifacts/.in -xf - && mv ~/jsr-engine-data/artifacts/.in/$(basename "$A") ~/jsr-engine-data/artifacts/$NAME"
  ssh "$HOST" "cd ~/jsr-engine-data/artifacts/$NAME && sha256sum -c --quiet -" < "$LOG/$NAME.SHA256SUMS"
  echo "artifact $NAME verified"
done

# 4. Caches.
for S in "${CACHES[@]}"; do
  F=$H0_CACHE/$S.npz
  WANT=$(sha256sum "$F" | cut -c1-64)
  PANEL=$(printf '%s' "$PANEL_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(next((s['cache_sha256'] for s in d['splits'] if s['split']=='$S'),''))")
  [ -z "$PANEL" ] || [ "$PANEL" = "$WANT" ] || { echo "$S: H0 cache differs from the panel manifest"; exit 5; }
  ssh "$HOST" "mkdir -p ~/jsr-engine-data/cache/librispeech_pcm"
  scp -q "$F" "$HOST:jsr-engine-data/cache/librispeech_pcm/$S.npz.part"
  GOT=$(ssh "$HOST" "sha256sum ~/jsr-engine-data/cache/librispeech_pcm/$S.npz.part | cut -c1-64")
  [ "$GOT" = "$WANT" ] || { echo "$S: copy hash mismatch"; exit 5; }
  ssh "$HOST" "mv ~/jsr-engine-data/cache/librispeech_pcm/$S.npz.part ~/jsr-engine-data/cache/librispeech_pcm/$S.npz"
  echo "cache $S verified ($WANT)"
done

# 5. Staging record.
python3 - "$LOG" "$HOST" "$COMMIT" <<'PY'
import glob, json, os, sys, datetime
log, host, commit = sys.argv[1:]
rec = {"schema": "ultra.engine-staging/v1", "host": host, "commit": commit,
       "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "artifacts": {os.path.basename(p)[:-len(".SHA256SUMS")]: open(p).read().split("\n")
                     for p in sorted(glob.glob(os.path.join(log, "*.SHA256SUMS")))}}
json.dump(rec, open(os.path.join(log, "staging.json"), "w"), indent=1)
PY
scp -q "$LOG/staging.json" "$HOST:jsr-engine-data/staging-$C12.json"
echo "DONE staging $HOST; record $LOG/staging.json"
