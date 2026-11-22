#!/usr/bin/env bash
# Stage the qualifying host (xbabe2) for the G6 window. PREPARED, not run until the lead calls it.
#
# usage (from H0):
#   stage_qualifying_host.sh <freeze-commit> <host> <host-checkout> <artifact-dir>... [-- <cache>...]
#   e.g. stage_qualifying_host.sh <sha> xbabe2 /home/ubuntu/jsr \
#          /mnt/xbabe3-external/jsr/runs/engine/d2mr-50000/weights-int8prep \
#          /mnt/xbabe3-external/jsr/runs/engine/d2mr-50000/weights \
#          -- dev-clean dev-other
#
# What it does, failing closed at every step:
#  1. On <host>: uses the owner-designated EXISTING checkout <host-checkout> only. It never
#     clones, fetches or checks anything out: it refuses unless that checkout's HEAD is
#     <freeze-commit> and its tracked tree is clean. Builds ultra-engine and ultra-cli release
#     binaries there (CARGO_TARGET_DIR=<host-checkout>/target, <= 12 jobs, nice 10).
#  2. Requires every expected binary (no optional copies) and records its sha256.
#  3. Copies each artifact directory to ~/jsr-engine-data/artifacts/<parent>-<name>/ and
#     verifies every file's sha256 against the H0 original (the engine's loader re-checks the
#     arch.json file hashes at every load).
#  4. Copies the named caches (librispeech_pcm/<split>.npz) and verifies them: DEV splits
#     against docs/ultra/data/dev_selection_panel.json cache_sha256, every split against H0.
#     test-* caches are copied only when listed explicitly (sealed reads are gated by the
#     engine/bench guards, not by this script).
#  5. Writes staging.json (host, commit, checkout, verified binary digests, verified artifact
#     file digests, verified cache digests with their reference source) next to this run's
#     log on H0 and to ~/jsr-engine-data/staging-<commit12>.json on the host.
set -euo pipefail

[ $# -ge 4 ] || { sed -n 2,28p "$0"; exit 2; }
COMMIT=$1 HOST=$2 CHECKOUT=$3; shift 3
ARTS=(); CACHES=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; CACHES=("$@"); break; fi
  ARTS+=("$1"); shift
done
H0_CACHE=/mnt/xbabe3-external/jsr/cache/librispeech_pcm
PANEL_JSON=$(git -C "$(dirname "$0")" show "$COMMIT:docs/ultra/data/dev_selection_panel.json")
C12=${COMMIT:0:12}
LOG=/mnt/xbabe3-external/jsr/runs/engine/staging-$HOST-$C12-$(date -u +%Y%m%dT%H%M%SZ)
[ -e "$LOG" ] && { echo "exists $LOG"; exit 1; }
mkdir -p "$LOG"
exec > >(tee "$LOG/stage.log") 2>&1
echo "staging $HOST at $COMMIT ($(date -u +%FT%TZ))"

# 1-2. Build in the existing checkout on the host (no clone, fetch or checkout).
BINS="ultra-engine-eval ultra-engine-density ultra-engine-prepare ultra-engine-gate ultra"
ssh "$HOST" bash -s "$CHECKOUT" "$COMMIT" "$BINS" > "$LOG/binaries.SHA256SUMS" <<'REMOTE'
set -euo pipefail
CO=$1 COMMIT=$2 BINS=$3
cd "$CO"
[ "$(git rev-parse HEAD)" = "$(git rev-parse --verify "$COMMIT^{commit}")" ] \
  || { echo "checkout HEAD is not the freeze commit" >&2; exit 3; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty checkout" >&2; exit 3; }
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$CO/target CARGO_BUILD_JOBS=12 CARGO_INCREMENTAL=0
nice -n 10 cargo build --locked --release -p ultra-engine -p ultra-cli --bins >&2
D=~/jsr-engine-data/bin-${COMMIT:0:12}
[ ! -e "$D" ] || { echo "$D exists (never overwrite)" >&2; exit 3; }
mkdir -p "$D"
for b in $BINS; do
  [ -x "$CARGO_TARGET_DIR/release/$b" ] || { echo "missing binary $b" >&2; exit 3; }
  cp "$CARGO_TARGET_DIR/release/$b" "$D/"
done
cd "$D" && sha256sum $BINS
REMOTE
[ "$(wc -l < "$LOG/binaries.SHA256SUMS")" -eq 5 ] || { echo "binary digests incomplete"; exit 3; }
cat "$LOG/binaries.SHA256SUMS"

# 3. Artifacts, verified file by file against H0.
for A in "${ARTS[@]}"; do
  NAME="$(basename "$(dirname "$A")")-$(basename "$A")"
  (cd "$A" && sha256sum -- *) > "$LOG/$NAME.SHA256SUMS"
  ssh "$HOST" "mkdir -p ~/jsr-engine-data/artifacts && [ ! -e ~/jsr-engine-data/artifacts/$NAME ]" \
    || { echo "artifact $NAME already staged on $HOST (never overwrite)"; exit 4; }
  tar -C "$(dirname "$A")" -cf - "$(basename "$A")" \
    | ssh "$HOST" "mkdir -p ~/jsr-engine-data/artifacts/.in && tar -C ~/jsr-engine-data/artifacts/.in -xf - && mv ~/jsr-engine-data/artifacts/.in/$(basename "$A") ~/jsr-engine-data/artifacts/$NAME"
  ssh "$HOST" "cd ~/jsr-engine-data/artifacts/$NAME && sha256sum -c --quiet -" < "$LOG/$NAME.SHA256SUMS"
  mv "$LOG/$NAME.SHA256SUMS" "$LOG/artifact.$NAME.SHA256SUMS"
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
  echo "$WANT  $S.npz  ${PANEL:+panel-manifest}${PANEL:-h0-only}" >> "$LOG/caches.SHA256SUMS"
  echo "cache $S verified ($WANT)"
done

# 5. Staging record (structured provenance of everything verified above).
touch "$LOG/caches.SHA256SUMS"
python3 - "$LOG" "$HOST" "$COMMIT" "$CHECKOUT" <<'PY'
import glob, json, os, sys, datetime
log, host, commit, checkout = sys.argv[1:]
def sums(path):
    out = {}
    for line in open(path).read().splitlines():
        if line.strip():
            parts = line.split()
            out[parts[1]] = {"sha256": parts[0], **({"reference": parts[2]} if len(parts) > 2 else {})}
    return out
rec = {"schema": "ultra.engine-staging/v2", "host": host, "commit": commit, "checkout": checkout,
       "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "binaries": sums(os.path.join(log, "binaries.SHA256SUMS")),
       "artifacts": {os.path.basename(p)[len("artifact."):-len(".SHA256SUMS")]: sums(p)
                     for p in sorted(glob.glob(os.path.join(log, "artifact.*.SHA256SUMS")))},
       "caches": sums(os.path.join(log, "caches.SHA256SUMS")),
       "verified": "binaries built from the clean freeze checkout; artifacts and caches "
                   "re-hashed on the host and equal to H0 (DEV caches also to the panel)"}
json.dump(rec, open(os.path.join(log, "staging.json"), "w"), indent=1)
PY
scp -q "$LOG/staging.json" "$HOST:jsr-engine-data/staging-$C12.json"
echo "DONE staging $HOST; record $LOG/staging.json"
