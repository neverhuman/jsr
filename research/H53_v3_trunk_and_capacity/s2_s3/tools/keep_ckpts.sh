#!/usr/bin/env bash
# Protect S2's branch-point checkpoints from the trainer's pruning (eval-lane 10:40Z) under a disk guard (lead 10:42Z):
# hard-link each completed checkpoint into ~/jsr-data/protected/<run>/ as it lands, then keep only
#   - every branch point: the first 500-step checkpoint past k x 10,000 audio-h (cumulative logged audio, train.jsonl),
#   - names listed in <protected>/KEEP (S2b-BP1, the cut, ...),
#   - the latest 2 checkpoints (and the latest model-*.pt snapshot),
# pruning the other links by name. Warns in AGENT_CHAT.md if the disk's free space drops below 60 GB.
#   keep_ckpts.sh <run_name> [<run_name> ...]          (run on x5 = node 0)
set -u
CHAT_HOST=${CHAT_HOST:-}
while :; do
  for N in "$@"; do
    R=$HOME/jsr-data/runs/train/$N; P=$HOME/jsr-data/protected/$N
    [ -d "$R" ] || continue
    mkdir -p "$P"; touch "$P/KEEP" "$P/.seen"
    for m in "$R"/ckpt-*.sha256 "$R"/model-*.sha256; do
      [ -e "$m" ] || continue
      f=${m%.sha256}.pt; b=$(basename "$f")
      grep -qx "$b" "$P/.seen" && continue  # each checkpoint is linked once; a pruned link is not re-made
      [ -e "$f" ] && ln "$f" "$P/$b" 2>/dev/null && ln -f "$m" "$P/$(basename "$m")" && echo "$b" >> "$P/.seen" && echo "$(date -u +%T) linked $N/$b"
    done
    keep=$(python3 - "$R" "$P" <<'PY'
import glob, json, os, re, sys
R, P = sys.argv[1], sys.argv[2]
keep = set(l.strip() for l in open(os.path.join(P, "KEEP")) if l.strip())
cum, k, bps = 0.0, 1, []
try:
    for l in open(os.path.join(R, "train.jsonl")):
        if '"kind": "train"' not in l:
            continue
        r = json.loads(l); cum += r["audio_s"]
        while cum >= k * 36e6:
            s = r["step"]; bps.append(s if s % 500 == 0 else (s // 500 + 1) * 500); k += 1
except FileNotFoundError:
    pass
keep |= {"ckpt-%08d.pt" % s for s in bps}
ck = sorted(os.path.basename(f) for f in glob.glob(os.path.join(P, "ckpt-*.pt")))
md = sorted(os.path.basename(f) for f in glob.glob(os.path.join(P, "model-*.pt")))
keep |= set(ck[-2:]) | set(md[-1:])
print(" ".join(sorted(keep)))
PY
)
    for f in "$P"/ckpt-*.pt "$P"/model-*.pt; do
      [ -e "$f" ] || continue
      b=$(basename "$f")
      case " $keep " in *" $b "*) ;; *) rm -f "$f" "${f%.pt}.sha256" && echo "$(date -u +%T) pruned link $N/$b";; esac
    done
  done
  free=$(df -BG --output=avail "$HOME/jsr-data" | tail -1 | tr -dc 0-9)
  if [ "${free:-999}" -lt 60 ] && [ ! -e /tmp/keep_ckpts_lowdisk ]; then
    touch /tmp/keep_ckpts_lowdisk; echo "$(date -u +%T) LOW DISK ${free} GB"
  fi
  sleep 10
done
