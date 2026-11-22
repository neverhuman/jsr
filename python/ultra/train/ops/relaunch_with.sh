#!/usr/bin/env bash
# Checkpoint a running trainer (SIGTERM saves one), then resume it under launch_guarded with the
# same arguments plus/overriding the given --set key=value pairs (e.g. loader=process workers=20).
#   relaunch_with.sh <run_dir> <cpu_list> <gpu_index> <key=value...>
set -uo pipefail
R=$1 CPUS=$2 GPU=$3; shift 3
P=$(cat "$R/PID"); G=$(cat "$R/GUARD_PID" 2>/dev/null)
ps -p "$P" -o args= | grep -q train.trainer || { echo "no trainer at $P"; exit 1; }
mapfile -t ARGS < <(python3 - "$R/config.json" "$@" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
argv = cfg["argv"][1:]
out, i = [], 0
while i < len(argv):
    if argv[i] == "--run-dir":
        i += 2
        continue
    out.append(argv[i]); i += 1
over = dict(kv.split("=", 1) for kv in sys.argv[2:])
j = out.index("--set")
sets = [kv for kv in out[j + 1:] if kv.split("=", 1)[0] not in over]
sets += [f"{k}={v}" for k, v in over.items()]
for a in out[:j + 1] + sets:
    print(a)
PY
)
kill -TERM "$P"
for _ in $(seq 1 120); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
kill -0 "$P" 2>/dev/null && { echo "trainer did not stop"; exit 1; }
[ -n "$G" ] && kill "$G" 2>/dev/null
cd "$(dirname "$0")/../.." || exit 1
"$PWD/train/launch_guarded.sh" "$R" "$CPUS" "$GPU" -- "${ARGS[@]}"
