#!/bin/bash
# h52-labeler: start one stage in its own process group (setsid) + fleet_guard on that PGID.
#   launch_worker.sh <base> <name> <gpu|front> <guard gpu index or -1> <gpu target> <cpu target> -- <args for lab_gpu.py / lab_front.py>
set -euo pipefail
base=$1; name=$2; kind=$3; gidx=$4; gt=$5; ct=$6; shift 6; [ "$1" = "--" ] && shift
source "$base/env.sh"
mkdir -p "$base/logs"
cd "$base/logs"
script=lab_gpu.py; [ "$kind" = front ] && script=lab_front.py
code=${H52LAB_CODE:-$base/code}
setsid nohup nice -n ${H52LAB_NICE:-10} $H52LAB_PY "$code/$script" "$@" > "$base/logs/$name.log" 2>&1 < /dev/null &
pid=$!
sleep 1
pgid=$(ps -o pgid= -p $pid | tr -d ' ')
setsid nohup /usr/bin/python3 "$code/fleet_guard.py" --pgid "$pgid" --gpu "$gidx" --gpu-target "$gt" --cpu-target "$ct" \
  --start-duty 1.0 --log "$base/logs/$name.guard.jsonl" > "$base/logs/$name.guard.out" 2>&1 < /dev/null &
gpid=$!
echo "{\"name\": \"$name\", \"host\": \"$(hostname)\", \"pid\": $pid, \"pgid\": $pgid, \"guard_pid\": $gpid, \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$base/logs/pids.jsonl"
