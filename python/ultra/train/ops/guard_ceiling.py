"""H0 watchdog for the owner's GPU ceiling (5-min mean <= 90%): every minute, read each guarded
run's guard log over the last 5 minutes; if the mean measured GPU utilisation exceeds the ceiling,
restart that run's guard with the target lowered by `step` (Linux: restart_guard.sh; macOS:
the mac_guard relaunch). Logs every decision as JSON lines.

  python guard_ceiling.py [--ceiling 0.90] [--step 0.01] [--floor 0.80]
"""
import argparse
import json
import subprocess
import sys
import time

RUNS = {  # label: (host, run_dir, guard log name, kind)
    "D3MRc": ("xbabe1", "/mnt/amphora-backup/jsr-data/runs/train/D3MRc_S_MR_I8S_960h", "fleet_guard.jsonl", "linux"),
    "H13a": ("xbabe2", "/home/ubuntu/jsr-data/runs/train/H13a_S_spm512_960h", "fleet_guard.jsonl", "linux"),
    "D4M": ("xbabe4", "/Users/agent/jsr-train/data/runs/train/D4M_M_MR_960h", "mac_guard.jsonl", "mac"),
}
MAC_RESTART = (
    'R={r}; PG=$(cat $R/PGID); PY=~/jsr-train/venv/bin/python; '
    'for g in $(pgrep -f "train.mac_guard --pgid $PG "); do kill -TERM $g; done; sleep 5; '
    'cd ~/jsr-train/jsr/python/ultra; nohup $PY -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" '
    'nice -n 10 $PY -m train.mac_guard --pgid $PG --gpu-target {t} --cpu-target 0.85 --min-duty 0.0 '
    '--log $R/mac_guard.jsonl >> $R/guard.log 2>&1 < /dev/null & sleep 3; '
    'pgrep -f "train.mac_guard --pgid $PG " | head -1 > $R/GUARD_PID')


def ssh(host, cmd, timeout=60):
    return subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=20", host, cmd], capture_output=True,
                          text=True, timeout=timeout)


def mean_gpu(host, path, window_s=300):
    r = ssh(host, f"tail -400 {path}")
    rows = []
    for line in r.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        return None
    t_end = rows[-1]["t"]
    vals = [x["gpu_ema"] for x in rows if x.get("gpu_ema") is not None and x["t"] >= t_end - window_s]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ceiling", type=float, default=0.90)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--start", type=float, default=0.89)
    a = ap.parse_args(argv)
    if not 0 < a.floor <= a.start <= a.ceiling <= 1 or a.step <= 0:
        ap.error("need 0 < floor <= start <= ceiling <= 1 and step > 0")
    target = {k: a.start for k in RUNS}
    while True:
        for label, (host, run, log, kind) in RUNS.items():
            try:
                m = mean_gpu(host, f"{run}/{log}")
            except subprocess.TimeoutExpired:
                m = None
            rec = {"t": time.time(), "run": label, "gpu_5min": m, "target": target[label]}
            if m is not None and m > a.ceiling and target[label] - a.step >= a.floor:
                target[label] = round(target[label] - a.step, 3)
                if kind == "linux":
                    ssh(host, f"~/jsr/python/ultra/train/ops/restart_guard.sh {run} {target[label]}", 120)
                else:
                    ssh(host, MAC_RESTART.format(r=run, t=target[label]), 120)
                rec["action"] = f"target -> {target[label]}"
            print(json.dumps(rec), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
