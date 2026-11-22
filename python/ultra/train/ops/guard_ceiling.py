"""H0 watchdog for the owner's GPU ceiling (5-min mean <= 90%): every minute, read each guarded
run's guard log over the last 5 minutes; if the mean measured GPU utilisation exceeds the ceiling (default 0.89, one point under the
owner's 90% cap, since the fleet monitor's window reads slightly higher than the guard log),
restart that run's guard with the target lowered by `step` (Linux: restart_guard.sh; macOS:
the mac_guard relaunch). Logs every decision as JSON lines.

  python guard_ceiling.py [--ceiling 0.90] [--step 0.01] [--floor 0.80]
"""
import argparse
import json
import subprocess
import sys
import time

RUNS = {  # label: (host, run_dir, guard log name, kind); host "local" = H0 itself
    "D2": ("local", "/mnt/xbabe3-external/jsr/runs/train/D2_S_960h", "fleet_guard.jsonl", "linux"),
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


LOCAL_OPS = "/home/ubuntu/jsr/.agent/worktrees/trainer/python/ultra/train/ops"


def ssh(host, cmd, timeout=60):
    if host == "local":
        cmd = cmd.replace("~/jsr/python/ultra/train/ops/", LOCAL_OPS + "/")
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=20", host, cmd], capture_output=True,
                          text=True, timeout=timeout)


IOREG = ('for i in $(seq 1 20); do ioreg -r -d 1 -w 0 -c IOAccelerator '
         '| grep -o \'"Device Utilization %"=[0-9]*\' | head -1 | cut -d= -f2; sleep 1; done')


def mac_device_util(host):
    """Mean of 20 one-second ioreg "Device Utilization %" samples (the fleet monitor's metric)."""
    r = ssh(host, IOREG, 60)
    vals = [int(v) for v in r.stdout.split() if v.isdigit()]
    return sum(vals) / len(vals) / 100.0 if vals else None


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
    ap.add_argument("--ceiling", type=float, default=0.89)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--start", type=float, default=0.89)
    ap.add_argument("--mac-ceiling", type=float, default=0.88,
                    help="macOS: step down while the 5-min mean of ioreg Device Utilization exceeds this")
    ap.add_argument("--mac-floor", type=float, default=0.60)
    ap.add_argument("--targets", default="", help='current per-run targets as JSON, e.g. {"H13a": 0.88}')
    a = ap.parse_args(argv)
    if not 0 < a.floor <= a.start <= a.ceiling <= 1 or a.step <= 0:
        ap.error("need 0 < floor <= start <= ceiling <= 1 and step > 0")
    ap_mac = {"ceiling": a.mac_ceiling, "floor": a.mac_floor}
    mac_hist = {k: [] for k, v in RUNS.items() if v[3] == "mac"}
    last_change = {k: 0.0 for k in RUNS}
    target = {k: a.start for k in RUNS}
    target.update(json.loads(a.targets) if a.targets else {})
    while True:
        for label, (host, run, log, kind) in RUNS.items():
            ceiling, floor = a.ceiling, a.floor
            try:
                if kind == "mac":  # the fleet monitor reads ioreg on the Mac; mac_guard reads lower
                    u = mac_device_util(host)
                    if u is not None:
                        mac_hist[label] = (mac_hist[label] + [u])[-5:]
                    m = sum(mac_hist[label]) / len(mac_hist[label]) if mac_hist[label] else None
                    ceiling, floor = ap_mac["ceiling"], ap_mac["floor"]
                else:
                    m = mean_gpu(host, f"{run}/{log}")
            except subprocess.TimeoutExpired:
                m = None
            rec = {"t": time.time(), "run": label, "gpu_5min": m, "target": target[label],
                   "metric": "ioreg_device_util" if kind == "mac" else "guard_gpu_ema"}
            settled = time.time() - last_change[label] > 180
            if m is not None and m > ceiling and settled and target[label] - a.step >= floor:
                last_change[label] = time.time()
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
