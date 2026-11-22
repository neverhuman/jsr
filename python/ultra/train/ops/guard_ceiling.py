"""H0 watchdog for the owner's GPU ceiling (5-min mean <= 90%). Every minute it samples each
guarded run's GPU with the fleet monitor's own metric: nvidia-smi utilization.gpu on Linux,
ioreg "Device Utilization %" on the Mac (twenty 1-second samples, all hosts in parallel). It keeps
a rolling 5-minute mean per run. While that mean exceeds the ceiling (default 0.88), it restarts
the run's guard with the target lowered by `step` (Linux: restart_guard.sh; macOS: a mac_guard
relaunch), at most once per 3 minutes and never below the floor (0.60). The guards' own
estimates read lower than the monitor, hence a separate check. Logs JSON lines.

  python guard_ceiling.py [--mac-ceiling 0.88] [--step 0.01] [--mac-floor 0.60] [--targets JSON]
"""
import argparse
import json
import subprocess
import sys
import time

RUNS = {  # label: (host, run_dir, guard log name, kind); host "local" = H0 itself
    "D2": ("local", "/mnt/xbabe3-external/jsr/runs/train/D2_S_960h", "fleet_guard.jsonl", "linux"),
    "H22t": ("xbabe1", "/mnt/amphora-backup/jsr-data/runs/train/H22t_kd*", "fleet_guard.jsonl", "linux"),
    "H13a": ("xbabe2", "/home/ubuntu/jsr-data/runs/train/H13a_S_spm512_960h", "fleet_guard.jsonl", "linux"),
    "L3": ("xbabe7", "/home/ubuntu/jsr-data/runs/train/L3_M_MR_spm512_pruned_960h", "fleet_guard.jsonl", "linux"),
    "H22ctl": ("xbabe7", "/home/ubuntu/jsr-data/runs/train/H22ctl_S_r0bpe_pruned_960h", "fleet_guard.jsonl", "linux"),
    "H23": ("xbabe6", "/home/ubuntu/jsr-data/runs/train/H23_current_gpu0", "fleet_guard.jsonl", "linux"),
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


RESTART = ('S=~/jsr/.agent/worktrees/trainer/python/ultra/train/ops/restart_guard.sh; '
           '[ -x $S ] || S=~/jsr/python/ultra/train/ops/restart_guard.sh; $S {run} {t} 0.85 {gpu}')


def resolve(host, run):
    """A run spec with '*' (a chain of arms) resolves to the arm whose trainer is alive."""
    if "*" not in run:
        return run
    r = ssh(host, f'for d in {run}; do kill -0 $(cat $d/PID 2>/dev/null) 2>/dev/null && echo $d; done | head -1')
    return r.stdout.strip() or None


def ssh(host, cmd, timeout=60):
    if host == "local":
        cmd = cmd.replace("~/jsr/.agent/worktrees/trainer/python/ultra/train/ops/", LOCAL_OPS + "/")
        cmd = cmd.replace("~/jsr/python/ultra/train/ops/", LOCAL_OPS + "/")
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=20", host, cmd], capture_output=True,
                          text=True, timeout=timeout)


IOREG = ('for i in $(seq 1 20); do ioreg -r -d 1 -w 0 -c IOAccelerator '
         '| grep -o \'"Device Utilization %"=[0-9]*\' | head -1 | cut -d= -f2; sleep 1; done')


NVSMI = ('for i in $(seq 1 20); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits '
         '-i {gpu}; sleep 1; done')


GPU_INDEX = {"H13c": 1, "L3": 1, "L2": 1}  # every other run uses GPU 0 of its host


def linux_gpu_util(host, gpu=0):
    """Mean of 20 one-second nvidia-smi utilization.gpu samples (the fleet monitor's metric)."""
    r = ssh(host, NVSMI.format(gpu=gpu), 60)
    vals = [int(v) for v in r.stdout.split() if v.isdigit()]
    return sum(vals) / len(vals) / 100.0 if vals else None


def mac_device_util(host):
    """Mean of 20 one-second ioreg "Device Utilization %" samples (the fleet monitor's metric)."""
    r = ssh(host, IOREG, 60)
    vals = [int(v) for v in r.stdout.split() if v.isdigit()]
    return sum(vals) / len(vals) / 100.0 if vals else None


def live_target(host, run, kind):
    """--gpu-target of the running guard for this run (None if not found)."""
    run = resolve(host, run)
    if run is None:
        return None
    mod = "train.mac_guard" if kind == "mac" else "ultra.fleet_guard"
    r = ssh(host, f"P=$(cat {run}/PGID); ps -eo args | grep \"[{mod[0]}]{mod[1:]} --pgid $P \" "
                  f"| grep -o 'gpu-target [0-9.]*' | head -1")
    try:
        return float(r.stdout.split()[1])
    except (IndexError, ValueError):
        return None


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
                    help="step down while the 5-min mean of the monitor metric exceeds this (all hosts)")
    ap.add_argument("--mac-floor", type=float, default=0.60, help="lowest target (all hosts)")
    ap.add_argument("--mac-step", type=float, default=0.03, help="macOS step size (2-min cooldown)")
    ap.add_argument("--linux-step", type=float, default=0.02, help="Linux step size (2-min cooldown)")
    ap.add_argument("--linux-ceiling", type=float, default=0.87,
                    help="Linux: step down while the nvidia-smi 5-min mean exceeds this")
    ap.add_argument("--mac-ioreg-ceiling", type=float, default=0.86,
                    help="macOS: step down while the ioreg 5-min mean exceeds this")
    ap.add_argument("--targets", default="", help='current per-run targets as JSON, e.g. {"H13a": 0.88}')
    a = ap.parse_args(argv)
    if not 0 < a.floor <= a.start <= a.ceiling <= 1 or a.step <= 0:
        ap.error("need 0 < floor <= start <= ceiling <= 1 and step > 0")
    ap_mac = {"ceiling": a.mac_ceiling, "floor": a.mac_floor}
    mac_hist = {k: [] for k in RUNS}
    last_change = {k: 0.0 for k in RUNS}
    target = {k: a.start for k in RUNS}
    target.update(json.loads(a.targets) if a.targets else {})
    for k, v in RUNS.items():  # start from the guards' real targets
        try:
            live = live_target(v[0], v[1], v[3])
        except (subprocess.SubprocessError, OSError, ValueError):
            live = None
        if live is not None:
            target[k] = live
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=len(RUNS))
    while True:
        t0 = time.time()
        futs = {k: (pool.submit(mac_device_util, v[0]) if v[3] == "mac"
                    else pool.submit(linux_gpu_util, v[0], GPU_INDEX.get(k, 0)))
                for k, v in RUNS.items()}
        for label, (host, run, log, kind) in RUNS.items():
            ceiling, floor = a.ceiling, a.floor
            try:
                # The fleet monitor's metrics: ioreg Device Utilization on the Mac, nvidia-smi
                # utilization.gpu on Linux. The guards' own estimates read lower than both.
                u = futs[label].result(timeout=90)
                if u is not None:
                    mac_hist[label] = (mac_hist[label] + [u])[-5:]
                m = sum(mac_hist[label]) / len(mac_hist[label]) if mac_hist[label] else None
                ceiling, floor = ap_mac["ceiling"], ap_mac["floor"]
            except (subprocess.SubprocessError, OSError, TimeoutError, ValueError):
                # an unreachable or slow host must not stop the watchdog
                m = None
            rec = {"t": time.time(), "run": label, "gpu_5min": m, "target": target[label],
                   "metric": "ioreg_device_util" if kind == "mac" else "nvidia_smi_util"}
            # The Mac's ioreg signal is bursty: bigger, faster steps there.
            step, cooldown = (a.mac_step, 120) if kind == "mac" else (a.linux_step, 120)
            ceiling = a.mac_ioreg_ceiling if kind == "mac" else a.linux_ceiling
            settled = time.time() - last_change[label] > cooldown
            if m is not None and m > ceiling and settled:
                live = live_target(host, run, kind)  # the guard's actual target, not our copy
                if live is not None:
                    target[label] = live
            if m is not None and m > ceiling and settled and target[label] - step >= floor:
                last_change[label] = time.time()
                target[label] = round(target[label] - step, 3)
                if kind == "linux":
                    rd = resolve(host, run)
                    if rd:
                        ssh(host, RESTART.format(run=rd, t=target[label], gpu=GPU_INDEX.get(label, 0)), 120)
                else:
                    ssh(host, MAC_RESTART.format(r=run, t=target[label]), 120)
                rec["action"] = f"target -> {target[label]}"
            print(json.dumps(rec), flush=True)
        time.sleep(max(1.0, 60 - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
