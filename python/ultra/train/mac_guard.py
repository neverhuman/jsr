"""macOS (Apple silicon) fleet guard: the same duty-cycle policy as ultra.fleet_guard, with the
host's GPU utilisation read from IOAccelerator ("Device Utilization %", all processes) and CPU
utilisation from `top`. Owner rule: never more than 90% mean CPU/GPU on any host.

  python -m train.mac_guard --pgid <PGID> [--gpu-target 0.74] [--cpu-target 0.85] [--min-duty 0]

Every period the job's process group gets SIGCONT for duty x period and SIGSTOP for the rest.
Duty falls by 0.05 while either smoothed reading is above its target and rises by 0.02 while both
are 0.05 below. On exit (signal or job gone) the job is always resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time

STEP_DOWN, STEP_UP, MARGIN, ALPHA, MAX_DUTY = 0.05, 0.02, 0.05, 0.3, 1.0


def gpu_util() -> float | None:
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    vals = [int(v) for v in re.findall(r'"Device Utilization %"=(\d+)', out)]
    return max(vals) / 100.0 if vals else None


def cpu_util() -> float | None:
    try:
        out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.findall(r"CPU usage: .*?([\d.]+)% idle", out)
    return 1.0 - float(m[-1]) / 100.0 if m else None


def next_duty(duty: float, g: float | None, c: float | None, gt: float, ct: float,
              min_duty: float) -> float:
    over = (g is not None and g > gt) or (c is not None and c > ct)
    roomy = (g is None or g < gt - MARGIN) and (c is None or c < ct - MARGIN)
    if over:
        duty -= STEP_DOWN
    elif roomy:
        duty += STEP_UP
    return min(MAX_DUTY, max(min_duty, duty))


def signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pgid", type=int, required=True)
    ap.add_argument("--gpu-target", type=float, default=0.74)
    ap.add_argument("--cpu-target", type=float, default=0.85)
    ap.add_argument("--min-duty", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=4.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--log-every", type=float, default=30.0)
    a = ap.parse_args()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    duty, g_ema, c_ema, last = 0.85, None, None, 0.0
    log = open(a.log, "a") if a.log else None
    try:
        while not stop["flag"]:
            if not signal_group(a.pgid, signal.SIGCONT):
                break
            t_end = time.monotonic() + duty * a.period
            gs = []
            while time.monotonic() < t_end:
                g = gpu_util()
                if g is not None:
                    gs.append(g)
                time.sleep(0.25)
            if duty < MAX_DUTY:
                signal_group(a.pgid, signal.SIGSTOP)
                t_end = time.monotonic() + (1 - duty) * a.period
                while time.monotonic() < t_end:
                    g = gpu_util()
                    if g is not None:
                        gs.append(g)
                    time.sleep(0.25)
            g = sum(gs) / len(gs) if gs else None
            c = cpu_util()
            g_ema = g if g_ema is None else (g_ema if g is None else (1 - ALPHA) * g_ema + ALPHA * g)
            c_ema = c if c_ema is None else (c_ema if c is None else (1 - ALPHA) * c_ema + ALPHA * c)
            duty = next_duty(duty, g_ema, c_ema, a.gpu_target, a.cpu_target, a.min_duty)
            now = time.time()
            if log and now - last >= a.log_every:
                log.write(json.dumps({"t": round(now, 1), "host": os.uname().nodename,
                                      "pgid": a.pgid, "duty": round(duty, 3),
                                      "gpu_ema": None if g_ema is None else round(g_ema, 3),
                                      "cpu_ema": None if c_ema is None else round(c_ema, 3)})
                          + "\n")
                log.flush()
                last = now
    finally:
        signal_group(a.pgid, signal.SIGCONT)
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
