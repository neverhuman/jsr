"""Fleet guard: keep a host's mean GPU and CPU utilisation under a ceiling while our job runs.

The owner rule (2026-11-20): never more than 90% mean CPU/GPU utilisation on any xbabe host;
we compete with other people's processes. This guard throttles ONE of our jobs (a whole process
group) by a duty cycle of SIGCONT/SIGSTOP. Every period it measures the host's GPU utilisation
(nvidia-smi, all processes, not just ours) and mean CPU utilisation (/proc/stat), smooths them,
and lowers the job's duty when either exceeds its target, raising it again when there is
headroom. On exit it always resumes the job, so a crash of the guard never leaves training
frozen. CUDA kernels already queued complete during a stop; only the host side pauses.

    python -m ultra.fleet_guard --pgid <PGID> [--gpu 0] [--gpu-target 0.85] [--cpu-target 0.85]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

MIN_DUTY, MAX_DUTY = 0.10, 1.0


@dataclass
class Policy:
    gpu_target: float = 0.85
    cpu_target: float = 0.85
    step_down: float = 0.05
    step_up: float = 0.02
    margin: float = 0.05
    alpha: float = 0.3  # EMA weight of the newest sample


def next_duty(duty: float, gpu_ema: float | None, cpu_ema: float, p: Policy) -> float:
    """Lower duty when the host is over either target; raise it slowly when both have margin."""
    over = cpu_ema > p.cpu_target or (gpu_ema is not None and gpu_ema > p.gpu_target)
    roomy = cpu_ema < p.cpu_target - p.margin and (gpu_ema is None or gpu_ema < p.gpu_target - p.margin)
    if over:
        duty -= p.step_down
    elif roomy:
        duty += p.step_up
    return min(MAX_DUTY, max(MIN_DUTY, duty))


def ema(prev: float | None, x: float, alpha: float) -> float:
    return x if prev is None else (1 - alpha) * prev + alpha * x


def gpu_util(index: int) -> float | None:
    """Instant utilisation of one GPU in [0, 1], or None if there is no GPU / nvidia-smi failed."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip()) / 100.0 if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def cpu_times() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    return sum(v), v[3] + v[4]  # total, idle + iowait


def signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def run(args) -> int:
    p = Policy(gpu_target=args.gpu_target, cpu_target=args.cpu_target)
    duty, g_ema, c_ema = args.start_duty, None, None
    t_prev, i_prev = cpu_times()
    last_log = 0.0
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log = open(args.log, "a") if args.log else None
    try:
        while not stopping["flag"]:
            if not signal_group(args.pgid, signal.SIGCONT):
                break  # job finished
            # Sample the GPU across the WHOLE cycle (run and stop phases): an instant reading
            # taken right after a stop is biased low and would push the duty back up.
            samples: list[float] = []
            run_s, stop_s = duty * args.period, (1 - duty) * args.period
            end_run = time.monotonic() + run_s
            while time.monotonic() < end_run:
                g = gpu_util(args.gpu) if args.gpu >= 0 else None
                if g is not None:
                    samples.append(g)
                time.sleep(min(args.sample, max(0.0, end_run - time.monotonic())))
            if duty < MAX_DUTY:
                signal_group(args.pgid, signal.SIGSTOP)
                end_stop = time.monotonic() + stop_s
                while time.monotonic() < end_stop:
                    g = gpu_util(args.gpu) if args.gpu >= 0 else None
                    if g is not None:
                        samples.append(g)
                    time.sleep(min(args.sample, max(0.0, end_stop - time.monotonic())))
            t, i = cpu_times()
            cpu = 1 - (i - i_prev) / max(1, t - t_prev)
            t_prev, i_prev = t, i
            g = sum(samples) / len(samples) if samples else None
            c_ema = ema(c_ema, cpu, p.alpha)
            g_ema = ema(g_ema, g, p.alpha) if g is not None else g_ema
            duty = next_duty(duty, g_ema, c_ema, p)
            now = time.time()
            if log and now - last_log >= args.log_every:
                log.write(json.dumps({"t": round(now, 1), "host": os.uname().nodename, "pgid": args.pgid,
                                      "duty": round(duty, 3), "gpu_ema": None if g_ema is None else round(g_ema, 3),
                                      "cpu_ema": round(c_ema, 3)}) + "\n")
                log.flush()
                last_log = now
    finally:
        signal_group(args.pgid, signal.SIGCONT)  # never leave the job frozen
        if log:
            log.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgid", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to watch; -1 for CPU-only jobs")
    ap.add_argument("--gpu-target", type=float, default=0.85)
    ap.add_argument("--cpu-target", type=float, default=0.85)
    ap.add_argument("--period", type=float, default=4.0, help="cycle length; stops must outlast the CUDA queue")
    ap.add_argument("--sample", type=float, default=0.25, help="GPU sampling interval within a cycle")
    ap.add_argument("--start-duty", type=float, default=0.85)
    ap.add_argument("--log", help="append JSON lines here")
    ap.add_argument("--log-every", type=float, default=30.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
