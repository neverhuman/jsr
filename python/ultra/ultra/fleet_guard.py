"""Fleet guard: keep a host's mean GPU and CPU utilisation under a ceiling while our job runs.

The owner rule (2026-11-20): never more than 90% mean CPU/GPU utilisation on any xbabe host;
we compete with other people's processes. This guard throttles ONE of our jobs (a whole process
group) by a duty cycle of SIGCONT/SIGSTOP. Every period it measures the host's GPU utilisation
(nvidia-smi, all processes, not just ours) and mean CPU utilisation (/proc/stat), smooths them,
and lowers the job's duty when either exceeds its target, raising it again when there is
headroom.

GPU measurement (2026-11-22): a long-lived `nvidia-smi -lms 2000` stream feeds a rolling 5-minute mean, the same statistic the
owner's cap watch uses; paused time is included because sampling never stops. The controller
steps on that mean (slowly) and on a 30-second mean (fast, only well above target). The old
per-cycle EMA tracked bursts and read ~10 points below the 5-minute nvidia-smi mean on xbabe2.
`--gpu` is the nvidia-smi (physical, PCI-order) index; nvidia-smi ignores CUDA_VISIBLE_DEVICES. On exit it always resumes the job, so a crash of the guard never leaves training
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


def next_duty(duty: float, gpu_ema: float | None, cpu_ema: float, p: Policy, min_duty: float = MIN_DUTY) -> float:
    """Lower duty when the host is over either target; raise it slowly when both have margin."""
    over = cpu_ema > p.cpu_target or (gpu_ema is not None and gpu_ema > p.gpu_target)
    roomy = cpu_ema < p.cpu_target - p.margin and (gpu_ema is None or gpu_ema < p.gpu_target - p.margin)
    if over:
        duty -= p.step_down
    elif roomy:
        duty += p.step_up
    return min(MAX_DUTY, max(min_duty, duty))


def next_duty_windowed(duty: float, gpu_5min: float | None, gpu_30s: float | None, cpu_ema: float,
                       p: Policy, min_duty: float = MIN_DUTY) -> float:
    """GPU control on the 5-minute mean (small steps, deadband 0.03), plus a fast step down when
    the 30-second mean is far over target; CPU as before."""
    if cpu_ema > p.cpu_target or (gpu_30s is not None and gpu_30s > p.gpu_target + 0.10):
        duty -= p.step_down
    elif gpu_5min is not None and gpu_5min > p.gpu_target:
        duty -= 0.01
    elif cpu_ema < p.cpu_target - p.margin and (gpu_5min is None or gpu_5min < p.gpu_target - 0.03):
        duty += 0.005
    return min(MAX_DUTY, max(min_duty, duty))


class GpuWindow:
    """Background nvidia-smi stream (one reading per `period`), rolling means over the last N seconds."""

    def __init__(self, index: int, period: float = 2.0, horizon: float = 300.0):
        import collections
        import threading

        self.index, self.period, self.horizon = index, period, horizon
        self.samples: collections.deque = collections.deque()
        self._lock = threading.Lock()
        if index >= 0:
            threading.Thread(target=self._loop, name="gpu-window", daemon=True).start()

    def add(self, t: float, u: float) -> None:
        with self._lock:
            self.samples.append((t, u))
            while self.samples and self.samples[0][0] < t - self.horizon:
                self.samples.popleft()

    def mean(self, seconds: float, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        with self._lock:
            xs = [u for t, u in self.samples if t >= now - seconds]
        return sum(xs) / len(xs) if xs else None

    def _loop(self) -> None:
        """One long-lived `nvidia-smi -lms` stream: readings arrive on nvidia-smi's own clock,
        independent of this process's scheduling and of the run/stop cycle. Note that a stopped
        job's already-queued kernels keep the GPU busy, which the stream reports as busy."""
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
               "-i", str(self.index), "-lms", str(int(self.period * 1000))]
        while True:
            try:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True) as proc:
                    for line in proc.stdout:
                        try:
                            self.add(time.monotonic(), float(line.split()[0]) / 100.0)
                        except (IndexError, ValueError):
                            continue
            except OSError:
                pass
            time.sleep(5.0)  # nvidia-smi exited or is missing: retry


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
    gw = GpuWindow(args.gpu)
    try:
        while not stopping["flag"]:
            if not signal_group(args.pgid, signal.SIGCONT):
                break  # job finished
            run_s, stop_s = duty * args.period, (1 - duty) * args.period
            time.sleep(run_s)
            if duty < MAX_DUTY:
                signal_group(args.pgid, signal.SIGSTOP)
                time.sleep(stop_s)
            t, i = cpu_times()
            cpu = 1 - (i - i_prev) / max(1, t - t_prev)
            t_prev, i_prev = t, i
            c_ema = ema(c_ema, cpu, p.alpha)
            g_ema, g_30 = gw.mean(300.0), gw.mean(30.0)
            if g_ema is not None and len(gw.samples) < 15:  # < 30 s of samples: hold the GPU term
                g_ema = None
            duty = next_duty_windowed(duty, g_ema, g_30, c_ema, p, args.min_duty)
            now = time.time()
            if log and now - last_log >= args.log_every:
                log.write(json.dumps({"t": round(now, 1), "host": os.uname().nodename, "pgid": args.pgid,
                                      "duty": round(duty, 3), "gpu_ema": None if g_ema is None else round(g_ema, 3),
                                      "gpu_5min": None if g_ema is None else round(g_ema, 3),
                                      "gpu_30s": None if g_30 is None else round(g_30, 3),
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
    ap.add_argument("--min-duty", type=float, default=MIN_DUTY,
                    help="floor of the duty; 0 lets the guard pause our job completely while others fill the host")
    ap.add_argument("--log", help="append JSON lines here")
    ap.add_argument("--log-every", type=float, default=30.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
