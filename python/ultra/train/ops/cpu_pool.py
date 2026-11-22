"""Self-regulating elastic CPU worker pool targeting ~80% total host CPU (`nice -n 15`).

Runs disjoint background CPU tasks (e.g., R0 teacher-cache shards, full-DEV beam-4 evals,
sMBR N-best generation) across explicit pinned cores without interfering with GPU trainers
or co-tenant workloads:
  * samples host CPU utilisation (`/proc/stat` on Linux, loadavg fallback on macOS) and
    available RAM (`/proc/meminfo` on Linux) every `period_s` seconds;
  * suspends (`SIGSTOP`) excess owned background workers whenever host CPU > `cpu_high`
    or available RAM < `min_free_ram_gb`, and resumes (`SIGCONT`) / spawns workers whenever
    host CPU < `cpu_low` and RAM has headroom;
  * NEVER signals processes outside its own spawned worker pool, and ALWAYS resumes /
    terminates its workers cleanly on exit (`SIGTERM`/`SIGINT`).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class PoolPolicy:
    cpu_low: float = 0.75
    cpu_high: float = 0.83
    min_free_ram_gb: float = 12.0
    max_workers: int = 16


def parse_cores(spec: str) -> list[int]:
    """Parse a taskset-style core list like '0-7,16-23,32' into sorted unique ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid core range: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def read_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as fh:
            v = list(map(int, fh.readline().split()[1:]))
        return sum(v), v[3] + v[4]
    except (OSError, ValueError, IndexError):
        return None


def read_free_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return None


def sample_host_cpu(prev: tuple[int, int] | None) -> tuple[float, tuple[int, int] | None]:
    cur = read_cpu_times()
    if cur is not None and prev is not None:
        dt = max(1, cur[0] - prev[0])
        di = max(0, cur[1] - prev[1])
        return max(0.0, min(1.0, 1.0 - di / dt)), cur
    if cur is not None:
        return 0.0, cur
    ncpu = os.cpu_count() or 1
    la1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    return max(0.0, min(1.0, la1 / ncpu)), None


def decide_target_active(active: int, total_slots: int, cpu: float, free_ram_gb: float | None,
                         p: PoolPolicy) -> int:
    """Closed-loop controller keeping total host CPU in [cpu_low, cpu_high] with RAM guard."""
    cap = min(total_slots, p.max_workers)
    if cap <= 0:
        return 0
    low_ram = free_ram_gb is not None and free_ram_gb < p.min_free_ram_gb
    if low_ram or cpu > p.cpu_high:
        step = 2 if (low_ram or cpu > p.cpu_high + 0.05) else 1
        return max(0, active - step)
    if cpu < p.cpu_low and active < cap:
        step = 2 if cpu < p.cpu_low - 0.15 else 1
        return min(cap, active + step)
    return min(active, cap)


def _signal_worker(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def _build_cmd(cmd_template: str, core: int, slot: int, nice_val: int) -> list[str]:
    rendered = cmd_template.replace("{core}", str(core)).replace("{slot}", str(slot))
    parts = shlex.split(rendered)
    prefix: list[str] = []
    if platform.system() == "Linux":
        prefix.extend(["taskset", "-c", str(core)])
    prefix.extend(["nice", "-n", str(nice_val)])
    return prefix + parts


def run_pool(cores: list[int], cmd_template: str, policy: PoolPolicy, *,
             period_s: float = 2.0, nice_val: int = 15, log_path: str | None = None,
             log_every: float = 30.0) -> int:
    """Spawn one worker per core (up to `policy.max_workers`) and regulate active count."""
    slots = cores[:policy.max_workers]
    procs: list[subprocess.Popen] = []
    paused: list[bool] = []
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    prev_cpu = read_cpu_times()
    time.sleep(0.25)
    log_fh = open(log_path, "a") if log_path else None
    last_log = 0.0
    try:
        for slot, core in enumerate(slots):
            cmd = _build_cmd(cmd_template, core, slot, nice_val)
            p = subprocess.Popen(cmd, start_new_session=True)
            procs.append(p)
            paused.append(False)

        while not stopping["flag"]:
            alive_idxs = [i for i, p in enumerate(procs) if p.poll() is None]
            if not alive_idxs:
                break
            cpu, prev_cpu = sample_host_cpu(prev_cpu)
            ram = read_free_ram_gb()
            running_idxs = [i for i in alive_idxs if not paused[i]]
            paused_idxs = [i for i in alive_idxs if paused[i]]
            want = decide_target_active(len(running_idxs), len(alive_idxs), cpu, ram, policy)
            while len(running_idxs) > want and running_idxs:
                idx = running_idxs.pop()
                _signal_worker(procs[idx], signal.SIGSTOP)
                paused[idx] = True
            while len(running_idxs) < want and paused_idxs:
                idx = paused_idxs.pop(0)
                _signal_worker(procs[idx], signal.SIGCONT)
                paused[idx] = False
                running_idxs.append(idx)
            now = time.time()
            if log_fh and now - last_log >= log_every:
                log_fh.write(json.dumps({
                    "t": round(now, 1), "host": platform.node(),
                    "cpu": round(cpu, 3), "free_ram_gb": None if ram is None else round(ram, 1),
                    "alive": len(alive_idxs), "active": len(running_idxs),
                    "paused": len(alive_idxs) - len(running_idxs),
                    "target_band": [policy.cpu_low, policy.cpu_high],
                }) + "\n")
                log_fh.flush()
                last_log = now
            time.sleep(period_s)
    finally:
        for i, p in enumerate(procs):
            if p.poll() is None:
                if paused[i]:
                    _signal_worker(p, signal.SIGCONT)
                if stopping["flag"]:
                    _signal_worker(p, signal.SIGTERM)
        if stopping["flag"]:
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _signal_worker(p, signal.SIGKILL)
        if log_fh:
            log_fh.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", required=True, help="taskset-style core list, e.g. 0-31,48-63")
    ap.add_argument("--cmd", required=True, help="worker command template ({core}, {slot})")
    ap.add_argument("--max-workers", type=int, default=0, help="0 = len(cores)")
    ap.add_argument("--cpu-low", type=float, default=0.75)
    ap.add_argument("--cpu-high", type=float, default=0.83)
    ap.add_argument("--min-free-ram-gb", type=float, default=12.0)
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--log", default=None)
    ap.add_argument("--log-every", type=float, default=30.0)
    a = ap.parse_args(argv)
    cores = parse_cores(a.cores)
    policy = PoolPolicy(
        cpu_low=a.cpu_low, cpu_high=a.cpu_high,
        min_free_ram_gb=a.min_free_ram_gb,
        max_workers=a.max_workers or len(cores),
    )
    return run_pool(cores, a.cmd, policy, period_s=a.period, nice_val=a.nice,
                    log_path=a.log, log_every=a.log_every)


if __name__ == "__main__":
    sys.exit(main())
