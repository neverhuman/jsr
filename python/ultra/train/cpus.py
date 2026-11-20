"""Pick idle logical CPUs from /proc/stat (two samples) for pinning CPU training with taskset.

  python -m train.cpus --n 32 --interval 5   ->  prints a taskset list such as 64-79,96-111
Prefers whole physical cores (both SMT siblings idle) when enough exist.
"""
from __future__ import annotations

import argparse
import os
import time


def _sample() -> dict[int, tuple[int, int]]:
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                vals = list(map(int, f[1:]))
                idle = vals[3] + vals[4]
                out[int(f[0][3:])] = (idle, sum(vals))
    return out


def busy_fractions(interval: float) -> dict[int, float]:
    a = _sample()
    time.sleep(interval)
    b = _sample()
    return {c: 1.0 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}


def siblings(cpu: int) -> list[int]:
    p = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(p) as fh:
            txt = fh.read().strip()
    except OSError:
        return [cpu]
    out = []
    for part in txt.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def pick(n: int, interval: float = 5.0, max_busy: float = 0.10) -> list[int]:
    busy = busy_fractions(interval)
    idle = {c for c, f in busy.items() if f <= max_busy}
    whole = sorted({min(siblings(c)) for c in idle if set(siblings(c)) <= idle})
    chosen: list[int] = []
    for c in whole:  # one thread per idle physical core first
        if len(chosen) < n:
            chosen.append(c)
    for c in sorted(idle):
        if len(chosen) < n and c not in chosen:
            chosen.append(c)
    return sorted(chosen)


def fmt(cpus: list[int]) -> str:
    parts, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-busy", type=float, default=0.10)
    a = ap.parse_args()
    cpus = pick(a.n, a.interval, a.max_busy)
    if len(cpus) < a.n:
        raise SystemExit(f"only {len(cpus)} idle CPUs (wanted {a.n}): {fmt(cpus)}")
    print(fmt(cpus))
    print(f"# {len(cpus)} cpus; load {os.getloadavg()}", flush=True)


if __name__ == "__main__":
    main()
