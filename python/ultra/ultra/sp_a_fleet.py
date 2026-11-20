"""SP-A fleet: paced real-time density across many pinned physical cores.

One worker process per physical core (SMT sibling left idle), each running `ultra.sp_a --paced`
with N streams for a fixed duration. A worker sustains its N streams iff p99 lateness and final
lateness are both below the bound. Total sustained streams = sum of N over sustaining workers.
Process-tree CPU is each worker's own getrusage (inference threads live in the worker).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def idle_cores(n: int, exclude: set[int], window_s: float = 3.0) -> list[int]:
    """Physical cores (0..63) whose two SMT threads were least busy over a short window."""
    def snap():
        out = {}
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("cpu") and line[3].isdigit():
                    p = line.split()
                    v = list(map(int, p[1:]))
                    out[int(p[0][3:])] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(window_s)
    b = snap()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a}
    half = len(busy) // 2
    ranked = sorted((busy[c] + busy[c + half], c) for c in range(half) if c not in exclude)
    return [c for _, c in ranked[:n]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--streams-per-core", type=int, default=12)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--duration-s", type=float, default=180.0)
    ap.add_argument("--decode", choices=["single", "batch"], default="batch")
    ap.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--bound-ms", type=float, default=320.0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    exclude = {int(x) for x in args.exclude.split(",") if x}
    cores = idle_cores(args.cores, exclude)
    tmp = args.out + ".workers.jsonl"
    procs = []
    for i, c in enumerate(cores):
        cmd = ["taskset", "-c", str(c), sys.executable, "-m", "ultra.sp_a", "--paced", "--split", args.split,
               "--streams", str(args.streams_per_core), "--decode", args.decode, "--precision", args.precision,
               "--duration-s", str(args.duration_s), "--offset", str(i * 97), "--out", tmp]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    codes = [p.wait() for p in procs]
    workers = [json.loads(line) for line in open(tmp)]
    ok = [w for w in workers if w["lateness_ms"]["p99"] < args.bound_ms and w["lateness_ms"]["final"] < args.bound_ms]
    refs = sum(w["wer"]["ref_words"] for w in workers)
    errs = sum(w["wer"]["sub"] + w["wer"]["del"] + w["wer"]["ins"] for w in workers)
    summary = dict(schema="ultra.sp-a-fleet/v1", cores=cores, streams_per_core=args.streams_per_core,
                   decode=args.decode, precision=args.precision, split=args.split, duration_s=args.duration_s,
                   workers=len(workers), exit_codes=codes, sustaining_workers=len(ok),
                   sustained_streams=sum(w["streams"] for w in ok), offered_streams=sum(w["streams"] for w in workers),
                   pooled_wer=round(100.0 * errs / max(refs, 1), 3), ref_words=refs,
                   worst_p99_ms=max(w["lateness_ms"]["p99"] for w in workers),
                   cpu_s_per_audio_s=round(sum(w["cpu_seconds"] for w in workers)
                                           / max(sum(w["audio_seconds"] for w in workers), 1e-9), 4))
    with open(args.out, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
