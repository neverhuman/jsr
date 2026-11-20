"""SP-A fleet: paced real-time density across many pinned physical cores.

One worker process per physical core (SMT sibling left idle), each running `ultra.sp_a --paced`
with N streams for a fixed duration, writing into its own run directory. The run is VALID only if
every worker exited 0 and wrote a complete result file for this run_id. A worker sustains its N
streams iff its p99 lateness and last lateness are both below the bound and it drained every stream.

Timing labels: `--qualifying` takes the single host lease (BENCH_LEASE.host) and should only be used
inside a quiet window announced in AGENT_CHAT.md; otherwise the per-owner lease is taken and the
result is labelled exploratory.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import uuid

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")


def idle_cores(n: int, allowed: list[int], window_s: float = 3.0) -> list[int]:
    """The n physical cores in `allowed` whose two SMT threads were least busy over a short window."""
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
    ranked = sorted((busy[c] + busy[c + half], c) for c in allowed)
    return [c for _, c in ranked[:n]]


def _finite(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


GROWTH_FRACTION = 0.25  # last-third mean may exceed first-third mean by at most this share of the bound


def sustains(w: dict, bound_ms: float) -> bool:
    """Real time held: bounded tail and last lateness, no growing backlog, every stream drained."""
    lat = w.get("lateness_ms")
    if not isinstance(lat, dict) or w.get("drained") is not True:
        return False
    keys = ("p99", "last", "first_third_mean", "last_third_mean")
    if not all(_finite(lat.get(k)) and lat[k] >= 0 for k in keys):
        return False
    grows = lat["last_third_mean"] - lat["first_third_mean"] > GROWTH_FRACTION * bound_ms
    return lat["p99"] < bound_ms and lat["last"] < bound_ms and not grows


def _row_problem(w) -> str | None:
    """Structural validation of one worker result; None when usable."""
    for k in ("streams", "cpu_seconds", "audio_accepted_s", "rtf_cpu", "max_ready_batch"):
        if not _finite(w.get(k)) or w[k] < 0:
            return f"missing, non-finite or negative {k}"
    wer = w.get("wer")
    if not isinstance(wer, dict) or not all(_finite(wer.get(k)) and wer[k] >= 0 for k in ("ref_words", "sub", "del", "ins")):
        return "missing or malformed wer"
    return None


def summarise(run_dir: str, run_id: str, requested: int, cores: list[int], codes: list[int],
              bound_ms: float) -> dict:
    workers, problems = [], []
    if len(cores) != requested:
        problems.append(f"requested {requested} cores, got {len(cores)}")
    if len(codes) != requested:
        problems.append(f"expected {requested} exit codes, got {len(codes)}")
    problems += [f"worker {i}: exit {c}" for i, c in enumerate(codes) if c != 0]
    for i in range(requested):
        p = os.path.join(run_dir, f"worker_{i}.json")
        if not os.path.exists(p):
            problems.append(f"worker {i}: no result file")
            continue
        try:
            w = json.load(open(p))
        except ValueError:
            problems.append(f"worker {i}: malformed result")
            continue
        if not isinstance(w, dict):
            problems.append(f"worker {i}: result is not an object")
            continue
        if w.get("run_id") != run_id or w.get("worker_id") != i:
            problems.append(f"worker {i}: result belongs to another run")
            continue
        if i < len(cores) and w.get("affinity") != [cores[i]]:
            problems.append(f"worker {i}: ran on {w.get('affinity')}, expected [{cores[i]}]")
        bad = _row_problem(w)
        if bad:
            problems.append(f"worker {i}: {bad}")
            continue
        workers.append(w)
    valid = not problems
    ok = [w for w in workers if sustains(w, bound_ms)] if valid else []
    refs = sum(w["wer"]["ref_words"] for w in workers)
    errs = sum(w["wer"]["sub"] + w["wer"]["del"] + w["wer"]["ins"] for w in workers)
    acc = sum(w["audio_accepted_s"] for w in workers)
    return dict(valid=valid, problems=problems, workers=len(workers),
                sustaining_workers=len(ok), sustained_streams=sum(w["streams"] for w in ok),
                offered_streams=sum(w["streams"] for w in workers),
                pooled_wer=round(100.0 * errs / max(refs, 1), 3), ref_words=refs,
                audio_accepted_s=round(acc, 2),
                cpu_s_per_audio_s=round(sum(w["cpu_seconds"] for w in workers) / max(acc, 1e-9), 4),
                worst_p99_ms=max((w.get("lateness_ms", {}).get("p99", float("inf")) for w in workers), default=None),
                max_ready_batch=max((w["max_ready_batch"] for w in workers), default=0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--streams-per-core", type=int, default=12)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--duration-s", type=float, default=180.0)
    ap.add_argument("--decode", choices=["single", "batch"], default="batch")
    ap.add_argument("--grid-ms", type=int, default=320)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--bound-ms", type=float, default=320.0)
    ap.add_argument("--allowed", default="32-47", help="physical-core block owned by this lane (AGENT_CHAT partition)")
    ap.add_argument("--qualifying", action="store_true", help="take the host lease (announced quiet window only)")
    ap.add_argument("--out", default=os.path.join(DATA, "runs", "sp_a", "fleet.jsonl"))
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.allowed.split("-"))
    lease_path = os.path.join(DATA, "BENCH_LEASE.host" if args.qualifying else "BENCH_LEASE.lead")
    lease = open(lease_path, "w")
    fcntl.flock(lease, fcntl.LOCK_EX)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(DATA, "runs", "sp_a", "fleet", run_id)
    os.makedirs(run_dir)
    load_before = os.getloadavg()
    cores = idle_cores(args.cores, list(range(lo, hi + 1)))
    procs = []
    for i, c in enumerate(cores):
        cmd = ["taskset", "-c", str(c), sys.executable, "-m", "ultra.sp_a", "--paced", "--split", args.split,
               "--streams", str(args.streams_per_core), "--decode", args.decode, "--precision", args.precision,
               "--grid-ms", str(args.grid_ms), "--duration-s", str(args.duration_s), "--offset", str(i * 97),
               "--run-id", run_id, "--worker-id", str(i), "--out-dir", run_dir]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    codes = [p.wait() for p in procs]
    summary = dict(schema="ultra.sp-a-fleet/v2", run_id=run_id, run_dir=run_dir,
                   label="qualifying" if args.qualifying else "exploratory", cores=cores,
                   streams_per_core=args.streams_per_core, decode=args.decode, grid_ms=args.grid_ms,
                   precision=args.precision, split=args.split, duration_s=args.duration_s, bound_ms=args.bound_ms,
                   exit_codes=codes, load_before=load_before, load_after=os.getloadavg(),
                   **summarise(run_dir, run_id, args.cores, cores, codes, args.bound_ms))
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
