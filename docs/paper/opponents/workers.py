"""Run N single-thread recognisers side by side, one per shard, as one process tree.

    python workers.py CPUS SHARDS RUN -- COMMAND...

CPUS is a comma list with one CPU per worker (Linux: each worker is pinned to its own CPU
before exec) or `none:N` for N unpinned workers (macOS has no affinity call). SHARDS is a
`shards-N` directory from workload.py. In COMMAND, `{wav}` becomes the worker's shard
directory, `{out}` its own output path (`RUN/worker-K.out`) and `{trace}` its trace path.
Each worker's stderr goes to `RUN/worker-K.stderr`. The launcher only waits; the caller
meters the whole tree, so N model loads are inside the cost, as they are for N services.
With `--time-l` every worker runs under `/usr/bin/time -l` so that a host without procfs
still yields a per-worker peak RSS (the tail of its stderr file).
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    args = sys.argv[1:]
    time_l = args[:1] == ["--time-l"]
    if time_l:
        args = args[1:]
    if len(args) < 5 or args[3] != "--":
        sys.exit("usage: workers.py [--time-l] CPUS SHARDS RUN -- COMMAND...")
    cpus, shards, run, command = args[0], Path(args[1]), Path(args[2]), args[4:]
    if cpus.startswith("none:"):
        slots = [None] * int(cpus.split(":")[1])
    else:
        slots = [int(cpu) for cpu in cpus.split(",")]
    parts = sorted(shards.glob("shard-*"), key=lambda path: int(path.name.split("-")[1]))
    if len(parts) != len(slots):
        sys.exit(f"{len(parts)} shards for {len(slots)} workers")
    children = []
    for index, (cpu, part) in enumerate(zip(slots, parts)):
        fill = {"wav": str(part), "out": str(run / f"worker-{index}.out"),
                "trace": str(run / f"worker-{index}.trace.jsonl")}
        argv = [token.format(**fill) for token in command]
        if time_l:
            argv = ["/usr/bin/time", "-l", *argv]
        pin = None if cpu is None else (lambda cpu=cpu: os.sched_setaffinity(0, {cpu}))
        with (run / f"worker-{index}.stderr").open("xb") as stderr:
            children.append(subprocess.Popen(argv, stdout=stderr, stderr=stderr, preexec_fn=pin))
    codes = [child.wait() for child in children]
    if any(codes):
        sys.exit(f"worker exit codes {codes}")


if __name__ == "__main__":
    main()
