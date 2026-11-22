"""Meter a command on macOS, where `ultra-measure` cannot run (it reads Linux procfs).

    python bsd_time_measure.py SERVICE_STDERR [--workers RUN] -- COMMAND...

The command runs under `/usr/bin/time -l`. The JSON on stdout uses `ultra-measure`'s field
names so the receipts read the same, with these differences, all recorded in the output:

- CPU seconds are the kernel's rusage for the waited-for process tree (read through
  getrusage for microsecond precision; `/usr/bin/time -l` prints the same counters to two
  decimals and both are kept). There is no 200 ms sampling and no thread count.
- `peak_rss_bytes` is `maximum resident set size` of the largest process, not a sampled
  sum over the tree. With `--workers RUN` (N recognisers side by side, each under its own
  `/usr/bin/time -l`) it is the sum of the workers' maxima, an upper bound on the tree.
- `peak_footprint_bytes` is macOS's `peak memory footprint` for the same process(es).
- macOS has no CPU affinity: `placement` is null and the thread count is the only bound.
"""

import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

REAL = re.compile(r"^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys\s*$")
FIELD = re.compile(r"^\s*(\d+)\s+(maximum resident set size|peak memory footprint)\s*$")


def split_time_block(text):
    """Return (command stderr, parsed `/usr/bin/time -l` block) from a captured stream."""
    lines = text.splitlines(keepends=True)
    start = max(index for index, line in enumerate(lines) if REAL.match(line))
    real, user, system = map(float, REAL.match(lines[start]).groups())
    block = {"real": real, "user": user, "sys": system}
    for line in lines[start + 1:]:
        found = FIELD.match(line)
        if found:
            block[found.group(2).replace(" ", "_")] = int(found.group(1))
    return "".join(lines[:start]), block


def main():
    args = sys.argv[1:]
    service_stderr = Path(args[0])
    workers = None
    rest = args[1:]
    if rest[:1] == ["--workers"]:
        workers = Path(rest[1])
        rest = rest[2:]
    if rest[:1] != ["--"] or len(rest) < 2:
        sys.exit("usage: bsd_time_measure.py SERVICE_STDERR [--workers RUN] -- COMMAND...")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    done = subprocess.run(["/usr/bin/time", "-l", *rest[1:]], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    wall = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    own, block = split_time_block(done.stdout.decode("utf-8", "replace"))
    service_stderr.write_text(own, encoding="utf-8")
    peak_rss = block["maximum_resident_set_size"]
    footprint = block.get("peak_memory_footprint")
    scope = "maximum resident set size of the largest process in the tree"
    if workers is not None:
        blocks = [split_time_block(path.read_text(encoding="utf-8", errors="replace"))[1]
                  for path in sorted(workers.glob("worker-*.stderr"))]
        peak_rss = sum(item["maximum_resident_set_size"] for item in blocks)
        footprint = sum(item.get("peak_memory_footprint", 0) for item in blocks)
        scope = f"sum of {len(blocks)} workers' maximum resident set size"
    print(json.dumps({
        "cpu_seconds_user": after.ru_utime - before.ru_utime,
        "cpu_seconds_sys": after.ru_stime - before.ru_stime,
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss,
        "peak_footprint_bytes": footprint,
        "peak_rss_scope": scope,
        "exit_code": done.returncode,
        "meter": "/usr/bin/time -l + getrusage(RUSAGE_CHILDREN)",
        "bsd_time": block,
        "placement": None,
    }))
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
