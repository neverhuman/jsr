"""Host-state evidence and scoring for the open opponent receipts (standard library only).

    open_receipt.py snapshot                                  per-CPU counters from /proc/stat, as JSON
    open_receipt.py cpus LIST                                 expand a CPU list ("8-15" -> "8 9 ... 15")
    open_receipt.py gate --need K [--limit 25]                exit 0 if K more benchmark cores fit on the host
    open_receipt.py guard BEFORE AFTER --service-cpus LIST    was the arm alone on its CCX? exit 3 if not
    open_receipt.py score PANEL RUN SCORER SERVICE_CPUS       `paper_dev_audio.py score` for any CPU list
    open_receipt.py trace-summary TRACE_JSONL                 totals of a trace v1 file, as JSON

`guard` implements the placement rule for plotted rows: the arm alone on its CCX, the other
cores and every SMT sibling of that CCX idle. It also records every other CPU that was busy
during the repetition, so the receipt says which other benchmark cores were active.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

SYS_CPU = Path("/sys/devices/system/cpu")
ACTIVE = 0.5  # a core this busy counts as an active benchmark core


def parse_cpus(text):
    cpus = set()
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return sorted(cpus)


def snapshot():
    counters = {}
    for line in Path("/proc/stat").read_text().splitlines():
        name, *values = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            counters[name[3:]] = [int(value) for value in values[:8]]
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "monotonic_s": time.monotonic(),
        "loadavg": Path("/proc/loadavg").read_text().split()[:3],
        "cpus": counters,
    }


def busy_fractions(before, after):
    """Fraction of each CPU's time spent outside idle and iowait between two snapshots."""
    busy = {}
    for cpu, start in before["cpus"].items():
        delta = [b - a for a, b in zip(start, after["cpus"][cpu], strict=True)]
        total = sum(delta)
        busy[int(cpu)] = 0.0 if total <= 0 else 1.0 - (delta[3] + delta[4]) / total
    return busy


def topology_list(cpu, relative, root=SYS_CPU):
    return parse_cpus((root / f"cpu{cpu}" / relative).read_text().strip())


def placement(service, root=SYS_CPU):
    """SMT siblings and the remaining CPUs of the service CPUs' L3 domains (their CCXs)."""
    siblings, domain = set(), set()
    for cpu in service:
        siblings.update(topology_list(cpu, "topology/thread_siblings_list", root))
        domain.update(topology_list(cpu, "cache/index3/shared_cpu_list", root))
    siblings -= set(service)
    return sorted(siblings), sorted(domain - set(service) - siblings)


def active_cores(busy, exclude=(), root=SYS_CPU):
    """Physical cores (as lowest sibling CPU) with a logical CPU at least ACTIVE busy."""
    cores = set()
    for cpu, fraction in busy.items():
        if fraction >= ACTIVE and cpu not in exclude:
            cores.add(min(topology_list(cpu, "topology/thread_siblings_list", root)))
    return sorted(cores)


def guard(before, after, service, sibling_limit, neighbour_limit, root=SYS_CPU):
    busy = busy_fractions(before, after)
    siblings, neighbours = placement(service, root)
    reasons = []
    for cpu in siblings:
        if busy[cpu] > sibling_limit:
            reasons.append(f"SMT sibling cpu{cpu} was {busy[cpu]:.2%} busy")
    for cpu in neighbours:
        if busy[cpu] > neighbour_limit:
            reasons.append(f"CCX neighbour cpu{cpu} was {busy[cpu]:.2%} busy")
    others = {cpu: round(fraction, 4) for cpu, fraction in sorted(busy.items())
              if cpu not in service and fraction >= 0.05}
    return {
        "schema": "open-opponents.host-guard/v1",
        "rule": "arm alone on its CCX: other cores and all SMT siblings of the CCX idle",
        "started_utc": before["utc"], "finished_utc": after["utc"],
        "seconds": round(after["monotonic_s"] - before["monotonic_s"], 3),
        "loadavg_before": before["loadavg"], "loadavg_after": after["loadavg"],
        "service_cpus": service,
        "service_busy": {str(cpu): round(busy[cpu], 4) for cpu in service},
        "smt_siblings_busy": {str(cpu): round(busy[cpu], 4) for cpu in siblings},
        "ccx_neighbours_busy": {str(cpu): round(busy[cpu], 4) for cpu in neighbours},
        "limits": {"smt_sibling": sibling_limit, "ccx_neighbour": neighbour_limit},
        "other_busy_cpus": {str(cpu): fraction for cpu, fraction in others.items()},
        "other_active_cores": active_cores(busy, exclude=service, root=root),
        "host_busy_mean": round(sum(busy.values()) / len(busy), 4),
        "verdict": "contaminated" if reasons else "clean",
        "reasons": reasons,
    }


def gate(need, limit, seconds):
    before = snapshot()
    time.sleep(seconds)
    cores = active_cores(busy_fractions(before, snapshot()))
    return {"active_cores": cores, "need": need, "limit": limit, "fits": len(cores) + need <= limit}


def score(staged, run, scorer, service):
    """`paper_dev_audio.score` with the placement check widened from one CPU to a CPU list."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/ultra/tools"))
    import paper_dev_audio as audio

    measurement = audio.read_json(run / "measurement.json")
    placed = measurement.get("placement") or {}
    cpu = measurement["cpu_seconds_user"] + measurement["cpu_seconds_sys"]
    if (measurement["exit_code"] != 0 or not math.isfinite(cpu) or cpu <= 0
            or measurement["peak_rss_bytes"] <= 0 or not placed.get("separate_l3_and_smt")
            or placed.get("effective_service_cpus_at_spawn") != service):
        raise ValueError("cold service failed, lacked resource accounting, or was misplaced")
    index = audio.read_json(staged / "index.json")
    if [row["id"] for row in audio.read_json(staged / "audio.json")] != [row["id"] for row in index]:
        raise ValueError("audio manifest and score index disagree")
    hyps = audio.hypotheses(run)
    ids = {row["id"] for row in index}
    if len(ids) != sum(audio.SPLIT_COUNTS.values()) or ids != hyps.keys():
        raise ValueError(f"wrong hypothesis population, missing {ids - hyps.keys()}, extra {hyps.keys() - ids}")
    with (run / "scores.jsonl").open("x", encoding="utf-8") as out:
        for reference in index:
            row = {key: value for key, value in reference.items() if key not in {"id", "uid"}}
            json.dump({**row, "id": reference["uid"], "hyp": hyps[reference["id"]]}, out)
            out.write("\n")
    scoring = subprocess.run([scorer, run / "scores.jsonl"], capture_output=True, check=True)
    if json.loads(scoring.stdout)["normaliser_sha256"] != audio.NORMALIZER_SHA256:
        raise ValueError("unexpected Rust normalizer")
    (run / "score.json").write_bytes(scoring.stdout)


def trace_summary(path):
    """Totals of a trace: time inside the recogniser (final `wall_ms` per utterance) against audio fed."""
    events = finals = 0
    audio_ms = wall_ms = 0.0
    with open(path, encoding="utf-8") as trace:
        for line in trace:
            event = json.loads(line)
            events += 1
            if event["final"]:
                finals += 1
                audio_ms += event["audio_ms"]
                wall_ms += event["wall_ms"]
    return {
        "schema": "open-opponents.trace-summary/v1", "events": events, "utterances": finals,
        "audio_seconds": round(audio_ms / 1000, 3), "recogniser_seconds": round(wall_ms / 1000, 3),
        "note": "recogniser_seconds is the unpaced time from each utterance's first packet to its final "
                "result, summed; model load and process start are outside it",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("cpus").add_argument("list")
    fit = sub.add_parser("gate")
    fit.add_argument("--need", type=int, required=True)
    fit.add_argument("--limit", type=int, default=25)
    fit.add_argument("--seconds", type=float, default=3.0)
    check = sub.add_parser("guard")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    check.add_argument("--service-cpus", required=True)
    check.add_argument("--sibling-limit", type=float, default=0.01)
    check.add_argument("--neighbour-limit", type=float, default=0.02)
    scoring = sub.add_parser("score")
    scoring.add_argument("staged", type=Path)
    scoring.add_argument("run", type=Path)
    scoring.add_argument("scorer", type=Path)
    scoring.add_argument("service_cpus")
    sub.add_parser("trace-summary").add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        json.dump(snapshot(), sys.stdout)
    elif args.command == "cpus":
        print(*parse_cpus(args.list))
    elif args.command == "gate":
        result = gate(args.need, args.limit, args.seconds)
        json.dump(result, sys.stdout)
        sys.exit(0 if result["fits"] else 4)
    elif args.command == "guard":
        result = guard(json.loads(args.before.read_text()), json.loads(args.after.read_text()),
                       parse_cpus(args.service_cpus), args.sibling_limit, args.neighbour_limit)
        json.dump(result, sys.stdout, indent=1)
        sys.exit(3 if result["reasons"] else 0)
    elif args.command == "trace-summary":
        json.dump(trace_summary(args.trace), sys.stdout, indent=1)
    else:
        score(args.staged, args.run, args.scorer, parse_cpus(args.service_cpus))


if __name__ == "__main__":
    main()
