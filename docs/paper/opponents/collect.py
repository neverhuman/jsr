"""Pull, score and summarise the opponent receipts written by run-opponent.sh.

    python docs/paper/opponents/collect.py pull   STAGE
    python docs/paper/opponents/collect.py score  STAGE PANEL SCORER
    python docs/paper/opponents/collect.py table  STAGE PANEL
    python docs/paper/opponents/collect.py export STAGE RAW
    python docs/paper/opponents/collect.py rows   STAGE PANEL RAW
    python docs/paper/opponents/collect.py origins RAW

STAGE is a local directory (`<machine>/<arm>-<budget>[-<workload>]/round-N/`), PANEL the
staged panel with `index.json`, SCORER the pinned `ultra-score` executable. `score` writes
`scores.jsonl` and `score.json` beside each finished repetition exactly as
`paper_dev_audio.py score` does, for the utterances of that repetition's workload view.
`table` prints one JSON object per arm and budget (every repetition and the median).
`export` copies the small receipt files under RAW (`docs/paper/data/raw/opponents`):
the hypotheses once per arm, their digest in every repetition. `rows` writes the evidence
rows (schema v2 of agent_paper.md) under RAW/rows: `<machine>-bench-cross.jsonl` for the
730-utterance arms, `.slow.jsonl` for arms measured on a smaller view (never the money
plot's group) and `.loaded.jsonl` for xbabe0 repetitions taken outside the host's timing lock
(cross-checks only); rows with fewer than three repetitions go to `.partial.jsonl`. Arms on hold
in arms.json (licence under review) are neither exported nor
written as rows; their receipts stay on the benchmark machines.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "python/ultra/tools"))

from paper_dev_audio import NORMALIZER_SHA256, hypotheses, read_json  # noqa: E402
from workload import VIEWS  # noqa: E402

MACHINES = {
    "zen2": "xbabe0:jsr-data/opponents/bench-cross/receipts/zen2/",
    "xeon8358": "xbabe6:jsr-data/opponents/bench-cross/receipts/xeon8358/",
    "m3ultra": "xbabe4:/Users/agent/jsr-train/data/opponents/bench-cross/receipts/m3ultra/",
}
KEEP = ("timing.lock", "command", "environment", "config.sha256", "runtime-libraries.sha256", "workload.sha256", "host",
        "host.before", "host.after", "round", "started", "finished", "budget.json", "measurement.json",
        "service.stderr", "score.json", "trace.sha256", "trace.lines", "engine_eval.summary.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(stage):
    """Finished repetitions; an arm directory holding a file INVALID (with the reason) is left out."""
    for done in sorted(stage.glob("*/*/round-*/finished")):
        if not (done.parent.parent / "INVALID").is_file():
            yield done.parent


def pull(stage):
    skip = [option for name in sorted(held_arms()) for option in ("--exclude", f"{name}-*")]
    for machine, source in MACHINES.items():
        target = stage / machine
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--exclude", "trace.jsonl", "--exclude", "worker-*", "--exclude",
                        "*.incomplete.*", *skip, source, str(target) + "/"], check=False)


def busy_neighbours(run, budget):
    """CPUs other than the service and observer CPUs that were at least half busy during the run."""
    before, after = run / "cpustat.before", run / "cpustat.after"
    if not before.is_file() or not after.is_file():
        return None
    def load(path):
        table = {}
        for line in path.read_text().splitlines():
            name, *fields = line.split()
            if name != "cpu":
                values = list(map(int, fields[:8]))
                table[int(name[3:])] = (sum(values), values[3] + values[4])
        return table
    first, last = load(before), load(after)
    own = {int(cpu) for cpu in str(budget["service_cpus"]).split(",") if cpu} | {int(budget["observer_cpu"])}
    busy = []
    for cpu, (total, idle) in last.items():
        span = total - first[cpu][0]
        if cpu not in own and span > 0 and 1 - (idle - first[cpu][1]) / span >= 0.5:
            busy.append(cpu)
    return busy


def service_mhz(run):
    """Mean clock of the service CPUs over the repetition, from run-opponent.sh's sampler."""
    path = run / "freq.tsv"
    if not path.is_file():
        return None
    values = [int(field) for line in path.read_text().splitlines() for field in line.split()[1:]]
    return sum(values) / len(values) / 1000 if values else None


def score(stage, panel, scorer):
    index = read_json(panel / "index.json")
    for run in runs(stage):
        if (run / "score.json").is_file():
            continue
        budget = read_json(run / "budget.json")
        measurement = read_json(run / "measurement.json")
        cpu = measurement["cpu_seconds_user"] + measurement["cpu_seconds_sys"]
        if measurement["exit_code"] != 0 or not math.isfinite(cpu) or cpu <= 0 or measurement["peak_rss_bytes"] <= 0:
            raise ValueError(f"failed or unmetered repetition {run}")
        rows = [index[position] for position in VIEWS[budget["workload"]](len(index))]
        hyps = hypotheses(run)
        if {row["id"] for row in rows} != hyps.keys():
            raise ValueError(f"wrong hypothesis population in {run}")
        with (run / "scores.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                record = {key: value for key, value in row.items() if key not in {"id", "uid"}}
                json.dump({**record, "id": row["uid"], "hyp": hyps[row["id"]]}, out)
                out.write("\n")
        done = subprocess.run([scorer, run / "scores.jsonl"], capture_output=True, check=True)
        if json.loads(done.stdout)["normaliser_sha256"] != NORMALIZER_SHA256:
            raise ValueError("unexpected Rust normalizer")
        (run / "score.json").write_bytes(done.stdout)


def repetition(run, panel_seconds):
    budget = read_json(run / "budget.json")
    measurement = read_json(run / "measurement.json")
    report = read_json(run / "score.json")["system_a"]
    cpu = measurement["cpu_seconds_user"] + measurement["cpu_seconds_sys"]
    audio = panel_seconds[budget["workload"]]
    sets = report["datasets"]
    return {
        "round": (run / "round").read_text().strip(),
        "cpu_seconds": cpu,
        "wall_seconds": measurement["wall_seconds"],
        "cpu_s_per_audio_s": cpu / audio,
        "wall_s_per_audio_s": measurement["wall_seconds"] / audio,
        "rtfx": audio / measurement["wall_seconds"],
        "peak_rss_bytes": measurement["peak_rss_bytes"],
        "wer_pct": 100 * report["pooled_wer"],
        "wer_clean_pct": 100 * sets["dev-clean"]["wer"] if "dev-clean" in sets else None,
        "wer_other_pct": 100 * sets["dev-other"]["wer"] if "dev-other" in sets else None,
        "utterances": report["utterances"],
        "scores_sha256": sha256(run / "scores.jsonl"),
        "service_cpus": budget["service_cpus"],
        "busy_neighbours": busy_neighbours(run, budget) if budget["service_cpus"] else None,
        "service_mhz": service_mhz(run),
        "started": (run / "started").read_text().strip(),
        "finished": (run / "finished").read_text().strip(),
        "measurement_sha256": sha256(run / "measurement.json"),
    }


def table(stage, panel_seconds):
    rows = []
    for arm in sorted(path for path in stage.glob("*/*") if path.is_dir() and not (path / "INVALID").is_file()):
        reps = [repetition(run, panel_seconds) for run in sorted(arm.glob("round-*"))
                if (run / "score.json").is_file() and (run / "round").read_text().strip().isdigit()]
        if not reps:
            continue
        budget = read_json(next(iter(sorted(arm.glob("round-*")))) / "budget.json")
        cpus = [rep["cpu_s_per_audio_s"] for rep in reps]
        walls = [rep["wall_s_per_audio_s"] for rep in reps]
        middle = sorted(range(len(cpus)), key=cpus.__getitem__)[len(cpus) // 2]
        rows.append({
            "machine": arm.parent.name, "arm": budget["arm"], "budget": budget["budget"],
            "class": budget["class"], "threads": budget["threads"], "parallel": budget["parallel"],
            "workload": budget["workload"], "repetitions": len(reps),
            "cpu_s_per_audio_s": statistics.median(cpus), "cpu_reps": cpus,
            "cpu_spread_pct": 100 * (max(cpus) - min(cpus)) / statistics.median(cpus),
            "wall_s_per_audio_s": statistics.median(walls), "wall_reps": walls,
            "rtfx": 1 / statistics.median(walls), "speed_1core": 1 / statistics.median(cpus),
            "peak_rss_bytes": max(rep["peak_rss_bytes"] for rep in reps),
            "wer_pct": reps[middle]["wer_pct"], "wer_clean_pct": reps[middle]["wer_clean_pct"],
            "wer_other_pct": reps[middle]["wer_other_pct"],
            "wer_reps": [rep["wer_pct"] for rep in reps],
            "identical_hypotheses": len({rep["scores_sha256"] for rep in reps}) == 1,
            "median_round": reps[middle]["round"], "receipt_sha256": reps[middle]["measurement_sha256"],
            "service_cpus": [rep["service_cpus"] for rep in reps],
            "busy_neighbours": [rep["busy_neighbours"] for rep in reps],
            "n_busy_cores": [None if rep["busy_neighbours"] is None else len(rep["busy_neighbours"]) for rep in reps],
            "service_mhz": [rep["service_mhz"] for rep in reps],
            "started": min(rep["started"] for rep in reps), "finished": max(rep["finished"] for rep in reps),
            "source": str(arm),
        })
    return rows


def held_arms():
    return {name for name, arm in read_json(HERE / "arms.json")["arms"].items() if "hold" in arm}


def export(stage, raw):
    held = held_arms()
    for run in runs(stage):
        if not (run / "score.json").is_file() or read_json(run / "budget.json")["arm"] in held:
            continue
        target = raw / run.relative_to(stage)
        target.mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (run / name).is_file():
                shutil.copyfile(run / name, target / name)
        hyps = hypotheses(run)
        text = "".join(json.dumps({"id": key, "hyp": hyps[key]}, ensure_ascii=False) + "\n" for key in sorted(hyps))
        digest = hashlib.sha256(text.encode()).hexdigest()
        (target / "predictions.sha256").write_text(f"{digest}  ../predictions.{digest[:12]}.jsonl\n")
        shared = target.parent / f"predictions.{digest[:12]}.jsonl"
        if not shared.is_file():
            shared.write_text(text, encoding="utf-8")


def evidence(stage, panel_seconds, raw):
    registry = read_json(HERE / "arms.json")
    files = {}
    by_id = {}
    for row in table(stage, panel_seconds):
        if row["arm"] in held_arms():
            continue
        arm, machine = registry["arms"][row["arm"]], registry["machines"][row["machine"]]
        run = Path(row["source"]) / f"round-{row['median_round']}"
        view_digest = (run / "workload.sha256").read_text().split()[0]
        separate = row["budget"] == "w8" and "variant_w8" in arm
        variant = arm["variant_w8"] if separate else arm["variant"]
        record = {
            "id": f"{variant}-{row['machine']}-t{row['threads']}",
            "variant": variant, "system": arm["system_w8"] if separate else arm["system"],
            "short": arm["short"], "macro": arm["macro"] + ("Procs" if separate else ""),
            "runtime": arm["runtime"], "role": arm["role"], "configuration": arm["configuration"],
            "class": row["class"], "machine": row["machine"], "host": machine["host"],
            "hardware": machine["hardware"], "threads": row["threads"],
            "parallel": arm.get("parallel_override", row["parallel"]),
            "workload": registry["workloads"][row["workload"]], "workload_view": row["workload"],
            "workload_digest": view_digest,
            "scope": "cold_process_tree_launch_to_exit",
            "cpu_s_per_audio_s": row["cpu_s_per_audio_s"], "cpu_reps": row["cpu_reps"],
            "wall_s_per_audio_s": row["wall_s_per_audio_s"], "wall_reps": row["wall_reps"],
            "wer_pct": row["wer_pct"], "wer_clean_pct": row["wer_clean_pct"], "wer_other_pct": row["wer_other_pct"],
            "peak_rss_bytes": row["peak_rss_bytes"], "artifact_bytes": arm["artifact_bytes"],
            "n_busy_cores": row["n_busy_cores"], "service_mhz": row["service_mhz"],
            "service_cpus": row["service_cpus"], "repeated_hypotheses_identical": row["identical_hypotheses"],
            "source": str(Path("docs/paper/data/raw/opponents") / run.relative_to(stage) / "measurement.json"),
            "receipt_sha256": row["receipt_sha256"], "label": "exploratory",
            "session": f"{row['started'][:10]} {machine['host']} bench-cross harness e2ef608",
        }
        kind = "main" if record["workload_view"] == "panel730" else "slow"
        if row["repetitions"] < 3:
            kind = "partial"  # the evidence schema wants three repetitions; kept for the record
        if row["machine"] == "zen2":
            # xbabe0 is shared by four benchmark lanes: only repetitions taken under the host's
            # timing lock (one arm on the whole host) are primary; the rest are loaded cross-checks.
            locked = all((Path(row["source"]) / f"round-{number}" / "timing.lock").is_file()
                         for number in range(1, row["repetitions"] + 1))
            record["host_condition"] = "quiet: one arm on the host (TIMING.lock)" if locked else "loaded: other lanes running"
            if not locked:
                kind = "loaded"
        key = (kind, record["id"], record["workload"])
        other = by_id.get(key)
        if other is not None:
            # Eight cores measured both ways: keep the faster as the row, name the other inside it.
            better, worse = sorted((other, record), key=lambda item: item["wall_s_per_audio_s"])
            better["other_parallel"] = {name: worse[name] for name in (
                "parallel", "cpu_s_per_audio_s", "cpu_reps", "wall_s_per_audio_s", "wall_reps", "peak_rss_bytes",
                "source", "receipt_sha256")}
            record = better
        by_id[key] = record
    for (kind, _, _), record in sorted(by_id.items()):
        suffix = {"main": "", "slow": ".slow", "loaded": ".loaded", "partial": ".partial"}[kind]
        files.setdefault(raw / "rows" / f"{record['machine']}-bench-cross{suffix}.jsonl", []).append(record)
    (raw / "rows").mkdir(parents=True, exist_ok=True)
    for path, records in files.items():
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def origins(raw):
    """Name the origin of every exported file in raw/MANIFEST.txt (rebuilt by paper_data first)."""
    manifest = raw.parent / "MANIFEST.txt"
    lines = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, rest = line.partition("  ")
        path, _, origin = rest.partition("  ")
        parts = path.split("/")
        if parts[0] == raw.name and len(parts) > 2 and parts[1] in MACHINES:
            source = MACHINES[parts[1]]
            if parts[-1].startswith("predictions."):
                origin = f"collect.py export: final hypotheses of {source}{parts[2]}, sorted by id"
            else:
                origin = source + "/".join(parts[2:])
        elif parts[0] == raw.name and len(parts) > 1 and parts[1] == "rows":
            origin = "collect.py rows: evidence rows derived from the receipts beside this directory"
        lines.append(f"{digest}  {path}  {origin}" if rest else line)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def view_seconds(panel):
    audio = read_json(panel / "audio.json")
    return {name: sum(audio[position]["samples"] for position in view(len(audio))) / 16000
            for name, view in VIEWS.items()}


def main():
    command, stage = sys.argv[1], Path(sys.argv[2])
    if command == "pull":
        pull(stage)
    elif command == "score":
        score(stage, Path(sys.argv[3]), sys.argv[4])
    elif command == "table":
        for row in table(stage, view_seconds(Path(sys.argv[3]))):
            print(json.dumps(row))
    elif command == "rows":
        evidence(stage, view_seconds(Path(sys.argv[3])), Path(sys.argv[4]))
    elif command == "origins":
        origins(stage)
    elif command == "export":
        export(stage, Path(sys.argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
