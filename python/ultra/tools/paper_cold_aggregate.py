"""Bind a completed three-repeat DEV rotation to the paper evidence manifest."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from paper_dev_audio import NORMALIZER_SHA256, SPLIT_COUNTS, read_json, sha256


HOST = "xbabe3"  # the rotation's host; its CPU is the paper's machine key below
MACHINE = "zen2"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def proof(path):
    return {"path": str(path), "sha256": sha256(path)}


def pin_values(path):
    return {line.split(maxsplit=1)[0] for line in path.read_text().splitlines() if line}


def arm_result(run, workload, configuration):
    check((run / "finished").is_file(), f"incomplete timing repetition {run}")
    check((run / "host").read_text().strip() == HOST, f"unexpected host {run}")
    check(pin_values(run / "workload.sha256") == {workload["digest"]}, f"wrong workload {run}")
    pins = pin_values(run / "config.sha256")
    expected = {workload["measure"]["sha256"], configuration["executable"]["sha256"]}
    if "model_manifest" in configuration:
        expected.add(configuration["model_manifest"]["sha256"])
    else:
        expected.update(row["sha256"] for row in configuration["artifacts"])
    check(expected <= pins, f"configuration digest mismatch {run}")
    state = (run / "host.before").read_text().splitlines()
    check(state[1:] == [workload["governor"], str(int(workload["boost"]))],
          f"governor or boost mismatch {run}")
    measure = read_json(run / "measurement.json")
    placement = measure["placement"]
    check(measure["exit_code"] == 0 and measure["samples"] > 0
          and measure["interval_ms"] == workload["rss_sampling_interval_ms"]
          and placement["separate_l3_and_smt"]
          and placement["effective_service_cpus_at_spawn"] == [workload["service_cpu"]]
          and placement["effective_observer_cpus"] == [workload["observer_cpu"]], f"invalid timing {run}")
    cpu = measure["cpu_seconds_user"] + measure["cpu_seconds_sys"]
    rss = measure["peak_rss_bytes"]
    check(math.isfinite(cpu) and cpu > 0 and rss > 0, f"invalid resource data {run}")
    raw_score = read_json(run / "score.json")
    check(raw_score["normaliser_sha256"] == NORMALIZER_SHA256, f"wrong scorer {run}")
    score = raw_score["system_a"]
    check(score["utterances"] == workload["utterances"] and
          {key: item["utterances"] for key, item in score["datasets"].items()} == SPLIT_COUNTS,
          f"wrong score coverage {run}")
    receipt = {
        "round": int((run / "round").read_text()),
        "cpu_seconds": cpu,
        "cpu_seconds_per_audio_second": cpu / workload["audio_seconds"],
        "wall_seconds": measure["wall_seconds"],
        "peak_rss_bytes": rss,
        "measurement": proof(run / "measurement.json"),
        "command": proof(run / "command"),
        "source": proof(run / "service.stderr"),
        "score": proof(run / "score.json"),
        "predictions": proof(run / "scores.jsonl"),
        "configuration_pin": proof(run / "config.sha256"),
        "runtime_libraries": proof(run / "runtime-libraries.sha256"),
        "host_before": proof(run / "host.before"),
        "host_after": proof(run / "host.after"),
        "started_utc": (run / "started").read_text().strip(),
        "finished_utc": (run / "finished").read_text().strip(),
    }
    if (run / "predictions.jsonl").is_file():
        receipt["recognizer_output"] = proof(run / "predictions.jsonl")
    else:
        # Hash the exact byte stream per result file, including each final newline.
        outputs = sorted((run / "outputs").glob("*.txt"))
        check(len(outputs) == workload["utterances"], f"missing raw final outputs {run}")
        inventory = "".join(f"{path.name} {sha256(path)}\n" for path in outputs)
        receipt["recognizer_output"] = {
            "path": str(run / "outputs"),
            "sha256": hashlib.sha256(inventory.encode()).hexdigest(),
            "digest_scope": "lexically sorted lines of filename, space, SHA-256, newline",
        }
    return receipt, score


def aggregate(evidence, root):
    data = read_json(evidence)
    workload_key = "cold-panel"
    workload = data["workloads"][workload_key]
    check(sha256(root / "panel/audio.json") == workload["digest"], "staged manifest changed")
    configs = {name: config for name, config in data["configurations"].items() if "run_alias" in config}
    aliases = workload["first_round_order"]
    check(len(aliases) == len(configs)
          and set(aliases) == {row["run_alias"] for row in configs.values()}, "rotation manifest differs from configurations")
    check(workload["rounds"] == 3, "need exactly three rounds")
    results = []
    by_alias = {}
    for name, config in configs.items():
        alias = config["run_alias"]
        entries = [arm_result(root / f"cold/round-{round_id}/{alias}", workload, config)
                   for round_id in range(1, workload["rounds"] + 1)]
        receipts, scores = zip(*entries, strict=True)
        signatures = {row["predictions"]["sha256"] for row in receipts}
        library_signatures = {row["runtime_libraries"]["sha256"] for row in receipts}
        check(len(library_signatures) == 1, f"runtime libraries differed across rounds for {alias}")
        check([r["round"] for r in receipts] == [1, 2, 3], f"repetition indexes differ for {alias}")
        values = [row["cpu_seconds_per_audio_second"] for row in receipts]
        median_index = sorted(range(len(values)), key=values.__getitem__)[1]
        score = scores[median_index]
        wers = [100 * row["pooled_wer"] for row in scores]
        results.append({
            "id": name, "scope": "cold-dev", "configuration": name,
            "workload": workload_key, "workload_digest": workload["digest"],
            "source": receipts[median_index]["measurement"]["path"], "status": "matched-exploratory",
            "cpu_seconds_per_audio_second": statistics.median(values),
            "cpu_seconds_per_audio_second_range": [min(values), max(values)],
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in receipts),
            "artifact_bytes": config["artifact_bytes"],
            "wer_round": median_index + 1,
            "repeated_hypotheses_identical": len(signatures) == 1,
            "wer_pooled_pct": 100 * score["pooled_wer"],
            "wer_pooled_pct_range": [min(wers), max(wers)],
            "wer_clean_pct": 100 * score["datasets"]["dev-clean"]["wer"],
            "wer_other_pct": 100 * score["datasets"]["dev-other"]["wer"],
            "counts": {key: row["counts"] for key, row in score["datasets"].items()},
            "repetitions": receipts,
        })
        by_alias[alias] = receipts
    previous = ""
    for i in range(workload["rounds"]):
        for alias in aliases[i:] + aliases[:i]:
            receipt = by_alias[alias][i]
            check(previous <= receipt["started_utc"] <= receipt["finished_utc"],
                  f"rotation ordering or one-core serialization differs at {alias}")
            previous = receipt["finished_utc"]
    data["measurements"] = [r for r in data["measurements"] if r["scope"] != "cold-dev"] + results
    data["pending"].pop("cold-dev", None)
    return data


def manuscript_rows(data):
    """Evidence rows for the paper generator (schema v2, docs/paper/data/README.md): one line per
    (system, runtime, machine, threads). `class` must be declared by the configuration."""
    for row in data["measurements"]:
        if row["scope"] != "cold-dev":
            continue
        name = row["configuration"]
        config = data["configurations"][name]
        workload = data["workloads"][row["workload"]]
        receipt = row["repetitions"][row["wer_round"] - 1]
        check(config.get("class") in ("streaming", "whole_utterance"),
              f"configuration {name} must declare class: streaming or whole_utterance")
        variant = config.get("variant", name.removeprefix("cold-"))
        walls = [rep["wall_seconds"] / workload["audio_seconds"] for rep in row["repetitions"]]
        yield {
            "id": f"{variant}-{MACHINE}-t1",
            "variant": variant,
            "system": config["label"],
            "short": config.get("figure_label", config["label"]),
            "runtime": config["runtime"],
            "role": config.get("role") or ("ours" if "checkpoint_sha256" in config else "comparator"),
            "class": config["class"],
            "configuration": (f"{config['numeric_mode']}; {config['decoder']}; chunk {config['chunk_ms']} ms, "
                              f"right context {config['right_context_ms']} ms; evidence.json {name}"),
            "machine": MACHINE,
            "host": HOST,
            "hardware": workload["hardware_label"],
            "threads": 1,
            "parallel": "intra_op",
            "session": f"{row['repetitions'][0]['started_utc'][:10]} {HOST} rotation {workload['measure']['sha256'][:12]}",
            "workload": workload["label"],
            "workload_digest": workload["digest"],
            "scope": "cold_process_tree_launch_to_exit",
            "cpu_s_per_audio_s": row["cpu_seconds_per_audio_second"],
            "cpu_reps": [rep["cpu_seconds_per_audio_second"] for rep in row["repetitions"]],
            "wall_s_per_audio_s": statistics.median(walls),
            "wall_reps": walls,
            "wer_pct": row["wer_pooled_pct"],
            "wer_clean_pct": row["wer_clean_pct"],
            "wer_other_pct": row["wer_other_pct"],
            "peak_rss_bytes": row["peak_rss_bytes"],
            "artifact_bytes": row["artifact_bytes"],
            "label": "exploratory",
            "source": receipt["measurement"]["path"],
            "receipt_sha256": receipt["measurement"]["sha256"],
            "score": receipt["score"],
            "predictions": receipt["predictions"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("manifest", "manuscript-jsonl"), default="manifest")
    args = parser.parse_args()
    result = aggregate(args.evidence, args.run_root.resolve())
    with args.output.open("x", encoding="utf-8") as output:
        if args.format == "manuscript-jsonl":
            for row in manuscript_rows(result):
                output.write(json.dumps(row) + "\n")
        else:
            json.dump(result, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
