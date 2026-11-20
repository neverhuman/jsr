"""Verify retained M1 artifacts and combine Rust scores; performs no scoring.

Usage from the active root:
  .venv/bin/python docs/ultra/reviews/verify-m1-evidence.py RUN_ROOT NEW_OUTPUT.json
"""

import hashlib
import json
import math
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load(root, name):
    run = root / name
    for index in ["checksums.json", "score-checksums.json"]:
        for relative, expected in read(run / index).items():
            path = run / relative
            require(path.resolve().is_relative_to(run.resolve()), "escaping artifact")
            require(digest(path) == expected, f"changed artifact: {path}")
    plan, summary = read(run / "plan.json"), read(run / "summary.json")
    score, support = read(run / "score.json"), read(run / "support.json")
    inputs_path = Path(plan["job"]["inputs"])
    require(digest(inputs_path) == plan["workload_sha256"], "changed workload")
    expected = read(inputs_path)["inputs"]
    outcomes, records = rows(run / "outcomes.jsonl"), rows(run / "scores.jsonl")
    require(summary["valid"] and not summary["qualifying"], "wrong M1 status")
    require(not summary["evidence_valid"], "M1 must remain exploratory")
    require(len(expected) == len(outcomes) == len(records), "missing outcome/score")
    by_id = {row["input_id"]: row for row in outcomes}
    require(len(by_id) == len(expected), "duplicate outcome")
    for item in expected:
        row = by_id[item["id"]]
        require(row["run_id"] == plan["run_id"], "wrong run")
        require(row["source_index"] == item["index"], "wrong source index")
        require(row["pcm_sha256"] == item["sha256"], "wrong PCM")
        require(row["accepted_samples"] == row["processed_samples"] == item["samples"],
                "unfinished input")
    require(len({row["id"] for row in records}) == len(records), "duplicate score")
    audio = sum(item["samples"] for item in expected) / 16000
    require(math.isclose(summary["audio_s"], audio), "wrong duration")
    dataset = score["system_a"]["datasets"]["dev-clean"]
    measurement = summary["measurement"]
    result = dict(run=name, run_id=plan["run_id"],
                  workload_sha256=plan["workload_sha256"], audio_s=audio,
                  utterances=len(records), speaker_clusters=support["independent_speaker_clusters"],
                  counts=dataset["counts"], wer_percent=100 * dataset["wer"],
                  cpu_s_per_audio_s=summary["cpu_s_per_audio_s"],
                  warm_service_cost=summary["warm_service_cost"],
                  peak_rss_bytes=measurement["peak_rss_bytes"],
                  peak_pss_bytes=measurement["peak_pss_bytes"],
                  peak_threads=measurement["max_threads"],
                  decoder_cap_hits=summary["decoder_cap_hits"],
                  qualifying=False, evidence_valid=False,
                  scores_sha256=digest(run / "scores.jsonl"),
                  artifact_hashes={key: digest(run / key) for key in
                                   ["plan.json", "summary.json", "checksums.json", "score-checksums.json"]})
    reference = [(row["id"], row["speaker"], row["dataset"], row["ref"], row["duration_s"])
                 for row in records]
    return result, plan, reference


def main():
    root, output = map(Path, sys.argv[1:])
    names = ["m1-corrected-r0-dev400", "m1-corrected-r1-dev400", "m1-corrected-n560-dev400",
             "m1-corrected-r1-aa8", "m1-corrected-r1-aa8-repeat", "m1-corrected-r1-paced8"]
    loaded = [load(root, name) for name in names]
    require(all(row[1]["workload_sha256"] == loaded[0][1]["workload_sha256"] and
                row[2] == loaded[0][2] for row in loaded[:3]), "unmatched quality panel")
    first, repeat, paced = [row[1] for row in loaded[3:]]
    fields = ["bundle_manifest_sha256", "workload_sha256", "executable_sha256",
              "worker_sha256", "consumer_sha256"]
    for key in fields:
        require(first[key] == repeat[key] == paced[key], f"different runtime/input: {key}")
    for key in ["precision", "streams", "packet_ms", "threads", "guarded"]:
        require(first["job"][key] == repeat["job"][key] == paced["job"][key], key)
    require(not first["job"]["paced"] and not repeat["job"]["paced"] and paced["job"]["paced"],
            "wrong pacing configuration")
    require(len({p["host"]["measurement_binary_sha256"] for p in [first, repeat, paced]}) == 1,
            "different measurement binaries")
    aa = {}
    for name in ["final.jsonl", "commit.jsonl"]:
        value = digest(root / names[3] / name)
        require(value == digest(root / names[4] / name), "A/A transcript disagreement")
        aa[name] = value
    observations = rows(root / names[5] / "observations.jsonl")
    slacks = [row["observed_s"] - row["source_start_s"] -
              row["observation"]["accepted_samples"] / 16000 for row in observations]
    require(slacks and all(math.isfinite(s) and s >= -1e-9 for s in slacks),
            "paced observation precedes capture deadline")
    report = dict(schema="ultra.m1-evidence/v1", protocol="M1-DEV-20261120",
                  verifier_sha256=digest(Path(__file__)), rows=[row[0] for row in loaded],
                  aa_transcript_hashes=aa, paced_observations=len(slacks),
                  paced_min_capture_slack_s=min(slacks),
                  paced_check_scope="Necessary observation deadline check; not a packet-release trace",
                  latency_cq_fc=None, capacity=None, sustained_operation=None,
                  second_day_reproduction=None, qualifying=False)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(loaded)} runs; wrote {output}")


if __name__ == "__main__":
    main()
