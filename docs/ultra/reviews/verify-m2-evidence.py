"""Combine checked Rust scores for the native FP32 extension; performs no scoring.

Usage: .venv/bin/python docs/ultra/reviews/verify-m2-evidence.py RUN_ROOT NEW_OUTPUT.json
"""

import json
import runpy
import sys
from pathlib import Path


def main():
    root, output = map(Path, sys.argv[1:])
    base_script = Path(__file__).with_name("verify-m1-evidence.py")
    base = runpy.run_path(str(base_script), run_name="m1_verifier")
    require, digest = base["require"], base["digest"]
    names = ["m1-corrected-r0-dev400", "m1-corrected-r1-dev400",
             "m1-corrected-n560-dev400", "m2-native-fp32-dev400"]
    loaded = [base["load"](root, name) for name in names]
    for row in loaded:
        require(row[1]["workload_sha256"] == loaded[0][1]["workload_sha256"] and
                row[2] == loaded[0][2], "unmatched quality panel")
    native = loaded[-1][1]
    require(native["job"]["engine"] == "r1-native" and native["job"]["precision"] == "fp32",
            "wrong native operating point")
    require(not native["source_dirty"], "native row needs committed runtime source")
    require(native["bundle_manifest_sha256"] == loaded[1][1]["bundle_manifest_sha256"],
            "different R1 bundle")
    rows = []
    normalizer = None
    for name, (row, plan, _) in zip(names, loaded):
        summary = base["read"](root / name / "summary.json")
        score = base["read"](root / name / "score.json")
        identity = {key: score[key] for key in
                    ["normaliser_id", "normaliser_sha256", "weights_sha256"]}
        if normalizer is None:
            normalizer = identity
        require(identity == normalizer, "different scorer normalization or dataset weights")
        streams = plan["job"].get("streams", plan["job"].get("concurrency"))
        require(streams == 8 and plan["job"]["threads"] == 1, "different primary resource configuration")
        row.update(runtime_path=summary["runtime_path"], precision=summary["precision"],
                   cost_scope=summary["cost_scope"], source_commit=plan["source_commit"],
                   source_dirty=plan["source_dirty"],
                   executable_sha256=plan["executable_sha256"],
                   bundle_manifest_sha256=plan.get("bundle_manifest_sha256"),
                   model_sha256=plan.get("model_sha256"),
                   configured_threads=plan["job"]["threads"], configured_streams=streams,
                   guarded=summary["guarded"])
        rows.append(row)
    build_path = root / "m2-build-provenance-20261120.json"
    build = base["read"](build_path)
    require(build["build_success"] and build["source_unchanged_after_build"] and
            build["source_commit"] == native["source_commit"] and
            build["executable_sha256"] == native["executable_sha256"], "native build identity mismatch")
    fixtures = []
    final_hash = None
    paced_check = None
    fixture_names = ["m2-native-aa8", "m2-native-aa8-repeat", "m2-native-packet10-8",
                     "m2-native-packet37-8", "m2-native-paced8"]
    for name in fixture_names:
        row, plan, references = base["load"](root, name)
        require(references == loaded[-1][2][:8], "different native fixture population")
        for key in ["bundle_manifest_sha256", "executable_sha256"]:
            require(plan[key] == native[key], "different native fixture runtime")
        require(plan["job"]["engine"] == "r1-native" and plan["job"]["precision"] == "fp32" and
                plan["job"]["streams"] == 8 and plan["job"]["threads"] == 1,
                "different native fixture configuration")
        observed = digest(root / name / "final.jsonl")
        if final_hash is None:
            final_hash = observed
        require(observed == final_hash == digest(root / name / "commit.jsonl"),
                "native A/A, packetization or paced transcript mismatch")
        fixtures.append(dict(run=name, packet_ms=plan["job"]["packet_ms"],
                             paced=plan["job"]["paced"], final_sha256=observed,
                             wer_percent=row["wer_percent"], scores_sha256=row["scores_sha256"]))
        if plan["job"]["paced"]:
            observations = base["rows"](root / name / "observations.jsonl")
            slacks = [x["observed_s"] - x["source_start_s"] -
                      x["observation"]["accepted_samples"] / 16000 for x in observations]
            require(slacks and min(slacks) >= -1e-9, "paced native observation before capture")
            paced_check = dict(observations=len(slacks), min_capture_slack_s=min(slacks),
                               scope="Necessary observation deadline check; not a packet-release trace")
    report = dict(schema="ultra.m2-evidence/v1", workload_protocol="M1-DEV-20261120",
                  scope="Exploratory native FP32 extension; precision/runtime differ across rows",
                  verifier_sha256=digest(Path(__file__)), base_verifier_sha256=digest(base_script),
                  build_provenance_sha256=digest(build_path), normalizer=normalizer,
                  reference_comparison_sha256=digest(root / "m2-native-fp32-reference-comparison.json"),
                  native_repetition_checks=fixtures, paced_observation_check=paced_check,
                  repetition_support="Same eight source examples / one speaker; no added independent support",
                  rows=rows, latency_cq_fc=None, capacity=None,
                  sustained_operation=None, second_day_reproduction=None, qualifying=False)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(rows)} matched full-panel runs; wrote {output}")


if __name__ == "__main__":
    main()
