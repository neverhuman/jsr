"""Verify the first D2 snapshot's exploratory native row; performs no scoring.

Usage: .venv/bin/python docs/ultra/reviews/verify-d2-evidence.py RUN_ROOT NEW_OUTPUT.json
"""

import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path


def main():
    root, output = map(Path, sys.argv[1:])
    helper = Path(__file__).with_name("verify-m1-evidence.py")
    base = runpy.run_path(str(helper), run_name="m1_verifier")
    require, digest, read = base["require"], base["digest"], base["read"]
    names = ["m1-corrected-r0-dev400", "m1-corrected-r1-dev400",
             "m2-native-fp32-dev400", "m1-corrected-n560-dev400",
             "m2-d2-s5000-bridge-dev400",
             "m2-d2-s5000-native-dev400"]
    loaded = [base["load"](root, name) for name in names]
    native = loaded[-1][1]
    d1_native = loaded[2][1]
    rows, normalizer = [], None
    for name, (row, plan, references) in zip(names, loaded):
        require(references == loaded[0][2] and
                plan["workload_sha256"] == loaded[0][1]["workload_sha256"],
                "different quality population or audio")
        summary, score = [read(root / name / f"{key}.json") for key in ["summary", "score"]]
        identity = {key: score[key] for key in
                    ["normaliser_id", "normaliser_sha256", "weights_sha256"]}
        if normalizer is None:
            normalizer = identity
        require(identity == normalizer, "different scorer configuration")
        streams = plan["job"].get("streams", plan["job"].get("concurrency"))
        require(streams == 8 and plan["job"]["threads"] == 1 and summary["guarded"],
                "different primary resource configuration")
        outcomes = sorted(base["rows"](root / name / "outcomes.jsonl"),
                          key=lambda item: item["source_index"])
        records = base["rows"](root / name / "scores.jsonl")
        final = base["rows"](root / name / "final.jsonl")
        require(len(final) == len(outcomes) and
                len({item["input_id"] for item in final}) == len(final),
                "missing or duplicate final output")
        final_by_id = {item["input_id"]: item["text"] for item in final}
        require(all(record["hyp"] == outcome["final_text"] ==
                    final_by_id[outcome["input_id"]]
                    for record, outcome in zip(records, outcomes)),
                "score does not use actual final output")
        row.update(runtime_path=summary["runtime_path"], precision=summary["precision"],
                   cost_scope=summary["cost_scope"], launch_checkout_commit=plan["source_commit"],
                   launch_checkout_dirty=plan["source_dirty"],
                   executable_sha256=plan["executable_sha256"],
                   bundle_manifest_sha256=plan.get("bundle_manifest_sha256"),
                   model_sha256=plan.get("model_sha256"),
                   observer_placement="observer co-located on the service core",
                   scorer_executable_sha256=read(root / name / "support.json")["scorer_executable_sha256"],
                   peak_processes=None, wall_timing_evidence=False)
        rows.append(row)
    require(native["job"]["engine"] == "r1-native" and
            native["job"]["precision"] == "fp32" and not native["source_dirty"],
            "wrong D2 runtime operating point")
    for key in ["executable_sha256", "source_commit"]:
        require(native[key] == d1_native[key], "different native implementation")
    build_path = root / "m2-build-provenance-20261120.json"
    build = read(build_path)
    require(build["build_success"] and build["source_unchanged_after_build"] and
            all(build[key] == native[key] for key in ["source_commit", "executable_sha256"]),
            "native build identity mismatch")
    d2_summary = read(root / names[-1] / "summary.json")
    d1_summary = read(root / names[2] / "summary.json")
    require(d2_summary["cold_load"]["ort_library_sha256"] ==
            d1_summary["cold_load"]["ort_library_sha256"] and
            d2_summary["cold_load"]["cpu_only_verified"] and
            not d2_summary["cold_load"]["python_runtime_used"], "different native ORT runtime")
    bundle = Path(native["job"]["bundle"])
    manifest = read(bundle / "manifest.json")
    provenance_path = bundle / "source-provenance.json"
    provenance = read(provenance_path)
    require(digest(bundle / "manifest.json") == native["bundle_manifest_sha256"] ==
            provenance["manifest_sha256"], "changed manifest")
    model = manifest["model"]
    require(model["checkpoint_step"] == 5000 and model["family"] == "dense" and
            digest(Path(model["checkpoint"])) == model["checkpoint_sha256"] ==
            provenance["checkpoint_sha256"], "different D2 snapshot")
    require(model["exporter_commit"] == provenance["exporter_commit"] ==
            provenance["source_head_at_start"] == native["source_commit"],
            "different export source")
    for item in manifest["files"].values():
        path = bundle / item["path"]
        require(path.resolve().is_relative_to(bundle.resolve()) and
                digest(path) == item["sha256"], "changed or escaping bundle payload")
    for path, expected in provenance["source_sha256"].items():
        require(path.startswith("python/ultra/train/") and ".." not in Path(path).parts,
                "unexpected export source path")
        blob = subprocess.check_output(["rtk", "proxy", "git", "show",
                                        f"{provenance['exporter_commit']}:{path}"])
        require(hashlib.sha256(blob).hexdigest() == expected, "export source differs from Git")
    smoke_names = ["m2-d2-s5000-native-smoke8", "m2-d2-s5000-bridge-smoke8"]
    smoke = [base["load"](root, name) for name in smoke_names]
    expected_texts = base["rows"](root / smoke_names[0] / "final.jsonl")
    for name, (_, plan, refs) in zip(smoke_names, smoke):
        require(refs == loaded[-1][2][:8] and
                plan["bundle_manifest_sha256"] == native["bundle_manifest_sha256"] and
                plan["job"]["precision"] == "fp32", "different smoke population/model")
        require(base["rows"](root / name / "final.jsonl") == expected_texts,
                "native/NumPy smoke mismatch")
    require([item[1]["job"]["engine"] for item in smoke] == ["r1-native", "r1-bridge"],
            "smoke engines do not establish native/NumPy comparison")
    smoke_to_full_matches = sum(a == b for a, b in zip(
        expected_texts, base["rows"](root / names[-1] / "final.jsonl")[:8]))
    bridge = loaded[-2][1]
    require(bridge["job"]["engine"] == "r1-bridge" and
            bridge["job"]["precision"] == "fp32" and
            bridge["bundle_manifest_sha256"] == native["bundle_manifest_sha256"] and
            bridge["executable_sha256"] == native["executable_sha256"],
            "different D2 bridge operating point")
    for key in ["worker_sha256", "consumer_sha256"]:
        require(bridge[key] == smoke[1][1][key], "different full-panel reference consumer")
    native_texts = base["rows"](root / names[-1] / "final.jsonl")
    bridge_texts = base["rows"](root / names[-2] / "final.jsonl")
    require([item["input_id"] for item in native_texts] ==
            [item["input_id"] for item in bridge_texts], "different full-panel output IDs")
    mismatches = [dict(input_id=a["input_id"], native=a["text"], numpy=b["text"])
                  for a, b in zip(native_texts, bridge_texts) if a["text"] != b["text"]]
    native_wer, bridge_wer = rows[-1]["wer_percent"], rows[-2]["wer_percent"]
    matches = len(native_texts) - len(mismatches)
    parity = dict(matches=matches, examples=len(native_texts), mismatches=mismatches,
                  transcript_agreement=matches / len(native_texts),
                  absolute_wer_delta_pp=abs(native_wer - bridge_wer),
                  exact_transcript_equivalence=not mismatches,
                  handoff_minimum_pass=(matches / len(native_texts) >= 0.98 and
                                        abs(native_wer - bridge_wer) <= 0.10),
                  native_final_sha256=digest(root / names[-1] / "final.jsonl"),
                  numpy_final_sha256=digest(root / names[-2] / "final.jsonl"),
                  scope="Native versus retained NumPy FP32 consumer; trainer checkpoint parity separate")
    account_path = root / "d2-s5000-training-account.json"
    account = read(account_path)
    prefix = root / "d2-s5000-training-prefix.jsonl"
    require(digest(prefix) == account["prefix_sha256"] and
            account["checkpoint_sha256"] == model["checkpoint_sha256"] and
            account["checkpoint_step"] == 5000 and not account["selected"],
            "different training account")
    training = base["rows"](prefix)
    intervals = [item for item in training if item["kind"] == "train"]
    require(intervals[-1]["step"] == 5000 and len(intervals) == account["intervals"] and
            math.isclose(sum(item["audio_s"] for item in intervals) / 3600,
                         account["logged_audio_hours"]) and
            math.isclose((training[-1]["time"] - training[0]["time"]) / 3600,
                         account["start_to_progress_eval_hours"]), "wrong training totals")
    paired = root / "m2-r0-d2-s5000-paired-20261120.json"
    paired_provenance_path = root / "m2-r0-d2-s5000-paired-provenance.json"
    paired_provenance = read(paired_provenance_path)
    require(digest(Path(paired_provenance["executable"])) ==
            paired_provenance["executable_sha256"], "changed paired scorer")
    for key in ["system_a", "system_b", "output"]:
        require(digest(root / paired_provenance[key]) == paired_provenance["sha256"][key],
                "changed paired input or output")
    comparison = read(paired)
    require(comparison["system_a"] == read(root / names[0] / "score.json")["system_a"] and
            comparison["system_b"] == read(root / names[-1] / "score.json")["system_a"],
            "paired comparison does not match actual scores")
    require(all(comparison[key] == value for key, value in normalizer.items()),
            "different paired scorer configuration")
    bootstrap = comparison["comparison"]["pooled_delta"]
    require(bootstrap["replicates"] == 10000 and bootstrap["seed"] == 20261120 and
            bootstrap["alpha"] == 0.05, "different descriptive bootstrap protocol")
    report = dict(schema="ultra.d2-snapshot-evidence/v1", protocol="M1-DEV-20261120",
                  scope="First retained D2 step-5000 snapshot; unselected, exploratory DEV research",
                  verifier_sha256=digest(Path(__file__)), base_verifier_sha256=digest(helper),
                  build_provenance_sha256=digest(build_path),
                  export_provenance_sha256=digest(provenance_path),
                  training_account_sha256=digest(account_path),
                  executable_build_source_commit=build["source_commit"],
                  source_field_scope="plan.source_commit is launch checkout; build provenance binds executable",
                  checkpoint_sha256=model["checkpoint_sha256"], normalizer=normalizer,
                  rows=rows, paired_comparison_sha256=digest(paired),
                  paired_provenance_sha256=digest(paired_provenance_path),
                  smoke_matches=8, smoke_examples=8,
                  smoke_to_full_native_matches=smoke_to_full_matches,
                  smoke_final_sha256=digest(root / smoke_names[0] / "final.jsonl"),
                  full_panel_d2_reference_parity=parity,
                  full_panel_d2_checkpoint_parity=None, selected_d2=False,
                  latency_cq_fc=None, capacity=None, sustained_operation=None,
                  second_day_reproduction=None, qualifying=False)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(rows)} full-panel rows and D2 smoke agreement; wrote {output}")


if __name__ == "__main__":
    main()
