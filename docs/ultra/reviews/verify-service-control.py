"""Verify an output-preserving service control and retained Rust scores; no scoring.

Usage: python verify-service-control.py ROOT BINDING.json RUN BASELINE NEW_OUTPUT.json
"""

import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path


def fused_identity(root, run_name, baseline_name, plan, baseline, binding, base):
    require, digest, read = [base[k] for k in ["require", "digest", "read"]]
    arch = read(root / run_name / "bundle-manifest.json")
    onnx = read(root / baseline_name / "bundle-manifest.json")
    require(arch["schema"] == "ultra.r1-weights/v1" and baseline["job"]["engine"] in
            ["r1-native", "r1-bridge"] and plan["job"]["precision"] == "fp32",
            "unsupported representation comparison")
    checkpoint = arch["source"]["checkpoint_sha256"]
    require(checkpoint == onnx["model"]["checkpoint_sha256"] and
            arch["source"]["bundle"]["manifest_sha256"] == baseline["bundle_manifest_sha256"],
            "fused weights do not bind the reference checkpoint/export")
    require(arch["features"]["cmvn"] == onnx["features"]["cmvn"], "different CMVN")
    for key in ["blank_id", "vocab_size", "ctx_init", "max_symbols"]:
        require(arch["decoder"][key] == onnx["decoder"][key], "different decoder rule")
    for key in ["tokens", "bpe", "mel_fbank"]:
        require(arch["files"][key]["sha256"] == onnx["files"][key]["sha256"],
                "different tokenizer/frontend payload")
    bundle = Path(plan["job"]["bundle"])
    require(digest(bundle / "arch.json") == plan["bundle_manifest_sha256"], "changed fused manifest")
    for item in arch["files"].values():
        path = bundle / item["path"]
        require(path.resolve().is_relative_to(bundle.resolve()) and digest(path) == item["sha256"],
                "changed/escaping fused payload")
    model_proof = binding["model_provenance"]
    path = Path(model_proof["path"])
    require(path.resolve() == (bundle / "source-provenance.json").resolve() and
            digest(path) == model_proof["sha256"], "changed model export proof")
    proof = read(path)
    require(proof["export_format"] == "fused-fp32" and proof["checkpoint_sha256"] == checkpoint and
            proof["manifest_sha256"] == plan["bundle_manifest_sha256"] and
            proof["exporter_commit"] == arch["converter"]["commit"], "different fused export provenance")
    require(digest(Path(arch["source"]["checkpoint"])) == checkpoint, "changed retained checkpoint")
    for source, expected in proof["source_sha256"].items():
        require(source.startswith("python/ultra/") and ".." not in Path(source).parts,
                "bad fused exporter source path")
        blob = subprocess.check_output(["rtk", "proxy", "git", "show",
                                        f"{proof['exporter_commit']}:{source}"])
        require(hashlib.sha256(blob).hexdigest() == expected, "fused export source differs from Git")
    return dict(checkpoint_sha256=checkpoint, model_export_proof_sha256=digest(path),
                reference_manifest_sha256=baseline["bundle_manifest_sha256"])


def main():
    root, binding_path, run_name, baseline_name, output = sys.argv[1:]
    root, binding_path, output = map(Path, [root, binding_path, output])
    helper = Path(__file__).with_name("verify-m1-evidence.py")
    base = runpy.run_path(str(helper), run_name="m1_verifier")
    require, digest, read, rows = [base[k] for k in ["require", "digest", "read", "rows"]]
    current, previous = [base["load"](root, name) for name in [run_name, baseline_name]]
    row, plan, references = current
    require(references == previous[2], "different reference population or clusters")
    fused = plan["job"]["engine"] == "r1-fused"
    for field in ["workload_sha256"] + ([] if fused else ["bundle_manifest_sha256"]):
        require(plan[field] == previous[1][field], f"different {field}")
    for field in ([] if fused else ["engine"]) + ["precision", "streams", "packet_ms", "threads", "paced", "guarded"]:
        require(plan["job"][field] == previous[1]["job"][field], f"different {field}")
    run = root / run_name
    final = rows(run / "final.jsonl")
    reference_final = rows(root / baseline_name / "final.jsonl")
    require([r["input_id"] for r in final] == [r["input_id"] for r in reference_final],
            "different final output population/order")
    mismatches = [dict(input_id=a["input_id"], current=a["text"], reference=b["text"])
                  for a, b in zip(final, reference_final) if a["text"] != b["text"]]
    require(fused or not mismatches, "actual Final texts differ")
    require(final == rows(run / "commit.jsonl"), "terminal committed view differs")
    by_id = {item["input_id"]: item["text"] for item in final}
    outcomes = sorted(rows(run / "outcomes.jsonl"), key=lambda item: item["source_index"])
    records = rows(run / "scores.jsonl")
    require(len(by_id) == len(outcomes) == len(records), "missing or duplicate Final")
    require(all(record["hyp"] == outcome["final_text"] == by_id[outcome["input_id"]]
                for record, outcome in zip(records, outcomes)), "score differs from actual output")
    score, old_score = [read(root / name / "score.json") for name in [run_name, baseline_name]]
    for key in ["normaliser_id", "normaliser_sha256", "weights_sha256"] + ([] if fused else ["system_a"]):
        require(score[key] == old_score[key], f"different Rust score or normalization: {key}")
    binding = read(binding_path)
    model_identity = fused_identity(root, run_name, baseline_name, plan, previous[1], binding, base) if fused else None
    binary = Path(binding["binaries"]["ultra-bridge"]["path"])
    proof_path = binary.parent.parent / "build-provenance.json"
    proof = read(proof_path)
    require(digest(proof_path) == binding["build_provenance_sha256"], "changed build proof")
    require(proof["build_exit_code"] == 0 and proof["source_unchanged_after_build"] and
            not proof["source_dirty"] and not plan["source_dirty"], "unverified or dirty build")
    require(proof["source_commit"] == binding["source_commit"], "different build source")
    require(binding["binaries"] == proof["binaries"], "different retained binaries")
    for item in binding["binaries"].values():
        require(digest(Path(item["path"])) == item["sha256"], "changed retained executable")
    require(plan["executable_sha256"] == digest(binary), "run did not use bound binary")
    measure_sha = binding["binaries"]["ultra-measure"]["sha256"]
    require(plan["host"]["measurement_binary_sha256"] == measure_sha, "different measurer")
    require(read(run / "support.json")["scorer_executable_sha256"] == digest(binary),
            "different Rust scorer executable")
    # Verify the recorded clean tree against Git content, not the checkout's later HEAD.
    for path, expected in proof["source_sha256"].items():
        require("\n" not in path and ".." not in Path(path).parts, "bad source path")
        blob = subprocess.check_output(["rtk", "proxy", "git", "show",
                                        f"{proof['source_commit']}:{path}"])
        require(hashlib.sha256(blob).hexdigest() == expected, "build source differs from Git")
    summary = read(run / "summary.json")
    require(summary["guarded"] and not summary["wall_timing_evidence"], "wrong timing scope")
    measurement = summary["measurement"]
    placement = measurement["placement"]
    require(placement["configured_service_cpus"] == placement["effective_service_cpus_at_spawn"] == [52]
            and placement["configured_observer_cpus"] == placement["effective_observer_cpus"] == [48]
            and placement["separate_l3_and_smt"], "different or unverified placement")
    require(measurement["max_processes"] >= 1 and measurement["interval_ms"] == 20 and
            math.isfinite(measurement["observer_cpu_seconds"]) and
            measurement["observer_cpu_seconds"] > 0, "missing observer accounting")
    service = read(run / "service.json")
    require(service["complete"] and service["unfinished_samples"] == 0 and
            service["accepted_samples"] == service["processed_samples"], "unfinished work")
    governance = read(run.with_name(run.name + ".governance") / "governance.json")
    require(governance["job_exit_status"] == 0 and governance["gpu"] == -1 and
            governance["cpu_target"] == 0.85 and not governance["wall_timing_evidence"],
            "missing or different governance")
    matching_jobs = []
    for item in binding["resolved_jobs"].values():
        path = Path(item["path"])
        require(digest(path) == item["sha256"], "changed resolved job")
        job = read(path)
        if Path(job["out"]) == run:
            matching_jobs.append(job)
    require(len(matching_jobs) == 1, "run absent or duplicated in binding")
    effective_job = matching_jobs[0]
    effective_job["guarded"] = True
    effective_job["timeout_s"] = max(3600, effective_job["timeout_s"])
    require(effective_job == read(run.with_name(run.name + ".governance") / "job.json"),
            "different effective guarded job")
    effective_job["measure"] = plan["job"]["measure"]
    if effective_job["engine"] != "r1-bridge":
        # Native jobs may omit these unused paths; Rust's typed Job defaults them
        # to empty PathBufs. Preserve any supplied value rather than ignoring it.
        for key in ["python", "worker"]:
            effective_job.setdefault(key, "")
    require(effective_job == plan["job"], "different executed job")
    for item in binding["helpers"].values():
        require(digest(Path(item["path"])) == item["sha256"], "changed governance helper")
    require(governance["guard_source_sha256"] == binding["helpers"]["ultra/fleet_guard.py"]["sha256"],
            "different executed guard source")
    row.update(measurement=measurement, executable_sha256=plan["executable_sha256"],
               build_source_commit=proof["source_commit"], launch_checkout_commit=plan["source_commit"],
               final_sha256=digest(run / "final.jsonl"), actual_output_matches=len(final) - len(mismatches))
    agreement = row["actual_output_matches"] / len(final)
    wer_delta = row["wer_percent"] - previous[0]["wer_percent"]
    report = dict(schema="ultra.service-control-review/v1", current=row, baseline=previous[0],
                  binding_sha256=digest(binding_path), build_proof_sha256=digest(proof_path),
                  verifier_sha256=digest(Path(__file__)), base_verifier_sha256=digest(helper),
                  model_identity=model_identity, mismatches=mismatches,
                  parity=dict(exact_transcripts=not mismatches, agreement=agreement,
                              wer_delta_pp=wer_delta, minimum_pass=agreement >= 0.98 and abs(wer_delta) <= 0.10),
                  qualifying=False, wall_timing_evidence=False,
                  interpretation="Single shared-host observations; historical date/source/placement can differ."
                  " Actual-output joins and parity are recorded; CPU differences alone do not establish a repeatable speedup.")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(final)} actual outputs and Rust scores; wrote {output}")


if __name__ == "__main__":
    main()
