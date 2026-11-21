"""Verify an output-preserving service control and retained Rust scores; no scoring.

Usage: python verify-service-control.py ROOT BINDING.json RUN BASELINE NEW_OUTPUT.json
"""

import hashlib
import json
import math
import runpy
import struct
import subprocess
import sys
from pathlib import Path


def verify_sources(revision, sources, require):
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "invalid immutable source revision")
    require(bool(sources), "missing source-content proof")
    for path, expected in sources.items():
        require("\n" not in path and ".." not in Path(path).parts, "bad source path")
        blob = subprocess.check_output(["rtk", "proxy", "git", "show", f"{revision}:{path}"])
        require(hashlib.sha256(blob).hexdigest() == expected, "source differs from Git")


def export_identity(bundle, arch, manifest_sha, binding, base):
    require, digest, read = [base[k] for k in ["require", "digest", "read"]]
    item = binding["model_provenance"]
    path = Path(item["path"])
    require(path.resolve() == (bundle / "source-provenance.json").resolve() and
            digest(path) == item["sha256"], "changed model export proof")
    proof = read(path)
    checkpoint = arch["source"]["checkpoint_sha256"]
    require(proof["export_format"] == "fused-fp32" and proof["checkpoint_sha256"] == checkpoint and
            proof["manifest_sha256"] == manifest_sha and
            proof["exporter_commit"] == arch["converter"]["commit"], "different fused export provenance")
    require(digest(Path(arch["source"]["checkpoint"])) == checkpoint, "changed retained checkpoint")
    require(all(p.startswith("python/ultra/") for p in proof["source_sha256"]),
            "bad fused exporter source path")
    verify_sources(proof["exporter_commit"], proof["source_sha256"], require)
    return dict(checkpoint_sha256=checkpoint, model_export_proof_sha256=digest(path))


def prepared_identity(root, run_name, baseline_name, plan, baseline, binding, base, arch):
    require, digest, read = [base[k] for k in ["require", "digest", "read"]]
    require(baseline["job"]["engine"] == "r1-fused" and baseline["job"]["precision"] == "fp32",
            "prepared INT8 needs the actual same-checkpoint fused FP32 control")
    bundle, parent = [Path(p["job"]["bundle"]) for p in [plan, baseline]]
    item = binding["preparation_provenance"]
    path = Path(item["path"])
    require(path.resolve() == (bundle / "preparation-provenance.json").resolve() and
            digest(path) == item["sha256"], "changed preparation proof")
    prep = read(path)
    old = read(root / baseline_name / "bundle-manifest.json")
    require(Path(prep["source_fp32_bundle"]).resolve() == parent.resolve() and
            digest(parent / "arch.json") == prep["source_fp32_manifest_sha256"] ==
            baseline["bundle_manifest_sha256"], "different prepared source bundle")
    for item in old["files"].values():
        require(digest(parent / item["path"]) == item["sha256"], "changed source FP32 payload")
    normalized = json.loads(json.dumps(arch))
    del normalized["int8"]
    del normalized["files"]["weights_int8"]
    cmvn = b""
    for name in ["mean", "istd"]:
        a, b = [x["features"]["cmvn"][name] for x in [old, normalized]]
        require(len(a) == len(b) == 80, "different CMVN shape")
        for x, y in zip(a, b):
            bits = struct.pack("<f", x)
            require(bits == struct.pack("<f", y), "different operational CMVN bits")
            cmvn += bits
        normalized["features"]["cmvn"][name] = a
    require(normalized == old and hashlib.sha256(cmvn).hexdigest() == prep["cmvn_effective_f32_sha256"],
            "prepared artifact changed a non-INT8 field")
    require(prep["manifest_sha256"] == plan["bundle_manifest_sha256"] and
            prep["weights_int8_sha256"] == arch["files"]["weights_int8"]["sha256"] and
            prep["source_weights_sha256"] == arch["int8"]["source_weights_sha256"] ==
            old["files"]["weights"]["sha256"] and
            prep["int8_fp32_kinds"] == arch["int8"]["fp32_kinds"] == [] and
            prep["producer_verified_readback"] and prep["job_exit_status"] == 0,
            "different or incomplete prepared artifact")
    require(prep["governance"]["cpu_target"] == 0.85 and prep["governance"]["gpu"] == -1 and
            prep["governance"]["guard_source_sha256"] ==
            binding["helpers"]["ultra/fleet_guard.py"]["sha256"], "different preparation governance")
    producer_proof = prep["producer_build_proof"]
    require(digest(Path(producer_proof["path"])) == producer_proof["sha256"], "changed producer build proof")
    build = read(Path(producer_proof["path"]))
    require(build["build_exit_code"] == 0 and build["source_unchanged_after_build"] and
            not build["source_dirty"], "unverified preparation producer")
    producer = build["binaries"]["ultra-engine-prepare"]
    require(digest(Path(producer["path"])) == producer["sha256"] == prep["producer_executable_sha256"] and
            prep["command"] == [producer["path"], "--weights", str(parent), "--out", str(bundle)],
            "different executed preparation command")
    verify_sources(build["source_commit"], build["source_sha256"], require)
    require(digest(parent / "source-provenance.json") == prep["source_export_proof_sha256"],
            "changed original FP32 export proof")
    result = export_identity(parent, old, baseline["bundle_manifest_sha256"], binding, base)
    load = read(root / run_name / "load.json")
    require(load["int8_source"] == "prepared" and load["weights_int8_sha256"] ==
            prep["weights_int8_sha256"] and load["int8_fp32_kinds"] == [],
            "service did not use the prepared precision policy")
    return dict(**result, preparation_proof_sha256=digest(path),
                reference_manifest_sha256=baseline["bundle_manifest_sha256"],
                weights_int8_sha256=prep["weights_int8_sha256"])


def fused_identity(root, run_name, baseline_name, plan, baseline, binding, base):
    require, digest, read = [base[k] for k in ["require", "digest", "read"]]
    arch = read(root / run_name / "bundle-manifest.json")
    require(arch["schema"] == "ultra.r1-weights/v1", "unsupported fused schema")
    bundle = Path(plan["job"]["bundle"])
    require(digest(bundle / "arch.json") == plan["bundle_manifest_sha256"], "changed fused manifest")
    for item in arch["files"].values():
        path = bundle / item["path"]
        require(path.resolve().is_relative_to(bundle.resolve()) and digest(path) == item["sha256"],
                "changed/escaping fused payload")
    load = read(root / run_name / "load.json")
    require(load["precision"] == plan["job"]["precision"] and load["cpu_only_verified"] and
            load["runtime_weight_quantization"] is False and
            load["manifest_sha256"] == plan["bundle_manifest_sha256"] and
            load["checkpoint_sha256"] == arch["source"]["checkpoint_sha256"] and
            load["weights_sha256"] == arch["files"]["weights"]["sha256"],
            "different executed fused model/precision")
    if plan["job"]["precision"] == "int8":
        return prepared_identity(root, run_name, baseline_name, plan, baseline, binding, base, arch)
    onnx = read(root / baseline_name / "bundle-manifest.json")
    require(baseline["job"]["engine"] in ["r1-native", "r1-bridge"] and
            baseline["job"]["precision"] == "fp32", "unsupported representation comparison")
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
    return dict(**export_identity(bundle, arch, plan["bundle_manifest_sha256"], binding, base),
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
    precision_control = fused and plan["job"]["precision"] == "int8"
    for field in ["workload_sha256"] + ([] if fused else ["bundle_manifest_sha256"]):
        require(plan[field] == previous[1][field], f"different {field}")
    for field in (([] if fused else ["engine"]) + ([] if precision_control else ["precision"]) +
                  ["streams", "packet_ms", "threads", "paced", "guarded"]):
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
    verify_sources(proof["source_commit"], proof["source_sha256"], require)
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
                              wer_delta_pp=wer_delta, consumer_equivalence_evaluated=not precision_control,
                              minimum_pass=None if precision_control else agreement >= 0.98 and abs(wer_delta) <= 0.10),
                  precision_quality=dict(wer_delta_pp=wer_delta, point_margin_pp=0.10,
                                         point_gate_pass=wer_delta <= 0.10,
                                         scope="DEV descriptive point gate; no formal noninferiority")
                  if precision_control else None,
                  qualifying=False, wall_timing_evidence=False,
                  interpretation="Single shared-host observations; historical date/source/placement can differ."
                  " Actual-output joins and parity are recorded; CPU differences alone do not establish a repeatable speedup.")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(final)} actual outputs and Rust scores; wrote {output}")


if __name__ == "__main__":
    main()
