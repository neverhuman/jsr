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


def main():
    root, binding_path, run_name, baseline_name, output = sys.argv[1:]
    root, binding_path, output = map(Path, [root, binding_path, output])
    helper = Path(__file__).with_name("verify-m1-evidence.py")
    base = runpy.run_path(str(helper), run_name="m1_verifier")
    require, digest, read, rows = [base[k] for k in ["require", "digest", "read", "rows"]]
    current, previous = [base["load"](root, name) for name in [run_name, baseline_name]]
    row, plan, references = current
    require(references == previous[2], "different reference population or clusters")
    for field in ["workload_sha256", "bundle_manifest_sha256"]:
        require(plan[field] == previous[1][field], f"different {field}")
    for field in ["engine", "precision", "streams", "packet_ms", "threads", "paced", "guarded"]:
        require(plan["job"][field] == previous[1]["job"][field], f"different {field}")
    run = root / run_name
    final = rows(run / "final.jsonl")
    require(final == rows(root / baseline_name / "final.jsonl"), "actual Final texts differ")
    require(final == rows(run / "commit.jsonl"), "terminal committed view differs")
    by_id = {item["input_id"]: item["text"] for item in final}
    outcomes = sorted(rows(run / "outcomes.jsonl"), key=lambda item: item["source_index"])
    records = rows(run / "scores.jsonl")
    require(len(by_id) == len(outcomes) == len(records), "missing or duplicate Final")
    require(all(record["hyp"] == outcome["final_text"] == by_id[outcome["input_id"]]
                for record, outcome in zip(records, outcomes)), "score differs from actual output")
    score, old_score = [read(root / name / "score.json") for name in [run_name, baseline_name]]
    for key in ["normaliser_id", "normaliser_sha256", "weights_sha256", "system_a"]:
        require(score[key] == old_score[key], f"different Rust score or normalization: {key}")
    binding = read(binding_path)
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
               final_sha256=digest(run / "final.jsonl"), actual_output_matches=len(final))
    report = dict(schema="ultra.service-control-review/v1", current=row, baseline=previous[0],
                  binding_sha256=digest(binding_path), build_proof_sha256=digest(proof_path),
                  verifier_sha256=digest(Path(__file__)), base_verifier_sha256=digest(helper),
                  qualifying=False, wall_timing_evidence=False,
                  interpretation="Single shared-host observations; historical date/source/placement can differ."
                  " Exact output is verified; CPU differences alone do not establish causation or a speedup.")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(final)} actual outputs and Rust scores; wrote {output}")


if __name__ == "__main__":
    main()
