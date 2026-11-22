"""Collect, tabulate and export the open opponent receipts (standard library only).

    open_summary.py collect RUNS RAW_DIR      copy every finished, clean repetition from
                                              RUNS/{1c,8c}/cold/round-N/ARM to RAW_DIR/ARM/round-N
                                              in compact form (no audio, models or traces)
    open_summary.py expand RAW_DIR PANEL OUT  rebuild the run-cold.sh layout OUT/{1c,8c}/cold/round-N/ARM
                                              from the compact form and the staged panel, hash-checked
    open_summary.py table DIR                 one line per repetition and per arm (markdown)
    open_summary.py rows DIR SESSION          evidence rows (schema v2 of agent_paper.md) as JSONL
    open_summary.py anchors DIR SESSION       the same rows for the session anchors (our engine and
                                              the Zipformer), kept apart from the opponents' rows

Compact form: the receipt files of run-cold.sh plus the host records of open-run.sh, except
that `scores.jsonl` (hypotheses joined with the references) is replaced by its hash in
`scores.jsonl.sha256`, and a `predictions.jsonl` identical to round 1's is replaced by
`predictions.jsonl.ref`. `expand` restores both byte for byte and verifies the hashes.

DIR is searched for repetition directories (those with `measurement.json`, `finished` and
`arm`), so both layouts work. The workload constants come from docs/paper/evidence.json.
A row's cost is the median repetition; its WER is that repetition's score; its peak memory
is the largest of the three (as paper_cold_aggregate.py does).
"""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
RECEIPT_FILES = (
    "arm", "command", "config.sha256", "runtime-libraries.sha256", "workload.sha256", "host", "host.before",
    "host.after", "round", "started", "finished", "measurement.json", "service.stderr", "score.json",
    "host.cpu.json", "host.freq.tsv", "host.gate.json", "artifact.bytes", "trace.sha256", "trace.summary.json",
)
# variant -> table name, plot label, macro stem, runtime, class, role, configuration
SYSTEMS = {
    "vosk-large": (
        "Vosk Large (en-us 0.22)", "Vosk Large", "VoskLarge", "vosk 0.3.45 (Kaldi nnet3)", "streaming", "comparator",
        "vosk-model-en-us-0.22, FP32, 20 ms packets, static HCLG graph, finals rescored (G.carpa + RNNLM), "
        "model endpoint rules"),
    "vosk-small": (
        "Vosk Small (en-us 0.15)", "Vosk Small", "VoskSmall", "vosk 0.3.45 (Kaldi nnet3)", "streaming", "comparator",
        "vosk-model-small-en-us-0.15, FP32, 20 ms packets, lookahead graph, model endpoint rules"),
    "whispercpp-stream-base": (
        "Whisper.cpp Streaming Base", "Whisper.cpp Base", "WhisperCppBase",
        "pywhispercpp 1.4.1 (whisper.cpp 1.8.2)", "streaming", "comparator",
        "ggml-base.en FP16, greedy, independent 3 s chunks with no context carried (Picovoice engine at 272be8f)"),
    "whispercpp-stream-tiny": (
        "Whisper.cpp Streaming Tiny", "Whisper.cpp Tiny", "WhisperCppTiny",
        "pywhispercpp 1.4.1 (whisper.cpp 1.8.2)", "streaming", "comparator",
        "ggml-tiny.en FP16, greedy, independent 3 s chunks with no context carried (Picovoice engine at 272be8f)"),
    "parakeet-v3-onnxasr-int8": (
        "Parakeet TDT 0.6B v3 (onnx-asr INT8)", "Parakeet v3 ORT", "ParakeetOnnxAsr",
        "onnx-asr 0.12.0, onnxruntime 1.32.0", "whole_utterance", "comparator",
        "INT8 ONNX export (istupakov 8f23f0c), TDT greedy, whole utterance"),
    "jsr-d2e-16457-int8": (
        "JSR D2e 16457 INT8", "JSR D2e INT8", "JsrDtwoE", "ultra-engine e2ef608 (ultra-cold-wav)", "streaming", "ours",
        "D2e step 16457, prepared INT8, greedy, 20 ms packets (session anchor)"),
    "r0-zipformer-int8": (
        "Zipformer INT8", "Zipformer INT8", "RzeroZipformer", "sherpa-onnx 2.0.2 (ultra-cold-wav e2ef608)",
        "streaming", "comparator",
        "streaming Zipformer 2023-06-26 INT8, greedy, 320 ms chunks, 20 ms packets (session anchor)"),
}
MODES = {"": (1, "intra_op"), "-t8": (8, "intra_op"), "-w8": (8, "workers")}
ANCHORS = {"jsr-d2e-16457-int8", "r0-zipformer-int8"}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def workload():
    return json.loads((REPO / "docs/paper/evidence.json").read_text())["workloads"]["cold-panel"]


def split_arm(arm):
    for suffix in ("-t8", "-w8"):
        if arm.endswith(suffix):
            return arm[:-len(suffix)], suffix
    return arm, ""


def repetitions(root):
    """Finished repetitions under `root`, keyed by arm, in round order."""
    found = {}
    for measurement in sorted(root.rglob("measurement.json")):
        run = measurement.parent
        if ".contaminated." in str(run) or ".incomplete." in str(run):
            continue
        if not (run / "finished").is_file() or not (run / "arm").is_file():
            continue
        found.setdefault((run / "arm").read_text().strip(), []).append(run)
    return {arm: sorted(runs, key=lambda run: int((run / "round").read_text())) for arm, runs in sorted(found.items())}


def mean_frequency_khz(run):
    values = []
    for line in (run / "host.freq.tsv").read_text().splitlines():
        fields = line.split("\t")
        values.extend(int(value) for value in fields[1:-1])
    return statistics.fmean(values) if values else None


def repetition(run, load):
    measure = json.loads((run / "measurement.json").read_text())
    score = json.loads((run / "score.json").read_text())["system_a"]
    guard = json.loads((run / "host.cpu.json").read_text())
    if (run / "workload.sha256").read_text().split()[0] != load["digest"] or score["utterances"] != load["utterances"]:
        raise ValueError(f"wrong workload in {run}")
    if measure["exit_code"] != 0 or guard["verdict"] != "clean":
        raise ValueError(f"failed or contaminated repetition {run}")
    cpu = measure["cpu_seconds_user"] + measure["cpu_seconds_sys"]
    return {
        "run": run, "round": int((run / "round").read_text()), "host": (run / "host").read_text().strip(),
        "service_cpus": measure["placement"]["effective_service_cpus_at_spawn"],
        "cpu_s": cpu, "wall_s": measure["wall_seconds"],
        "cpu_per_audio": cpu / load["audio_seconds"], "wall_per_audio": measure["wall_seconds"] / load["audio_seconds"],
        "peak_rss_bytes": measure["peak_rss_bytes"], "max_threads": measure["max_threads"],
        "max_processes": measure["max_processes"],
        "wer": 100 * score["pooled_wer"], "wer_clean": 100 * score["datasets"]["dev-clean"]["wer"],
        "wer_other": 100 * score["datasets"]["dev-other"]["wer"],
        "predictions_sha256": predictions_sha256(run),
        "artifact_bytes": int((run / "artifact.bytes").read_text()),
        "started": (run / "started").read_text().strip(), "finished": (run / "finished").read_text().strip(),
        "other_active_cores": guard["other_active_cores"], "sibling_busy_max": max(guard["smt_siblings_busy"].values()),
        "frequency_khz": mean_frequency_khz(run), "loadavg": [guard["loadavg_before"][0], guard["loadavg_after"][0]],
        "recogniser_s": (json.loads((run / "trace.summary.json").read_text())["recogniser_seconds"]
                         if (run / "trace.summary.json").is_file() else None),
    }


def arm_summary(reps):
    order = sorted(range(len(reps)), key=lambda i: reps[i]["cpu_per_audio"])
    median = reps[order[len(order) // 2]]
    cpus = [rep["cpu_per_audio"] for rep in reps]
    return {
        "median": median, "cpu_reps": cpus, "wall_reps": [rep["wall_per_audio"] for rep in reps],
        "cpu_per_audio": statistics.median(cpus),
        "wall_per_audio": statistics.median(rep["wall_per_audio"] for rep in reps),
        "peak_rss_bytes": max(rep["peak_rss_bytes"] for rep in reps),
        "spread": (max(cpus) - min(cpus)) / statistics.median(cpus),
        "identical_hypotheses": len({rep["predictions_sha256"] for rep in reps}) == 1,
    }


def predictions_sha256(run):
    """Hash of a repetition's predictions.jsonl, in either form."""
    if (run / "predictions.jsonl").is_file():
        return sha256(run / "predictions.jsonl")
    return (run / "predictions.jsonl.ref").read_text().split()[1]


def collect(runs, raw):
    copied = 0
    for group in ("1c", "8c"):
        for run in sorted((runs / group / "cold").glob("round-*/*")):
            if ".contaminated." in run.name or ".incomplete." in run.name or not (run / "finished").is_file():
                continue
            if json.loads((run / "host.cpu.json").read_text())["verdict"] != "clean":
                continue
            target = raw / run.name / run.parent.name
            target.mkdir(parents=True, exist_ok=True)
            for name in RECEIPT_FILES:
                if (run / name).is_file():
                    shutil.copyfile(run / name, target / name)
            (target / "scores.jsonl.sha256").write_text(sha256(run / "scores.jsonl") + "\n")
            first = raw / run.name / "round-1" / "predictions.jsonl"
            digest = sha256(run / "predictions.jsonl")
            for stale in ("predictions.jsonl", "predictions.jsonl.ref"):
                (target / stale).unlink(missing_ok=True)
            if target.name != "round-1" and first.is_file() and sha256(first) == digest:
                (target / "predictions.jsonl.ref").write_text(f"round-1 {digest}\n")
            else:
                shutil.copyfile(run / "predictions.jsonl", target / "predictions.jsonl")
            copied += 1
    print(f"copied {copied} repetitions to {raw}")


def expand(raw, panel, out):
    """Rebuild the full receipts: every compact file, predictions.jsonl and the scorer input."""
    index = json.loads((panel / "index.json").read_text())
    for arm, runs in repetitions(raw).items():
        for run in runs:
            measure = json.loads((run / "measurement.json").read_text())
            group = "1c" if len(measure["placement"]["effective_service_cpus_at_spawn"]) == 1 else "8c"
            target = out / group / "cold" / run.name / arm
            target.mkdir(parents=True)
            for name in RECEIPT_FILES:
                if (run / name).is_file():
                    shutil.copyfile(run / name, target / name)
            source = run / "predictions.jsonl"
            if not source.is_file():
                source = run.parent / (run / "predictions.jsonl.ref").read_text().split()[0] / "predictions.jsonl"
            shutil.copyfile(source, target / "predictions.jsonl")
            if sha256(target / "predictions.jsonl") != predictions_sha256(run):
                raise ValueError(f"predictions of {run} do not match their recorded hash")
            hyps = {row["id"]: row["hyp"] for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())}
            # Same bytes as paper_dev_audio.score writes: reference row without id/uid, then uid and hypothesis.
            with (target / "scores.jsonl").open("x", encoding="utf-8") as scores:
                for reference in index:
                    row = {key: value for key, value in reference.items() if key not in {"id", "uid"}}
                    json.dump({**row, "id": reference["uid"], "hyp": hyps[reference["id"]]}, scores)
                    scores.write("\n")
            if sha256(target / "scores.jsonl") != (run / "scores.jsonl.sha256").read_text().strip():
                raise ValueError(f"rebuilt scorer input of {run} does not match its recorded hash")
    print(f"expanded receipts under {out}")


def table(root):
    load = workload()
    lines = ["| arm | round | CPUs | CPU-s/audio-s | x real time (wall) | peak RSS MB | WER clean / other / pooled | "
             "MHz | other active cores |", "|---|---|---|---|---|---|---|---|---|"]
    summary = ["| arm | reps | median CPU-s/audio-s | spread | x real time | peak RSS MB | model MB | "
               "WER clean / other / pooled | same text |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, runs in repetitions(root).items():
        reps = [repetition(run, load) for run in runs]
        for rep in reps:
            cpus = ",".join(map(str, rep["service_cpus"])) if len(rep["service_cpus"]) < 3 else \
                f"{rep['service_cpus'][0]}-{rep['service_cpus'][-1]}"
            lines.append(f"| {arm} | {rep['round']} | {cpus} | {rep['cpu_per_audio']:.4f} | "
                         f"{1 / rep['wall_per_audio']:.2f} | {rep['peak_rss_bytes'] / 1e6:.0f} | "
                         f"{rep['wer_clean']:.2f} / {rep['wer_other']:.2f} / {rep['wer']:.2f} | "
                         f"{(rep['frequency_khz'] or 0) / 1000:.0f} | {len(rep['other_active_cores'])} |")
        total = arm_summary(reps)
        mid = total["median"]
        summary.append(f"| {arm} | {len(reps)} | {total['cpu_per_audio']:.4f} | {100 * total['spread']:.1f}% | "
                       f"{1 / total['wall_per_audio']:.2f} | {total['peak_rss_bytes'] / 1e6:.0f} | "
                       f"{mid['artifact_bytes'] / 1e6:.0f} | {mid['wer_clean']:.2f} / {mid['wer_other']:.2f} / "
                       f"{mid['wer']:.2f} | {'yes' if total['identical_hypotheses'] else 'no'} |")
    print("\n".join(lines + [""] + summary))


def rows(root, session, anchors):
    load = workload()
    by_variant = {}
    for arm, runs in repetitions(root).items():
        variant, suffix = split_arm(arm)
        if len(runs) != load["rounds"] or (variant in ANCHORS) != anchors:
            continue
        reps = [repetition(run, load) for run in runs]
        by_variant.setdefault((variant, MODES[suffix][0]), []).append((arm, suffix, reps, arm_summary(reps)))
    for (variant, threads), candidates in sorted(by_variant.items()):
        candidates.sort(key=lambda item: item[3]["wall_per_audio"])
        arm, suffix, reps, total = candidates[0]
        system, short, macro, runtime, kind, role, configuration = SYSTEMS[variant]
        mid = total["median"]
        cpus = sorted({tuple(rep["service_cpus"]) for rep in reps})
        row = {
            "id": f"{variant}-zen2-t{threads}", "variant": variant, "system": system, "short": short, "macro": macro,
            "runtime": runtime, "role": role, "class": kind, "machine": "zen2", "host": mid["host"],
            "hardware": "AMD Ryzen Threadripper PRO 3995WX, CPUs " + "; ".join(
                ",".join(map(str, group)) for group in cpus) + " (one set per repetition; SMT siblings and the "
                "rest of each CCX idle)",
            "threads": threads, "parallel": MODES[suffix][1], "configuration": configuration, "arm": arm,
            "workload": "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)",
            "workload_digest": load["digest"], "scope": "cold_process_tree_launch_to_exit",
            "cpu_s_per_audio_s": total["cpu_per_audio"], "cpu_reps": total["cpu_reps"],
            "wall_s_per_audio_s": total["wall_per_audio"], "wall_reps": total["wall_reps"],
            "wer_pct": mid["wer"], "wer_clean_pct": mid["wer_clean"], "wer_other_pct": mid["wer_other"],
            "wer_reps": [rep["wer"] for rep in reps], "repeated_hypotheses_identical": total["identical_hypotheses"],
            "peak_rss_bytes": total["peak_rss_bytes"], "artifact_bytes": mid["artifact_bytes"],
            "n_busy_cores": [len(rep["other_active_cores"]) + len(rep["service_cpus"]) for rep in reps],
            "service_mhz": [round(rep["frequency_khz"] / 1000) if rep["frequency_khz"] else None for rep in reps],
            "recogniser_s_per_audio_s": (mid["recogniser_s"] / load["audio_seconds"]
                                         if mid["recogniser_s"] is not None else None),
            "label": "exploratory", "session": session,
            "source": str(mid["run"] / "measurement.json"), "receipt_sha256": sha256(mid["run"] / "measurement.json"),
        }
        if len(candidates) > 1:
            row["alternatives"] = [{
                "arm": other_arm, "parallel": MODES[other_suffix][1],
                "cpu_s_per_audio_s": other["cpu_per_audio"], "cpu_reps": other["cpu_reps"],
                "wall_s_per_audio_s": other["wall_per_audio"], "wall_reps": other["wall_reps"],
                "peak_rss_bytes": other["peak_rss_bytes"],
            } for other_arm, other_suffix, _, other in candidates[1:]]
        print(json.dumps(row))


def main():
    command = sys.argv[1:2]
    if command == ["collect"] and len(sys.argv) == 4:
        collect(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == ["expand"] and len(sys.argv) == 5:
        expand(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif command == ["table"] and len(sys.argv) == 3:
        table(Path(sys.argv[2]))
    elif command in (["rows"], ["anchors"]) and len(sys.argv) == 4:
        rows(Path(sys.argv[2]), sys.argv[3], anchors=command == ["anchors"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
