"""Collect, tabulate and export the open opponent receipts (standard library only).

    open_summary.py collect RUNS RAW_DIR          copy every finished, clean repetition from
                                                  RUNS/{1c,8c}[-p73]/cold/round-N/ARM to
                                                  RAW_DIR/ARM/[p73-]round-N in compact form
    open_summary.py collect-wer PASS RAW_DIR ARM  copy a full-panel WER-only pass to RAW_DIR/ARM/wer-full
    open_summary.py expand RAW_DIR PANEL OUT      rebuild the run-cold.sh layout from the compact form
                                                  and the staged full panel, hash-checked
    open_summary.py table DIR                     one line per repetition and per arm (markdown)
    open_summary.py rows DIR SESSION              evidence rows (data/README.md schema) as JSONL
    open_summary.py anchors DIR SESSION           our engine's anchor repetitions, one row per panel

Timing panels: `full` (730 utterances; cost and WER from the same repetitions) and `p73`
(every tenth utterance; cost only). A system timed on p73 takes its WER from one full-panel
pass (`ARM/wer-full`, same binary and model hashes, any host load); its 73 shared hypotheses
are compared with every timed repetition and the counts are in the row.

Compact form: the receipt files of run-cold.sh plus the host records of open-run.sh, except
that `scores.jsonl` (hypotheses joined with the references) is replaced by its hash in
`scores.jsonl.sha256`, and a `predictions.jsonl` identical to the first round's is replaced by
`predictions.jsonl.ref`. `expand` restores both byte for byte and verifies the hashes.
Directories whose name starts with `loaded` hold cross-checks measured on a busy host; they
are listed by `table` when named directly and never enter `rows`.

A row's cost is the median repetition; its peak memory is the largest of the repetitions (as
paper_cold_aggregate.py does); on the full panel its WER is the median repetition's score.
"""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
RECEIPT_FILES = (
    "arm", "panel", "audio.seconds", "command", "config.sha256", "runtime-libraries.sha256", "workload.sha256",
    "host", "host.before", "host.after", "round", "started", "finished", "measurement.json", "service.stderr",
    "score.json", "host.cpu.json", "host.freq.tsv", "host.gate.json", "artifact.bytes", "trace.sha256",
    "trace.summary.json", "harness.commit",
)
WER_PASS_FILES = (
    "command", "config.sha256", "workload.sha256", "host", "host.before", "host.after", "started", "finished",
    "harness.commit", "service.stderr", "predictions.jsonl", "score.json", "trace.sha256",
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
    "jsr-h22ctl-79800-ema-int8": (
        "JSR H22ctl 79800 EMA INT8", "JSR H22ctl INT8", "JsrHtwentytwo", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours", "H22ctl step 79800 EMA, prepared INT8, greedy, 20 ms packets (session anchor)"),
    "jsr-a1-30000-ema-int8": (
        "JSR A1 30000 EMA INT8", "JSR A1 30k INT8", "JsrAoneThirtyK", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours", "A1 step 30000 EMA, prepared INT8, greedy, 20 ms packets (TIMING entry 6b)"),
    "jsr-a1-49000-ema-int8": (
        "JSR A1 49000 EMA INT8", "JSR A1 final INT8", "JsrAoneFinal", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours", "A1 step 49000 EMA (A1 final), prepared INT8, greedy, 20 ms packets (TIMING entry 14b)"),
    "jsr-f1-60000-ema-int8": (
        "JSR F1 60000 EMA INT8", "JSR F1 final INT8", "JsrFoneFinal", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours",
        "F1 step 60000 EMA (F1 final, multirate S_MR), prepared INT8, greedy, 20 ms packets (TIMING entry 14c)"),
    "jsr-arm13a-ema-int8-c1": (
        "JSR arm 13a EMA INT8, chunk 1", "JSR 13a INT8 c1", "JsrThirteenAcOne", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours",
        "arm 13a (delay penalty 0.01) step 8000 EMA, prepared INT8, greedy, chunk 1 = 40 ms (arch.json chunk fields "
        "set to 1, weights unchanged), 20 ms packets (TIMING entry 14d)"),
    "jsr-arm13a-ema-int8-c2": (
        "JSR arm 13a EMA INT8, chunk 2", "JSR 13a INT8 c2", "JsrThirteenAcTwo", "ultra-engine e2ef608 (ultra-cold-wav)",
        "streaming", "ours",
        "arm 13a (delay penalty 0.01) step 8000 EMA, prepared INT8, greedy, chunk 2 = 80 ms (arch.json chunk fields "
        "set to 2, weights unchanged), 20 ms packets (TIMING entry 14d)"),
    "jsr-rc1-ema-int8": (
        "ULTRA v3.0.0 (live)", "ULTRA v3.0.0 (live)", "UltraLive", "ultra-engine v3.0.0 9c8c05e (ultra-cold-wav)",
        "streaming", "ours",
        "RC1-EMA (weights.int8 edac19d8), prepared INT8, greedy, chunk 4 = 160 ms, 20 ms packets, the live stream "
        "only (no final pass, no LM) (TIMING entry 14q)"),
    "jsr-d2e-16457-int8": (
        "JSR D2e 16457 INT8", "JSR D2e INT8", "JsrDtwoE", "ultra-engine e2ef608 (ultra-cold-wav)", "streaming", "ours",
        "D2e step 16457, prepared INT8, greedy, 20 ms packets (session anchor)"),
    "r0-zipformer-int8": (
        "Zipformer INT8", "Zipformer INT8", "RzeroZipformer", "sherpa-onnx 2.0.2 (ultra-cold-wav e2ef608)",
        "streaming", "comparator",
        "streaming Zipformer 2023-06-26 INT8, greedy, 320 ms chunks, 20 ms packets (session anchor)"),
}
ANCHORS = {"jsr-h22ctl-79800-ema-int8", "jsr-d2e-16457-int8", "r0-zipformer-int8"}
# The paper's variant names where they differ from the arm names (the anchors join the primary rows).
PAPER_VARIANT = {"jsr-h22ctl-79800-ema-int8": "h22ctl-79800-ema-int8", "jsr-d2e-16457-int8": "d2e-16457-int8",
                 "r0-zipformer-int8": "r0-int8", "jsr-a1-30000-ema-int8": "a1-30000-ema-int8",
                 "jsr-a1-49000-ema-int8": "a1-49000-ema-int8", "jsr-f1-60000-ema-int8": "f1-60000-ema-int8",
                 "jsr-arm13a-ema-int8-c1": "arm13a-ema-int8-c1", "jsr-arm13a-ema-int8-c2": "arm13a-ema-int8-c2",
                 "jsr-rc1-ema-int8": "rc1-ema-live"}
SHARED_MODEL = ("vosk-",)  # --workers N shares one loaded model between N recogniser threads
PANEL_LABEL = {
    "full": "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)",
    "p73": "panel73: every tenth utterance of the DEV selection panel, 73 utt (40 dev-clean + 33 dev-other)",
}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def workload():
    return json.loads((REPO / "docs/paper/evidence.json").read_text())["workloads"]["cold-panel"]


def split_arm(arm):
    """`(variant, threads, parallel)` of an arm name."""
    for suffix, parallel in (("-t8", "intra_op"), ("-w8", None), ("-s8", "workers")):
        if arm.endswith(suffix):
            variant = arm[:-len(suffix)]
            return variant, 8, parallel or ("workers" if variant.startswith(SHARED_MODEL) else "processes")
    return arm, 1, "intra_op"


def loaded(path):
    return any(part.startswith("loaded") for part in path.parts)


def repetitions(root):
    """Finished repetitions under `root`, keyed by `(arm, panel)`, in round order."""
    found = {}
    for measurement in sorted(root.rglob("measurement.json")):
        run = measurement.parent
        if ".contaminated." in str(run) or ".incomplete." in str(run) or loaded(run.relative_to(root)):
            continue
        if not (run / "finished").is_file() or not (run / "arm").is_file():
            continue
        key = ((run / "arm").read_text().strip(), panel_of(run))
        found.setdefault(key, []).append(run)
    return {key: sorted(runs, key=lambda run: int((run / "round").read_text())) for key, runs in sorted(found.items())}


def panel_of(run):
    """Timing panel of a repetition; receipts written before protocol v3 are full-panel runs."""
    return (run / "panel").read_text().strip() if (run / "panel").is_file() else "full"


def mean_frequency_khz(run):
    values = []
    for line in (run / "host.freq.tsv").read_text().splitlines():
        fields = line.split("\t")
        values.extend(int(value) for value in fields[1:-1])
    return statistics.fmean(values) if values else None


def predictions_path(run):
    if (run / "predictions.jsonl").is_file():
        return run / "predictions.jsonl"
    return run.parent / (run / "predictions.jsonl.ref").read_text().split()[0] / "predictions.jsonl"


def predictions_sha256(run):
    """Hash of a repetition's predictions.jsonl, in either form."""
    if (run / "predictions.jsonl").is_file():
        return sha256(run / "predictions.jsonl")
    return (run / "predictions.jsonl.ref").read_text().split()[1]


def hypotheses(path):
    return {row["id"]: row["hyp"] for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def score_fields(score):
    return {"wer": 100 * score["pooled_wer"], "wer_clean": 100 * score["datasets"]["dev-clean"]["wer"],
            "wer_other": 100 * score["datasets"]["dev-other"]["wer"]}


def repetition(run):
    measure = json.loads((run / "measurement.json").read_text())
    guard = json.loads((run / "host.cpu.json").read_text())
    if measure["exit_code"] != 0:
        raise ValueError(f"failed repetition {run}")
    audio = (float((run / "audio.seconds").read_text()) if (run / "audio.seconds").is_file()
             else workload()["audio_seconds"])
    cpu = measure["cpu_seconds_user"] + measure["cpu_seconds_sys"]
    result = {
        "run": run, "round": int((run / "round").read_text()), "host": (run / "host").read_text().strip(),
        "panel": panel_of(run), "audio_seconds": audio,
        "workload_digest": (run / "workload.sha256").read_text().split()[0],
        "service_cpus": measure["placement"]["effective_service_cpus_at_spawn"],
        "cpu_s": cpu, "wall_s": measure["wall_seconds"],
        "cpu_per_audio": cpu / audio, "wall_per_audio": measure["wall_seconds"] / audio,
        "peak_rss_bytes": measure["peak_rss_bytes"], "max_threads": measure["max_threads"],
        "max_processes": measure["max_processes"],
        "predictions_sha256": predictions_sha256(run),
        "artifact_bytes": int((run / "artifact.bytes").read_text()),
        "started": (run / "started").read_text().strip(), "finished": (run / "finished").read_text().strip(),
        "verdict": guard["verdict"], "other_active_cores": guard["other_active_cores"],
        "sibling_busy_max": max(guard["smt_siblings_busy"].values()),
        "frequency_khz": mean_frequency_khz(run), "loadavg": [guard["loadavg_before"][0], guard["loadavg_after"][0]],
        "recogniser_s": (json.loads((run / "trace.summary.json").read_text())["recogniser_seconds"]
                         if (run / "trace.summary.json").is_file() else None),
        "harness_commit": (run / "harness.commit").read_text().strip() if (run / "harness.commit").is_file() else None,
    }
    if (run / "score.json").is_file():
        score = json.loads((run / "score.json").read_text())["system_a"]
        if score["utterances"] != workload()["utterances"]:
            raise ValueError(f"wrong score coverage in {run}")
        result.update(score_fields(score))
    return result


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


def wer_pass(root, variant):
    """The full-panel WER-only pass of a system, or None."""
    run = root / variant / "wer-full"
    if not (run / "score.json").is_file():
        return None
    score = json.loads((run / "score.json").read_text())["system_a"]
    return {"run": run, "host": (run / "host").read_text().strip(), "hypotheses": hypotheses(run / "predictions.jsonl"),
            "command": (run / "command").read_text().strip(), **score_fields(score)}


def shared_equal(rep, full):
    """How many of a repetition's hypotheses equal the full-panel pass's for the same utterance."""
    hyps = hypotheses(predictions_path(rep["run"]))
    return sum(full["hypotheses"].get(ident) == text for ident, text in hyps.items()), len(hyps)


def compact(run, target, first):
    target.mkdir(parents=True, exist_ok=True)
    for name in RECEIPT_FILES:
        if (run / name).is_file():
            shutil.copyfile(run / name, target / name)
    if (run / "scores.jsonl").is_file():
        (target / "scores.jsonl.sha256").write_text(sha256(run / "scores.jsonl") + "\n")
    digest = sha256(run / "predictions.jsonl")
    for stale in ("predictions.jsonl", "predictions.jsonl.ref"):
        (target / stale).unlink(missing_ok=True)
    if target != first and (first / "predictions.jsonl").is_file() and sha256(first / "predictions.jsonl") == digest:
        (target / "predictions.jsonl.ref").write_text(f"{first.name} {digest}\n")
    else:
        shutil.copyfile(run / "predictions.jsonl", target / "predictions.jsonl")


def collect(runs, raw):
    copied = 0
    for group, prefix in (("1c", ""), ("8c", ""), ("1c-p73", "p73-"), ("8c-p73", "p73-")):
        for run in sorted((runs / group / "cold").glob("round-*/*")):
            if ".contaminated." in run.name or ".incomplete." in run.name or not (run / "finished").is_file():
                continue
            if not (run / "host.cpu.json").is_file():
                continue
            arm_dir = raw / run.name
            rounds = sorted(path.parent.parent.name
                            for path in (runs / group / "cold").glob(f"round-*/{run.name}/host.cpu.json"))
            compact(run, arm_dir / f"{prefix}{run.parent.name}", arm_dir / f"{prefix}{rounds[0]}")
            copied += 1
    print(f"copied {copied} repetitions to {raw}")


def collect_wer(source, raw, arm):
    target = raw / arm / "wer-full"
    target.mkdir(parents=True, exist_ok=True)
    for name in WER_PASS_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    (target / "scores.jsonl.sha256").write_text(sha256(source / "scores.jsonl") + "\n")
    print(f"copied the WER pass of {arm} to {target}")


def scorer_input(index, hyps, path):
    """Same bytes as paper_dev_audio.score writes: reference row without id/uid, then uid and hypothesis."""
    with path.open("x", encoding="utf-8") as scores:
        for reference in index:
            row = {key: value for key, value in reference.items() if key not in {"id", "uid"}}
            json.dump({**row, "id": reference["uid"], "hyp": hyps[reference["id"]]}, scores)
            scores.write("\n")


def expand(raw, panel, out):
    """Rebuild the full receipts: every compact file, predictions.jsonl and the scorer input."""
    index = json.loads((panel / "index.json").read_text())
    for (arm, kind), runs in repetitions(raw).items():
        for run in runs:
            measure = json.loads((run / "measurement.json").read_text())
            group = "1c" if len(measure["placement"]["effective_service_cpus_at_spawn"]) == 1 else "8c"
            group += "" if kind == "full" else f"-{kind}"
            target = out / group / "cold" / f"round-{int((run / 'round').read_text())}" / arm
            target.mkdir(parents=True)
            for name in RECEIPT_FILES:
                if (run / name).is_file():
                    shutil.copyfile(run / name, target / name)
            shutil.copyfile(predictions_path(run), target / "predictions.jsonl")
            if sha256(target / "predictions.jsonl") != predictions_sha256(run):
                raise ValueError(f"predictions of {run} do not match their recorded hash")
            if (run / "scores.jsonl.sha256").is_file():
                scorer_input(index, hypotheses(target / "predictions.jsonl"), target / "scores.jsonl")
                if sha256(target / "scores.jsonl") != (run / "scores.jsonl.sha256").read_text().strip():
                    raise ValueError(f"rebuilt scorer input of {run} does not match its recorded hash")
    print(f"expanded receipts under {out}")


def cpu_list(cpus):
    return ",".join(map(str, cpus)) if len(cpus) < 3 else f"{cpus[0]}-{cpus[-1]}"


def table(root):
    lines = ["| arm | panel | round | CPUs | CPU-s/audio-s | x real time (wall) | peak RSS MB | WER clean / other / "
             "pooled | MHz | other busy cores | alone |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = ["| arm | panel | reps | median CPU-s/audio-s | spread | x real time | peak RSS MB | model MB | "
               "same text |", "|---|---|---|---|---|---|---|---|---|"]
    for (arm, kind), runs in repetitions(root).items():
        reps = [repetition(run) for run in runs]
        for rep in reps:
            wer = f"{rep['wer_clean']:.2f} / {rep['wer_other']:.2f} / {rep['wer']:.2f}" if "wer" in rep else "-"
            lines.append(f"| {arm} | {kind} | {rep['round']} | {cpu_list(rep['service_cpus'])} | "
                         f"{rep['cpu_per_audio']:.4f} | {1 / rep['wall_per_audio']:.2f} | "
                         f"{rep['peak_rss_bytes'] / 1e6:.0f} | {wer} | {(rep['frequency_khz'] or 0) / 1000:.0f} | "
                         f"{len(rep['other_active_cores'])} | {'yes' if rep['verdict'] == 'clean' else 'NO'} |")
        total = arm_summary(reps)
        summary.append(f"| {arm} | {kind} | {len(reps)} | {total['cpu_per_audio']:.4f} | {100 * total['spread']:.1f}% | "
                       f"{1 / total['wall_per_audio']:.2f} | {total['peak_rss_bytes'] / 1e6:.0f} | "
                       f"{total['median']['artifact_bytes'] / 1e6:.0f} | "
                       f"{'yes' if total['identical_hypotheses'] else 'no'} |")
    print("\n".join(lines + [""] + summary))


def base_row(variant, arm, threads, parallel, reps, total, session):
    system, short, macro, runtime, kind, role, configuration = SYSTEMS[variant]
    load = workload()
    mid = total["median"]
    cpus = sorted({tuple(rep["service_cpus"]) for rep in reps})
    paper_variant = PAPER_VARIANT.get(variant, variant)
    row = {
        "id": f"{paper_variant}-zen2-t{threads}", "variant": paper_variant, "system": system, "short": short,
        "macro": macro, "runtime": runtime, "role": role, "class": kind, "machine": "zen2", "host": mid["host"],
        "hardware": "AMD Ryzen Threadripper PRO 3995WX, CPUs " + "; ".join(cpu_list(group) for group in cpus)
                    + " (one set per repetition; the arm alone on the host)",
        "threads": threads, "parallel": parallel, "configuration": configuration, "arm": arm,
        "workload": PANEL_LABEL["full"], "workload_digest": load["digest"], "audio_seconds": load["audio_seconds"],
        "timing_panel": mid["panel"], "scope": "cold_process_tree_launch_to_exit",
        "cpu_s_per_audio_s": total["cpu_per_audio"], "cpu_reps": total["cpu_reps"],
        "wall_s_per_audio_s": total["wall_per_audio"], "wall_reps": total["wall_reps"],
        "peak_rss_bytes": total["peak_rss_bytes"], "artifact_bytes": mid["artifact_bytes"],
        "repeated_hypotheses_identical": total["identical_hypotheses"],
        "n_busy_cores_reps": [len(rep["other_active_cores"]) + len(rep["service_cpus"]) for rep in reps],
        "service_mhz": [round(rep["frequency_khz"] / 1000) if rep["frequency_khz"] else None for rep in reps],
        "repetition_windows": [[rep["started"], rep["finished"]] for rep in reps],
        "recogniser_s_per_audio_s": (mid["recogniser_s"] / mid["audio_seconds"]
                                     if mid["recogniser_s"] is not None else None),
        "label": "exploratory", "session": session,
        "source": str(mid["run"] / "measurement.json"), "receipt_sha256": sha256(mid["run"] / "measurement.json"),
    }
    commits = sorted({rep["harness_commit"] for rep in reps if rep.get("harness_commit")})
    if commits:
        row["harness_commits"] = commits
    if mid["panel"] != "full":
        row.update(timing_workload=PANEL_LABEL[mid["panel"]], timing_workload_digest=mid["workload_digest"],
                   timing_audio_seconds=mid["audio_seconds"])
    return row


def full_panel_wer(root, variant, groups, rounds):
    """Where a system timed on panel73 takes its WER from: `(scores, hypotheses, description)` or None."""
    full = wer_pass(root, variant)
    if full is not None:
        return full, f"one full-panel pass on {full['host']} (same binary and model hashes; {full['command']})"
    timed = groups.get((variant, "full"), [])
    if len(timed) != rounds or any(rep["verdict"] != "clean" for rep in timed):
        return None
    chosen = arm_summary(timed)["median"]
    full = {"hypotheses": hypotheses(predictions_path(chosen["run"])), "wer": chosen["wer"],
            "wer_clean": chosen["wer_clean"], "wer_other": chosen["wer_other"]}
    return full, f"the system's one-core full-panel repetitions on {chosen['host']} (median repetition)"


def rows(root, session):
    load = workload()
    groups = {key: [repetition(run) for run in runs] for key, runs in repetitions(root).items()}
    by_variant = {}
    for (arm, kind), reps in groups.items():
        variant, threads, parallel = split_arm(arm)
        reps = [rep for rep in reps if rep["verdict"] == "clean"]
        if variant in ANCHORS or not reps:
            continue
        by_variant.setdefault((variant, threads), []).append((arm, parallel, reps, arm_summary(reps)))
    for (variant, threads), measured in sorted(by_variant.items()):
        # The primary row is the fastest mode with all its repetitions; every other mode, with any
        # number of repetitions, is listed under alternatives.
        complete = sorted((item for item in measured if len(item[2]) == load["rounds"]),
                          key=lambda item: item[3]["wall_per_audio"])
        if not complete:
            continue
        candidates = complete + [item for item in measured if item not in complete]
        arm, parallel, reps, total = candidates[0]
        row = base_row(variant, arm, threads, parallel, reps, total, session)
        mid = total["median"]
        if mid["panel"] == "full":
            row.update(wer_pct=mid["wer"], wer_clean_pct=mid["wer_clean"], wer_other_pct=mid["wer_other"],
                       wer_reps=[rep["wer"] for rep in reps], wer_source="the timed repetitions (median repetition)")
        else:
            source = full_panel_wer(root, variant, groups, load["rounds"])
            if source is None:
                continue
            full, description = source
            row.update(
                wer_pct=full["wer"], wer_clean_pct=full["wer_clean"], wer_other_pct=full["wer_other"],
                wer_source=description,
                shared_hypotheses_equal_full_pass=["%d/%d" % shared_equal(rep, full) for rep in reps])
        if len(candidates) > 1:
            row["alternatives"] = [{
                "arm": other_arm, "parallel": other_parallel, "reps": len(other_reps),
                "cpu_s_per_audio_s": other["cpu_per_audio"], "cpu_reps": other["cpu_reps"],
                "wall_s_per_audio_s": other["wall_per_audio"], "wall_reps": other["wall_reps"],
                "peak_rss_bytes": other["peak_rss_bytes"],
            } for other_arm, other_parallel, other_reps, other in candidates[1:]]
        print(json.dumps(row))


def anchors(root, session):
    """One anchor row per (arm, panel): every repetition of our engine in this session's holds."""
    groups = {key: [repetition(run) for run in runs] for key, runs in repetitions(root).items()}
    full_wer = {}
    for (arm, kind), reps in sorted(groups.items(), key=lambda item: item[0][1] != "full"):
        variant, threads, parallel = split_arm(arm)
        reps = [rep for rep in reps if rep["verdict"] == "clean"]
        if variant not in ANCHORS or len(reps) < workload()["rounds"]:
            continue
        total = arm_summary(reps)
        row = base_row(variant, arm, threads, parallel, reps, total, session)
        mid = total["median"]
        row.update(id=f"{row['id']}-anchor-bench-open-{kind}", anchor=True, anchor_holds=[{
            "window": [rep["started"], rep["finished"]], "cpus": rep["service_cpus"],
            "cpu_s_per_audio_s": rep["cpu_per_audio"], "wall_s_per_audio_s": rep["wall_per_audio"]} for rep in reps])
        if kind == "full":
            full_wer[variant] = (mid, total)
            row.update(wer_pct=mid["wer"], wer_clean_pct=mid["wer_clean"], wer_other_pct=mid["wer_other"],
                       wer_source="the anchor repetitions (median repetition)")
        elif variant in full_wer:
            chosen, _ = full_wer[variant]
            row.update(wer_pct=chosen["wer"], wer_clean_pct=chosen["wer_clean"], wer_other_pct=chosen["wer_other"],
                       wer_source="this session's full-panel anchor repetitions (median repetition)")
        else:
            continue
        print(json.dumps(row))


def main():
    command, arguments = sys.argv[1:2], sys.argv[2:]
    if command == ["collect"] and len(arguments) == 2:
        collect(Path(arguments[0]), Path(arguments[1]))
    elif command == ["collect-wer"] and len(arguments) == 3:
        collect_wer(Path(arguments[0]), Path(arguments[1]), arguments[2])
    elif command == ["expand"] and len(arguments) == 3:
        expand(*map(Path, arguments))
    elif command == ["table"] and len(arguments) == 1:
        table(Path(arguments[0]))
    elif command == ["rows"] and len(arguments) == 2:
        rows(Path(arguments[0]), arguments[1])
    elif command == ["anchors"] and len(arguments) == 2:
        anchors(Path(arguments[0]), arguments[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
