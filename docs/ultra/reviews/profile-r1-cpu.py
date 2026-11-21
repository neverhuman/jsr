"""Reference-blind CPU/operator diagnostic for an existing R1 bundle consumer.

Run with JOB.json (or `run JOB.json` through run_guarded.py). Inputs are an
audio-only bridge manifest, bundle, retained consumer source and diagnostic flags.
This is not a service benchmark or scoring implementation.
"""

import collections
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    args = sys.argv[1:]
    if args[0] == "run":
        args = args[1:]
    job_path = Path(args[0])
    job = json.loads(job_path.read_text())
    source_bytes = Path(__file__).read_bytes()
    out = Path(job["out"])
    out.mkdir()
    (out / "diagnostic-source.py").write_bytes(source_bytes)
    source = Path(job["consumer"])
    if digest(source) != job["consumer_sha256"]:
        raise ValueError("consumer source changed")
    spec = importlib.util.spec_from_file_location("profile_bundle_consumer", source)
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    workload_path = Path(job["inputs"])
    workload = json.loads(workload_path.read_text())
    if workload["sample_rate"] != 16000 or not workload["inputs"]:
        raise ValueError("unsupported audio workload")
    inputs = workload["inputs"]
    if len({item["id"] for item in inputs}) != len(inputs):
        raise ValueError("duplicate input ID")
    audio = []
    for item in inputs:
        path = workload_path.parent / item["path"]
        if not path.resolve().is_relative_to(workload_path.parent.resolve()):
            raise ValueError("escaping audio path")
        if digest(path) != item["sha256"]:
            raise ValueError("changed audio")
        pcm = np.fromfile(path, dtype="<i2")
        if len(pcm) != item["samples"]:
            raise ValueError("wrong PCM sample count")
        audio.append(pcm.astype(np.float32) / np.float32(32768))
    seconds, calls = collections.Counter(), collections.Counter()
    cache_bytes = collections.Counter()
    contexts = set()
    decoder_rows = 0
    sessions = {}
    original_session = ort.InferenceSession

    class TimedSession:
        def __init__(self, path, options, **kwargs):
            self.tag = Path(path).name.split(".")[0]
            options.enable_profiling = bool(job["operator_profile"])
            options.profile_file_prefix = str(out / self.tag)
            self.inner = original_session(path, options, **kwargs)
            if self.inner.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("non-CPU provider")
            sessions[self.tag] = self

        def run(self, outputs, feed):
            nonlocal decoder_rows
            start = time.process_time()
            result = self.inner.run(outputs, feed)
            seconds[self.tag] += time.process_time() - start
            calls[self.tag] += 1
            if self.tag == "encoder":
                cache_bytes["input_payload_bytes"] += sum(
                    value.nbytes for key, value in feed.items() if key not in ["feats", "t0"])
            if self.tag == "decoder":
                decoder_rows += len(feed["ctx"])
                contexts.update(map(tuple, feed["ctx"].tolist()))
            return result

        def get_outputs(self):
            return self.inner.get_outputs()

    load_start = time.process_time()
    with patch.object(ort, "InferenceSession", TimedSession):
        rec = consumer.BundleRecognizer(job["bundle"], job["precision"], job["threads"])
    load_cpu = time.process_time() - load_start
    featurize, greedy = rec._featurize, rec._greedy

    def timed_feature(stream):
        start = time.process_time()
        try:
            return featurize(stream)
        finally:
            seconds["frontend"] += time.process_time() - start

    def timed_greedy(streams, encoded):
        start = time.process_time()
        try:
            return greedy(streams, encoded)
        finally:
            seconds["greedy_inclusive"] += time.process_time() - start

    rec._featurize, rec._greedy = timed_feature, timed_greedy
    packet = job["packet_ms"] * 16
    if packet <= 0 or job["streams"] <= 0 or job["threads"] != 1:
        raise ValueError("diagnostic requires positive packet/streams and one inference thread")
    next_input, active, final = 0, [], [None] * len(inputs)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    while next_input < len(inputs) or active:
        while next_input < len(inputs) and len(active) < job["streams"]:
            active.append([next_input, 0, rec.create_stream()])
            next_input += 1
        for item in active:
            index, cursor, stream = item
            end = min(cursor + packet, len(audio[index]))
            if end > cursor:
                stream.accept_waveform(16000, audio[index][cursor:end])
                item[1] = end
                if end == len(audio[index]):
                    stream.input_finished()
            elif not stream.finished:
                stream.input_finished()
        while True:
            ready = [item[2] for item in active if rec.is_ready(item[2])]
            if not ready:
                break
            rec.decode_streams(ready)
        remaining = []
        for item in active:
            index, cursor, stream = item
            if cursor == len(audio[index]) and not rec.is_ready(stream):
                final[index] = dict(input_id=inputs[index]["id"], text=rec.get_result(stream))
            else:
                remaining.append(item)
        active = remaining
    cpu = time.process_time() - start_cpu
    wall = time.monotonic() - start_wall
    if any(item is None for item in final):
        raise ValueError("missing diagnostic output")
    exclusive = {key: seconds[key] for key in ["encoder", "decoder", "joiner", "frontend"]}
    exclusive["other_python_driver_and_packing"] = cpu - sum(exclusive.values())
    operators, traces = {}, {}
    for tag, session in sessions.items():
        if not job["operator_profile"]:
            continue
        trace_path = Path(session.inner.end_profiling())
        op_time, op_calls = collections.Counter(), collections.Counter()
        for event in json.loads(trace_path.read_text()):
            if (event.get("cat") == "Node" and event.get("ph") == "X" and
                    event.get("args", {}).get("provider") == "CPUExecutionProvider"):
                operation = event["args"].get("op_name", event["name"])
                op_time[operation] += event["dur"]
                op_calls[operation] += 1
        operators[tag] = [dict(operation=op, wall_us=duration, calls=op_calls[op])
                          for op, duration in op_time.most_common()]
        traces[trace_path.name] = digest(trace_path)
    with (out / "final.jsonl").open("x") as stream:
        for item in final:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = dict(schema="ultra.r1-cpu-diagnostic/v1", job=job,
                  scope="Consumer diagnostic only; excludes service/IPC and measurement observer",
                  cpu_scope="Current process including ORT CPU kernels; nested greedy timer is not additive",
                  operator_scope="Instrumented operator WALL durations, not CPU fractions or latency evidence",
                  operator_profile=job["operator_profile"], inference_cpu_s=cpu,
                  wall_s=wall, wall_timing_evidence=False, qualifying=False,
                  cold_load_cpu_s=load_cpu, stage_cpu_s=exclusive,
                  stage_cpu_fraction={key: value / cpu for key, value in exclusive.items()},
                  greedy_inclusive_cpu_s=seconds["greedy_inclusive"], session_calls=dict(calls),
                  decoder_context_rows=decoder_rows, unique_decoder_contexts=len(contexts),
                  decoder_context_scope="Exact context-pair repetition count; not measured cache speedup",
                  cache_io=dict(cache_bytes),
                  cache_scope="Encoder input state payload sum; not measured memory traffic/bandwidth",
                  audio_s=sum(len(item) for item in audio) / 16000,
                  examples=len(final), forced_advances=rec.forced_advances,
                  manifest_sha256=digest(Path(job["bundle"]) / "manifest.json"),
                  workload_sha256=digest(workload_path), consumer_sha256=digest(source),
                  diagnostic_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  final_sha256=digest(out / "final.jsonl"),
                  numpy=np.__version__, ort=ort.__version__,
                  affinity=sorted(os.sched_getaffinity(0)), operator_traces=traces,
                  operators=operators)
    with (out / "profile.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ["examples", "inference_cpu_s", "stage_cpu_fraction", "session_calls"]}))


if __name__ == "__main__":
    main()
