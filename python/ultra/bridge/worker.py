"""Binary PCM IPC for the delivered NumPy/ORT bundle consumer; no dataset/scorer imports.

Requests: little-endian u32 JSON-header size, UTF-8 JSON, then each item's s16le PCM.
Responses: one JSON line, echoing run/worker/sequence. EOF before shutdown is an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

MAX_HEADER = 65536


def exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError("truncated IPC frame")
    return data


def request(stream):
    n, = struct.unpack("<I", exact(stream, 4))
    if not 0 < n <= MAX_HEADER:
        raise ValueError("invalid IPC header length")
    return json.loads(exact(stream, n))


def serve(rec, stream_in, stream_out, run_id, manifest_sha, load_s, calls=None, timings=None):
    import numpy as np

    states = {}
    seen = set()
    seq = 0
    while True:
        q = request(stream_in)
        timing_start = len(timings) if timings is not None else 0
        if set(q) != {"run_id", "worker_id", "seq", "kind", "items"}:
            raise ValueError("unexpected request fields")
        if type(q["seq"]) is not int or q["run_id"] != run_id or q["worker_id"] != "w0" or q["seq"] != seq:
            raise ValueError("stale worker/run/sequence")
        reply = dict(run_id=run_id, worker_id="w0", seq=seq, kind=q["kind"], rows=[])
        seq += 1
        if q["kind"] == "init":
            if seq != 1 or q["items"]:
                raise ValueError("invalid init")
            providers = [s.get_providers() for s in (rec.enc, rec.dec, rec.joi)]
            if any(p != ["CPUExecutionProvider"] for p in providers):
                raise ValueError("non-CPU execution provider")
            import onnxruntime as ort

            paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if line.split()[-1].startswith("/") and "onnxruntime" in line.split()[-1]
                     and ".so" in Path(line.split()[-1]).name}
            reply["info"] = dict(manifest_sha256=manifest_sha, load_s=load_s,
                                 providers=providers, numpy=np.__version__, ort=ort.__version__,
                                 runtime_libraries={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                    for p in paths}, cpu_only_verified=True)
        elif q["kind"] == "shutdown":
            if q["items"] or states:
                raise ValueError("unfinished streams at shutdown")
        elif q["kind"] == "batch":
            items = q["items"]
            if not 0 < len(items) <= 32 or len({i["id"] for i in items}) != len(items):
                raise ValueError("invalid batch")
            for i in items:
                if set(i) != {"id", "offset", "samples", "start", "finish"}:
                    raise ValueError("unexpected audio fields")
                if (type(i["samples"]) is not int or not 0 < i["samples"] <= 1600
                        or type(i["offset"]) is not int or i["offset"] < 0
                        or type(i["id"]) is not str or not i["id"]
                        or type(i["start"]) is not bool or type(i["finish"]) is not bool):
                    raise ValueError("invalid packet length")
                if i["start"]:
                    if i["id"] in seen:
                        raise ValueError("reused stream instance")
                    seen.add(i["id"])
                    states[i["id"]] = [rec.create_stream(), 0, hashlib.sha256()]
                s, accepted, digest = states[i["id"]]
                if i["offset"] != accepted or s.finished:
                    raise ValueError("out of order or closed audio")
                raw = exact(stream_in, i["samples"] * 2)
                digest.update(raw)
                s.accept_waveform(16000, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768)
                states[i["id"]][1] += i["samples"]
                if i["finish"]:
                    s.input_finished()
            active = [states[i["id"]][0] for i in items]
            while True:
                ready = [s for s in active if rec.is_ready(s)]
                if not ready:
                    break
                before = [s.t_next for s in ready]
                rec.decode_streams(ready)
                if any(s.t_next <= t for s, t in zip(ready, before, strict=True)):
                    raise ValueError("decoder made no progress")
            for i in items:
                s, accepted, digest = states[i["id"]]
                done = s.finished and not rec.is_ready(s)
                # The consumer has no authoritative per-sample processing watermark.
                # Keep streaming work conservatively unfinished; credit only actual EOF drain.
                reply["rows"].append(dict(id=i["id"], accepted_samples=accepted,
                    processed_samples=accepted if done else 0, terminal=done, drained=done,
                    hyp=rec.get_result(s), forced_advances=s.forced,
                    pcm_sha256=digest.hexdigest() if done else None))
                if done:
                    del states[i["id"]]
        else:
            raise ValueError("unknown request")
        reply["encoder_calls"] = calls[0] if calls else 0
        reply["model_batch_ms"] = timings[timing_start:] if timings is not None else []
        stream_out.write(json.dumps(reply, allow_nan=False, separators=(",", ":")).encode() + b"\n")
        stream_out.flush()
        if q["kind"] == "shutdown":
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--precision", choices=["fp32", "int8"], required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    if a.threads <= 0:
        raise ValueError("threads must be positive")
    # -I ignores user PYTHONPATH; this exact repo-relative consumer is the sole import root.
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
    started = time.monotonic()
    manifest = Path(a.bundle, "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != a.manifest_sha256:
        raise ValueError("bundle identity changed")
    from train.bundle_infer import BundleRecognizer

    rec = BundleRecognizer(a.bundle, a.precision, a.threads)
    calls = [0]
    timings = []
    original_run = rec._run

    def counted_run(*args, **kwargs):
        calls[0] += 1
        started = time.monotonic()
        result = original_run(*args, **kwargs)
        timings.append((time.monotonic() - started) * 1000)
        return result

    rec._run = counted_run
    # CPUExecutionProvider is verified on all three actual sessions in the init response.
    serve(rec, sys.stdin.buffer, sys.stdout.buffer, a.run_id, digest, time.monotonic() - started, calls, timings)


if __name__ == "__main__":
    main()
