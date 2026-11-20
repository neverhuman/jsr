"""SP-A: real-time stream density of a streaming transducer on CPU, at measured WER.

Modes
  throughput  unpaced; N concurrent streams; reports audio-seconds per CPU-second
              (= streams one core can sustain at 100% utilisation) and WER.
  paced       real-time arrival for N streams; reports lateness percentiles and backlog,
              which decide whether N streams are actually sustained.

decode = "single" decodes each ready stream alone (one weight pass per stream per chunk);
decode = "batch" decodes all ready streams in one call (one weight pass per tick).
CPU is getrusage(RUSAGE_SELF) user+sys: every inference thread lives in this process.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import sherpa_onnx

from ultra.libri import load_split
from ultra.wer import corpus_wer

MODELS = os.path.join(os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr"), "models")


def make_recognizer(model_dir: str, precision: str, threads: int):
    tag = "epoch-99-avg-1-chunk-16-left-128" + (".int8" if precision == "int8" else "")
    p = lambda kind: os.path.join(model_dir, f"{kind}-{tag}.onnx")  # noqa: E731
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(model_dir, "tokens.txt"), encoder=p("encoder"), decoder=p("decoder"),
        joiner=p("joiner"), num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu")


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run(args) -> dict:
    utts = load_split(args.split, args.limit)
    rec = make_recognizer(args.model_dir, args.precision, args.threads)
    packet = int(16000 * args.packet_ms / 1000)
    queue = list(range(len(utts)))
    active: dict[int, list] = {}  # slot -> [utt_index, stream, cursor]
    hyps: dict[int, str] = {}
    lateness: list[float] = []
    max_backlog = 0
    audio_done = 0.0

    def refill():
        for slot in range(args.streams):
            if slot not in active and queue:
                i = queue.pop(0)
                active[slot] = [i, rec.create_stream(), 0]

    def decode(ready):
        if not ready:
            return
        if args.decode == "batch":
            for k in range(0, len(ready), args.max_batch):
                rec.decode_streams(ready[k:k + args.max_batch])
        else:
            for s in ready:
                rec.decode_stream(s)

    refill()
    c0, w0 = cpu_seconds(), time.perf_counter()
    tick = 0
    while active:
        # 1. arrival: one packet per active stream
        for slot, st in list(active.items()):
            i, s, cur = st
            pcm = utts[i].pcm
            if cur < len(pcm):
                chunk = pcm[cur:cur + packet].astype(np.float32) / 32768.0
                s.accept_waveform(16000, chunk)
                st[2] = cur + packet
                if st[2] >= len(pcm):
                    # Right-context flush: the model needs audio beyond the last chunk. The padding's
                    # CPU is charged; its duration is not counted as audio.
                    s.accept_waveform(16000, np.zeros(int(16000 * args.tail_pad_s), dtype=np.float32))
                    s.input_finished()
        # 2. compute: everything that is ready
        while True:
            ready = [st[1] for st in active.values() if rec.is_ready(st[1])]
            max_backlog = max(max_backlog, len(ready))
            if not ready:
                break
            decode(ready)
        # 3. retire finished streams
        for slot, st in list(active.items()):
            i, s, cur = st
            if cur >= len(utts[i].pcm) and not rec.is_ready(s):
                hyps[i] = rec.get_result(s).strip()
                audio_done += utts[i].seconds
                del active[slot]
        refill()
        tick += 1
        if args.paced:
            due = w0 + tick * args.packet_ms / 1000.0
            now = time.perf_counter()
            lateness.append(now - due)
            if now < due:
                time.sleep(due - now)
    cpu = cpu_seconds() - c0
    wall = time.perf_counter() - w0
    refs = [utts[i].text for i in sorted(hyps)]
    hyp = [hyps[i] for i in sorted(hyps)]
    w = corpus_wer(refs, hyp)
    out = dict(schema="ultra.sp-a/v1", split=args.split, utterances=len(hyps), audio_seconds=round(audio_done, 2),
               streams=args.streams, decode=args.decode, max_batch=args.max_batch, precision=args.precision,
               threads=args.threads, paced=args.paced, tail_pad_s=args.tail_pad_s, packet_ms=args.packet_ms,
               cpu_seconds=round(cpu, 3), wall_seconds=round(wall, 3),
               rtf_cpu=round(cpu / audio_done, 5), streams_per_core_at_full_load=round(audio_done / cpu, 2),
               wer=w, max_ready_backlog=max_backlog, affinity=sorted(os.sched_getaffinity(0)),
               host=os.uname().nodename, commit=_commit())
    if lateness:
        lat = np.array(lateness)
        out["lateness_ms"] = {q: round(float(np.percentile(lat, p)) * 1000, 2)
                              for q, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
        out["lateness_ms"]["final"] = round(float(lat[-1]) * 1000, 2)
    return out


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "zipformer-en-2023-06-26"))
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--decode", choices=["single", "batch"], default="batch")
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--packet-ms", type=int, default=20)
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--tail-pad-s", type=float, default=0.8)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = run(args)
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
