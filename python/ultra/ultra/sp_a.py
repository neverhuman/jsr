"""SP-A: real-time stream density of a streaming transducer on CPU, at measured WER.

Arrival: packet k of every active stream is accepted at t0 + (k+1)*packet (after its last sample
exists). Unpaced runs accept packets as fast as compute allows.
Decode: with --grid-ms G, decoding runs only on grid ticks (every G ms of audio time) and batches
every ready stream into one call (decode=batch) or decodes them one by one (decode=single).
Cutoff: with --duration-s, arrival stops at the cutoff, every accepted stream is flushed (tail pad +
input_finished) and drained. CPU per audio-second counts ALL accepted audio; WER counts only
utterances whose audio fully arrived.
Lateness: at each grid tick that decoded something, completion time minus the arrival time of the
newest accepted sample. CPU is getrusage(RUSAGE_SELF) user+sys (inference threads are in-process).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import uuid

import numpy as np
import sherpa_onnx

from ultra.libri import load_split
from ultra.wer import corpus_wer

MODELS = os.path.join(os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr"), "models")
SR = 16000


def make_recognizer(model_dir: str, precision: str, threads: int):
    tag = "epoch-99-avg-1-chunk-16-left-128" + (".int8" if precision == "int8" else "")
    p = lambda kind: os.path.join(model_dir, f"{kind}-{tag}.onnx")  # noqa: E731
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(model_dir, "tokens.txt"), encoder=p("encoder"), decoder=p("decoder"),
        joiner=p("joiner"), num_threads=threads, sample_rate=SR, feature_dim=80,
        decoding_method="greedy_search", provider="cpu")


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Slot:
    """One concurrent stream position: the utterance it carries and how much has been accepted."""

    def __init__(self, idx: int, stream):
        self.idx, self.stream, self.cursor, self.finished_input = idx, stream, 0, False


def run(args) -> dict:
    utts = load_split(args.split, args.limit)
    if args.offset:
        k = args.offset % len(utts)
        utts = utts[k:] + utts[:k]  # workers start at different utterances
    rec = make_recognizer(args.model_dir, args.precision, args.threads)
    packet = int(SR * args.packet_ms / 1000)
    grid_every = max(1, round(args.grid_ms / args.packet_ms)) if args.grid_ms else 1
    tail = np.zeros(int(SR * args.tail_pad_s), dtype=np.float32)
    queue = list(range(len(utts)))
    active: dict[int, Slot] = {}
    hyps: dict[int, str] = {}
    lateness: list[float] = []
    series: list[dict] = []
    accepted = 0  # samples
    max_ready = 0
    cutoff = False

    def refill():
        if cutoff:
            return
        for slot in range(args.streams):
            if slot not in active and queue:
                active[slot] = Slot(queue.pop(0), rec.create_stream())

    def finish(sl: Slot):
        # Right-context flush; padding CPU is charged, its duration is not counted as audio.
        sl.stream.accept_waveform(SR, tail)
        sl.stream.input_finished()
        sl.finished_input = True

    def decode_ready() -> int:
        n = 0
        while True:
            ready = [sl.stream for sl in active.values() if rec.is_ready(sl.stream)]
            if not ready:
                return n
            n += len(ready)
            nonlocal max_ready
            max_ready = max(max_ready, len(ready))
            if args.decode == "batch":
                for k in range(0, len(ready), args.max_batch):
                    rec.decode_streams(ready[k:k + args.max_batch])
            else:
                for s in ready:
                    rec.decode_stream(s)

    def retire():
        for slot, sl in list(active.items()):
            if sl.finished_input and not rec.is_ready(sl.stream):
                if sl.cursor >= len(utts[sl.idx].pcm):  # only fully-arrived utterances are scored
                    hyps[sl.idx] = rec.get_result(sl.stream).strip()
                del active[slot]

    refill()
    load_before = os.getloadavg()
    c0, w0 = cpu_seconds(), time.perf_counter()
    tick = 0
    last_series = w0
    while active:
        due = w0 + (tick + 1) * args.packet_ms / 1000.0  # end of packet `tick`
        if args.paced and not cutoff:
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)
        arrival_t = max(due, time.perf_counter()) if args.paced else time.perf_counter()
        if args.duration_s and not cutoff and arrival_t - w0 >= args.duration_s:
            cutoff = True
            for sl in active.values():
                if not sl.finished_input:
                    finish(sl)
        if not cutoff:
            for sl in active.values():
                pcm = utts[sl.idx].pcm
                if sl.cursor < len(pcm):
                    chunk = pcm[sl.cursor:sl.cursor + packet]
                    sl.stream.accept_waveform(SR, chunk.astype(np.float32) / 32768.0)
                    sl.cursor += len(chunk)
                    accepted += len(chunk)
                    if sl.cursor >= len(pcm):
                        finish(sl)
        tick += 1
        if cutoff or tick % grid_every == 0:
            if decode_ready() and not cutoff:  # drain-phase decodes are not real-time ticks
                lateness.append(time.perf_counter() - arrival_t)
            retire()
            refill()
        now = time.perf_counter()
        if now - last_series >= 1.0:
            series.append(dict(t=round(now - w0, 3), accepted_s=round(accepted / SR, 3),
                               completed=len(hyps), active=len(active),
                               lateness_s=round(lateness[-1], 4) if lateness else None))
            last_series = now
    cpu = cpu_seconds() - c0
    wall = time.perf_counter() - w0
    order = sorted(hyps)
    refs = [utts[i].text for i in order]
    w = corpus_wer(refs, [hyps[i] for i in order])
    accepted_s = accepted / SR
    completed_s = sum(utts[i].seconds for i in order)
    out = dict(schema="ultra.sp-a/v2", run_id=args.run_id, worker_id=args.worker_id, split=args.split,
               streams=args.streams, decode=args.decode, max_batch=args.max_batch, precision=args.precision,
               threads=args.threads, paced=args.paced, packet_ms=args.packet_ms, grid_ms=args.grid_ms,
               tail_pad_s=args.tail_pad_s, offset=args.offset, duration_s=args.duration_s,
               utterances_scored=len(order), audio_accepted_s=round(accepted_s, 3),
               audio_completed_s=round(completed_s, 3), cpu_seconds=round(cpu, 3), wall_seconds=round(wall, 3),
               rtf_cpu=round(cpu / max(accepted_s, 1e-9), 5),
               streams_per_core_at_full_load=round(accepted_s / max(cpu, 1e-9), 2),
               wer=w, max_ready_batch=max_ready, load_before=load_before, load_after=os.getloadavg(),
               affinity=sorted(os.sched_getaffinity(0)), host=os.uname().nodename, commit=_commit(),
               sherpa_onnx=getattr(sherpa_onnx, "__version__", "unknown"))
    if lateness:
        lat = np.array(lateness)
        out["lateness_ms"] = {q: round(float(np.percentile(lat, p)) * 1000, 2)
                              for q, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
        out["lateness_ms"]["last"] = round(float(lat[-1]) * 1000, 2)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.join(args.out_dir, f"worker_{args.worker_id}")
        with open(stem + ".predictions.jsonl", "w") as fh:
            for i in order:
                fh.write(json.dumps(dict(id=utts[i].uid, speaker=str(utts[i].speaker), dataset=args.split,
                                         ref=utts[i].text, hyp=hyps[i])) + "\n")
        with open(stem + ".series.jsonl", "w") as fh:
            fh.writelines(json.dumps(s) + "\n" for s in series)
        tmp = stem + ".json.tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, stem + ".json")  # atomic: a present result file is a complete one
    return out


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "zipformer-en-2023-06-26"))
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--decode", choices=["single", "batch"], default="batch")
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--packet-ms", type=int, default=20)
    ap.add_argument("--grid-ms", type=int, default=320, help="decode only every G ms of audio time (0 = every packet)")
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--tail-pad-s", type=float, default=0.8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--run-id", default=uuid.uuid4().hex[:12])
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--out-dir", help="write worker_<id>.{json,predictions.jsonl,series.jsonl} here")
    ap.add_argument("--out", help="append the summary line to this JSONL (single runs)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    res = run(args)
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
