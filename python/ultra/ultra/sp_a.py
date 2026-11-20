"""SP-A: real-time stream density of a streaming transducer on CPU, at measured WER.

Clock model (fixed capture deadlines, never rebased):
  packet k of the shared stream timeline is captured at  deadline(k) = t0 + (k+1) * packet.
  Paced runs accept every packet whose deadline has passed (catching up in bulk when behind);
  unpaced runs accept one packet per iteration as fast as compute allows.
Decode: with --grid-ms G, decoding runs only when the accepted timeline crosses a grid boundary
  (or during the drain), batching every ready stream into one call (decode=batch) or one by one.
Latency: at each decode, `lateness` = completion time - deadline of the OLDEST packet not yet
  decoded at decode start, i.e. the age of the oldest unprocessed audio when it is finally handled.
  Backlog that has accumulated is therefore never hidden.
Cutoff: with --duration-s, arrival stops at the cutoff; every accepted stream is flushed (tail pad +
  input_finished) and drained. Steady-state metrics exclude the drain; drain cost is reported
  separately. CPU per audio-second counts all accepted audio; WER counts fully-arrived utterances.
CPU is getrusage(RUSAGE_SELF) user+sys (inference threads are in-process).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

SR = 16000
DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
MODELS = os.path.join(DATA, "models")


@dataclass
class Clock:
    """Wall clock and CPU clock; tests inject a fake one."""
    now: callable = time.perf_counter
    sleep: callable = time.sleep
    cpu: callable = field(default=lambda: sum(resource.getrusage(resource.RUSAGE_SELF)[:2]))


@dataclass
class Item:
    uid: str
    speaker: str
    text: str
    pcm: np.ndarray  # int16

    @property
    def seconds(self) -> float:
        return len(self.pcm) / SR


class Slot:
    """One concurrent stream position: the utterance it carries and how much has been accepted."""

    def __init__(self, idx: int, stream):
        self.idx, self.stream, self.cursor, self.finished_input = idx, stream, 0, False


def simulate(items: list[Item], rec, opt, clock: Clock) -> dict:
    """Run the arrival/decode loop. `rec` follows the sherpa-onnx OnlineRecognizer interface."""
    packet = int(SR * opt.packet_ms / 1000)
    pkt_s = opt.packet_ms / 1000.0
    grid_every = max(1, round(opt.grid_ms / opt.packet_ms)) if opt.grid_ms else 1
    tail = np.zeros(int(SR * opt.tail_pad_s), dtype=np.float32)
    queue = list(range(len(items)))
    active: dict[int, Slot] = {}
    hyps: dict[int, str] = {}
    steady_lat: list[float] = []
    series: list[dict] = []
    accepted = 0        # samples, all streams
    k_acc = 0           # packets accepted on the shared timeline
    k_dec = 0           # packets of the timeline already handed to a decode
    max_ready = 0
    cutoff = False
    drain_cpu0 = drain_t0 = None

    def refill():
        if not cutoff:
            for slot in range(opt.streams):
                if slot not in active and queue:
                    active[slot] = Slot(queue.pop(0), rec.create_stream())

    def finish(sl: Slot):
        sl.stream.accept_waveform(SR, tail)  # right-context flush; CPU charged, not counted as audio
        sl.stream.input_finished()
        sl.finished_input = True

    def decode_ready() -> int:
        nonlocal max_ready
        n = 0
        while True:
            ready = [sl.stream for sl in active.values() if rec.is_ready(sl.stream)]
            if not ready:
                return n
            n += len(ready)
            max_ready = max(max_ready, len(ready))
            if opt.decode == "batch":
                for k in range(0, len(ready), opt.max_batch):
                    rec.decode_streams(ready[k:k + opt.max_batch])
            else:
                for s in ready:
                    rec.decode_stream(s)

    def retire():
        for slot, sl in list(active.items()):
            if sl.finished_input and not rec.is_ready(sl.stream):
                if sl.cursor >= len(items[sl.idx].pcm):  # only fully-arrived utterances are scored
                    hyps[sl.idx] = rec.get_result(sl.stream).strip()
                del active[slot]

    def accept_one_packet():
        nonlocal accepted, k_acc
        for sl in active.values():
            pcm = items[sl.idx].pcm
            if sl.cursor < len(pcm):
                chunk = pcm[sl.cursor:sl.cursor + packet]
                sl.stream.accept_waveform(SR, chunk.astype(np.float32) / 32768.0)
                sl.cursor += len(chunk)
                accepted += len(chunk)
                if sl.cursor >= len(pcm):
                    finish(sl)
        k_acc += 1

    refill()
    c0, t0 = clock.cpu(), clock.now()
    deadline = lambda k: t0 + (k + 1) * pkt_s  # noqa: E731  capture time of packet k
    last_series = t0
    while active:
        now = clock.now()
        if not cutoff and opt.duration_s and now - t0 >= opt.duration_s:
            cutoff = True
            drain_cpu0, drain_t0 = clock.cpu(), now
            for sl in active.values():
                if not sl.finished_input:
                    finish(sl)
        if not cutoff:
            if opt.paced:
                if now < deadline(k_acc):
                    clock.sleep(deadline(k_acc) - now)
                    now = clock.now()
                while deadline(k_acc) <= now and active:  # accept everything already captured
                    accept_one_packet()
                    refill()
            else:
                accept_one_packet()
        crossed = k_acc // grid_every > k_dec // grid_every
        if cutoff or crossed:
            oldest = deadline(k_dec)  # capture deadline of the oldest packet not yet decoded
            if decode_ready() and not cutoff and opt.paced:
                steady_lat.append(clock.now() - oldest)
            k_dec = k_acc
            retire()
            refill()
        now = clock.now()
        if now - last_series >= 1.0:
            series.append(dict(t=round(now - t0, 3), accepted_s=round(accepted / SR, 3),
                               captured_packets=int((now - t0) / pkt_s), accepted_packets=k_acc,
                               decoded_packets=k_dec, completed=len(hyps), active=len(active),
                               oldest_unprocessed_s=round(max(0.0, now - deadline(k_dec)), 4)))
            last_series = now
    cpu, wall = clock.cpu() - c0, clock.now() - t0
    order = sorted(hyps)
    accepted_s = accepted / SR
    out = dict(utterances_scored=len(order), audio_accepted_s=round(accepted_s, 3),
               audio_completed_s=round(sum(items[i].seconds for i in order), 3),
               cpu_seconds=round(cpu, 3), wall_seconds=round(wall, 3),
               rtf_cpu=round(cpu / max(accepted_s, 1e-9), 5),
               streams_per_core_at_full_load=round(accepted_s / max(cpu, 1e-9), 2),
               max_ready_batch=max_ready, drained=not active, accepted_packets=k_acc,
               hyps={items[i].uid: hyps[i] for i in order}, series=series)
    if drain_t0 is not None:
        out["drain"] = dict(cpu_seconds=round(clock.cpu() - drain_cpu0, 3),
                            wall_seconds=round(clock.now() - drain_t0, 3))
    if steady_lat:
        lat = np.array(steady_lat)
        out["lateness_ms"] = {q: round(float(np.percentile(lat, p)) * 1000, 2)
                              for q, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
        out["lateness_ms"]["last"] = round(float(lat[-1]) * 1000, 2)
        n = len(lat)
        out["lateness_ms"]["first_third_mean"] = round(float(lat[: max(1, n // 3)].mean()) * 1000, 2)
        out["lateness_ms"]["last_third_mean"] = round(float(lat[-max(1, n // 3):].mean()) * 1000, 2)
    return out


def make_recognizer(model_dir: str, precision: str, threads: int):
    import sherpa_onnx
    tag = "epoch-99-avg-1-chunk-16-left-128" + (".int8" if precision == "int8" else "")
    p = lambda kind: os.path.join(model_dir, f"{kind}-{tag}.onnx")  # noqa: E731
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(model_dir, "tokens.txt"), encoder=p("encoder"), decoder=p("decoder"),
        joiner=p("joiner"), num_threads=threads, sample_rate=SR, feature_dim=80,
        decoding_method="greedy_search", provider="cpu")


def run(args) -> dict:
    from ultra.libri import load_split
    from ultra.wer import corpus_wer
    utts = load_split(args.split, args.limit)
    if args.offset:
        k = args.offset % len(utts)
        utts = utts[k:] + utts[:k]  # workers start at different utterances
    items = [Item(u.uid, str(u.speaker), u.text, u.pcm) for u in utts]
    rec = make_recognizer(args.model_dir, args.precision, args.threads)
    load_before = os.getloadavg()
    sim = simulate(items, rec, args, Clock())
    hyps, series = sim.pop("hyps"), sim.pop("series")
    by_id = {it.uid: it for it in items}
    ids = list(hyps)
    w = corpus_wer([by_id[i].text for i in ids], [hyps[i] for i in ids])
    out = dict(schema="ultra.sp-a/v3", run_id=args.run_id, worker_id=args.worker_id, split=args.split,
               streams=args.streams, decode=args.decode, max_batch=args.max_batch, precision=args.precision,
               threads=args.threads, paced=args.paced, packet_ms=args.packet_ms, grid_ms=args.grid_ms,
               tail_pad_s=args.tail_pad_s, offset=args.offset, duration_s=args.duration_s, **sim,
               wer=w, load_before=load_before, load_after=os.getloadavg(),
               affinity=sorted(os.sched_getaffinity(0)), host=os.uname().nodename, commit=_commit())
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.join(args.out_dir, f"worker_{args.worker_id}")
        with open(stem + ".predictions.jsonl", "w") as fh:
            for i in ids:
                it = by_id[i]
                fh.write(json.dumps(dict(id=i, speaker=it.speaker, dataset=args.split, ref=it.text,
                                         hyp=hyps[i])) + "\n")
        with open(stem + ".series.jsonl", "w") as fh:
            fh.writelines(json.dumps(s) + "\n" for s in series)
        tmp = stem + ".json.tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, allow_nan=False)
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
    ap.add_argument("--grid-ms", type=int, default=320, help="decode when the timeline crosses a G ms boundary (0 = every packet)")
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
    if not all(math.isfinite(v) for v in (res["cpu_seconds"], res["rtf_cpu"])):
        raise ValueError("non-finite result")
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
