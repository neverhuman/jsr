"""Shared pieces of the open-source opponent runners (standard library only).

Each runner in this directory feeds the staged 730-utterance DEV panel to one recogniser,
writes `predictions.jsonl` (`{id, hyp}`) for `paper_dev_audio.py score`, and writes a trace of
the results it read while feeding (trace v1, see `Trace`). A runner is launched cold under
`ultra-measure`, so model load, the whole pass and the flush are inside the measured tree.
Nothing here imports a recogniser.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

SAMPLE_RATE = 16000
PACKET_MS = 20
PACKET_SAMPLES = SAMPLE_RATE * PACKET_MS // 1000
PACKET_BYTES = 2 * PACKET_SAMPLES
WAV_HEADER_BYTES = 44


def read_staged_wav(path):
    """Return the PCM16 payload of a staged WAV; reject anything else (as moonshine_stream.cc does)."""
    data = Path(path).read_bytes()
    header = data[:WAV_HEADER_BYTES]

    def field(at, count):
        return int.from_bytes(header[at:at + count], "little")

    size = field(40, 4) if len(header) == WAV_HEADER_BYTES else 0
    if (len(header) != WAV_HEADER_BYTES or header[0:4] != b"RIFF" or header[8:16] != b"WAVEfmt "
            or header[36:40] != b"data" or field(16, 4) != 16 or field(20, 2) != 1 or field(22, 2) != 1
            or field(24, 4) != SAMPLE_RATE or field(34, 2) != 16 or size == 0 or size % 2 != 0
            or size + WAV_HEADER_BYTES != len(data)):
        raise ValueError(f"expected a staged 16 kHz mono PCM16 WAV: {path}")
    return data[WAV_HEADER_BYTES:]


def staged_wavs(wav_dir):
    """Lexically sorted staged WAV paths; any other directory entry is an error."""
    paths = sorted(Path(wav_dir).iterdir())
    if not paths or any(not path.is_file() or path.suffix != ".wav" for path in paths):
        raise ValueError(f"expected only staged WAV files in {wav_dir}")
    return paths


def wav_samples(path):
    """Sample count from the file size (the header is validated when the audio is read)."""
    return (os.stat(path).st_size - WAV_HEADER_BYTES) // 2


def packets(pcm):
    """Yield `(samples_fed_so_far, packet_bytes)` for consecutive 20 ms packets; the last may be short."""
    for start in range(0, len(pcm), PACKET_BYTES):
        packet = pcm[start:start + PACKET_BYTES]
        yield (start + len(packet)) // 2, packet


def balanced_shards(costs, workers):
    """Split item indexes into `workers` shards of near-equal total cost (longest first, least loaded).

    Deterministic: ties go to the lower item index and the lower shard index. Each shard keeps
    its items in the original order.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    shards = [[] for _ in range(workers)]
    loads = [0] * workers
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        target = min(range(workers), key=lambda w: (loads[w], w))
        shards[target].append(index)
        loads[target] += costs[index]
    return [sorted(shard) for shard in shards]


def millis(samples):
    value = samples * 1000 / SAMPLE_RATE
    return int(value) if value == int(value) else round(value, 3)


class Trace:
    """Trace v1: one JSON object per result read, in feed order.

    `id`            staged utterance id (file stem)
    `audio_ms`      audio handed to the recogniser so far in this utterance
    `wall_ms`       monotonic ms since this utterance's first packet, read right after the result;
                    the feed is unpaced, so this is cumulative recogniser time, not a live clock
    `text`          the whole hypothesis so far (not a delta), raw recogniser output
    `final`         true exactly once per utterance: the last event, after the end-of-stream flush;
                    its text is the scored hypothesis
    `audio_s`, `chunk_wall_ms`, `cum_wall_ms`
                    the word-lag harness's names: `audio_ms / 1000`, the time spent in the
                    recogniser for the packet that produced the event, and `wall_ms` again
    """

    def __init__(self, path):
        self._out = open(path, "x", encoding="utf-8")

    def event(self, ident, samples, wall_s, chunk_s, text, final):
        wall_ms = round(wall_s * 1000, 3)
        json.dump({
            "id": ident, "audio_ms": millis(samples), "wall_ms": wall_ms, "text": text, "final": final,
            "audio_s": round(samples / SAMPLE_RATE, 6), "chunk_wall_ms": round(chunk_s * 1000, 3),
            "cum_wall_ms": wall_ms,
        }, self._out, ensure_ascii=False)
        self._out.write("\n")

    def close(self):
        self._out.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_predictions(path, hypotheses):
    """Write `{id, hyp}` lines sorted by id; the file must not exist yet."""
    with open(path, "x", encoding="utf-8") as out:
        for ident in sorted(hypotheses):
            json.dump({"id": ident, "hyp": hypotheses[ident]}, out, ensure_ascii=False)
            out.write("\n")


def read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return {row["id"]: row["hyp"] for row in rows}


def thread_environment(threads):
    """Pin every native thread pool to `threads`. Call before importing the recogniser."""
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)


def pin_worker(index):
    """Pin the calling thread to the `index`-th CPU of the mask the process was started with."""
    cpus = sorted(os.sched_getaffinity(os.getpid()))
    if index >= len(cpus):
        raise ValueError(f"worker {index} has no CPU in the affinity mask {cpus}")
    os.sched_setaffinity(threading.get_native_id(), {cpus[index]})
    return cpus[index]


def parse_args(description, model_help, thread_knob):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", type=Path, help=model_help)
    parser.add_argument("wav_dir", type=Path, help="staged panel WAV directory")
    parser.add_argument("predictions", type=Path, help="output predictions.jsonl (must not exist)")
    parser.add_argument("trace", type=Path, help="output trace.jsonl (must not exist)")
    if thread_knob:
        parser.add_argument("--threads", type=int, default=1, help="intra-op threads per recogniser")
    parser.add_argument("--workers", type=int, default=1,
                        help="single-thread recognisers side by side, one pinned CPU each")
    parser.add_argument("--shard", default=None, help=argparse.SUPPRESS)  # K/N, set by the parent
    args = parser.parse_args()
    if args.workers < 1 or getattr(args, "threads", 1) < 1:
        parser.error("--threads and --workers must be at least 1")
    if args.workers > 1 and getattr(args, "threads", 1) != 1:
        parser.error("worker mode runs single-thread recognisers: use --threads 1")
    return args


def shard_of(args, paths, cost):
    """`(worker_index, paths)` for this process: everything, or shard K of N when `--shard K/N` is set."""
    if args.shard is None:
        return None, paths
    index, count = (int(part) for part in args.shard.split("/"))
    shards = balanced_shards([cost(path) for path in paths], count)
    return index, [paths[i] for i in shards[index]]


def worker_path(path, index):
    return path.with_name(f"{path.name}.w{index}")


def merge_worker_outputs(args, workers):
    """Join the per-worker prediction and trace files into the two requested outputs."""
    hypotheses = {}
    with open(args.trace, "x", encoding="utf-8") as trace:
        for index in range(workers):
            part = read_predictions(worker_path(args.predictions, index))
            if hypotheses.keys() & part.keys():
                raise ValueError("an utterance was recognised by two workers")
            hypotheses.update(part)
            trace.write(worker_path(args.trace, index).read_text(encoding="utf-8"))
    write_predictions(args.predictions, hypotheses)
    for index in range(workers):
        worker_path(args.predictions, index).unlink()
        worker_path(args.trace, index).unlink()
    return hypotheses


def run_worker_processes(args, script, extra=()):
    """Start `args.workers` copies of `script`, one shard and one CPU each, then merge their outputs."""
    if args.predictions.exists() or args.trace.exists():
        raise FileExistsError("outputs must not exist yet")
    children = []
    for index in range(args.workers):
        command = [sys.executable, "-I", str(script), str(args.model), str(args.wav_dir),
                   str(worker_path(args.predictions, index)), str(worker_path(args.trace, index)),
                   "--shard", f"{index}/{args.workers}", *extra]
        children.append(subprocess.Popen(command))
    failed = [child.args for child in children if child.wait() != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed")
    return merge_worker_outputs(args, args.workers)
