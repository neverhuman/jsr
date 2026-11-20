"""Process adapter for Nemotron Speech Streaming EN 0.6B through NeMo-Speech.cpp (P04).

Boundary (spec v1.0-B §12.1): this module receives audio only. It never sees reference
transcripts, utterance ids or file names that carry text. Inputs are anonymised to
`u%05d.wav` in the order given; results come back in the same order.

The recogniser runs out of process: the upstream, unmodified `nemo-speech bench asr`
command in cache-aware streaming mode. That command

- loads one recogniser (CPU backend, `--asr.backend.threads N`);
- pushes each file in chunks of (right_context + 1) x 80 ms;
- calls `finish()` at end of input, which makes the cache-aware RNNT runner decode
  zero padding to flush the last tokens (runtime docs, "Word timestamps and progress");
- with `--concurrency C > 1`, runs C streams at once and turns on the runtime's dynamic
  microbatching (one encoder/predictor/joint graph for all compatible streams).

Receipts: process-tree CPU is os.wait4's rusage for the child (user + system of the child and
every descendant it reaped), never wall x threads. Peak RSS is the child's `ru_maxrss`.
GPU use is checked with `nvidia-smi` while the child runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
EXT = os.path.join(DATA, "ext", "NeMo-Speech.cpp")
DEFAULT_BIN = os.path.join(EXT, "build", "cpu-asr", "bin", "nemo-speech")
DEFAULT_MODEL = os.path.join(DATA, "models", "nemotron-0.6b", "nemotron-speech-streaming-en-0.6b.q8_0.gguf")
# Supported cache-aware operating points of this checkpoint: att_context_size [70, R],
# chunk = (R + 1) x 80 ms. R=1 is 160 ms, R=6 is 560 ms.
CHUNK_MS_TO_R = {80: 0, 160: 1, 560: 6, 1120: 13}


@dataclass
class Receipt:
    argv: list[str]
    returncode: int
    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    peak_rss_kib: int
    max_threads_observed: int
    gpu_pids_seen: list[int]
    affinity: list[int]
    bench: dict = field(default_factory=dict)

    @property
    def cpu_seconds(self) -> float:
        return self.cpu_user_seconds + self.cpu_system_seconds


def write_wavs(pcms: list[np.ndarray], wav_dir: str) -> list[str]:
    """Write int16 16 kHz mono PCM as anonymised WAVs; reuse files that already match."""
    os.makedirs(wav_dir, exist_ok=True)
    paths = []
    for k, pcm in enumerate(pcms):
        if pcm.dtype != np.int16 or pcm.ndim != 1:
            raise ValueError("expected 1-D int16 PCM at 16 kHz")
        p = os.path.join(wav_dir, f"u{k:05d}.wav")
        if not (os.path.exists(p) and os.path.getsize(p) == 44 + 2 * len(pcm)):
            with wave.open(p + ".tmp", "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            os.replace(p + ".tmp", p)
        paths.append(p)
    return paths


def _gpu_pids() -> set[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {int(x) for x in out.split() if x.strip().isdigit()}


def _descendants(pid: int) -> set[int]:
    out, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f"/proc/{p}/task/{p}/children") as fh:
                kids = [int(x) for x in fh.read().split()]
        except OSError:
            kids = []
        for k in kids:
            if k not in out:
                out.add(k), todo.append(k)
    return out


def run_measured(argv: list[str], log_path: str, poll_s: float = 5.0):
    """Run argv; return (returncode, wall_s, stdout, rusage, gpu_pids_seen, max_threads).

    rusage is os.wait4's for this child alone: user + system of the child and every descendant
    it reaped, and the child's peak RSS.
    """
    gpu_seen: set[int] = set()
    max_threads = 0
    with open(log_path, "w") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True)
        stdout_chunks: list[str] = []
        reader = threading.Thread(target=lambda: stdout_chunks.append(proc.stdout.read()), daemon=True)
        reader.start()
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid == proc.pid:
                break
            tree = _descendants(proc.pid)
            gpu_seen |= tree & _gpu_pids()
            for p in tree:
                try:
                    max_threads = max(max_threads, len(os.listdir(f"/proc/{p}/task")))
                except OSError:
                    pass
            time.sleep(poll_s)
        wall = time.perf_counter() - t0
        proc.returncode = os.waitstatus_to_exitcode(status)
        reader.join()
    return proc.returncode, wall, "".join(stdout_chunks), ru, sorted(gpu_seen), max_threads


class NemotronAdapter:
    def __init__(self, chunk_ms: int = 160, threads: int = 1, cpus: list[int] | None = None,
                 binary: str = DEFAULT_BIN, model: str = DEFAULT_MODEL):
        if chunk_ms not in CHUNK_MS_TO_R:
            raise ValueError(f"unsupported chunk_ms {chunk_ms}; supported: {sorted(CHUNK_MS_TO_R)}")
        self.chunk_ms, self.right_context = chunk_ms, CHUNK_MS_TO_R[chunk_ms]
        self.threads, self.cpus, self.binary, self.model = threads, cpus, binary, model

    def argv(self, wav_dir: str, out_dir: str, concurrency: int, trace: str | None = None) -> list[str]:
        a = [self.binary, "--json", "bench", "asr", wav_dir, "--model", self.model, "--device", "cpu",
             "--mode", "stream", "--asr.backend.threads", str(self.threads),
             "--asr.streaming.rnnt_right_context", str(self.right_context),
             "--concurrency", str(concurrency), "--repetitions", "1", "--warmup", "0",
             "--save", out_dir]
        if trace:
            a += ["--trace", trace]
        if self.cpus:
            a = ["taskset", "-c", ",".join(map(str, self.cpus))] + a
        return a

    def transcribe(self, pcms: list[np.ndarray], work_dir: str, concurrency: int = 1,
                   trace: bool = False) -> tuple[list[str], Receipt]:
        """Stream every input through the recogniser; return hypotheses (input order) and a receipt."""
        wav_dir = os.path.join(work_dir, "wav")
        write_wavs(pcms, wav_dir)
        out_dir = os.path.join(work_dir, f"hyp_{self.chunk_ms}ms_c{concurrency}")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        argv = self.argv(wav_dir, out_dir, concurrency,
                         os.path.join(out_dir, "trace.jsonl") if trace else None)
        rc, wall, stdout, ru, gpu_seen, max_threads = run_measured(argv, os.path.join(out_dir, "stderr.log"))
        try:
            bench = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError:
            bench = {"unparsed_stdout": stdout[-2000:]}
        rec = Receipt(argv=argv, returncode=rc, wall_seconds=round(wall, 3),
                      cpu_user_seconds=round(ru.ru_utime, 3),
                      cpu_system_seconds=round(ru.ru_stime, 3),
                      peak_rss_kib=int(ru.ru_maxrss), max_threads_observed=max_threads,
                      gpu_pids_seen=gpu_seen, affinity=list(self.cpus or []), bench=bench)
        if rc != 0:
            raise RuntimeError(f"nemo-speech failed rc={rc}; see {out_dir}/stderr.log")
        hyps = []
        for k in range(len(pcms)):
            p = os.path.join(out_dir, f"u{k:05d}.txt")
            hyps.append(open(p).read().strip() if os.path.exists(p) else "")
        return hyps, rec
