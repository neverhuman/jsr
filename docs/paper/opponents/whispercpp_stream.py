"""Whisper.cpp "streaming" runner for the cold DEV panel: the Picovoice benchmark configuration.

    whispercpp_stream.py MODEL_BIN WAV_DIR PREDICTIONS_JSONL TRACE_JSONL [--threads N | --workers N]

Replicates `WhisperCppStreamingEngine` of Picovoice/speech-to-text-benchmark (commit
272be8f2244c10dfb2994f192bd6390982b17a5d, engine.py): `pywhispercpp` `Model(model,
n_threads=N, single_segment=True, print_realtime=False, print_progress=False,
redirect_whispercpp_logs_to=None)`; the audio is cut into consecutive, non-overlapping 3 s
chunks, each transcribed on its own with `Model.transcribe` (no context is carried over); a
tail shorter than 0.2 s is zero-padded to 0.2 s; `[BLANK_AUDIO]` is removed and the chunk
texts are joined by spaces.

Here the audio arrives in 20 ms packets, as fast as possible. A chunk is decoded as soon as
its last packet is in, the tail at end of stream; one trace event is written per decode.
`--workers N` starts N single-thread copies of this script (one shard, one CPU each).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_common as common  # noqa: E402

CHUNK_MS = 3000  # WhisperCppStreamingEngine.DEFAULT_CHUNK_SIZE_MS
CHUNK_SAMPLES = CHUNK_MS * common.SAMPLE_RATE // 1000
MIN_CHUNK_SAMPLES = int(0.2 * common.SAMPLE_RATE)


def chunk_count(samples):
    """Decodes needed for an utterance of `samples` samples (the cost unit for balanced shards)."""
    return -(-samples // CHUNK_SAMPLES)


def chunk_text(segments):
    return " ".join(segment.text.replace("[BLANK_AUDIO]", "") for segment in segments).strip()


def recognise(model, np, path, trace):
    pcm = common.read_staged_wav(path)
    ident = path.stem
    words = []
    pending = []
    pending_samples = 0
    start = time.perf_counter()

    def decode(fed, final):
        nonlocal pending, pending_samples
        before = time.perf_counter()
        if pending_samples:
            chunk = np.frombuffer(b"".join(pending), dtype="<i2").astype(np.float32) / 32768.0
            if len(chunk) < MIN_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, MIN_CHUNK_SAMPLES - len(chunk)))
            text = chunk_text(model.transcribe(chunk))
            if text:
                words.extend(text.split())
            pending, pending_samples = [], 0
        now = time.perf_counter()
        trace.event(ident, fed, now - start, now - before, " ".join(words), final)

    total = len(pcm) // 2
    for fed, packet in common.packets(pcm):
        pending.append(packet)
        pending_samples += len(packet) // 2
        if pending_samples == CHUNK_SAMPLES and fed < total:
            decode(fed, False)
    decode(total, True)
    return ident, " ".join(words)


def main():
    args = common.parse_args(__doc__.splitlines()[0], "ggml model file (ggml-base.en.bin, ggml-tiny.en.bin)",
                             thread_knob=True)
    if args.workers > 1:
        common.run_worker_processes(args, os.path.abspath(__file__))
        return
    common.thread_environment(args.threads)
    import numpy as np
    from pywhispercpp.model import Model

    paths = common.staged_wavs(args.wav_dir)
    worker, paths = common.shard_of(args, paths, lambda path: chunk_count(common.wav_samples(path)))
    if worker is not None:
        common.pin_worker(worker)
    model = Model(str(args.model), n_threads=args.threads, single_segment=True, print_realtime=False,
                  print_progress=False, redirect_whispercpp_logs_to=None)
    hypotheses = {}
    with common.Trace(args.trace) as trace:
        for path in paths:
            ident, text = recognise(model, np, path, trace)
            hypotheses[ident] = text
    common.write_predictions(args.predictions, hypotheses)


if __name__ == "__main__":
    main()
