"""Vosk streaming runner for the cold DEV panel.

    vosk_stream.py MODEL_DIR WAV_DIR PREDICTIONS_JSONL TRACE_JSONL [--workers N]

One `KaldiRecognizer` per utterance, fed 20 ms packets through `AcceptWaveform` as fast as
possible. After every packet the runner reads the recogniser's result (`Result` at an
endpoint, otherwise `PartialResult`) and writes one trace event; `FinalResult` flushes the
stream. The transcript is the endpoint segments joined by spaces, as in the Picovoice
benchmark's `VoskStreamingEngine` (speech-to-text-benchmark 272be8f, engine.py).

Kaldi decoding has no intra-op thread knob. `--workers N` loads the model once and runs N
recogniser threads side by side over N balanced shards, each pinned to its own CPU of the
affinity mask (the documented multi-stream use: one `Model`, many recognisers).
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_common as common  # noqa: E402


def recognise(model, recogniser_type, path, trace):
    pcm = common.read_staged_wav(path)
    ident = path.stem
    recogniser = recogniser_type(model, common.SAMPLE_RATE)
    segments = []
    start = time.perf_counter()
    for fed, packet in common.packets(pcm):
        before = time.perf_counter()
        if recogniser.AcceptWaveform(packet):
            segment = json.loads(recogniser.Result()).get("text", "").strip()
            if segment:
                segments.append(segment)
            text = " ".join(segments)
        else:
            partial = json.loads(recogniser.PartialResult()).get("partial", "").strip()
            text = " ".join(segments + [partial]) if partial else " ".join(segments)
        now = time.perf_counter()
        trace.event(ident, fed, now - start, now - before, text, False)
    before = time.perf_counter()
    segment = json.loads(recogniser.FinalResult()).get("text", "").strip()
    if segment:
        segments.append(segment)
    now = time.perf_counter()
    text = " ".join(segments)
    trace.event(ident, len(pcm) // 2, now - start, now - before, text, True)
    return ident, text


def run_shard(model, recogniser_type, paths, trace_path, pin=None):
    if pin is not None:
        common.pin_worker(pin)
    hypotheses = {}
    with common.Trace(trace_path) as trace:
        for path in paths:
            ident, text = recognise(model, recogniser_type, path, trace)
            hypotheses[ident] = text
    return hypotheses


def main():
    args = common.parse_args(__doc__.splitlines()[0], "unpacked Vosk model directory", thread_knob=False)
    common.thread_environment(1)
    from vosk import KaldiRecognizer, Model

    paths = common.staged_wavs(args.wav_dir)
    model = Model(model_path=str(args.model))
    if args.workers == 1:
        common.write_predictions(args.predictions, run_shard(model, KaldiRecognizer, paths, args.trace))
        return
    shards = common.balanced_shards([common.wav_samples(path) for path in paths], args.workers)
    results = [None] * args.workers
    errors = []

    def work(index):
        try:
            results[index] = run_shard(model, KaldiRecognizer, [paths[i] for i in shards[index]],
                                       common.worker_path(args.trace, index), pin=index)
        except BaseException as error:  # a worker failure must fail the whole run
            errors.append(error)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    for index, hypotheses in enumerate(results):
        common.write_predictions(common.worker_path(args.predictions, index), hypotheses)
    common.merge_worker_outputs(args, args.workers)


if __name__ == "__main__":
    main()
