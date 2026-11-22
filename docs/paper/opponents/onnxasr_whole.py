"""onnx-asr whole-utterance runner for the cold DEV panel (Parakeet TDT 0.6B v3, INT8).

    onnxasr_whole.py MODEL_DIR WAV_DIR PREDICTIONS_JSONL TRACE_JSONL [--threads N | --workers N]

`onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path=MODEL_DIR, quantization="int8")` on the
CPU execution provider with `intra_op_num_threads = N` and one inter-op thread; each clip is
recognised whole with `recognize(waveform, sample_rate=16000)`. This is not a streaming
system: the trace holds one final event per utterance (the time to recognise the clip).
`--workers N` starts N single-thread copies of this script (one shard, one CPU each).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_common as common  # noqa: E402

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"


def main():
    args = common.parse_args(__doc__.splitlines()[0], "directory with the INT8 ONNX export", thread_knob=True)
    if args.workers > 1:
        common.run_worker_processes(args, os.path.abspath(__file__))
        return
    common.thread_environment(args.threads)
    import numpy as np
    import onnx_asr
    import onnxruntime

    paths = common.staged_wavs(args.wav_dir)
    worker, paths = common.shard_of(args, paths, common.wav_samples)
    if worker is not None:
        common.pin_worker(worker)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    model = onnx_asr.load_model(MODEL_NAME, path=args.model, quantization="int8", sess_options=options,
                                providers=["CPUExecutionProvider"])
    hypotheses = {}
    with common.Trace(args.trace) as trace:
        for path in paths:
            pcm = common.read_staged_wav(path)
            start = time.perf_counter()
            waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            text = model.recognize(waveform, sample_rate=common.SAMPLE_RATE)
            elapsed = time.perf_counter() - start
            trace.event(path.stem, len(pcm) // 2, elapsed, elapsed, text, True)
            hypotheses[path.stem] = text
    common.write_predictions(args.predictions, hypotheses)


if __name__ == "__main__":
    main()
