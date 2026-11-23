"""Cold streaming service around sherpa-onnx for the streaming Zipformer int8 bundle.

    python sherpa_zipformer.py BUNDLE WAV_DIR PREDICTIONS.jsonl [--trace TRACE.jsonl]

The Linux rows use `ultra-cold-wav zipformer-int8`, which drives the same sherpa-onnx C
library from Rust. That binary cannot be built on macOS, so this adapter repeats its
configuration through sherpa-onnx's Python binding: one recogniser per process, one thread,
CPU provider, greedy search, no endpointing, 16 kHz / 80-dim features, 20 ms packets,
every ready frame decoded after each packet, 0.8 s of zero padding before the final flush.
On Linux it also yields the partials trace (bench-open's v1 keys plus `paced_ms`), which
the Rust binary does not write.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time
import wave

PACKET = 320
TAIL = int(16000 * 0.8)
STEM = "epoch-99-avg-1-chunk-16-left-128.int8.onnx"


def staged(wav_dir):
    wavs = sorted(path for path in wav_dir.iterdir())
    if not wavs or any(not re.fullmatch(r"u\d{5}\.wav", path.name) or not path.is_file() for path in wavs):
        raise ValueError(f"unexpected staged input in {wav_dir}")
    return wavs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    wavs = staged(args.wav_dir)

    import numpy as np
    import sherpa_onnx

    model = args.bundle / "model"
    recogniser = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model / "tokens.txt"), encoder=str(model / f"encoder-{STEM}"),
        decoder=str(model / f"decoder-{STEM}"), joiner=str(model / f"joiner-{STEM}"),
        num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
        provider="cpu", enable_endpoint_detection=False)
    print(json.dumps({"runtime": "sherpa-onnx", "version": sherpa_onnx.__version__, "threads": 1,
                      "input_packet_ms": 20, "eof_tail_ms": 800}), file=sys.stderr)
    tail = np.zeros(TAIL, dtype=np.float32)
    trace = args.trace.open("x", encoding="utf-8") if args.trace else None
    with args.predictions.open("x", encoding="utf-8") as out:
        for path in wavs:
            with wave.open(str(path), "rb") as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                    raise ValueError(f"expected 16 kHz mono PCM16: {path}")
                audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            begin = time.perf_counter()
            stream = recogniser.create_stream()
            paced_ms, last = 0.0, ""

            def step(samples, audio_ms, final, stream=stream, path=path, begin=begin):
                nonlocal paced_ms, last
                start = time.perf_counter()
                stream.accept_waveform(16000, samples)
                if final:
                    stream.input_finished()
                while recogniser.is_ready(stream):
                    recogniser.decode_stream(stream)
                text = recogniser.get_result(stream) if final or trace is not None else last
                finish = time.perf_counter()
                compute_ms = (finish - start) * 1000.0
                paced_ms = max(paced_ms, audio_ms) + compute_ms
                if trace is not None and (final or text != last):
                    wall_ms = (finish - begin) * 1000.0
                    json.dump({"id": path.stem, "audio_ms": round(audio_ms, 3), "wall_ms": round(wall_ms, 3),
                               "paced_ms": round(paced_ms, 3), "compute_ms": round(compute_ms, 3), "text": text,
                               "final": final, "audio_s": round(audio_ms / 1000.0, 4),
                               "chunk_wall_ms": round(compute_ms, 3), "cum_wall_ms": round(wall_ms, 3)},
                              trace, ensure_ascii=False)
                    trace.write("\n")
                last = text
                return text

            for offset in range(0, len(audio), PACKET):
                packet = audio[offset:offset + PACKET]
                step(packet, (offset + len(packet)) / 16.0, False)
            text = step(tail, len(audio) / 16.0, True)
            json.dump({"id": path.stem, "hyp": text}, out, ensure_ascii=False)
            out.write("\n")
    if trace is not None:
        trace.close()


if __name__ == "__main__":
    main()
