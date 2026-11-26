import array, json, pathlib, sys, wave
import numpy as np
import sherpa_onnx

model_dir = pathlib.Path(sys.argv[1])
wav_dir = pathlib.Path(sys.argv[2])
out_jsonl = pathlib.Path(sys.argv[3])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 1

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(model_dir / "encoder.int8.onnx"),
    decoder=str(model_dir / "decoder.int8.onnx"),
    joiner=str(model_dir / "joiner.int8.onnx"),
    tokens=str(model_dir / "tokens.txt"),
    num_threads=threads,
    provider="cpu",
    model_type="nemo_transducer",
    decoding_method="greedy_search",
)

out_jsonl.parent.mkdir(parents=True, exist_ok=True)
wavs = sorted(wav_dir.glob("*.wav"))
with out_jsonl.open("w", encoding="utf-8") as f:
    for wp in wavs:
        with wave.open(str(wp), "rb") as wf:
            sr = wf.getframerate()
            pcm = array.array("h")
            pcm.frombytes(wf.readframes(wf.getnframes()))
        samples = np.asarray(pcm, dtype=np.float32) / 32768.0
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        f.write(json.dumps({"id": wp.stem, "text": text}) + "\n")
