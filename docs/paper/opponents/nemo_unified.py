"""Cold service around NeMo (PyTorch, CPU) for the unified offline/streaming RNN-T models.

    python nemo_unified.py MODE MODEL.nemo WAV_DIR PREDICTIONS.jsonl THREADS [--lang en] [--trace T.jsonl] [--batch N]

MODE is `offline` (the model card's `asr_model.transcribe`, one utterance per batch) or
`stream:LEFT_S:CHUNK_S:RIGHT_S` (buffered streaming: the encoder is re-run over
[left | chunk | right] for every chunk). The streaming loop is NeMo's reference loop
(examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py at the pinned
commit) with batch size 1, `att_context_size_as_chunk=true`, greedy label-looping decoding
carried across chunks. One model load per process, the staged u%05d.wav files in name order,
one final hypothesis per file as `{"id", "hyp"}`. `--batch N` (streaming only, no trace) runs
N utterances side by side through the same loop, longest first: an accuracy pass for the
full panel, never a timed row.

The trace has one line per changed partial and one final line with bench-open's v1 keys:
audio_ms is the audio the step had to wait for (the right edge of its chunk plus right
context), wall_ms the measured wall clock since the utterance began in this unpaced run,
paced_ms the time the same text would appear with audio arriving in real time.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time
import wave


def staged(wav_dir):
    wavs = sorted(path for path in wav_dir.iterdir())
    if not wavs or any(not re.fullmatch(r"u\d{5}\.wav", path.name) or not path.is_file() for path in wavs):
        raise ValueError(f"unexpected staged input in {wav_dir}")
    return wavs


def read_pcm(path):
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"expected 16 kHz mono PCM16: {path}")
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return pcm.astype("float32") / 32768.0


def load(model_path, threads):
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu")
    model.freeze()
    model.eval()
    return torch, model


def offline(model, wavs, lang, out):
    extra = {"target_lang": lang} if getattr(model, "concat", False) else {}
    results = model.transcribe([str(path) for path in wavs], batch_size=1, num_workers=0, verbose=False, **extra)
    for path, result in zip(wavs, results, strict=True):
        json.dump({"id": path.stem, "hyp": result.text}, out, ensure_ascii=False)
        out.write("\n")


def streaming(torch, model, wavs, context_secs, lang, out, trace, batch):
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

    decoding = OmegaConf.structured(RNNTDecodingConfig())
    with open_dict(decoding):
        decoding.greedy.preserve_alignments = False
        decoding.fused_batch_size = -1
    model.change_decoding_strategy(decoding)
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    computer = model.decoding.decoding.decoding_computer
    rate = model.cfg.preprocessor.sample_rate
    stride = model.cfg.preprocessor.window_stride
    factor = model.encoder.subsampling_factor
    feature_samples = (int(rate * stride) // factor) * factor
    frame_samples = feature_samples * factor
    per_sec = 1.0 / stride
    frames = ContextSize(*(int(secs * per_sec / factor) for secs in context_secs))
    samples = ContextSize(*(count * frame_samples for count in (frames.left, frames.chunk, frames.right)))
    if model.cfg.encoder.att_context_style == "chunked_limited_with_rc":
        model.encoder.set_default_att_context_size(att_context_size=[frames.left, frames.chunk, frames.right])
    prompted = getattr(model, "concat", False)
    language = model.resolve_prompt_language(lang) if prompted else None
    print(json.dumps({"context_frames": [frames.left, frames.chunk, frames.right],
                      "context_samples": [samples.left, samples.chunk, samples.right],
                      "att_context_style": str(model.cfg.encoder.att_context_style),
                      "prompt_language": lang if prompted else None, "batch": batch}), file=sys.stderr)
    audios = {path: torch.from_numpy(read_pcm(path)) for path in wavs}
    order = sorted(wavs, key=lambda path: -len(audios[path])) if batch > 1 else list(wavs)
    texts = {}
    for first in range(0, len(order), batch):
        group = order[first:first + batch]
        size = len(group)
        total = max(len(audios[path]) for path in group)
        audio = torch.zeros(size, total)
        for row, path in enumerate(group):
            audio[row, :len(audios[path])] = audios[path]
        rest = torch.tensor([len(audios[path]) for path in group])
        prompt = torch.full((size,), language, dtype=torch.long) if prompted else None
        begin = time.perf_counter()
        buffer = StreamingBatchedAudioBuffer(batch_size=size, context_samples=samples, dtype=audio.dtype,
                                             device=audio.device)
        hyps, state, left, right = None, None, 0, min(samples.chunk + samples.right, total)
        paced_ms, last = 0.0, None
        while left < total:
            start = time.perf_counter()
            length = min(right, total) - left
            last_batch = length >= rest
            lengths = torch.where(last_batch, rest, torch.full_like(rest, fill_value=length))
            buffer.add_audio_batch_(audio[:, left:right], audio_lengths=lengths, is_last_chunk=right >= total,
                                    is_last_chunk_batch=last_batch)
            kwargs = {"input_signal": buffer.samples, "input_signal_length": buffer.context_size_batch.total()}
            if prompt is not None:
                kwargs["language_prompt_indices"] = prompt
            encoded, encoded_len = model(**kwargs)
            encoded = encoded.transpose(1, 2)
            context = buffer.context_size.subsample(factor=frame_samples)
            context_batch = buffer.context_size_batch.subsample(factor=frame_samples)
            encoded = encoded[:, context.left:]
            decode_len = torch.where(last_batch, encoded_len - context_batch.left, context_batch.chunk)
            chunk_hyps, state = computer(x=encoded, out_len=decode_len, prev_batched_state=state,
                                         multi_biasing_ids=None)
            if hyps is None:
                hyps = chunk_hyps
            else:
                hyps.merge_(chunk_hyps)
            finish = time.perf_counter()
            audio_ms = min(right, total) * 1000.0 / rate
            compute_ms = (finish - start) * 1000.0
            paced_ms = max(paced_ms, audio_ms) + compute_ms
            final = right >= total
            if trace is not None:
                text = model.tokenizer.ids_to_text(
                    batched_hyps_to_hypotheses(hyps, batch_size=1)[0].y_sequence.tolist())
                if final or text != last:
                    wall_ms = (finish - begin) * 1000.0
                    json.dump({"id": group[0].stem, "audio_ms": round(audio_ms, 3), "wall_ms": round(wall_ms, 3),
                               "paced_ms": round(paced_ms, 3), "compute_ms": round(compute_ms, 3), "text": text,
                               "final": final, "audio_s": round(audio_ms / 1000.0, 4),
                               "chunk_wall_ms": round(compute_ms, 3), "cum_wall_ms": round(wall_ms, 3)},
                              trace, ensure_ascii=False)
                    trace.write("\n")
                    last = text
            rest = rest - lengths
            left = right
            right = min(right + samples.chunk, total)
        for path, hyp in zip(group, batched_hyps_to_hypotheses(hyps, batch_size=size), strict=True):
            texts[path] = model.tokenizer.ids_to_text(hyp.y_sequence.tolist())
    for path in wavs:
        json.dump({"id": path.stem, "hyp": texts[path]}, out, ensure_ascii=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode")
    parser.add_argument("model", type=Path)
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("threads", type=int)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    wavs = staged(args.wav_dir)
    context = None
    if args.mode.startswith("stream:"):
        context = tuple(float(part) for part in args.mode.split(":")[1:])
        if len(context) != 3:
            raise ValueError(f"invalid streaming mode {args.mode}")
    elif args.mode != "offline" or args.trace is not None:
        raise ValueError("MODE is offline or stream:LEFT_S:CHUNK_S:RIGHT_S; a trace needs a streaming mode")
    if args.batch < 1 or (args.batch > 1 and (context is None or args.trace is not None)):
        raise ValueError("--batch is for streaming without a trace")
    torch, model = load(args.model, args.threads)
    from importlib.metadata import version

    print(json.dumps({"runtime": "nemo", "nemo_toolkit": version("nemo_toolkit"), "torch": torch.__version__,
                      "threads": torch.get_num_threads(), "mode": args.mode,
                      "model_class": type(model).__name__}), file=sys.stderr)
    with args.predictions.open("x", encoding="utf-8") as out:
        if context is None:
            offline(model, wavs, args.lang, out)
        elif args.trace is None:
            streaming(torch, model, wavs, context, args.lang, out, None, args.batch)
        else:
            with args.trace.open("x", encoding="utf-8") as trace:
                streaming(torch, model, wavs, context, args.lang, out, trace, 1)


if __name__ == "__main__":
    main()
