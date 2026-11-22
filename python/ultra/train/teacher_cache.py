"""H22 teacher cache: run R0 (icefall streaming Zipformer2, sherpa-onnx export, chunk 16 / left 128)
over LibriSpeech TRAIN shards and store compact distillation targets. TRAIN only: the script
refuses any split that is not train-*.

For every utterance and speed factor (0.9 / 1.0 / 1.1, perturbed exactly as train.data does), R0
runs streaming greedy search (one symbol per frame, as its ONNX recipe), and for every encoder
frame the cache keeps:
  * the top-k (default 8) joiner log-probabilities and their R0 token ids at the greedy context;
  * the token emitted on that frame (0 = blank).
Per utterance it also keeps the 1-best text. One .npz per (shard, speed), plus a SHA256SUMS line.

  python -m train.teacher_cache --model-dir <R0 dir> --split train-clean-100 --shard 0 \
      --speed 1.0 --out <dir> [--topk 8] [--threads 1]

R0 token ids are its own (blank 0, BPE-500 piece ids). A student sharing R0's bpe.model through
train.tokenizer has output index = piece id + 1 for pieces and 0 for blank. The mapping is
recorded in the H22 README, and the cache stays in R0's own ids.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

TAIL_PAD_S = 0.3


class R0:
    def __init__(self, model_dir: str, threads: int = 1, precision: str = "int8"):
        import onnxruntime as ort

        tag = "epoch-99-avg-1-chunk-16-left-128" + (".int8" if precision == "int8" else "")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        mk = lambda n: ort.InferenceSession(os.path.join(model_dir, f"{n}-{tag}.onnx"), so,  # noqa: E731
                                            providers=["CPUExecutionProvider"])
        self.enc, self.dec, self.join = mk("encoder"), mk("decoder"), mk("joiner")
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.T = int(meta["T"])
        self.chunk = int(meta["decode_chunk_len"])
        self.inputs = [i for i in self.enc.get_inputs() if i.name != "x"]
        self.outputs = [o.name for o in self.enc.get_outputs()]
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "bpe.model"))
        self.context = int(self.dec.get_modelmeta().custom_metadata_map["context_size"])

    def _init_states(self):
        st = []
        for i in self.inputs:
            shape = [1 if not isinstance(d, int) else d for d in i.shape]
            dt = np.int64 if "int64" in i.type else np.float32
            st.append(np.zeros(shape, dtype=dt))
        return st

    @staticmethod
    def fbank(pcm16: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        fb = knf.OnlineFbank(opts)
        x = pcm16.astype(np.float32) / 32768.0
        x = np.concatenate([x, np.zeros(int(TAIL_PAD_S * 16000), dtype=np.float32)])
        fb.accept_waveform(16000, x.tolist())
        fb.input_finished()
        return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)

    def encode(self, feats: np.ndarray) -> np.ndarray:
        states, outs, off = self._init_states(), [], 0
        while off + self.T <= feats.shape[0]:
            feed = {"x": feats[None, off:off + self.T]}
            feed.update({i.name: s for i, s in zip(self.inputs, states)})
            res = self.enc.run(None, feed)
            outs.append(res[0][0])
            states = res[1:]
            off += self.chunk
        return np.concatenate(outs) if outs else np.zeros((0, 512), np.float32)

    def greedy_topk(self, enc: np.ndarray, k: int):
        ctx = [0] * self.context
        dec = self.dec.run(None, {"y": np.array([ctx], dtype=np.int64)})[0]
        ids = np.zeros((enc.shape[0], k), np.int16)
        lps = np.zeros((enc.shape[0], k), np.float16)
        emitted = np.zeros(enc.shape[0], np.int16)
        hyp = []
        for t in range(enc.shape[0]):
            logit = self.join.run(None, {"encoder_out": enc[t:t + 1], "decoder_out": dec})[0][0]
            m = logit.max()
            lp = logit - (m + np.log(np.exp(logit - m).sum()))
            top = np.argpartition(-lp, k)[:k]
            top = top[np.argsort(-lp[top])]
            ids[t], lps[t] = top, lp[top]
            y = int(top[0])
            emitted[t] = y
            if y != 0:
                hyp.append(y)
                ctx = ctx[1:] + [y]
                dec = self.dec.run(None, {"y": np.array([ctx], dtype=np.int64)})[0]
        return ids, lps, emitted, self.sp.decode(hyp)


def perturb(pcm: np.ndarray, f: float) -> np.ndarray:
    """Identical to train.data.ParquetBatches._perturb for a given factor."""
    if f == 1.0:
        return pcm
    from fractions import Fraction

    from scipy.signal import resample_poly

    r = Fraction(f).limit_denominator(100)
    y = resample_poly(pcm.astype(np.float32), r.denominator, r.numerator)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def shard_files(split: str) -> list[str]:
    from ultra.libri import PARQUET, SPLIT_DIRS

    if not split.startswith("train-"):
        raise ValueError(f"split {split!r} refused: the teacher only sees TRAIN splits")
    return sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke tests)")
    a = ap.parse_args(argv)
    if a.speed not in (0.9, 1.0, 1.1):
        ap.error("--speed must be 0.9, 1.0 or 1.1 (the training perturbation factors)")
    if not 1 <= a.topk <= 64:
        ap.error("--topk must be in 1..64")
    files = shard_files(a.split)
    if not 0 <= a.shard < len(files):
        ap.error(f"--shard out of range (0..{len(files) - 1})")
    import pyarrow.parquet as pq
    import soundfile as sf

    r0 = R0(a.model_dir, a.threads)
    tab = pq.read_table(files[a.shard], columns=["id", "text", "audio"])
    rows = tab.to_pylist()[: a.limit or None]
    uids, texts, hyps, nfr = [], [], [], []
    all_ids, all_lps, all_emit = [], [], []
    t0, audio_s = time.time(), 0.0
    for row in rows:
        pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if sr != 16000 or pcm.ndim != 1:
            raise ValueError(f"{row['id']}: expected 16 kHz mono")
        pcm = perturb(pcm, a.speed)
        audio_s += len(pcm) / 16000.0
        ids, lps, emitted, hyp = r0.greedy_topk(r0.encode(r0.fbank(pcm)), a.topk)
        uids.append(row["id"]); texts.append(row["text"]); hyps.append(hyp); nfr.append(len(emitted))
        all_ids.append(ids); all_lps.append(lps); all_emit.append(emitted)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.split}_{a.shard:04d}_sp{a.speed:.1f}.npz"
    tmp = os.path.join(a.out, name + ".tmp.npz")
    np.savez(tmp, uid=np.array(uids), text=np.array(texts), hyp=np.array(hyps),
             n_frames=np.array(nfr, np.int32), topk_ids=np.concatenate(all_ids),
             topk_lp=np.concatenate(all_lps), emitted=np.concatenate(all_emit),
             meta=np.array(json.dumps({"teacher": "R0 zipformer-en-2023-06-26 chunk-16-left-128 int8",
                                       "speed": a.speed, "topk": a.topk, "split": a.split,
                                       "shard": os.path.basename(files[a.shard])})))
    os.replace(tmp, os.path.join(a.out, name))
    h = hashlib.sha256(open(os.path.join(a.out, name), "rb").read()).hexdigest()
    with open(os.path.join(a.out, "SHA256SUMS"), "a") as fh:
        fh.write(f"{h}  {name}\n")
    print(json.dumps({"file": name, "utts": len(uids), "audio_s": round(audio_s, 1),
                      "wall_s": round(time.time() - t0, 1), "rtf": round((time.time() - t0) / max(audio_s, 1e-9), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
