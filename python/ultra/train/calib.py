"""Calibration data for static (batch-invariant) INT8 quantisation of an R1 export.

Calibration audio comes from the TRAIN partition only (never DEV/test): a stratified pick from a
candidate pool of train-clean-100 and train-other-500 utterances covering quiet (lowest RMS),
noisy/hard (train-other), fast (highest tokens per second), long (longest) and random examples.
The chosen utterance ids are written to a manifest whose SHA-256 goes into the bundle.

Encoder and joiner inputs are captured by running the FP32 streaming path over those utterances:
a seeded sample of real chunk inputs (features slice, t0, caches) and of real joiner inputs
(projected encoder rows and decoder outputs met during greedy decoding).
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from train.infer import StreamingRecognizer, run_streaming

STRATA = ("quiet", "other", "fast", "long", "random")


class _Utt:
    def __init__(self, uid: str, text: str, pcm: np.ndarray):
        self.uid, self.text, self.pcm = uid, text, pcm

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 16000.0


def calibration_set(n: int = 300, pool_per_split: int = 1500, seed: int = 20261120,
                    tokenizer=None) -> tuple[list[_Utt], dict]:
    """Stratified TRAIN-only calibration utterances and their manifest."""
    from ultra.libri import PARQUET, SPLIT_DIRS

    rng = np.random.default_rng(seed)
    pool: list[tuple[str, str, np.ndarray, str]] = []
    for split in ("train-clean-100", "train-other-500"):
        files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))
        picked = 0
        for f in [files[i] for i in rng.permutation(len(files))[:3]]:
            pf = pq.ParquetFile(f)
            for rg in rng.permutation(pf.metadata.num_row_groups)[:6]:
                for row in pf.read_row_group(int(rg), columns=["id", "text", "audio"]).to_pylist():
                    pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
                    if sr == 16000 and pcm.ndim == 1 and len(pcm) <= 20 * 16000:
                        pool.append((row["id"], row["text"], pcm, split))
                        picked += 1
                if picked >= pool_per_split:
                    break
            if picked >= pool_per_split:
                break
    secs = np.array([len(p[2]) / 16000 for p in pool])
    rms = np.array([float(np.sqrt(np.mean(p[2].astype(np.float64) ** 2))) for p in pool])
    words = np.array([len(p[1].split()) for p in pool])
    rate = words / np.maximum(secs, 0.5)
    k = n // len(STRATA)
    chosen: dict[int, str] = {}

    def take(order, name):
        for i in order:
            if len(chosen) >= n or sum(v == name for v in chosen.values()) >= k:
                return
            chosen.setdefault(int(i), name)

    take(np.argsort(rms), "quiet")
    take([i for i in rng.permutation(len(pool)) if pool[i][3] == "train-other-500"], "other")
    take(np.argsort(-rate), "fast")
    take(np.argsort(-secs), "long")
    take(rng.permutation(len(pool)), "random")
    idx = sorted(chosen)
    utts = [_Utt(pool[i][0], pool[i][1], pool[i][2]) for i in idx]
    manifest = {"seed": seed, "partition": "train (train-clean-100, train-other-500)",
                "n": len(utts), "strata": {s: sum(chosen[i] == s for i in idx) for s in STRATA},
                "utterances": [{"uid": pool[i][0], "split": pool[i][3], "stratum": chosen[i],
                                "seconds": round(float(secs[i]), 3)} for i in idx]}
    blob = json.dumps(manifest["utterances"], sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(blob).hexdigest()
    return utts, manifest


class _Capture(StreamingRecognizer):
    """FP32 streaming recogniser that records a seeded sample of encoder and joiner inputs."""

    def __init__(self, *a, p_chunk: float = 0.05, p_joiner: float = 0.05, seed: int = 0, **kw):
        super().__init__(*a, **kw)
        self.rng = np.random.default_rng(seed)
        self.p_chunk, self.p_joiner = p_chunk, p_joiner
        self.enc_rows: list[dict] = []
        self.joiner_rows: list[tuple[np.ndarray, np.ndarray]] = []
        orig = self.encoder.forward

        def wrapped(*args):
            take = self.rng.random(args[0].shape[0]) < self.p_chunk
            for i in np.nonzero(take)[0]:
                row = {}
                for name, t in zip(("feats", "t0", "k", "v", "kv_len", "conv"), args, strict=False):
                    ax = 1 if name in ("k", "v", "conv") else 0
                    row[name] = t.select(ax, int(i)).unsqueeze(ax).numpy().copy()
                self.enc_rows.append(row)
            return orig(*args)

        self.encoder.forward = wrapped

    def _greedy(self, streams, enc_p):
        if self.rng.random() < self.p_joiner * 10:
            with torch.no_grad():
                ctx = torch.tensor([s.ctx for s in streams])
                dec = self.model.predictor(ctx)
            for i in range(len(streams)):
                t = int(self.rng.integers(enc_p.shape[1]))
                self.joiner_rows.append((enc_p[i, t].numpy().copy(), dec[i].numpy().copy()))
        super()._greedy(streams, enc_p)


def capture(model, tokenizer, utts, p_chunk: float = 0.03, p_joiner: float = 0.05,
            seed: int = 0, max_enc_rows: int = 384, max_joiner_rows: int = 4096):
    """(encoder input rows, joiner input rows) sampled from FP32 streaming over `utts`."""
    if model.cfg.encoder.family != "dense":
        raise ValueError("calibration capture implements the dense family; extend for multirate")
    rec = _Capture(model, tokenizer, threads=torch.get_num_threads(), p_chunk=p_chunk,
                   p_joiner=p_joiner, seed=seed)
    run_streaming(rec, utts, streams=8, packet_ms=160)
    return rec.enc_rows[:max_enc_rows], rec.joiner_rows[:max_joiner_rows]
