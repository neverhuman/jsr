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
    # Hash of the selected-utterance list (the selection identity); distinct from the hash of
    # the calibration_manifest.json file recorded under manifest.files.
    manifest["selection_sha256"] = hashlib.sha256(blob).hexdigest()
    return utts, manifest


class _Reservoir:
    """Uniform sample of at most `cap` items from a stream of unknown length (Algorithm R)."""

    def __init__(self, cap: int, rng: np.random.Generator):
        self.cap, self.rng, self.seen, self.items = cap, rng, 0, []

    def offer(self, make):
        """`make()` builds the item only if it is kept (avoids copying rejected payloads)."""
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(make())
        else:
            j = int(self.rng.integers(self.seen))
            if j < self.cap:
                self.items[j] = make()


class _Capture(StreamingRecognizer):
    """FP32 streaming recogniser that reservoir-samples the inputs of real encoder calls (one row
    per stream-chunk) and of real joiner calls (one row per (stream, frame, symbol) query, with
    the context that call actually used), across the whole calibration population."""

    def __init__(self, *a, utts=None, seed: int = 0, enc_cap: int = 384, joiner_cap: int = 4096,
                 **kw):
        super().__init__(*a, **kw)
        rng = np.random.default_rng(seed)
        self.enc = _Reservoir(enc_cap, rng)
        self.joi = _Reservoir(joiner_cap, rng)
        self._utts = list(utts or [])
        self._made = 0

    def create_stream(self):
        s = super().create_stream()
        s.uid = self._utts[self._made].uid if self._made < len(self._utts) else f"#{self._made}"
        self._made += 1
        return s

    def _run_group(self, streams, t0s, n):
        self._group = (streams, t0s, n)
        super()._run_group(streams, t0s, n)

    def _greedy(self, streams, enc_p):
        # Same algorithm as StreamingRecognizer._greedy, recording every joiner call's inputs.
        from train.transducer import BLANK

        pred, joiner = self.model.predictor, self.model.joiner
        ctx = torch.tensor([s.ctx for s in streams])
        t0s = self._group[1]
        for t in range(enc_p.shape[1]):
            active = torch.ones(len(streams), dtype=torch.bool)
            for sym in range(self.max_symbols):
                idx = active.nonzero().squeeze(1)
                if idx.numel() == 0:
                    break
                e, d = enc_p[idx, t], pred(ctx[idx])
                for r, i in enumerate(idx.tolist()):
                    self.joi.offer(lambda r=r, i=i, sym=sym: {
                        "enc": e[r].numpy().copy(), "dec": d[r].numpy().copy(),
                        "uid": streams[i].uid, "frame": t0s[i] + t, "symbol": sym,
                        "ctx": ctx[i].tolist()})
                kk = joiner.logits(e, d).argmax(dim=-1)
                emit = kk != BLANK
                ei = idx[emit]
                for i, tok in zip(ei.tolist(), kk[emit].tolist(), strict=True):
                    streams[i].tokens.append(tok)
                ctx[ei] = torch.stack([ctx[ei, 1], kk[emit]], dim=1)
                active = torch.zeros_like(active)
                active[ei] = True
            else:
                for i in active.nonzero().squeeze(1).tolist():
                    streams[i].forced += 1
                    self.forced_advances += 1
        for i, s in enumerate(streams):
            s.ctx = ctx[i].tolist()


def _encoder_tap(rec: _Capture):
    orig = rec.encoder.forward
    names = ("feats", "t0", "k", "v", "kv_len", "conv")

    def wrapped(*args):
        streams, t0s, n = rec._group
        if n != rec.chunk:
            return orig(*args)  # partial final chunks: other shapes break histogram calibrators
        for i in range(args[0].shape[0]):
            def make(i=i):
                row = {}
                for name, t in zip(names, args, strict=False):
                    ax = 1 if name in ("k", "v", "conv") else 0
                    row[name] = t.select(ax, i).unsqueeze(ax).numpy().copy()
                row["_prov"] = {"uid": streams[i].uid, "t0": int(t0s[i]), "n": int(n),
                                "kv_len": int(args[4][i])}
                return row
            rec.enc.offer(make)
        return orig(*args)

    rec.encoder.forward = wrapped


def capture(model, tokenizer, utts, seed: int = 0, max_enc_rows: int = 384,
            max_joiner_rows: int = 4096):
    """Reservoir samples of real encoder-call rows and real joiner-call rows (FP32 streaming over
    `utts`), each with provenance (uid, chunk start / frame, cache occupancy or context).
    Returns (encoder feeds, joiner feeds, provenance dict)."""
    if model.cfg.encoder.family != "dense":
        raise ValueError("calibration capture implements the dense family; extend for multirate")
    rec = _Capture(model, tokenizer, threads=torch.get_num_threads(), utts=utts, seed=seed,
                   enc_cap=max_enc_rows, joiner_cap=max_joiner_rows)
    _encoder_tap(rec)
    run_streaming(rec, utts, streams=8, packet_ms=160)
    enc_rows = [{k: v for k, v in r.items() if k != "_prov"} for r in rec.enc.items]
    joi_rows = [{"enc": r["enc"][None], "dec": r["dec"][None]} for r in rec.joi.items]
    prov = {"encoder": {"offered": rec.enc.seen, "kept": len(rec.enc.items),
                        "rows": [r["_prov"] for r in rec.enc.items]},
            "joiner": {"offered": rec.joi.seen, "kept": len(rec.joi.items),
                       "rows": [{k: r[k] for k in ("uid", "frame", "symbol", "ctx")}
                                for r in rec.joi.items]},
            "sampling": "uniform reservoir (Algorithm R) over every call in the population; "
                        "encoder rows only from full chunks (n = chunk_frames)",
            "seed": seed}
    return enc_rows, joi_rows, prov
