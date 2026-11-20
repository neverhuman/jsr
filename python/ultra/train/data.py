"""Streaming LibriSpeech batches from parquet with duration bucketing (spec v1.0-B §26.4).

Order is a pure function of (seed, epoch): the shard row groups of a split are permuted per epoch,
grouped into windows of `window_groups` row groups, and each window is decoded, sorted by
duration (with seeded jitter) and packed into batches under an audio-seconds budget (padded
length x batch size). Batch order within a window is shuffled with a seed of (seed, epoch,
window). The sampler position is (epoch, window, batch): resuming replays the same order.

Only one window (a few hundred utterances) plus one prefetched window is held in memory.
"""
from __future__ import annotations

import glob
import io
import os
import queue
import threading
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from train.errors import DataError
from train.features import LogMel, pcm_to_float
from train.tokenizer import Tokenizer


@dataclass
class Example:
    uid: str
    text: str
    feats: torch.Tensor  # (T, 80)
    rnnt: list[int]
    ctc: list[int]

    @property
    def seconds(self) -> float:
        return self.feats.shape[0] / 100.0


@dataclass
class Batch:
    uids: list[str]
    texts: list[str]
    feats: torch.Tensor
    feat_lens: torch.Tensor
    rnnt: torch.Tensor
    rnnt_lens: torch.Tensor
    ctc: torch.Tensor
    ctc_lens: torch.Tensor

    @property
    def audio_seconds(self) -> float:
        return float(self.feat_lens.sum()) / 100.0

    def to(self, device) -> Batch:
        return Batch(self.uids, self.texts, *(t.to(device) for t in (
            self.feats, self.feat_lens, self.rnnt, self.rnnt_lens, self.ctc, self.ctc_lens)))


@dataclass(frozen=True)
class Position:
    epoch: int = 0
    window: int = 0
    batch: int = 0

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "window": self.window, "batch": self.batch}


def collate(exs: list[Example]) -> Batch:
    def pad(seqs, dtype):
        n = max(1, max(len(s) for s in seqs))
        out = torch.zeros(len(seqs), n, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.as_tensor(s, dtype=dtype)
        return out

    T = max(e.feats.shape[0] for e in exs)
    feats = torch.zeros(len(exs), T, exs[0].feats.shape[1])
    for i, e in enumerate(exs):
        feats[i, :e.feats.shape[0]] = e.feats
    return Batch([e.uid for e in exs], [e.text for e in exs], feats,
                 torch.tensor([e.feats.shape[0] for e in exs]),
                 pad([e.rnnt for e in exs], torch.long), torch.tensor([len(e.rnnt) for e in exs]),
                 pad([e.ctc for e in exs], torch.long), torch.tensor([len(e.ctc) for e in exs]))


def pack(durations: list[float], budget_s: float, max_batch: int) -> list[list[int]]:
    """Greedy packing of indices sorted by duration: padded seconds <= budget_s per batch."""
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    batches, cur, cur_max = [], [], 0.0
    for i in order:
        m = max(cur_max, durations[i])
        if cur and (m * (len(cur) + 1) > budget_s or len(cur) >= max_batch):
            batches.append(cur)
            cur, m = [], durations[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches


class Featurizer:
    def __init__(self, tok_rnnt: Tokenizer, tok_ctc: Tokenizer):
        self.logmel = LogMel()
        self.tok_rnnt, self.tok_ctc = tok_rnnt, tok_ctc

    def example(self, uid: str, text: str, pcm: np.ndarray) -> Example:
        return Example(uid, text, self.logmel(pcm_to_float(pcm)), self.tok_rnnt.encode(text),
                       self.tok_ctc.encode(text))


class ParquetBatches:
    """Endless, resumable batch stream over one or more LibriSpeech parquet splits."""

    def __init__(self, splits: list[str], feat: Featurizer, seed: int, budget_s: float = 200.0,
                 max_batch: int = 64, max_seconds: float = 20.0, window_groups: int = 4,
                 parquet_root: str | None = None, workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor

        from ultra.libri import PARQUET, SPLIT_DIRS

        root = parquet_root or PARQUET
        self.files = []
        for s in splits:
            fs = sorted(glob.glob(os.path.join(root, SPLIT_DIRS[s], "*.parquet")))
            if not fs:
                raise DataError(f"no parquet shards for {s} under {root}")
            self.files.extend(fs)
        self.units = [(fi, rg) for fi, f in enumerate(self.files)
                      for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]
        self.feat, self.seed = feat, seed
        self.budget_s, self.max_batch, self.max_seconds = budget_s, max_batch, max_seconds
        self.window_groups = window_groups
        self.dropped_long = 0
        self._pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.manifest = [f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
                         for f in self.files]

    @property
    def windows_per_epoch(self) -> int:
        return (len(self.units) + self.window_groups - 1) // self.window_groups

    def _window_units(self, epoch: int, window: int):
        perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.units))
        sel = perm[window * self.window_groups:(window + 1) * self.window_groups]
        return [self.units[i] for i in sorted(sel)]

    def _decode(self, row: dict) -> Example | None:
        pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if sr != 16000 or pcm.ndim != 1:
            raise DataError(f"{row['id']}: expected 16 kHz mono, got {sr} {pcm.shape}")
        if len(pcm) / 16000.0 > self.max_seconds:
            return None
        return self.feat.example(row["id"], row["text"], pcm)

    def _load_window(self, epoch: int, window: int) -> list[list[Example]]:
        rows: list[dict] = []
        for fi, rg in self._window_units(epoch, window):
            t = pq.ParquetFile(self.files[fi]).read_row_group(rg, columns=["id", "text", "audio"])
            rows.extend(t.to_pylist())
        # FLAC decode (libsndfile via cffi) and torch FFTs release the GIL: threads scale.
        decoded = list(self._pool.map(self._decode, rows)) if self._pool else \
            [self._decode(r) for r in rows]
        del rows
        self.dropped_long += sum(e is None for e in decoded)
        exs: list[Example] = [e for e in decoded if e is not None]
        rng = np.random.default_rng([self.seed, epoch, window])
        dur = [e.seconds * (1.0 + 0.05 * rng.random()) for e in exs]
        groups = pack(dur, self.budget_s, self.max_batch)
        rng.shuffle(groups)
        return [[exs[i] for i in g] for g in groups]

    def iterate(self, start: Position):
        """Yield (position after this batch, Batch) forever, from `start`; one window prefetched."""
        q: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def producer():
            ep, w = start.epoch, start.window
            try:
                while not stop.is_set():
                    item = (ep, w, self._load_window(ep, w))
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    w += 1
                    if w >= self.windows_per_epoch:
                        ep, w = ep + 1, 0
            except Exception as e:  # surfaced to the consumer, never swallowed
                q.put(e)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        first = True
        try:
            while True:
                item = q.get()
                if isinstance(item, Exception):
                    raise DataError(f"batch producer failed: {item!r}") from item
                ep, w, batches = item
                skip = start.batch if first else 0
                first = False
                for bi in range(skip, len(batches)):
                    nxt = Position(ep, w, bi + 1)
                    yield nxt, collate(batches[bi])
        finally:
            stop.set()


class FixedBatches:
    """A fixed list of examples served as the same batches every epoch (D0, tests)."""

    def __init__(self, examples: list[Example], batch_size: int, seed: int = 0, shuffle=False):
        self.examples, self.bs, self.seed, self.shuffle = examples, batch_size, seed, shuffle
        self.manifest = [e.uid for e in examples]

    @property
    def windows_per_epoch(self) -> int:
        return 1

    def _batches(self, epoch: int) -> list[list[int]]:
        idx = list(range(len(self.examples)))
        if self.shuffle:
            idx = list(np.random.default_rng([self.seed, epoch]).permutation(len(idx)))
        return [idx[i:i + self.bs] for i in range(0, len(idx), self.bs)]

    def iterate(self, start: Position):
        ep, skip = start.epoch, start.batch
        while True:
            bs = self._batches(ep)
            for bi in range(skip, len(bs)):
                yield Position(ep, 0, bi + 1), collate([self.examples[i] for i in bs[bi]])
            ep, skip = ep + 1, 0
