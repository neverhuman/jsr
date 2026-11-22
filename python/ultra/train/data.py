"""Streaming LibriSpeech batches from parquet with duration bucketing (spec v1.0-B §26.4).

Order is a pure function of (seed, epoch): the shard row groups of a split are permuted per epoch,
grouped into windows of `window_groups` row groups, and each window is decoded, sorted by
duration (with seeded jitter) and packed into batches under an audio-seconds budget (padded
length x batch size). Batch order within a window is shuffled with a seed of (seed, epoch,
window). The sampler position is (epoch, window, batch): resuming replays the same order.

Only one window (a few hundred utterances) plus one prefetched window is held in memory.
"""
from __future__ import annotations

import functools
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
    kd: tuple | None = None  # H22 teacher targets (ids, lp, u, ok) from train.kd.TeacherStore

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
    kd: tuple | None = None  # (kd_ids, kd_lp, kd_u, kd_mask), padded to the longest teacher track

    @property
    def audio_seconds(self) -> float:
        return float(self.feat_lens.sum()) / 100.0

    def to(self, device) -> Batch:
        return Batch(self.uids, self.texts, *(t.to(device) for t in (
            self.feats, self.feat_lens, self.rnnt, self.rnnt_lens, self.ctc, self.ctc_lens)),
            kd=tuple(t.to(device) for t in self.kd) if self.kd is not None else None)


@dataclass(frozen=True)
class Position:
    epoch: int = 0
    window: int = 0
    batch: int = 0

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "window": self.window, "batch": self.batch}


def _round_up(n: int, m: int) -> int:
    return n if m <= 1 else (n + m - 1) // m * m


def collate(exs: list[Example], pad_frames: int = 0, pad_tokens: int = 0) -> Batch:
    """Pad to the longest example; with pad_frames / pad_tokens > 1 the time and target axes are
    rounded up to those multiples, so a shape-specialising backend (Apple MPS) sees few distinct
    shapes. Lengths are unchanged, so the extra padding is masked everywhere."""
    def pad(seqs, dtype):
        n = _round_up(max(1, max(len(s) for s in seqs)), pad_tokens)
        out = torch.zeros(len(seqs), n, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.as_tensor(s, dtype=dtype)
        return out

    T = _round_up(max(e.feats.shape[0] for e in exs), pad_frames)
    feats = torch.zeros(len(exs), T, exs[0].feats.shape[1])
    for i, e in enumerate(exs):
        feats[i, :e.feats.shape[0]] = e.feats
    return Batch([e.uid for e in exs], [e.text for e in exs], feats,
                 torch.tensor([e.feats.shape[0] for e in exs]),
                 pad([e.rnnt for e in exs], torch.long), torch.tensor([len(e.rnnt) for e in exs]),
                 pad([e.ctc for e in exs], torch.long), torch.tensor([len(e.ctc) for e in exs]),
                 kd=_collate_kd(exs))


def _collate_kd(exs: list[Example]):
    if all(e.kd is None for e in exs):
        return None
    k = next(e.kd[0].shape[1] for e in exs if e.kd is not None)
    tt = max((len(e.kd[2]) for e in exs if e.kd is not None), default=0)
    ids = torch.zeros(len(exs), tt, k, dtype=torch.int16)
    lp = torch.zeros(len(exs), tt, k, dtype=torch.float16)
    u = torch.zeros(len(exs), tt, dtype=torch.long)
    mask = torch.zeros(len(exs), tt, dtype=torch.bool)
    for i, e in enumerate(exs):
        if e.kd is None or not e.kd[3]:
            continue  # no teacher entry, or R0's 1-best differs from the reference
        n = len(e.kd[2])
        ids[i, :n] = torch.from_numpy(np.asarray(e.kd[0]))
        lp[i, :n] = torch.from_numpy(np.asarray(e.kd[1]))
        u[i, :n] = torch.from_numpy(np.asarray(e.kd[2]))
        mask[i, :n] = True
    return ids, lp, u, mask


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


_WORKER: dict = {}


def _worker_init(tok_rnnt: str, tok_ctc: str, jsr_data: str) -> None:
    if jsr_data:
        os.environ["JSR_DATA"] = jsr_data
    torch.set_num_threads(1)
    from train.tokenizer import Tokenizer

    _WORKER["feat"] = Featurizer(Tokenizer(tok_rnnt), Tokenizer(tok_ctc))


def _decode_job(args):
    """Worker side of ParquetBatches(loader="process"): the thread path's _decode, returning
    plain arrays (uid, text, feats float32 (T, 80), rnnt ids, ctc ids, speed factor) or None."""
    uid, text, audio, epoch, seed, speeds, max_seconds = args
    pcm, sr = sf.read(io.BytesIO(audio), dtype="int16")
    if sr != 16000 or pcm.ndim != 1:
        raise DataError(f"{uid}: expected 16 kHz mono, got {sr} {pcm.shape}")
    f = 1.0
    if speeds:
        import hashlib

        h = hashlib.sha256(f"{seed}:{epoch}:{uid}".encode()).digest()
        f = speeds[h[0] % len(speeds)]
        if f != 1.0:
            from fractions import Fraction

            from scipy.signal import resample_poly

            r = Fraction(f).limit_denominator(100)
            y = resample_poly(pcm.astype(np.float32), r.denominator, r.numerator)
            pcm = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    if len(pcm) / 16000.0 > max_seconds:
        return None
    ex = _WORKER["feat"].example(uid, text, pcm)
    return uid, text, ex.feats.numpy(), ex.rnnt, ex.ctc, f


class ParquetBatches:
    """Endless, resumable batch stream over one or more LibriSpeech parquet splits."""

    def __init__(self, splits: list[str], feat: Featurizer, seed: int, budget_s: float = 200.0,
                 max_batch: int = 64, max_seconds: float = 20.0, window_groups: int = 4,
                 parquet_root: str | None = None, workers: int = 4, teacher=None,
                 speed_perturb: tuple[float, ...] = (), pad_frames: int = 0,
                 pad_tokens: int = 0, loader: str = "thread"):
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
        # (width, epoch): keep an older window width until that epoch ends (resume after a width
        # change), so a mid-epoch resume never skips or repeats data; set by the trainer.
        self.carry_width: tuple[int, int] | None = None
        self.dropped_long = 0
        self.speed_perturb = tuple(speed_perturb)
        self.teacher = teacher
        self.pad_frames, self.pad_tokens = pad_frames, pad_tokens
        if loader not in ("thread", "process"):
            raise DataError(f"unknown loader {loader!r}")
        self.loader = loader
        if loader == "process" and workers > 1:
            # Decode, speed perturbation and log-mel run in spawned worker processes (no GIL);
            # the same functions as the thread path, one torch thread each, so features match.
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                initargs=(feat.tok_rnnt.path, feat.tok_ctc.path, os.environ.get("JSR_DATA", "")))
        else:
            self._pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.manifest = [f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
                         for f in self.files]

    @property
    def windows_per_epoch(self) -> int:
        return (len(self.units) + self.window_groups - 1) // self.window_groups

    def _width(self, epoch: int) -> int:
        if self.carry_width is not None and epoch == self.carry_width[1]:
            return self.carry_width[0]
        return self.window_groups

    def _windows(self, epoch: int) -> int:
        wg = self._width(epoch)
        return (len(self.units) + wg - 1) // wg

    def _window_units(self, epoch: int, window: int):
        perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.units))
        wg = self._width(epoch)
        sel = perm[window * wg:(window + 1) * wg]
        return [self.units[i] for i in sorted(sel)]

    def _decode(self, row: dict, epoch: int = 0) -> Example | None:
        pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if sr != 16000 or pcm.ndim != 1:
            raise DataError(f"{row['id']}: expected 16 kHz mono, got {sr} {pcm.shape}")
        f = self._speed(row["id"], epoch) if self.speed_perturb else 1.0
        if f != 1.0:
            pcm = self._perturb(pcm, row["id"], epoch)
        if len(pcm) / 16000.0 > self.max_seconds:
            return None
        ex = self.feat.example(row["id"], row["text"], pcm)
        if self.teacher is not None:
            ex.kd = self.teacher.get(row["id"], f)
        return ex

    def _attach(self, x):
        """Process-path result -> Example in the main process (teacher lookup lives here)."""
        if x is None:
            return None
        uid, text, feats, rnnt, ctc, f = x
        ex = Example(uid, text, torch.from_numpy(feats), rnnt, ctc)
        if self.teacher is not None:
            ex.kd = self.teacher.get(uid, f)
        return ex

    def _speed(self, uid: str, epoch: int) -> float:
        import hashlib

        h = hashlib.sha256(f"{self.seed}:{epoch}:{uid}".encode()).digest()
        return self.speed_perturb[h[0] % len(self.speed_perturb)]

    def _perturb(self, pcm: np.ndarray, uid: str, epoch: int) -> np.ndarray:
        """Speed perturbation (tempo and pitch together, labels unchanged): resample by a
        factor chosen per (seed, epoch, utterance), e.g. 0.9 / 1.0 / 1.1."""
        import hashlib
        from fractions import Fraction

        from scipy.signal import resample_poly

        h = hashlib.sha256(f"{self.seed}:{epoch}:{uid}".encode()).digest()
        f = self.speed_perturb[h[0] % len(self.speed_perturb)]
        if f == 1.0:
            return pcm
        r = Fraction(f).limit_denominator(100)  # output length = input length / f
        y = resample_poly(pcm.astype(np.float32), r.denominator, r.numerator)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

    def _load_window(self, epoch: int, window: int) -> list[list[Example]]:
        rows: list[dict] = []
        for fi, rg in self._window_units(epoch, window):
            t = pq.ParquetFile(self.files[fi]).read_row_group(rg, columns=["id", "text", "audio"])
            rows.extend(t.to_pylist())
        # FLAC decode (libsndfile via cffi) and torch FFTs release the GIL: threads scale.
        if self.loader == "process" and self._pool is not None:
            args = [(r["id"], r["text"], r["audio"]["bytes"], epoch, self.seed, self.speed_perturb,
                     self.max_seconds) for r in rows]
            decoded = [self._attach(x) for x in self._pool.map(_decode_job, args, chunksize=16)]
        else:
            dec = functools.partial(self._decode, epoch=epoch)
            decoded = list(self._pool.map(dec, rows)) if self._pool else [dec(r) for r in rows]
        del rows
        self.dropped_long += sum(e is None for e in decoded)
        exs: list[Example] = [e for e in decoded if e is not None]
        rng = np.random.default_rng([self.seed, epoch, window])
        dur = [e.seconds * (1.0 + 0.05 * rng.random()) for e in exs]
        if self.pad_frames > 1:  # pack on the padded length the batch will really have
            dur = [_round_up(int(d * 100), self.pad_frames) / 100.0 for d in dur]
        groups = pack(dur, self.budget_s, self.max_batch)
        rng.shuffle(groups)
        return [[exs[i] for i in g] for g in groups]

    def iterate(self, start: Position):
        """Yield (position after this batch, Batch) forever, from `start`; one window prefetched."""
        q: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def producer():
            ep, w = start.epoch, start.window
            while not stop.is_set():
                item = (ep, w, self._load_window(ep, w))
                while not stop.is_set():
                    try:
                        q.put(item, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                w += 1
                if w >= self._windows(ep):
                    ep, w = ep + 1, 0

        # The producer runs as a future: a failure is kept on the future (never swallowed) and
        # re-raised here as a DataError as soon as the queue runs dry.
        from concurrent.futures import ThreadPoolExecutor

        runner = ThreadPoolExecutor(1, thread_name_prefix="batch-producer")
        fut = runner.submit(producer)
        first = True
        try:
            while True:
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    if fut.done() and fut.exception() is not None:
                        err = fut.exception()
                        raise DataError(f"batch producer failed: {err!r}") from err
                    continue
                ep, w, batches = item
                skip = start.batch if first else 0
                first = False
                for bi in range(skip, len(batches)):
                    nxt = Position(ep, w, bi + 1)
                    yield nxt, collate(batches[bi], self.pad_frames, self.pad_tokens)
        finally:
            stop.set()
            runner.shutdown(wait=False)


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
