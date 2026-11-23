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
    ctx: int = 0  # H41 arm 11: prepended stream-context feature frames (not part of the utterance)
    anchor: np.ndarray | None = None  # H48c: per-token MFA anchor frames (train.mfa_anchors)

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
    active: bool = True  # collective-only dummy inputs do not contribute audio or gradients
    ctx: torch.Tensor | None = None  # per-row prepended context frames (train.stream_ctx)
    anchor: torch.Tensor | None = None  # H48c (B, S) MFA anchor frames, -1 = none / padding

    @property
    def audio_seconds(self) -> float:
        n = float(self.feat_lens.sum()) - (float(self.ctx.sum()) if self.ctx is not None else 0.0)
        return n / 100.0 if self.active else 0.0

    def to(self, device) -> Batch:
        return Batch(self.uids, self.texts, *(t.to(device) for t in (
            self.feats, self.feat_lens, self.rnnt, self.rnnt_lens, self.ctc, self.ctc_lens)),
            kd=tuple(t.to(device) for t in self.kd) if self.kd is not None else None,
            active=self.active, ctx=self.ctx.to(device) if self.ctx is not None else None,
            anchor=self.anchor.to(device) if self.anchor is not None else None)


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
    rnnt = pad([e.rnnt for e in exs], torch.long)
    anchor = None
    if any(e.anchor is not None for e in exs):  # H48c: -1 marks unanchored tokens and padding
        anchor = torch.full(rnnt.shape, -1, dtype=torch.long)
        for i, e in enumerate(exs):
            if e.anchor is not None:
                if len(e.anchor) != len(e.rnnt):
                    raise DataError(f"{e.uid}: {len(e.anchor)} anchors for {len(e.rnnt)} tokens")
                anchor[i, :len(e.anchor)] = torch.as_tensor(np.asarray(e.anchor), dtype=torch.long)
    return Batch([e.uid for e in exs], [e.text for e in exs], feats,
                 torch.tensor([e.feats.shape[0] for e in exs]),
                 rnnt, torch.tensor([len(e.rnnt) for e in exs]),
                 pad([e.ctc for e in exs], torch.long), torch.tensor([len(e.ctc) for e in exs]),
                 kd=_collate_kd(exs),
                 ctx=torch.tensor([e.ctx for e in exs]) if any(e.ctx for e in exs) else None,
                 anchor=anchor)


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


def weighted_share(group: list[int], durations: list[float], weights, rank: int) -> list[int]:
    """This rank's part of one global batch when ranks differ in speed (throughput-weighted
    shards). Utterances are dealt in the group's order to the rank whose seconds so far, relative
    to its weight, would stay smallest, so each rank gets about weight / sum(weights) of the
    seconds and its part spans the group's duration range. Every rank computes the same deal, so
    the parts are disjoint and complete. Only used when weights are set; without them the
    loader keeps the plain group[rank::world] split."""
    given = [0.0] * len(weights)
    mine = []
    for i in group:
        d = durations[i]
        r = min(range(len(weights)), key=lambda k: ((given[k] + d) / weights[k], k))
        given[r] += d
        if r == rank:
            mine.append(i)
    return mine


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


def apply_synth_aug(pcm: np.ndarray, uid: str, epoch: int, seed: int) -> np.ndarray:
    """O2-clean mathematical acoustic augmentation (0 external audio files, exact length preserved):
    25% synthetic exponential-decay RIR (RT60 0.10-0.45 s) + 25% synthetic pink/white noise (10-28 dB).
    Deterministic from (seed, epoch, uid) so thread and process loaders match bit-for-bit."""
    if len(pcm) < 64:
        return pcm
    import hashlib

    h = hashlib.sha256(f"synth:{seed}:{epoch}:{uid}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    do_rir = (h[8] & 3) == 0  # 25%
    do_noise = (h[9] & 3) == 0  # 25%
    if not do_rir and not do_noise:
        return pcm
    y = pcm.astype(np.float32) / 32768.0
    if do_rir:
        from scipy.signal import fftconvolve

        rt60 = float(rng.uniform(0.10, 0.45))
        n_rir = int(rng.integers(512, 1921))  # 32-120 ms at 16 kHz
        t = np.arange(n_rir, dtype=np.float32) / 16000.0
        decay = np.exp(-6.9077553 * t / rt60)
        ir = rng.standard_normal(n_rir).astype(np.float32) * decay * 0.25
        ir[0] = 1.0  # direct path at tap 0 preserves causal onset alignment
        ir /= float(np.sqrt(np.sum(ir * ir)) + 1e-12)
        y = fftconvolve(y, ir, mode="full")[:len(pcm)].astype(np.float32)
    if do_noise:
        snr_db = float(rng.uniform(10.0, 28.0))
        sig_rms = float(np.sqrt(np.mean(y * y)) + 1e-8)
        n = rng.standard_normal(len(pcm)).astype(np.float32)
        if (h[10] & 1) == 0:  # pink 1/sqrt(f) spectral shaping
            spec = np.fft.rfft(n)
            freqs = np.arange(1, len(spec) + 1, dtype=np.float32)
            spec /= np.sqrt(freqs)
            n = np.fft.irfft(spec, n=len(pcm)).astype(np.float32)
        n_rms = float(np.sqrt(np.mean(n * n)) + 1e-8)
        y = y + n * (sig_rms / n_rms * (10.0 ** (-snr_db / 20.0)))
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def _decode_job(args):
    """Worker side of ParquetBatches(loader="process"): the thread path's _decode, returning
    plain arrays (uid, text, feats float32 (T, 80), rnnt ids, ctc ids, speed factor) or None."""
    uid, text, audio, epoch, seed, speeds, max_seconds, synth_aug = args
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
    if synth_aug:
        pcm = apply_synth_aug(pcm, uid, epoch, seed)
    ex = _WORKER["feat"].example(uid, text, pcm)
    return uid, text, ex.feats.numpy(), ex.rnnt, ex.ctc, f


def parse_extra_splits(spec: str) -> list[tuple[str, float | None]]:
    """`/abs/dir[@weight],...` -> [(dir, weight or None)] (H52). An extra split is a directory of
    parquet shards in the LibriSpeech schema (id, text, audio; optional conf), e.g. Libri-Light with
    R0 pseudo-label text. Without a weight it joins at its natural share (every row group once per
    epoch); a weight w in (0, 1) makes it w of each epoch's row groups instead."""
    out: list[tuple[str, float | None]] = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        path, _, w = item.partition("@")
        if not os.path.isabs(path):
            raise ValueError(f"extra split {path!r} must be an absolute directory")
        weight = float(w) if w else None
        if weight is not None and not 0.0 < weight < 1.0:
            raise ValueError(f"extra split weight {weight} must be in (0, 1)")
        out.append((path, weight))
    if sum(w or 0.0 for _, w in out) >= 1.0:
        raise ValueError("extra split weights must sum to less than 1")
    return out


class ParquetBatches:
    """Endless, resumable batch stream over one or more LibriSpeech parquet splits, plus optional
    extra splits (`extra`, see parse_extra_splits) whose rows below `min_conf` are dropped.
    `base_weight` > 0 fixes the LibriSpeech share of every epoch's row groups instead (e.g. 0.5, or
    LS 3 : LL 1 by hours); the extra splits then share the rest in proportion to their row groups."""

    def __init__(self, splits: list[str], feat: Featurizer, seed: int, budget_s: float = 200.0,
                 max_batch: int = 64, max_seconds: float = 20.0, window_groups: int = 4,
                 parquet_root: str | None = None, workers: int = 4, teacher=None,
                 speed_perturb: tuple[float, ...] = (), pad_frames: int = 0,
                 pad_tokens: int = 0, loader: str = "thread", synth_aug: bool = False,
                 extra: list[tuple[str, float | None]] = (), min_conf: float = 0.0,
                 base_weight: float = 0.0, anchors=None):
        from concurrent.futures import ThreadPoolExecutor

        from ultra.libri import PARQUET, SPLIT_DIRS

        root = parquet_root or PARQUET
        self.files = []
        for s in splits:
            fs = sorted(glob.glob(os.path.join(root, SPLIT_DIRS[s], "*.parquet")))
            if not fs:
                raise DataError(f"no parquet shards for {s} under {root}")
            self.files.extend(fs)
        self._first_extra, self.min_conf, self.dropped_conf = len(self.files), float(min_conf), 0
        spans = []  # (first file, end file, weight or None) per extra split
        for d, w in extra:
            fs = sorted(glob.glob(os.path.join(d, "*.parquet")))
            if not fs:
                raise DataError(f"no parquet shards in extra split {d}")
            for f in fs:
                if self.min_conf > 0 and "conf" not in pq.ParquetFile(f).schema_arrow.names:
                    raise DataError(f"{f}: min_conf {self.min_conf} needs a conf column")
            spans.append((len(self.files), len(self.files) + len(fs), w))
            self.files.extend(fs)
        units = [(fi, rg) for fi, f in enumerate(self.files)
                 for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]
        if base_weight:  # the base splits' share of each epoch; extras split the rest by row groups
            if not 0.0 < base_weight < 1.0 or not spans or any(w is not None for _, _, w in spans):
                raise DataError("base_weight needs 0 < w < 1 and extra splits without their own weights")
            size = [sum(1 for fi, _ in units if a <= fi < b) for a, b, _ in spans]
            spans = [(a, b, (1.0 - base_weight) * n / sum(size)) for (a, b, _), n in zip(spans, size, strict=True)]
        tags = [""] * self._first_extra
        for a, b, w in spans:
            tag = (f"@{w:.6g}" if w is not None else "") + (f"?conf>={self.min_conf}" if self.min_conf else "")
            tags += [tag] * (b - a)
        weighted = [s for s in spans if s[2] is not None]
        in_weighted = [any(a <= fi < b for a, b, _ in weighted) for fi, _ in units]
        self.units = [u for u, x in zip(units, in_weighted, strict=True) if not x]
        share = sum(w for _, _, w in weighted)
        # (row groups, draws per epoch) per weighted split: draws / epoch total = its weight.
        self._weighted = [([u for u in units if a <= u[0] < b], round(w * len(self.units) / (1.0 - share)))
                          for a, b, w in weighted]
        if any(not g for g, _ in self._weighted):
            raise DataError("a weighted extra split has no row groups")
        self.n_epoch_units = len(self.units) + sum(n for _, n in self._weighted)
        self.feat, self.seed = feat, seed
        self.budget_s, self.max_batch, self.max_seconds = budget_s, max_batch, max_seconds
        self.window_groups = window_groups
        # (width, epoch): keep an older window width until that epoch ends (resume after a width
        # change), so a mid-epoch resume never skips or repeats data; set by the trainer.
        self.carry_width: tuple[int, int] | None = None
        self.rank, self.world = 0, 1  # data-parallel shard (set by the trainer under torchrun)
        self.ddp_global_batches = False
        self.stream_ctx_s, self.stream_ctx_cap, self._ctx_bank = 0.0, 1.5, None  # H41 arm 11 (0 = off)
        self.rank_weights: tuple | None = None  # per-rank relative speed (None = equal shares)
        self.dropped_long = 0
        self.speed_perturb = tuple(speed_perturb)
        self.synth_aug = bool(synth_aug)
        self.teacher = teacher
        self.anchors = anchors  # H48c train.mfa_anchors.MfaAnchors (None = off)
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
        self.manifest = [f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}{t}"
                         for f, t in zip(self.files, tags, strict=True)]

    @property
    def windows_per_epoch(self) -> int:
        return (self.n_epoch_units + self.window_groups - 1) // self.window_groups

    def _width(self, epoch: int) -> int:
        if self.carry_width is not None and epoch == self.carry_width[1]:
            return self.carry_width[0]
        return self.window_groups

    def _windows(self, epoch: int) -> int:
        wg = self._width(epoch)
        return (self.n_epoch_units + wg - 1) // wg

    def _epoch_units(self, epoch: int) -> list[tuple[int, int]]:
        """One epoch's row groups: every unweighted group once, then each weighted extra split's
        draws for this epoch, read in order from an endless chain of seeded permutations of its
        groups (a down-weighted split rotates through all of them across epochs; an up-weighted
        one repeats them evenly). Without weighted splits this is `self.units`, unchanged."""
        if not self._weighted:
            return self.units
        out = list(self.units)
        for k, (groups, draws) in enumerate(self._weighted):
            first, n = epoch * draws, len(groups)
            for cyc in range(first // n, (first + draws - 1) // n + 1):
                perm = np.random.default_rng([self.seed, 52, k, cyc]).permutation(n)
                lo, hi = max(first, cyc * n) - cyc * n, min(first + draws, (cyc + 1) * n) - cyc * n
                out.extend(groups[i] for i in perm[lo:hi])
        return out

    def _window_units(self, epoch: int, window: int):
        units = self._epoch_units(epoch)
        perm = np.random.default_rng([self.seed, epoch]).permutation(len(units))
        wg = self._width(epoch)
        sel = perm[window * wg:(window + 1) * wg]
        return [units[i] for i in sorted(sel)]

    def _read_rows(self, fi: int, rg: int) -> list[dict]:
        """One row group's rows. In an extra split under min_conf > 0, rows whose conf is below it
        (or missing) are dropped and counted in dropped_conf; LibriSpeech rows are never filtered."""
        filt = self.min_conf > 0 and fi >= self._first_extra
        cols = ["id", "text", "audio", "conf"] if filt else ["id", "text", "audio"]
        rows = pq.ParquetFile(self.files[fi]).read_row_group(rg, columns=cols).to_pylist()
        if filt:
            kept = [r for r in rows if r["conf"] is not None and r["conf"] >= self.min_conf]
            self.dropped_conf += len(rows) - len(kept)
            rows = kept
        return rows

    def _decode(self, row: dict, epoch: int = 0) -> Example | None:
        pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if sr != 16000 or pcm.ndim != 1:
            raise DataError(f"{row['id']}: expected 16 kHz mono, got {sr} {pcm.shape}")
        f = self._speed(row["id"], epoch) if self.speed_perturb else 1.0
        if f != 1.0:
            pcm = self._perturb(pcm, row["id"], epoch)
        if len(pcm) / 16000.0 > self.max_seconds:
            return None
        if self.synth_aug:
            pcm = apply_synth_aug(pcm, row["id"], epoch, self.seed)
        ex = self.feat.example(row["id"], row["text"], pcm)
        if self.anchors is not None:  # H48c MFA anchors, keyed by (id, speed) like the teacher
            ex.anchor = self.anchors.token_anchors(row["id"], f, ex.rnnt, row["text"])
        if self.teacher is not None:
            ex.kd = self.teacher.get_checked(row["id"], f, ex.rnnt, n_mel=ex.feats.shape[0])
        return ex

    def _attach(self, x):
        """Process-path result -> Example in the main process (teacher lookup lives here)."""
        if x is None:
            return None
        uid, text, feats, rnnt, ctc, f = x
        ex = Example(uid, text, torch.from_numpy(feats), rnnt, ctc)
        if self.anchors is not None:  # H48c MFA anchors, keyed by (id, speed) like the teacher
            ex.anchor = self.anchors.token_anchors(uid, f, rnnt, text)
        if self.teacher is not None:
            ex.kd = self.teacher.get_checked(uid, f, rnnt, n_mel=feats.shape[0])
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
            rows.extend(self._read_rows(fi, rg))
        if self.world > 1:
            return self._load_window_sharded(rows, epoch, window)
        # FLAC decode (libsndfile via cffi) and torch FFTs release the GIL: threads scale.
        if self.loader == "process" and self._pool is not None:
            args = [(r["id"], r["text"], r["audio"]["bytes"], epoch, self.seed, self.speed_perturb,
                     self.max_seconds, self.synth_aug) for r in rows]
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

    def _frames_after(self, row: dict, epoch: int) -> int | None:
        """Log-mel frame count of a row after speed perturbation, from the FLAC header alone
        (None if the perturbed audio exceeds max_seconds), identical to decoding it."""
        n = sf.info(io.BytesIO(row["audio"]["bytes"])).frames
        if self.speed_perturb:
            f = self._speed(row["id"], epoch)
            if f != 1.0:
                from fractions import Fraction

                r = Fraction(f).limit_denominator(100)
                n = -(-n * r.denominator // r.numerator)  # resample_poly output length
        if n / 16000.0 > self.max_seconds:
            return None
        return self.feat.logmel.num_frames(n)

    def _load_window_sharded(self, rows: list[dict], epoch: int, window: int) -> list[list[Example]]:
        """Data-parallel window: pack on header-only durations (the same packing and shuffle as
        the single-process path), keep groups g with g % world == rank (truncated to a multiple
        of world so every rank has the same number of batches), and decode only those rows."""
        frames = [self._frames_after(r, epoch) for r in rows]
        self.dropped_long += sum(f is None for f in frames)
        keep = [i for i, f in enumerate(frames) if f is not None]
        rng = np.random.default_rng([self.seed, epoch, window])
        dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
        if self.pad_frames > 1:
            dur = [_round_up(int(d * 100), self.pad_frames) / 100.0 for d in dur]
        groups = pack(dur, self.budget_s, self.max_batch)
        rng.shuffle(groups)
        if self.ddp_global_batches:
            # Preserve the same global groups and cursor as the single-rank loader.
            if self.rank_weights:
                parts = [weighted_share(g, dur, self.rank_weights, self.rank) for g in groups]
            else:
                parts = [g[self.rank::self.world] for g in groups]
            active = [bool(g) for g in parts]
            parts = [g or [global_group[0]] for g, global_group in zip(parts, groups, strict=True)]
        else:
            n = len(groups)
            if n % self.world:
                target = n + self.world - n % self.world  # repeat the first groups up to a multiple
                groups = (groups * -(-target // max(1, n)))[:target]
            parts = groups[self.rank::self.world]
            active = [True] * len(parts)
        mine = [[rows[keep[i]] for i in g] for g in parts]
        flat = [r for g in mine for r in g]
        if self.loader == "process" and self._pool is not None:
            args = [(r["id"], r["text"], r["audio"]["bytes"], epoch, self.seed, self.speed_perturb,
                     self.max_seconds, self.synth_aug) for r in flat]
            decoded = [self._attach(x) for x in self._pool.map(_decode_job, args, chunksize=16)]
        else:
            dec = functools.partial(self._decode, epoch=epoch)
            decoded = list(self._pool.map(dec, flat)) if self._pool else [dec(r) for r in flat]
        out, k = [], 0
        for g in mine:
            out.append(decoded[k:k + len(g)])
            k += len(g)
        if any(e is None for g in out for e in g):
            raise DataError("header-predicted length disagrees with the decoded audio")
        if self.ddp_global_batches:
            return list(zip(out, active, strict=True))
        return out

    def _with_context(self, exs: list, epoch: int) -> list:
        if not self.stream_ctx_s:
            return exs
        from train.stream_ctx import ContextBank, add_context

        if self._ctx_bank is None:
            self._ctx_bank = ContextBank(self.feat.logmel)
        return add_context(exs, self._ctx_bank, self.seed, epoch, self.stream_ctx_s, self.stream_ctx_cap)

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
                    group = batches[bi]
                    if self.ddp_global_batches and self.world > 1:
                        examples, active = group
                        batch = collate(self._with_context(examples, ep), self.pad_frames, self.pad_tokens)
                        batch.active = active
                    else:
                        batch = collate(self._with_context(group, ep), self.pad_frames, self.pad_tokens)
                    yield nxt, batch
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
