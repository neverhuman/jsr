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


def stratified_parts(groups: list[list[int]], durations: list[float], world: int, rank: int,
                     seed) -> list[list[int]]:
    """This rank's groups under duration stratification (frontier OPT-2.2): groups (a multiple of
    `world`) are ordered by their longest member, cut into `world` equal strata and shuffled within
    each stratum; at step i rank r takes the i-th group of stratum (r + i) % world."""
    if len(groups) % world:
        raise DataError("stratified parts need a multiple of world groups")
    order = sorted(range(len(groups)), key=lambda k: (max(durations[i] for i in groups[k]), k))
    per = len(order) // world
    rng = np.random.default_rng(list(seed) + [7919])
    strata = []
    for j in range(world):
        s = order[j * per:(j + 1) * per]
        rng.shuffle(s)
        strata.append(s)
    return [groups[strata[(rank + i) % world][i]] for i in range(per)]


def apply_data_shard(source, cfg, world: int) -> None:
    """H50 DiLoCo island: one process (data_world > 0) reads exactly what DDP rank data_rank of
    data_world reads from the same ordered global batches (global-batch split, no rank weights)."""
    if getattr(cfg, "ddp_stratify_lengths", False) and cfg.ddp_global_batches:
        raise ValueError("ddp_stratify_lengths stratifies per-rank batches; it needs ddp_global_batches=false")
    if cfg.data_world <= 0:
        return
    if world != 1:
        raise ValueError("data_rank / data_world name a single-process island's shard; DDP ranks use their own")
    if not 0 <= cfg.data_rank < cfg.data_world or not hasattr(source, "world"):
        raise ValueError("need 0 <= data_rank < data_world and a source that shards (ParquetBatches)")
    if cfg.ddp_rank_weights or getattr(cfg, "ddp_stratify_lengths", False):
        raise ValueError("island shards use the plain global-batch split")
    source.rank, source.world, source.ddp_global_batches = cfg.data_rank, cfg.data_world, True


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
                 base_weight: float = 0.0, anchors=None, dur_tables: str = "",
                 node_shards: str = "", bpe_dropout_p: float = 0.0,
                 bpe_dropout_alpha: float = 0.1, collate_ahead: int = 0):
        from concurrent.futures import ThreadPoolExecutor

        from ultra.libri import PARQUET, SPLIT_DIRS

        self._node_plan = None
        if node_shards:
            from train.shard_sidecar import parse_node_shards, partition_items

            self._node_plan = parse_node_shards(node_shards)
        root = parquet_root or PARQUET
        self.files = []
        for s in splits:
            fs = sorted(glob.glob(os.path.join(root, SPLIT_DIRS[s], "*.parquet")))
            if not fs:
                raise DataError(f"no parquet shards for {s} under {root}")
            self.files.extend(fs)
        self._first_extra, self.min_conf, self.dropped_conf = len(self.files), float(min_conf), 0
        spans = []  # (first file, end file, weight or None) per extra split
        for si, (d, w) in enumerate(extra):
            fs = sorted(glob.glob(os.path.join(d, "*.parquet")))
            if not fs:
                raise DataError(f"no parquet shards in extra split {d}")
            if self._node_plan is not None and not self._node_plan.prestaged:
                fs = partition_items(fs, self._node_plan.node_world, weights=self._node_plan.weights,
                                     seed=self._node_plan.seed, salt=100 + si, preserve_order=True)[self._node_plan.node_rank]
                if not fs:
                    raise DataError(f"node_shards partition {self._node_plan.node_rank}/{self._node_plan.node_world} "
                                    f"received 0 shards from {d}")
            for f in fs:
                if self.min_conf > 0 and "conf" not in pq.ParquetFile(f).schema_arrow.names:
                    raise DataError(f"{f}: min_conf {self.min_conf} needs a conf column")
            spans.append((len(self.files), len(self.files) + len(fs), w))
            self.files.extend(fs)
        rg_counts = [pq.ParquetFile(f).metadata.num_row_groups for f in self.files]
        units = [(fi, rg) for fi, n_rg in enumerate(rg_counts) for rg in range(n_rg)]
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
        self._weighted_weights = [w for _, _, w in weighted]
        self._weighted_share = share
        # (row groups, draws per epoch) per weighted split: draws / epoch total = its weight.
        self._weighted = [([u for u in units if a <= u[0] < b], round(w * len(self.units) / (1.0 - share)))
                          for a, b, w in weighted]
        if any(not g for g, _ in self._weighted):
            raise DataError("a weighted extra split has no row groups")
        self.n_epoch_units = len(self.units) + sum(n for _, n in self._weighted)
        self._dur_table = None
        if dur_tables:
            from train.shard_sidecar import FrameSidecar

            self._dur_table = FrameSidecar.from_paths(dur_tables, self.files, rg_counts)
        self.feat, self.seed = feat, seed
        self.budget_s, self.max_batch, self.max_seconds = budget_s, max_batch, max_seconds
        self.window_groups = window_groups
        # (width, epoch): keep an older window width until that epoch ends (resume after a width
        # change), so a mid-epoch resume never skips or repeats data; set by the trainer.
        self.carry_width: tuple[int, int] | None = None
        # ((budget_s, max_batch), (epoch, window)): finish that window at a checkpoint's older batch shape after a
        # resume with a new one (train.resize), so nothing is skipped or repeated; set by the trainer.
        self.carry_shape: tuple | None = None
        self.rank, self.world = 0, 1  # data-parallel shard (set by the trainer under torchrun)
        self.ddp_global_batches = False
        self.stream_ctx_s, self.stream_ctx_cap, self._ctx_bank = 0.0, 1.5, None  # H41 arm 11 (0 = off)
        self.rank_weights: tuple | None = None  # per-rank relative speed (None = equal shares)
        self.stratify_lengths = False  # per-rank batches: every step spans every duration stratum
        self.dropped_long = 0
        self.speed_perturb = tuple(speed_perturb)
        self.synth_aug = bool(synth_aug)
        self.teacher = teacher
        self.anchors = anchors  # H48c train.mfa_anchors.MfaAnchors (None = off)
        self.bpe_dropout_p, self.bpe_dropout_alpha = float(bpe_dropout_p), float(bpe_dropout_alpha)
        self.pad_frames, self.pad_tokens = pad_frames, pad_tokens
        if int(collate_ahead) < 0:
            raise DataError(f"collate_ahead must be >= 0, got {collate_ahead}")
        self.collate_ahead = int(collate_ahead)  # >0: next N batches collated off the consumer's thread
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
        wg = self.carry_width[0] if (self.carry_width is not None and epoch == self.carry_width[1]) else self.window_groups
        if self._node_plan is not None:
            return max(1, round(wg * self._node_plan.share))
        return wg

    def _windows(self, epoch: int) -> int:
        wg = self._width(epoch)
        n = len(self._epoch_units(epoch)) if self._node_plan is not None else self.n_epoch_units
        return (n + wg - 1) // wg

    def _epoch_units(self, epoch: int) -> list[tuple[int, int]]:
        """One epoch's row groups: every unweighted group once, then each weighted extra split's
        draws for this epoch, read in order from an endless chain of seeded permutations of its
        groups (a down-weighted split rotates through all of them across epochs; an up-weighted
        one repeats them evenly). Without weighted splits this is `self.units`, unchanged."""
        if self._node_plan is not None:
            from train.shard_sidecar import partition_items

            plan = self._node_plan
            base_u = [u for u in self.units if u[0] < self._first_extra]
            extra_u = [u for u in self.units if u[0] >= self._first_extra]
            base_part = partition_items(base_u, plan.node_world, weights=plan.weights,
                                        seed=self.seed ^ plan.seed, salt=epoch, preserve_order=True)[plan.node_rank]
            out = base_part + extra_u
            if not self._weighted:
                return out
            denom = 1.0 - self._weighted_share
            for k, ((groups, _), w) in enumerate(zip(self._weighted, self._weighted_weights, strict=True)):
                draws = max(1, round(w * len(base_part) / denom)) if base_part else 0
                first, n = epoch * draws, len(groups)
                for cyc in range(first // n, (first + draws - 1) // n + 1):
                    perm = np.random.default_rng([self.seed, 52, plan.node_rank, k, cyc]).permutation(n)
                    lo, hi = max(first, cyc * n) - cyc * n, min(first + draws, (cyc + 1) * n) - cyc * n
                    out.extend(groups[i] for i in perm[lo:hi])
            return out
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
        seed_seq = [self.seed, epoch, self._node_plan.node_rank] if self._node_plan is not None else [self.seed, epoch]
        perm = np.random.default_rng(seed_seq).permutation(len(units))
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

    def _apply_bpe_dropout(self, ex: Example, epoch: int) -> None:
        """Wave 2D BPE-Dropout: auto-disabled on frame-KD rows to keep teacher lattice alignment."""
        if self.bpe_dropout_p <= 0.0 or (ex.kd is not None and bool(ex.kd[3])):
            return
        import hashlib

        h = hashlib.sha256(f"bpe:{self.seed}:{epoch}:{ex.uid}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        ex.rnnt = self.feat.tok_rnnt.encode(ex.text, dropout_p=self.bpe_dropout_p,
                                            alpha=self.bpe_dropout_alpha, rng=rng)
        ex.ctc = self.feat.tok_ctc.encode(ex.text, dropout_p=self.bpe_dropout_p,
                                          alpha=self.bpe_dropout_alpha, rng=rng)

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
        if getattr(self, "bpe_dropout_p", 0.0) > 0.0:
            self._apply_bpe_dropout(ex, epoch)
        return ex

    def _attach(self, x, epoch: int = 0):
        """Process-path result -> Example in the main process (teacher lookup lives here)."""
        if x is None:
            return None
        uid, text, feats, rnnt, ctc, f = x
        ex = Example(uid, text, torch.from_numpy(feats), rnnt, ctc)
        if self.anchors is not None:  # H48c MFA anchors, keyed by (id, speed) like the teacher
            ex.anchor = self.anchors.token_anchors(uid, f, rnnt, text)
        if self.teacher is not None:
            ex.kd = self.teacher.get_checked(uid, f, rnnt, n_mel=feats.shape[0])
        if getattr(self, "bpe_dropout_p", 0.0) > 0.0:
            self._apply_bpe_dropout(ex, epoch)
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

    def _eff_rank_world(self) -> tuple[int, int]:
        if self._node_plan is not None:
            return self._node_plan.local_rank, self._node_plan.local_world
        return self.rank, self.world

    def _partition_groups(self, groups: list[list[int]], dur: list[float], epoch: int, window: int):
        rank, world = self._eff_rank_world()
        if world <= 1:
            return groups, [True] * len(groups)
        if self.ddp_global_batches:
            if self.rank_weights and self._node_plan is None:
                parts = [weighted_share(g, dur, self.rank_weights, rank) for g in groups]
            else:
                parts = [g[rank::world] for g in groups]
            active = [bool(g) for g in parts]
            parts = [g or [gg[0]] for g, gg in zip(parts, groups, strict=True)]
            return parts, active
        n = len(groups)
        if n % world:
            target = n + world - n % world
            groups = (groups * -(-target // max(1, n)))[:target]
        parts = (stratified_parts(groups, dur, world, rank, [self.seed, epoch, window])
                 if self.stratify_lengths else groups[rank::world])
        return parts, [True] * len(parts)

    def _decode_groups(self, mine: list[list[dict]], epoch: int) -> list[list[Example]]:
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
        return out

    def _pack_window_sidecar(self, units: list[tuple[int, int]], epoch: int, window: int,
                             *, count_conf: bool = True):
        """Pack one window from precomputed per-row FLAC frame counts with zero audio reads."""
        locs: list[tuple[int, int, str]] = []
        frames: list[int | None] = []
        for ui, (fi, rg) in enumerate(units):
            rg_rows = self._dur_table.row_group(fi, rg)
            filt = self.min_conf > 0 and fi >= self._first_extra
            confs = None
            if filt and any(c is None for _, _, c in rg_rows):
                confs = pq.ParquetFile(self.files[fi]).read_row_group(rg, columns=["conf"]).column("conf").to_pylist()
            for ri, (uid, n_samples, c_side) in enumerate(rg_rows):
                if filt:
                    c_val = confs[ri] if confs is not None else c_side
                    if c_val is None or c_val < self.min_conf:
                        if count_conf:
                            self.dropped_conf += 1
                        continue
                locs.append((ui, ri, uid))
                frames.append(self._dur_table.frames_after(
                    uid, n_samples, epoch, seed=self.seed, speed_perturb=self.speed_perturb,
                    max_seconds=self.max_seconds, num_frames_fn=self.feat.logmel.num_frames))
        keep = [i for i, f in enumerate(frames) if f is not None]
        rng = np.random.default_rng([self.seed, epoch, window])
        dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
        if self.pad_frames > 1:
            dur = [_round_up(int(d * 100), self.pad_frames) / 100.0 for d in dur]
        groups = pack(dur, *self._shape(epoch, window))
        rng.shuffle(groups)
        return locs, keep, dur, groups

    def _read_selected_rows(self, units: list[tuple[int, int]], locs: list[tuple[int, int, str]],
                            keep: list[int], parts: list[list[int]]) -> list[list[dict]]:
        by_rg: dict[tuple[int, int], set[int]] = {}
        for g in parts:
            for i in g:
                ui, ri, _ = locs[keep[i]]
                by_rg.setdefault(units[ui], set()).add(ri)
        cache: dict[tuple[int, int, int], dict] = {}
        for (fi, rg), ridxs in by_rg.items():
            sorted_idx = sorted(ridxs)
            tbl = pq.ParquetFile(self.files[fi]).read_row_group(rg, columns=["id", "text", "audio"]).take(sorted_idx)
            for ri, row in zip(sorted_idx, tbl.to_pylist(), strict=True):
                cache[(fi, rg, ri)] = row
        mine: list[list[dict]] = []
        for g in parts:
            grp: list[dict] = []
            for i in g:
                ui, ri, uid = locs[keep[i]]
                fi, rg = units[ui]
                row = cache[(fi, rg, ri)]
                if row["id"] != uid:
                    raise DataError(f"duration sidecar uid mismatch at {self.files[fi]} rg={rg} row={ri}: "
                                    f"sidecar={uid!r} parquet={row['id']!r}")
                grp.append(row)
            mine.append(grp)
        return mine

    def _load_window_sidecar(self, units: list[tuple[int, int]], epoch: int, window: int,
                             skip: int = 0) -> list:
        locs, keep, dur, groups = self._pack_window_sidecar(units, epoch, window)
        self.dropped_long += len(locs) - len(keep)
        parts, active = self._partition_groups(groups, dur, epoch, window)
        if skip:
            parts, active = parts[skip:], active[skip:]
        out = self._decode_groups(self._read_selected_rows(units, locs, keep, parts), epoch)
        if self.ddp_global_batches and self.world > 1:
            return list(zip(out, active, strict=True))
        return out

    def _count_window_batches(self, epoch: int, window: int) -> int:
        """Metadata-only batch count for lockstep fast-forward on resume (zero audio decode)."""
        units = self._window_units(epoch, window)
        if self._dur_table is not None:
            _, _, dur, groups = self._pack_window_sidecar(units, epoch, window, count_conf=False)
        else:
            rows = [r for fi, rg in units for r in self._read_rows(fi, rg)]
            _, dur, groups = self._header_pack(rows, epoch, window)
        parts, _ = self._partition_groups(groups, dur, epoch, window)
        return len(parts)

    def _load_window(self, epoch: int, window: int, skip: int = 0) -> list[list[Example]]:
        units = self._window_units(epoch, window)
        if self._dur_table is not None:
            return self._load_window_sidecar(units, epoch, window, skip=skip)
        rows: list[dict] = []
        for fi, rg in units:
            rows.extend(self._read_rows(fi, rg))
        if self.world > 1 or self._node_plan is not None:
            return self._load_window_sharded(rows, epoch, window, skip=skip)
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
        groups = pack(dur, *self._shape(epoch, window))
        rng.shuffle(groups)
        return [[exs[i] for i in g] for g in groups[skip:]]

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

    def _header_pack(self, rows: list[dict], epoch: int, window: int):
        """(kept row indices, jittered durations, shuffled groups of positions in `kept`) for one window from
        header-only lengths: the same packing and shuffle as the single-process path."""
        frames = [self._frames_after(r, epoch) for r in rows]
        keep = [i for i, f in enumerate(frames) if f is not None]
        rng = np.random.default_rng([self.seed, epoch, window])
        dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
        if self.pad_frames > 1:
            dur = [_round_up(int(d * 100), self.pad_frames) / 100.0 for d in dur]
        groups = pack(dur, *self._shape(epoch, window))
        rng.shuffle(groups)
        return keep, dur, groups

    def _shape(self, epoch: int, window: int) -> tuple[float, int]:
        """(budget_s, max_batch) for packing one window: the carried older shape for that one window, else the
        current one."""
        if self.carry_shape is not None and tuple(self.carry_shape[1]) == (epoch, window):
            budget_s, max_batch = self.carry_shape[0]
        else:
            budget_s, max_batch = self.budget_s, self.max_batch
        if self._node_plan is not None and self.ddp_global_batches:
            gshare = self._node_plan.gpu_share(self.world)
            return budget_s * gshare, max(1, round(max_batch * gshare))
        return budget_s, max_batch

    def _header_groups(self, rows: list[dict], epoch: int, window: int) -> list[list[int]]:
        """One window's global groups as lists of indices into `rows` (H53 resize receipts)."""
        keep, _, groups = self._header_pack(rows, epoch, window)
        return [[keep[i] for i in g] for g in groups]

    def _load_window_sharded(self, rows: list[dict], epoch: int, window: int, skip: int = 0) -> list[list[Example]]:
        """Data-parallel window: pack on header-only durations (the same packing and shuffle as
        the single-process path), keep groups g with g % world == rank (truncated to a multiple
        of world so every rank has the same number of batches), and decode only those rows."""
        keep, dur, groups = self._header_pack(rows, epoch, window)
        self.dropped_long += len(rows) - len(keep)
        parts, active = self._partition_groups(groups, dur, epoch, window)
        if skip:
            parts, active = parts[skip:], active[skip:]
        mine = [[rows[keep[i]] for i in g] for g in parts]
        out = self._decode_groups(mine, epoch)
        if self.ddp_global_batches and self.world > 1:
            return list(zip(out, active, strict=True))
        return out

    def _with_context(self, exs: list, epoch: int) -> list:
        if not self.stream_ctx_s:
            return exs
        from train.stream_ctx import ContextBank, add_context

        if self._ctx_bank is None:
            self._ctx_bank = ContextBank(self.feat.logmel)
        return add_context(exs, self._ctx_bank, self.seed, epoch, self.stream_ctx_s, self.stream_ctx_cap)

    def _collate_group(self, group, epoch: int) -> Batch:
        """One packed group -> Batch: stream context (if any), then collate(); DDP-global groups carry `active`."""
        if self.ddp_global_batches and self.world > 1:
            examples, active = group
            batch = collate(self._with_context(examples, epoch), self.pad_frames, self.pad_tokens)
            batch.active = active
            return batch
        return collate(self._with_context(group, epoch), self.pad_frames, self.pad_tokens)

    def iterate(self, start: Position):
        """Yield (position after this batch, Batch) forever, from `start`; one window prefetched.

        With collate_ahead = N > 0, one collator thread runs the same per-batch context + collate() calls, in
        the same order, up to N batches ahead of the consumer, so the trainer's step overlaps the next batch's
        collation instead of waiting on it. The batches are identical; N = 0 (the default) collates inline."""
        q: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def producer():
            if self._node_plan is not None:
                if start.epoch != 0 or start.window != 0:
                    raise DataError(f"node_shards resume expects Position(0, 0, step), got {start}")
                ep, w, rem_skip = 0, 0, start.batch
                while not stop.is_set():
                    if rem_skip > 0:
                        nb = self._count_window_batches(ep, w)
                        if rem_skip >= nb:
                            rem_skip -= nb
                            w += 1
                            if w >= self._windows(ep):
                                ep, w = ep + 1, 0
                            continue
                    item = (ep, w, self._load_window(ep, w, skip=rem_skip))
                    rem_skip = 0
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    w += 1
                    if w >= self._windows(ep):
                        ep, w = ep + 1, 0
                return
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
        # re-raised as a DataError as soon as the queue runs dry.
        from concurrent.futures import ThreadPoolExecutor

        runner = ThreadPoolExecutor(1, thread_name_prefix="batch-producer")
        fut = runner.submit(producer)

        def windows():
            while True:
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    if fut.done() and fut.exception() is not None:
                        err = fut.exception()
                        raise DataError(f"batch producer failed: {err!r}") from err
                    if stop.is_set():  # only once the consumer has closed this iterator
                        return
                    continue
                yield item

        def batches():
            first, step_idx = True, start.batch
            for ep, w, groups in windows():
                skip = (0 if self._node_plan is not None else start.batch) if first else 0
                first = False
                for bi in range(skip, len(groups)):
                    step_idx += 1
                    nxt = Position(0, 0, step_idx) if self._node_plan is not None else Position(ep, w, bi + 1)
                    yield nxt, self._collate_group(groups[bi], ep)

        crunner, ahead = None, int(getattr(self, "collate_ahead", 0) or 0)
        try:
            if ahead <= 0:
                yield from batches()
                return
            cq: queue.Queue = queue.Queue(maxsize=ahead)

            def collator():
                for item in batches():
                    while True:
                        if stop.is_set():
                            return
                        try:
                            cq.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue

            crunner = ThreadPoolExecutor(1, thread_name_prefix="batch-collator")
            cfut = crunner.submit(collator)
            while True:
                try:
                    item = cq.get(timeout=1.0)
                except queue.Empty:
                    if cfut.done():  # the collator stops only on an error (the stream is endless)
                        err = cfut.exception()
                        if isinstance(err, DataError):
                            raise err
                        if err is None:
                            raise DataError("batch collator stopped without an error") from None
                        raise DataError(f"batch collator failed: {err!r}") from err
                    continue
                yield item
        finally:
            stop.set()
            runner.shutdown(wait=False)
            if crunner is not None:
                crunner.shutdown(wait=False)


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
