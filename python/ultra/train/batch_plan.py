"""Batch packing and data-parallel sharding plans (moved verbatim from train/data.py; data.py re-exports them)."""
from __future__ import annotations

import numpy as np

from train.errors import DataError


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
