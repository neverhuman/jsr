"""CPU/GPU training-throughput probe: audio-hours trained per wall-hour for a preset.

  taskset -c <cpus> python -m train.bench --preset S --threads 32 --steps 12 --budget-s 200

Runs real optimiser steps (SpecAugment on, the same code path as train.trainer) on
train-clean-100 batches and prints one JSON line per step plus a summary. Batches are fetched
before timing starts, so the number is model compute only; data loading runs on a separate
thread in real training.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import torch

from train.data import Featurizer, ParquetBatches, Position
from train.tokenizer import Tokenizer, tokenizer_path
from train.trainer import TrainConfig, Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", default="S")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--budget-s", type=float, default=200.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--run-dir", default=os.path.join(
        os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr"), "runs", "train", "bench"))
    ap.add_argument("--out", default=None, help="append the summary JSON line to this file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    feat = Featurizer(Tokenizer(tokenizer_path(4096)), Tokenizer(tokenizer_path(1024)))
    cfg = TrainConfig(preset=a.preset, budget_s=a.budget_s, eval_every=0)
    src = ParquetBatches(["train-clean-100"], feat, seed=7, budget_s=a.budget_s,
                         max_batch=a.max_batch)
    batches = []
    for _, b in src.iterate(Position()):
        batches.append(b)
        if len(batches) >= a.steps + a.warmup:
            break
    tr = Trainer(a.run_dir, cfg, src, device=a.device, log=False)
    tr.model.cmvn.set_stats([b.feats[0, :b.feat_lens[0]] for b in batches])
    audio, wall = 0.0, 0.0
    for i, b in enumerate(batches):
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = tr.train_step(b)
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec = {"i": i, "utts": r["utts"], "audio_s": round(r["audio_s"], 1), "wall_s": round(dt, 3),
               "x_realtime": round(r["audio_s"] / dt, 2), "loss": round(r["loss"], 3)}
        print(json.dumps(rec), flush=True)
        if i >= a.warmup:
            audio, wall = audio + r["audio_s"], wall + dt
    summ = {"kind": "bench", "preset": a.preset, "params": tr.model.num_params(),
            "threads": a.threads, "device": a.device, "budget_s": a.budget_s,
            "steps": a.steps, "audio_s": round(audio, 1), "wall_s": round(wall, 2),
            "audio_h_per_wall_h": round(audio / wall, 2),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if a.device == "cuda" else None,
            "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
            if a.device == "cuda" else None,
            "affinity": sorted(os.sched_getaffinity(0)), "loadavg": os.getloadavg()}
    print(json.dumps(summ), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
