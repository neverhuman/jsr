#!/usr/bin/env python3
"""Synthetic DDP step-time probe on the real S model (A1 shapes): exposed gradient-communication time.

Per config: the same model, the same synthetic batch per rank (random log-mel, random 500-piece
targets), native DDP exactly as trainer.py builds it (_ForwardLoss wrapper, broadcast_buffers=False,
gradient_as_bucket_view=True, optional bf16_compress_hook), bf16 autocast. Times
  t_sync   : forward + backward with the gradient all-reduce (what a DDP step pays)
  t_nosync : the same under ddp.no_sync() (no collective)
exposed comm = t_sync - t_nosync. Ranks from the environment (torchrun). Prints one JSON line (rank 0).

  torchrun --nproc_per_node 2 ddp_step_probe.py --code DIR --per-rank-s 375 --steps 30 --tag x
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="python/ultra directory (pinned snapshot)")
    ap.add_argument("--per-rank-s", type=float, default=375.0, help="audio seconds per rank per step")
    ap.add_argument("--utt-s", type=float, default=12.5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--bf16-comm", action="store_true")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from train.asr import ASRModel
    from train.trainer import TrainConfig, _ForwardLoss, model_config_for

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    cfg = TrainConfig(preset="S", rnnt_pieces=500, ctc_pieces=500, rnnt_loss="pruned", prune_range=5)
    torch.manual_seed(1234)
    model = ASRModel(model_config_for(cfg)).to(dev)
    ddp = DistributedDataParallel(_ForwardLoss(model), device_ids=[local], broadcast_buffers=False,
                                  gradient_as_bucket_view=True)
    if a.bf16_comm:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    g = torch.Generator().manual_seed(77 + rank)
    n = max(1, int(a.per_rank_s // a.utt_s))
    t_frames = int(a.utt_s * 100)
    feats = torch.randn(n, t_frames, 80, generator=g).to(dev)
    feat_lens = torch.full((n,), t_frames, dtype=torch.long, device=dev)
    u = int(a.utt_s * 3.8)
    rnnt = torch.randint(1, 501, (n, u), generator=g).to(dev)
    rnnt_lens = torch.full((n,), u, dtype=torch.long, device=dev)
    ctc, ctc_lens = rnnt.clone(), rnnt_lens.clone()

    def step(sync: bool) -> float:
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        ctx = ddp.no_sync() if not sync else _Null()
        with ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = ddp(feats, feat_lens, rnnt, rnnt_lens, ctc, ctc_lens, chunk=a.chunk)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t
        opt.step()
        return dt

    res = {}
    for mode in ("sync", "nosync", "sync"):
        for _ in range(a.warmup):
            step(mode == "sync")
        ts = sorted(step(mode == "sync") for _ in range(a.steps))
        med = ts[len(ts) // 2]
        t = torch.tensor([med], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        res.setdefault(mode, []).append(float(t[0]))
    t_sync = min(res["sync"])  # two sync passes bracket the no-sync pass (drift check)
    t_nosync = res["nosync"][0]
    if rank == 0:
        print(json.dumps({
            "tag": a.tag, "world": world, "host": socket.gethostname(), "per_rank_s": n * a.utt_s,
            "utts_per_rank": n, "bf16_comm": a.bf16_comm, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "t_sync_ms": round(t_sync * 1e3, 1), "t_sync_passes_ms": [round(v * 1e3, 1) for v in res["sync"]],
            "t_nosync_ms": round(t_nosync * 1e3, 1), "exposed_comm_ms": round((t_sync - t_nosync) * 1e3, 1),
            "exposed_pct": round(100 * (t_sync - t_nosync) / t_sync, 1),
            "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
    dist.destroy_process_group()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
