#!/usr/bin/env python3
"""All-reduce bus-bandwidth probe on torch's bundled NCCL (the trainer's own NCCL, 2.26.2 here).

nccl-tests' all_reduce_perf on the cloud hosts links libmpi.so.40, which is not installed, so it
cannot start. This reports the same quantities as nccl-tests: time per op, algbw = bytes / time,
busbw = algbw * 2 (n - 1) / n. One process per GPU, ranks from the environment (RANK, WORLD_SIZE,
LOCAL_RANK, MASTER_ADDR, MASTER_PORT), as launch_ddp_ranks.sh sets them.

  nccl_bench.py --sizes-mb 1,4,16,64,116,232 --dtype bf16 --iters 20 --warmup 5 --tag default
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

KNOBS = ("NCCL_SOCKET_NTHREADS", "NCCL_NSOCKS_PERTHREAD", "NCCL_BUFFSIZE", "NCCL_MIN_NCHANNELS",
         "NCCL_MAX_NCHANNELS", "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
         "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL", "NCCL_CROSS_NIC")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mb", default="1,4,16,64,116,232")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verify", action="store_true", help="exact-sum check after each size (fill = rank + 1)")
    a = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    t0 = time.time()
    dist.init_process_group("nccl", device_id=dev)
    init_s = time.time() - t0
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    esz = 2 if a.dtype == "bf16" else 4
    rows = []
    for mb in [float(v) for v in a.sizes_mb.split(",") if v.strip()]:
        n = int(mb * (1 << 20)) // esz
        x = torch.ones(n, dtype=dt, device=dev)
        for _ in range(a.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize(dev)
        dist.barrier()
        times = []
        for _ in range(a.iters):
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            dist.all_reduce(x)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t)
        # every rank's median, then the max over ranks (the collective finishes with the slowest)
        med = sorted(times)[len(times) // 2]
        tt = torch.tensor([med, min(times)], dtype=torch.float64, device=dev)
        dist.all_reduce(tt, op=dist.ReduceOp.MAX)
        med, best = float(tt[0]), float(tt[1])
        nbytes = n * esz
        ok = None
        if a.verify:  # every element must equal 1 + 2 + ... + world exactly (small integers: exact in bf16)
            x.fill_(float(rank + 1))
            dist.all_reduce(x)
            want = world * (world + 1) / 2
            bad = int((x != want).sum())
            flags = torch.tensor([float(bad)], dtype=torch.float64, device=dev)
            dist.all_reduce(flags)
            ok = int(flags[0]) == 0
        algbw = nbytes / med / 1e9
        rows.append({"size_mb": mb, "time_ms": round(med * 1e3, 3), "best_ms": round(best * 1e3, 3),
                     "algbw_GBs": round(algbw, 3), "busbw_GBs": round(algbw * 2 * (world - 1) / world, 3),
                     "busbw_Gbit": round(algbw * 2 * (world - 1) / world * 8, 2),
                     **({"exact": ok} if ok is not None else {})})
        del x
    if rank == 0:
        print(json.dumps({"tag": a.tag, "world": world, "host": socket.gethostname(), "dtype": a.dtype,
                          "nccl": ".".join(map(str, torch.cuda.nccl.version())), "init_s": round(init_s, 2),
                          "knobs": {k: os.environ[k] for k in KNOBS if k in os.environ},
                          "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
