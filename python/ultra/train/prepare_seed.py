"""Turn a trained checkpoint into an --init-from seed for a different training-loss setting
(e.g. D2c: exact -> pruned RNN-T). The deployed modules are copied unchanged; only the
training-only simple-joiner heads of the pruned loss are added (freshly initialised). The
source file is never modified; the seed is written to a new path with a sha256 marker.

  python -m train.prepare_seed --src <ckpt or average .pt> --out <seed.pt> --rnnt-loss pruned [--prune-range 5] [--seed 1234]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.errors import CheckpointError


def make_seed(src: str, out: str, rnnt_loss: str, prune_range: int, seed: int) -> dict:
    p = checkpoint.load(src, restore_rng=False)
    cfg = dataclasses.replace(ModelConfig.from_dict(p["model_config"]), rnnt_loss=rnnt_loss,
                              prune_range=prune_range)
    torch.manual_seed(seed)
    model = ASRModel(cfg)
    missing, unexpected = model.load_state_dict(p["model"], strict=False)
    allowed = {"simple_am.weight", "simple_am.bias", "simple_lm.weight", "simple_lm.bias"}
    if unexpected or not set(missing) <= allowed:
        raise CheckpointError(f"unexpected state difference: missing={missing} unexpected={unexpected}")
    payload = {"model": model.state_dict(), "model_config": cfg.to_dict(), "step": p.get("step", 0),
               "seed_from": {"path": os.path.abspath(src), "sha256": open(src[:-3] + ".sha256").read().strip(),
                             "added": sorted(missing)},
               "tokenizer_sha256": p.get("tokenizer_sha256")}
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    with open(out[:-3] + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(open(out, "rb").read()).hexdigest() + "\n")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rnnt-loss", default="pruned", choices=("exact", "pruned"))
    ap.add_argument("--prune-range", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)
    if not a.out.endswith(".pt") or os.path.abspath(a.out) == os.path.abspath(a.src):
        ap.error("--out must be a new .pt path (the source is never overwritten)")
    pl = make_seed(a.src, a.out, a.rnnt_loss, a.prune_range, a.seed)
    print({"out": a.out, "step": pl["step"], "added": pl["seed_from"]["added"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
