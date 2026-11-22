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


def linearise_simple_heads(model: ASRModel) -> None:
    """Initialise the simple joiner from the trained joiner with tanh linearised (tanh(x) ~ x):
    simple_am(enc) = W_out (P_enc enc + b_enc) + b_out and simple_lm(pred) = W_out pred, so the
    pruning band starts where the real joiner puts its mass (instead of from random heads)."""
    j = model.joiner
    w_out = j.out.weight.detach().float()
    with torch.no_grad():
        model.simple_am.weight.copy_(w_out @ j.enc_proj.weight.detach().float())
        model.simple_am.bias.copy_(w_out @ j.enc_proj.bias.detach().float() + j.out.bias.detach().float())
        model.simple_lm.weight.copy_(w_out)
        model.simple_lm.bias.zero_()


def make_seed(src: str, out: str, rnnt_loss: str, prune_range: int, seed: int,
              linearise: bool = True) -> dict:
    p = checkpoint.load(src, restore_rng=False)
    cfg = dataclasses.replace(ModelConfig.from_dict(p["model_config"]), rnnt_loss=rnnt_loss,
                              prune_range=prune_range)
    torch.manual_seed(seed)
    model = ASRModel(cfg)
    missing, unexpected = model.load_state_dict(p["model"], strict=False)
    allowed = {"simple_am.weight", "simple_am.bias", "simple_lm.weight", "simple_lm.bias"}
    if unexpected or not set(missing) <= allowed:
        raise CheckpointError(f"unexpected state difference: missing={missing} unexpected={unexpected}")
    if linearise and rnnt_loss == "pruned":
        linearise_simple_heads(model)
    payload = {"model": model.state_dict(), "model_config": cfg.to_dict(), "step": p.get("step", 0),
               "seed_from": {"path": os.path.abspath(src), "sha256": open(src[:-3] + ".sha256").read().strip(),
                             "added": sorted(missing), "simple_heads": "linearised" if linearise else "random"},
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
    ap.add_argument("--random-heads", action="store_true", help="random simple heads (default: linearised)")
    a = ap.parse_args(argv)
    if not a.out.endswith(".pt") or os.path.abspath(a.out) == os.path.abspath(a.src):
        ap.error("--out must be a new .pt path (the source is never overwritten)")
    pl = make_seed(a.src, a.out, a.rnnt_loss, a.prune_range, a.seed, linearise=not a.random_heads)
    print({"out": a.out, "step": pl["step"], "added": pl["seed_from"]["added"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
