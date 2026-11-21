"""Parameter averaging of checkpoints (finalist selection: average of the last K snapshots).

  python -m train.average --out <dir>/avg-last3.pt <model-...pt> <model-...pt> <model-...pt>

All inputs must share the model config, parameter names and shapes (checked); floating tensors
are averaged in float64 and cast back, other tensors must be identical. The result is written
like an eval-only snapshot (model, model_config, sources, sha256 marker) and loads anywhere a
checkpoint does (train.evaluate, train.export, train.infer).
"""
from __future__ import annotations

import argparse
import hashlib
import os

import torch

from train import checkpoint
from train.asr import ModelConfig
from train.errors import CheckpointError


def average(paths: list[str]) -> dict:
    if not paths:
        raise CheckpointError("no checkpoints to average")
    loaded = [checkpoint.load(p, restore_rng=False) for p in paths]
    cfgs = [ModelConfig.from_dict(p["model_config"]) for p in loaded]
    if any(c != cfgs[0] for c in cfgs):
        raise CheckpointError("checkpoints have different model configs")
    keys = list(loaded[0]["model"])
    for p, path in zip(loaded, paths, strict=True):
        if list(p["model"]) != keys:
            raise CheckpointError(f"parameter names differ in {path}")
        for k in keys:
            if p["model"][k].shape != loaded[0]["model"][k].shape:
                raise CheckpointError(f"shape of {k} differs in {path}")
    out = {}
    for k in keys:
        ts = [p["model"][k] for p in loaded]
        if torch.is_floating_point(ts[0]):
            out[k] = (sum(t.double() for t in ts) / len(ts)).to(ts[0].dtype)
        else:
            if any(not torch.equal(t, ts[0]) for t in ts):
                raise CheckpointError(f"non-float tensor {k} differs between checkpoints")
            out[k] = ts[0].clone()
    return {"model": out, "model_config": loaded[0]["model_config"],
            "averaged_from": [{"path": os.path.abspath(p), "step": q["step"]}
                              for p, q in zip(paths, loaded, strict=True)],
            "tokenizer_sha256": loaded[0].get("tokenizer_sha256")}


def save_average(paths: list[str], out_path: str) -> str:
    payload = average(paths)
    step = max(s["step"] for s in payload["averaged_from"])
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(dict(payload, step=step), fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    with open(out_path[:-3] + ".sha256", "w") as fh:
        fh.write(h + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("checkpoints", nargs="+")
    a = ap.parse_args()
    if not a.out.endswith(".pt"):
        raise SystemExit("--out must end in .pt")
    print(save_average(a.checkpoints, a.out))


if __name__ == "__main__":
    main()
