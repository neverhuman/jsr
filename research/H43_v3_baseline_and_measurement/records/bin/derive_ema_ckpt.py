#!/usr/bin/env python3
"""eval-lane helper: write a weights-only checkpoint whose 'model' is the source checkpoint's EMA weights.

  derive_ema_ckpt.py <source ckpt.pt> <out.pt>

The engine converter (tools/r1_to_safetensors.py convert) and train.export read p["model"] only; this file lets the
unmodified tools export the EMA weights. Keeps model_config, train_config, step, tokenizer_sha256; records the
source sha256. Writes <out>.sha256 (the completion marker train.checkpoint.load requires). Never overwrites.
"""
import hashlib, json, os, sys
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


src, dst = sys.argv[1:3]
if os.path.exists(dst):
    sys.exit(f"{dst} exists; never overwrite")
p = torch.load(src, map_location="cpu", weights_only=False)
if p.get("ema") is None:
    sys.exit("source has no EMA weights")
if sorted(p["ema"]) != sorted(p["model"]):
    sys.exit("EMA and model state dicts have different keys")
q = {k: p[k] for k in ("model_config", "train_config", "step", "tokenizer_sha256") if k in p}
q["model"] = p["ema"]
q["derived_from"] = {"checkpoint": os.path.abspath(src), "checkpoint_sha256": sha(src), "weights": "ema",
                     "ema_decay": (p.get("train_config") or {}).get("ema_decay"), "tool": "eval-lane derive_ema_ckpt.py"}
tmp = dst + ".tmp"
torch.save(q, tmp)
os.replace(tmp, dst)
d = sha(dst)
with open(dst[:-3] + ".sha256", "w") as fh:
    fh.write(d + "\n")
print(json.dumps({"out": dst, "sha256": d, **q["derived_from"], "step": q.get("step")}))
