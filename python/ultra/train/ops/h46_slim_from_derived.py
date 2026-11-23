"""H46: cut the decode-only checkpoint `train.lm_decode beam/encode` loads (the `slim` layout:
weights under "ema", "weights": "ema", "source_sha256" = the original checkpoint) from eval-lane's
derived "EMA-as-model" file, whose "model" tensors are the EMA tensors of the source checkpoint.

  cd python/ultra && python -m train.ops.h46_slim_from_derived <derived.pt> <out.pt> <source sha256>

The source sha256 must equal the one the derived file records ("derived_from"). Writes
<out>.sha256 (the completion marker train.checkpoint.load needs). Never overwrites.
"""
import hashlib
import json
import os
import sys

import torch

from train.errors import LmEvalError

src, dst, want = sys.argv[1:4]
if os.path.exists(dst):
    raise LmEvalError(f"{dst} exists; never overwrite")
p = torch.load(src, map_location="cpu", weights_only=False)
got = (p.get("derived_from") or {}).get("checkpoint_sha256")
if got != want or (p.get("derived_from") or {}).get("weights") != "ema":
    raise LmEvalError(f"derived file records source {got}, weights {p.get('derived_from')}; "
                      f"expected {want}, ema")
with open(src[:-3] + ".sha256") as fh:
    derived_sha = fh.read().strip()
keep = {"step": p["step"], "model_config": p["model_config"],
        "train_config": {"tokenizer": (p.get("train_config") or {}).get("tokenizer") or ""},
        "model": None, "ema": p["model"], "source_sha256": want, "weights": "ema",
        "derived_sha256": derived_sha}
torch.save(keep, dst + ".tmp")
os.replace(dst + ".tmp", dst)
with open(dst, "rb") as fh:
    digest = hashlib.sha256(fh.read()).hexdigest()
with open(dst[:-3] + ".sha256", "w") as fh:
    fh.write(digest + "\n")
print(json.dumps({"out": dst, "sha256": digest, "source_sha256": want,
                  "derived_sha256": derived_sha, "step": p["step"]}))
