#!/usr/bin/env python3
"""eval-lane: compat copy of a checkpoint for the PINNED d5f231e evaluator / exporter: drop model_config keys the pinned ModelConfig /
EncoderConfig do not know, ONLY when their value is the known identity default (the same set trainer-lane's compat copies drop);
any other unknown key or value -> refuse. Tensors untouched (verify with compat_check.py).
  make_compat.py SRC.pt OUT.pt        (writes OUT.sha256 and OUT.compat.json)"""
import dataclasses, hashlib, json, os, sys
SRC = "/dev/shm/eval-lane/src-d5f231e/python/ultra"; sys.path.insert(0, SRC)
import torch  # noqa: E402
from train.asr import ModelConfig, EncoderConfig  # noqa: E402
SAFE = {"kv_heads": 0, "ffn_act": "silu", "stem_type": "conv2d", "qk_rmsnorm": False, "norm_affine": True, "layer_scale_init": 0.0,
        "stage_left": [], "inter_ctc_vocab": 0, "deep_blocks": 8, "up_blocks": 2, "pred_ctx": 2, "inter_ctc": False}
norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v   # a tuple default and its JSON list are the same value
src, out = sys.argv[1:3]
p = torch.load(src, map_location="cpu", weights_only=False)
mc = p["model_config"]; enc_f = {f.name for f in dataclasses.fields(EncoderConfig)}; mc_f = {f.name for f in dataclasses.fields(ModelConfig)}
dropped = {}
for k in list(mc.get("encoder", {})):
    if k not in enc_f:
        v = mc["encoder"][k]
        if k not in SAFE or norm(v) != SAFE[k]:
            sys.exit(f"refuse: encoder.{k}={v!r} is unknown to the pinned evaluator and not an identity default")
        dropped[k] = norm(mc["encoder"].pop(k))
for k in list(mc):
    if k != "encoder" and k not in mc_f:
        v = mc[k]
        if k not in SAFE or norm(v) != SAFE[k]:
            sys.exit(f"refuse: {k}={v!r} is unknown to the pinned evaluator and not an identity default")
        dropped[k] = norm(mc.pop(k))
ModelConfig.from_dict(mc)                      # must construct now
torch.save(p, out)
h = hashlib.sha256(open(out, "rb").read()).hexdigest(); open(out[:-3] + ".sha256", "w").write(h + "\n")
rec = {"compat": True, "sha256": h, "checkpoint": os.path.abspath(src), "checkpoint_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(), "dropped": dropped, "tool": "eval-lane make_compat.py"}
json.dump(rec, open(out[:-3] + ".compat.json", "w")); print(json.dumps(rec))
