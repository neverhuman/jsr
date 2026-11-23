#!/usr/bin/env python3
"""Write a copy of an H41 checkpoint that eval-lane's pinned d5f231e evaluator can load.

  h41_compat_ckpt.py <model-00008000.pt> <out.pt>     (prints one JSON line; writes <out>.sha256 only on success)

Checkpoints from the current tree carry config keys that d5f231e's ModelConfig / EncoderConfig do not know. They
are dropped only when they hold the values that reproduce d5f231e's architecture and behaviour. The two multi-rate
block counts (deep_blocks, up_blocks) are never read by a dense encoder, and evaluate.py's strict load_state_dict
still rejects any parameter mismatch. Any other value (e.g. arm 16's pred_ctx 4) -> {"compat": false}: no file.
"""
import hashlib, json, os, sys
import torch

NEUTRAL_ENC = {"kv_heads": 0, "ffn_act": "silu", "stem_type": "conv2d", "qk_rmsnorm": False, "norm_affine": True,
               "layer_scale_init": 0.0, "stage_left": (), "inter_ctc_vocab": 0, "layer_drop": 0.0}
UNUSED_ENC = ("deep_blocks", "up_blocks")
NEUTRAL_MODEL = {"pred_ctx": 2, "inter_ctc": False, "text_tables_dim": 0}


def norm(v):
    return tuple(v) if isinstance(v, list) else v


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


src, dst = sys.argv[1:3]
p = torch.load(src, map_location="cpu", weights_only=False)
mc = dict(p["model_config"])
enc = dict(mc["encoder"])
bad = {k: enc[k] for k, v in NEUTRAL_ENC.items() if k in enc and norm(enc[k]) != v}
bad.update({k: mc[k] for k, v in NEUTRAL_MODEL.items() if k in mc and mc[k] != v})
if bad:
    print(json.dumps({"compat": False, "non_default": {k: list(v) if isinstance(v, tuple) else v for k, v in bad.items()}}))
    sys.exit(0)
dropped = {k: enc.pop(k) for k in list(NEUTRAL_ENC) + list(UNUSED_ENC) if k in enc}
dropped.update({k: mc.pop(k) for k in NEUTRAL_MODEL if k in mc})
mc["encoder"] = enc
q = dict(p, model_config=mc, compat_from={"checkpoint": os.path.abspath(src), "checkpoint_sha256": sha(src)})
torch.save(q, dst + ".tmp")
os.replace(dst + ".tmp", dst)
d = sha(dst)
with open(dst[:-3] + ".sha256", "w") as fh:
    fh.write(d + "\n")
print(json.dumps({"compat": True, "sha256": d, **q["compat_from"],
                  "dropped": {k: list(v) if isinstance(v, tuple) else v for k, v in dropped.items()}}))
