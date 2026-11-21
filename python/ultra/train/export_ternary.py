"""Export a ternary-QAT checkpoint as packed trits for the CPU engine (`ultra.r4-ternary/v0`).

  python -m train.export_ternary --checkpoint <ckpt.pt> --out <dir>

For every ternary layer (train.qat.ternary_targets) the bundle holds
  * trits in the T1 layout of `ultra-core::ternary` (four trits per byte, code = t + 1 in bits
    2k..2k+1, little end first; code 3 never appears); each output row is padded with zero
    trits to a multiple of 4 inputs, so a row is ceil(k / 4) bytes;
  * the per-output-channel scale s_n (float32) and the bias (float32, zeros if absent).
  y[r][n] = s_n * sum_k t[n][k] * x'[r][k] + b[n], where x' is the per-row centred int8
  quantisation of the input row (ultra-engine quant.rs), exactly the QAT forward pass.
All other tensors (stem, norms, depthwise kernels, embeddings, predictor, CMVN) are stored as
float32 in `fp32.bin`. `manifest.json` lists every tensor with its offset, shape, dtype and the
sha256 of both payload files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.qat import ternary_project, ternary_targets

SCHEMA = "ultra.r4-ternary/v0"


def pack_t1(trits: np.ndarray) -> np.ndarray:
    """(n, k) int8 trits -> (n, ceil(k/4)) uint8 in the ultra-core T1 layout."""
    n, k = trits.shape
    kp = (k + 3) // 4 * 4
    t = np.zeros((n, kp), dtype=np.int16)
    t[:, :k] = trits
    codes = (t + 1).astype(np.uint8).reshape(n, kp // 4, 4)
    return (codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4)
            | (codes[..., 3] << 6)).astype(np.uint8)


def unpack_t1(packed: np.ndarray, k: int) -> np.ndarray:
    codes = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(
        packed.shape[0], -1)
    if (codes == 3).any():
        raise ValueError("T1 code 3 in packed weights")
    return codes[:, :k].astype(np.int8) - 1


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def export(ckpt: str, out_dir: str) -> dict:
    p = checkpoint.load(ckpt, restore_rng=False)
    cfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(cfg)
    model.load_state_dict(p["model"])
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    tern = ternary_targets(model)
    mods = dict(model.named_modules())
    layers, tensors = [], []
    tpath, fpath = os.path.join(out_dir, "ternary.bin"), os.path.join(out_dir, "fp32.bin")
    covered = set()
    with open(tpath, "wb") as tf:
        for name in tern:
            lin = mods[name]
            t, s = ternary_project(lin.weight.detach().float())
            trits = t.numpy().astype(np.int8)
            packed = pack_t1(trits)
            if not np.array_equal(unpack_t1(packed, trits.shape[1]), trits):
                raise ValueError(f"pack round trip failed for {name}")
            bias = (lin.bias.detach().float().numpy() if lin.bias is not None
                    else np.zeros(trits.shape[0], dtype=np.float32))
            rec = {"name": name, "out": int(trits.shape[0]), "in": int(trits.shape[1]),
                   "row_bytes": int(packed.shape[1])}
            for key, arr in (("trits", packed), ("scale", s.numpy().reshape(-1).astype("<f4")),
                             ("bias", bias.astype("<f4"))):
                rec[key] = {"offset": tf.tell(), "bytes": int(arr.nbytes)}
                tf.write(arr.tobytes())
            rec["zero_fraction"] = round(float((trits == 0).mean()), 4)
            layers.append(rec)
            covered |= {f"{name}.weight", f"{name}.bias"}
    with open(fpath, "wb") as ff:
        for key, val in model.state_dict().items():
            if key in covered or not torch.is_floating_point(val):
                continue
            arr = val.detach().float().numpy().astype("<f4")
            tensors.append({"name": key, "shape": list(arr.shape), "offset": ff.tell(),
                            "bytes": int(arr.nbytes), "dtype": "float32"})
            ff.write(arr.tobytes())
    tbytes = os.path.getsize(tpath)
    manifest = {
        "schema": SCHEMA, "checkpoint": os.path.abspath(ckpt), "checkpoint_step": p["step"],
        "model_config": cfg.to_dict(),
        "files": {"ternary": {"path": "ternary.bin", "sha256": _sha(tpath), "bytes": tbytes},
                  "fp32": {"path": "fp32.bin", "sha256": _sha(fpath),
                           "bytes": os.path.getsize(fpath)}},
        "trit_layout": "ultra-core T1: 4 trits/byte, code = t + 1 at bits 2k..2k+1, rows padded "
                       "with zero trits to a multiple of 4",
        "linear_rule": "y[r][n] = scale[n] * sum_k t[n][k] * x'[r][k] + bias[n]; x' = per-row "
                       "centred int8 round trip of the input row (ultra-engine quant.rs)",
        "ternary_layers": layers, "fp32_tensors": tensors,
        "ternary_payload_mib": round(tbytes / 2**20, 3),
        "mean_zero_fraction": round(float(np.mean([r["zero_fraction"] for r in layers])), 4),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = export(a.checkpoint, a.out)
    print(json.dumps({"out": a.out, "ternary_layers": len(m["ternary_layers"]),
                      "ternary_payload_mib": m["ternary_payload_mib"],
                      "mean_zero_fraction": m["mean_zero_fraction"]}))


if __name__ == "__main__":
    main()
