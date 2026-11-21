"""R4 ternary bundle (`ultra.r4-ternary/v0`, trainer `train/export_ternary.py`) -> engine
weight directory. Used by `r1_to_safetensors.py ternary`; kept separate to keep files small.

Ternary layers become exact int8 codes {-1, 0, +1} in the engine's prepared-int8 panel layout
(scale = s_n, wsum = s_n * sum t, `codes: "ternary"`), so the engine's per-row centred int8
quantiser and fused epilogue run the trainer's QAT forward rule; the fp32 remainder and the
ternary biases go to weights.safetensors.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from r1_to_safetensors import (SCHEMA, ConvertError, sha256_file, source_identity,
                               write_safetensors)

# Must equal ultra-engine `model::prepared::INT8_LAYOUT` (the loader refuses anything else).
INT8_LAYOUT = ("ultra-engine/packed-i8 v1: panels NR=16, K groups of 4, per-channel symmetric "
               "weights, wsum")
R4_SCHEMA = "ultra.r4-ternary/v0"
_TAIL = {"ff1.w1": "ff1_w1", "ff1.w2": "ff1_w2", "att.qkv": "qkv", "att.out": "att_out",
         "conv.pw1": "pw1", "conv.pw2": "pw2", "ff2.w1": "ff2_w1", "ff2.w2": "ff2_w2"}


def engine_name(torch_name: str, enc: dict) -> str:
    """Engine layer name (`Model::linears_named`) of a ternary torch Linear."""
    special = {"encoder.pair": "pair", "encoder.restore": "restore",
               "joiner.enc_proj": "enc_proj", "joiner.out": "joiner_out"}
    if torch_name in special:
        return special[torch_name]
    parts = torch_name.split(".")
    group, idx, tail = parts[1], int(parts[2]), ".".join(parts[3:])
    lower = enc["lower_blocks"] if enc.get("family") == "multirate" else enc["n_blocks"]
    base = {"blocks": 0, "mid_blocks": lower,
            "final_blocks": lower + enc.get("middle_blocks", 0)}[group]
    return f"blocks.{base + idx}.{_TAIL[tail]}"


def unpack_t1(packed: np.ndarray, k: int) -> np.ndarray:
    codes = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(
        packed.shape[0], -1)
    if (codes == 3).any():
        raise ConvertError("T1 code 3 in packed trits")
    return codes[:, :k].astype(np.int8) - 1


def pack_panels(codes: np.ndarray) -> np.ndarray:
    """(n, k) int8 codes -> the engine's int8 panel layout [panel][k/4][16 cols][4]."""
    n, k = codes.shape
    kp, panels = (k + 3) // 4 * 4, (n + 15) // 16
    c = np.zeros((panels * 16, kp), dtype=np.int8)
    c[:n, :k] = codes
    return np.ascontiguousarray(c.reshape(panels, 16, kp // 4, 4).transpose(0, 2, 1, 3)).reshape(-1)


def ternary(bundle: str, assets: str, out: str) -> dict:
    """`ultra.r4-ternary/v0` bundle -> engine directory: ternary layers as exact int8 codes in
    a prepared-int8 artifact (scale = s_n, wsum = s_n * sum t), fp32 remainder in
    weights.safetensors, tokens / BPE / filterbank / feature spec from `assets` (an
    `ultra.r1-weights/v1` directory of the same tokenizer)."""
    if os.path.exists(out) and os.listdir(out):
        raise ConvertError(f"{out} is not empty; outputs go to a fresh directory")
    with open(os.path.join(bundle, "manifest.json")) as fh:
        man = json.load(fh)
    if man.get("schema") != R4_SCHEMA:
        raise ConvertError(f"bundle schema {man.get('schema')} != {R4_SCHEMA}")
    blobs = {}
    for key in ("ternary", "fp32"):
        f = man["files"][key]
        path = os.path.join(bundle, f["path"])
        if sha256_file(path) != f["sha256"]:
            raise ConvertError(f"{f['path']} hash mismatch")
        blobs[key] = open(path, "rb").read()
    with open(os.path.join(assets, "arch.json")) as fh:
        arch = json.load(fh)
    cfg = man["model_config"]
    enc = {**{"post_block_norm": True, "family": "dense", "lower_blocks": 4, "middle_blocks": 10,
              "final_blocks": 2}, **cfg["encoder"]}
    fp32 = {}
    for t in man["fp32_tensors"]:
        raw = blobs["fp32"][t["offset"]:t["offset"] + t["bytes"]]
        fp32[t["name"]] = np.frombuffer(raw, dtype="<f4").reshape(t["shape"]).copy()
    dh = enc["d_model"] // enc["n_heads"]
    import torch

    fp32["engine.rope_inv_freq"] = (1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=torch.float32)
                                                     / dh))).numpy()
    tb = blobs["ternary"]
    q8 = {}
    for rec in man["ternary_layers"]:
        n, k = rec["out"], rec["in"]

        def part(key, dtype):
            o = rec[key]
            return np.frombuffer(tb[o["offset"]:o["offset"] + o["bytes"]], dtype=dtype)

        trits = unpack_t1(part("trits", np.uint8).reshape(n, rec["row_bytes"]), k)
        scale = part("scale", "<f4")
        bias = part("bias", "<f4")
        nm = engine_name(rec["name"], enc)
        panels = (n + 15) // 16
        pad = lambda v: np.concatenate([v.astype("<f4"), np.zeros(panels * 16 - n, "<f4")])
        tsum = trits.astype(np.int64).sum(axis=1).astype(np.float32)
        q8[f"{nm}.q"] = pack_panels(trits)
        q8[f"{nm}.scale"] = pad(scale)
        q8[f"{nm}.wsum"] = pad(tsum * scale.astype(np.float32))
        fp32[f"{rec['name']}.bias"] = bias.copy()
    ckpt = man.get("checkpoint", "")
    ckpt_sha = sha256_file(ckpt) if os.path.exists(ckpt) else man["files"]["ternary"]["sha256"]
    os.makedirs(out, exist_ok=True)
    wpath = os.path.join(out, "weights.safetensors")
    write_safetensors(wpath, fp32, {"schema": SCHEMA, "checkpoint_sha256": ckpt_sha})
    wsha = sha256_file(wpath)
    ipath = os.path.join(out, "weights.int8.safetensors")
    write_safetensors(ipath, q8, {"layout": INT8_LAYOUT, "source_weights_sha256": wsha,
                                  "checkpoint_sha256": ckpt_sha})
    files = {"weights": {"path": "weights.safetensors", "sha256": wsha},
             "weights_int8": {"path": "weights.int8.safetensors", "sha256": sha256_file(ipath)}}
    for key in ("tokens", "bpe", "mel_fbank"):
        name = arch["files"][key]["path"]
        shutil.copyfile(os.path.join(assets, name), os.path.join(out, name))
        files[key] = {"path": name, "sha256": sha256_file(os.path.join(out, name))}
    arch["model"] = {"family": enc["family"], "config": cfg, "encoder": enc,
                     "checkpoint_step": int(man.get("checkpoint_step", -1)),
                     "tensors": len(fp32), "engine_tensors": ["engine.rope_inv_freq"]}
    arch["source"] = {"checkpoint": ckpt, "checkpoint_sha256": ckpt_sha,
                      "r4_bundle": os.path.abspath(bundle),
                      "r4_manifest_sha256": sha256_file(os.path.join(bundle, "manifest.json"))}
    arch["files"] = files
    arch["features"]["cmvn"] = {"mean": fp32["cmvn.mean"].tolist(), "istd": fp32["cmvn.istd"].tolist()}
    arch["int8"] = {"layout": INT8_LAYOUT, "source_weights_sha256": wsha,
                    "fp32_kinds": ["stem_c2", "stem_out"], "codes": "ternary",
                    "weights": "ternary {-1,0,+1} x per-output-channel absmean scale (R4 QAT); "
                               "codes stored in the int8 panel layout, exact",
                    "activations": "per-row centred dynamic int8 at inference time (not stored)",
                    "producer": "r1_to_safetensors.py ternary"}
    arch["converter"] = {"tool": "python/ultra/tools/r1_to_safetensors.py ternary",
                         **source_identity()}
    with open(os.path.join(out, "arch.json"), "w") as fh:
        json.dump(arch, fh, indent=1)
    return {"out": out, "ternary_layers": len(man["ternary_layers"]), "files": files}
