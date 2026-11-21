"""R1 trainer checkpoint -> `weights.safetensors` + `arch.json` (schema `ultra.r1-weights/v1`).

The pure-Rust engine (`crates/ultra-engine`) reads only this directory: no pickle, no ONNX.

  convert    python tools/r1_to_safetensors.py convert --checkpoint <ckpt.pt> --out <dir>
                 [--bundle <ultra.r1-export/v1 dir>]
  reference  python tools/r1_to_safetensors.py reference --checkpoint <ckpt.pt> --out <file>
                 [--split dev-clean --limit 50]

`convert` writes every state-dict tensor as little-endian F32 in a hand-written safetensors file
(8-byte LE header length, JSON header padded with spaces to a multiple of 8, raw data; tensors in
sorted name order, contiguous, so every offset is a multiple of 4). `arch.json` records the model
configuration, the streaming and greedy rules, the feature spec (copied from the bundle manifest
when `--bundle` is given; CMVN always from the checkpoint itself), the tokens/BPE/filterbank files
(copied next to the weights) and the SHA-256 of the source checkpoint and of every output file.
Checkpoints and bundles are only read.

`reference` runs the trainer's own streaming ChunkEncoder arithmetic chunk by chunk (torch fp32)
on DEV utterances and writes per-layer outputs (stem, every block, final norm, joiner projection)
plus the normalised features to one safetensors file, for the engine's per-layer parity test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SCHEMA = "ultra.r1-weights/v1"
BUNDLE_SCHEMAS = ("ultra.r1-export/v1", "ultra.r1-export/v1.1")
COPIED = {"tokens": "tokens.txt", "bpe": "bpe.model", "mel_fbank": "mel_fbank.f32"}


class ConvertError(Exception):
    """The checkpoint, bundle and requested output do not form a consistent weight set."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_safetensors(path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """Little-endian F32 safetensors writer (spec: u64 header length, JSON header, raw bytes)."""
    header: dict = {"__metadata__": metadata}
    off = 0
    blobs = []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name], dtype="<f4")
        nbytes = a.size * 4
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + nbytes]}
        blobs.append(a)
        off += nbytes
    hj = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hj += b" " * ((-len(hj)) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        for a in blobs:
            fh.write(a.tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Reader used for the round-trip self-check (F32 only)."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        data = fh.read()
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ConvertError(f"{name}: dtype {spec['dtype']} is not F32")
        a, b = spec["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype="<f4").reshape(spec["shape"])
    return out


def load_checkpoint(path: str) -> dict:
    """Trainer checkpoint (`ckpt-*.pt`, verified against its `.sha256` marker when present) or a
    weights-only snapshot (`model-*.pt`). Read-only."""
    import torch

    marker = path[: -len(".pt")] + ".sha256" if path.endswith(".pt") else None
    digest = sha256_file(path)
    if marker and os.path.exists(marker):
        with open(marker) as fh:
            want = fh.read().split()[0]
        if want != digest:
            raise ConvertError(f"checkpoint hash {digest} != marker {want}")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(p, dict) or "model" not in p or "model_config" not in p:
        raise ConvertError("checkpoint lacks 'model' or 'model_config'")
    p["_sha256"] = digest
    return p


def read_bundle(bundle: str, tokenizer_sha: str | None) -> dict:
    with open(os.path.join(bundle, "manifest.json")) as fh:
        man = json.load(fh)
    if man.get("schema") not in BUNDLE_SCHEMAS:
        raise ConvertError(f"bundle schema {man.get('schema')} not in {BUNDLE_SCHEMAS}")
    for key, name in COPIED.items():
        spec = man["files"].get(key)
        if spec is None or spec["path"] != name:
            raise ConvertError(f"bundle has no files.{key} = {name}")
        got = sha256_file(os.path.join(bundle, name))
        if got != spec["sha256"]:
            raise ConvertError(f"bundle {name} hash {got} != manifest {spec['sha256']}")
    bpe_sha = man["files"]["bpe"]["sha256"]
    if tokenizer_sha is not None and bpe_sha != tokenizer_sha:
        raise ConvertError(f"bundle bpe.model {bpe_sha} != checkpoint tokenizer {tokenizer_sha}")
    return man


def convert(ckpt: str, out: str, bundle: str | None) -> dict:
    p = load_checkpoint(ckpt)
    cfg = p["model_config"]
    enc = cfg["encoder"]
    if os.path.exists(os.path.join(out, "arch.json")):
        raise ConvertError(f"{out} already holds arch.json; never overwrite (use a new directory)")
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().to("cpu").float().numpy() for k, v in p["model"].items()}
    import torch

    # RoPE inverse frequencies exactly as torch computes them in float32 (train.model.rope_tables,
    # train.infer.ChunkEncoder._run_blocks), so the engine's angles pos * inv match bit for bit.
    dh = enc["d_model"] // enc["n_heads"]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
    sd["engine.rope_inv_freq"] = inv.numpy()
    meta = {"schema": SCHEMA, "checkpoint_sha256": p["_sha256"]}
    wpath = os.path.join(out, "weights.safetensors")
    write_safetensors(wpath, sd, meta)
    back = read_safetensors(wpath)
    if sorted(back) != sorted(sd) or any(not np.array_equal(back[k], sd[k]) for k in sd):
        raise ConvertError("safetensors round trip differs from the checkpoint")

    files = {"weights": {"path": "weights.safetensors", "sha256": sha256_file(wpath)}}
    features = None
    if bundle is not None:
        man = read_bundle(bundle, p.get("tokenizer_sha256"))
        for key, name in COPIED.items():
            shutil.copyfile(os.path.join(bundle, name), os.path.join(out, name))
            files[key] = {"path": name, "sha256": sha256_file(os.path.join(out, name))}
        features = dict(man["features"])
        features["cmvn_source"] = "checkpoint cmvn.mean / cmvn.istd (bundle values replaced)"
        bundle_rec = {"path": os.path.abspath(bundle),
                      "manifest_sha256": sha256_file(os.path.join(bundle, "manifest.json")),
                      "checkpoint_sha256": man["model"].get("checkpoint_sha256"),
                      "same_checkpoint": man["model"].get("checkpoint_sha256") == p["_sha256"]}
    else:
        bundle_rec = None
        features = {}
    features["cmvn"] = {"mean": sd["cmvn.mean"].tolist(), "istd": sd["cmvn.istd"].tolist()}

    arch = {
        "schema": SCHEMA,
        "model": {
            "family": enc.get("family", "dense"),
            "config": cfg,
            "encoder": {**{"post_block_norm": True, "family": "dense", "lower_blocks": 4,
                           "middle_blocks": 10, "final_blocks": 2}, **enc},
            "checkpoint_step": int(p.get("step", -1)),
            "tensors": len(sd),
            "engine_tensors": ["engine.rope_inv_freq"],
            "params": int(sum(v.size for v in sd.values())),
        },
        "source": {"checkpoint": os.path.abspath(ckpt), "checkpoint_sha256": p["_sha256"],
                   "tokenizer_sha256": p.get("tokenizer_sha256"), "bundle": bundle_rec},
        "files": files,
        "features": features,
        "streaming": {"encoder_rate_hz": 25, "subsampling": 4, "chunk_frames": enc["chunk"],
                      "left_context_frames": enc["left_context"],
                      "encoder_input_rule": "chunk [t0, t0+n) reads normalised feature frames "
                                            "4 t0 - 6 .. 4 (t0 + n) - 4 inclusive, zeros < 0",
                      "rope": "absolute encoder-frame positions, theta 10000, pairs (2i, 2i+1)"},
        "decoder": {"kind": "stateless", "context": 2, "blank_id": 0, "ctx_init": [0, 0],
                    "max_symbols": 8, "vocab_size": cfg["vocab_rnnt"],
                    "joiner_activation": cfg.get("joiner_act", "tanh")},
        "converter": {"tool": "python/ultra/tools/r1_to_safetensors.py", "commit": git_commit(),
                      "torch": torch.__version__, "numpy": np.__version__},
    }
    apath = os.path.join(out, "arch.json")
    with open(apath + ".tmp", "w") as fh:
        json.dump(arch, fh, indent=1)
    os.replace(apath + ".tmp", apath)
    return arch


def reference(ckpt: str, out: str, split: str, limit: int, start: int) -> dict:
    """Per-layer streaming outputs of the trainer's ChunkEncoder arithmetic (torch fp32)."""
    import torch
    import torch.nn.functional as F

    from train.asr import ASRModel, ModelConfig
    from train.features import LogMel
    from train.model import apply_rope, stem_out_len
    from train.infer import ChunkEncoder
    from ultra.libri import load_split

    if os.path.exists(out):
        raise ConvertError(f"{out} exists; never overwrite")
    torch.set_num_threads(1)
    p = load_checkpoint(ckpt)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval()
    c = model.cfg.encoder
    if c.family != "dense":
        raise ConvertError("reference dump supports the dense family")
    ce = ChunkEncoder(model.encoder).eval()
    logmel = LogMel()
    utts = load_split(split, start + limit)[start:]
    tensors: dict[str, np.ndarray] = {}
    ids = []
    H, dh, lc, K = c.n_heads, c.d_model // c.n_heads, c.left_context, c.conv_kernel
    with torch.no_grad():
        for ui, u in enumerate(utts):
            pcm = torch.from_numpy(u.pcm.astype(np.float32) / 32768.0)
            feats = model.cmvn(logmel(pcm))  # (F, 80) normalised
            n_feat = feats.shape[0]
            total = int(stem_out_len(n_feat))
            k = torch.zeros(c.n_blocks, 1, H, lc, dh)
            v = torch.zeros_like(k)
            kv_len = torch.zeros(1, dtype=torch.long)
            conv = torch.zeros(c.n_blocks, 1, K - 1, c.d_model)
            layers: list[list[torch.Tensor]] = [[] for _ in range(c.n_blocks + 3)]
            check = []
            t0 = 0
            while t0 < total:
                n = min(c.chunk, total - t0)
                lo, hi = 4 * t0 - 6, 4 * (t0 + n) - 4
                fs = torch.zeros(hi - lo + 1, c.n_mels)
                a = max(lo, 0)
                fs[a - lo:] = feats[a:hi + 1]
                fs = fs[None]
                tt = torch.tensor([t0])
                ref = ce(fs, tt, k, v, kv_len, conv)
                # Same arithmetic as ChunkEncoder._run_blocks, recording every block output.
                x = ce._stem(fs, tt, n)
                layers[0].append(x[0])
                inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
                pos = (tt[:, None] + torch.arange(n)[None]).float()
                ang = pos[..., None] * inv
                cos, sin = ang.cos()[:, None], ang.sin()[:, None]
                slot = torch.arange(lc)[None]
                key_ok = torch.cat([slot >= (lc - kv_len[:, None]),
                                    torch.ones(1, n, dtype=torch.bool)], dim=1)
                mask = key_ok[:, None, None, :]
                for li, blk in enumerate(model.encoder.blocks):
                    x = x + 0.5 * blk.ff1(x)
                    att = blk.att
                    q, kk, vv = att.qkv(att.norm(x)).view(1, n, 3, H, dh).permute(2, 0, 3, 1, 4)
                    q, kk = apply_rope(q, cos, sin), apply_rope(kk, cos, sin)
                    kall = torch.cat([k[li], kk], dim=2)
                    vall = torch.cat([v[li], vv], dim=2)
                    y = F.scaled_dot_product_attention(q, kall, vall, attn_mask=mask)
                    x = x + att.out(y.transpose(1, 2).reshape(1, n, -1))
                    cm = blk.conv
                    g = F.glu(cm.pw1(cm.norm(x)), dim=-1)
                    gall = torch.cat([conv[li], g], dim=1)
                    y = cm.dw(gall.transpose(1, 2)).transpose(1, 2)
                    x = x + cm.pw2(F.silu(cm.dnorm(y)))
                    x = x + 0.5 * blk.ff2(x)
                    x = blk.norm(x)
                    layers[li + 1].append(x[0])
                x = model.encoder.final_norm(x)
                layers[c.n_blocks + 1].append(x[0])
                layers[c.n_blocks + 2].append(model.joiner.enc_proj(x)[0])
                check.append(float((x - ref[0]).abs().max()))
                _, k, v, kv_len, conv = ref
                t0 += n
            if max(check) != 0.0:
                raise ConvertError(f"recorded path differs from ChunkEncoder by {max(check)}")
            tensors[f"u{ui:03d}.feats"] = feats.numpy()
            for li, rows in enumerate(layers):
                tensors[f"u{ui:03d}.layer{li:02d}"] = torch.cat(rows, 0).numpy()
            ids.append(u.uid)
    meta = {"schema": "ultra.r1-reference/v1", "checkpoint_sha256": p["_sha256"],
            "split": split, "start": str(start), "count": str(len(ids)), "ids": json.dumps(ids),
            "layers": "00 stem, 01..16 blocks, 17 final_norm, 18 joiner enc_proj",
            "commit": git_commit()}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_safetensors(out, tensors, meta)
    return {"out": out, "utterances": len(ids), "sha256": sha256_file(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv = sub.add_parser("convert")
    cv.add_argument("--checkpoint", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--bundle", default=None)
    rf = sub.add_parser("reference")
    rf.add_argument("--checkpoint", required=True)
    rf.add_argument("--out", required=True)
    rf.add_argument("--split", default="dev-clean")
    rf.add_argument("--start", type=int, default=0)
    rf.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "convert":
        arch = convert(a.checkpoint, a.out, a.bundle)
        print(json.dumps({"out": a.out, "files": arch["files"], "source": arch["source"]}, indent=1))
    else:
        print(json.dumps(reference(a.checkpoint, a.out, a.split, a.limit, a.start), indent=1))


if __name__ == "__main__":
    main()
