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


def source_identity() -> dict:
    """HEAD plus whether the converter/trainer sources differ from it, and the converter's own
    SHA-256, so a record binds the executed source rather than only the checkout HEAD."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".", "../train"],
                                        text=True, cwd=here).strip()
    except (OSError, subprocess.CalledProcessError):
        dirty = "unknown"
    return {"commit": git_commit(), "sources_dirty": dirty != "", "dirty_paths": dirty,
            "converter_sha256": sha256_file(os.path.abspath(__file__))}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_safetensors(path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """Little-endian F32 (int8 arrays as I8) safetensors writer (spec: u64 header length, JSON header, raw bytes)."""
    header: dict = {"__metadata__": metadata}
    off = 0
    blobs = []
    for name in sorted(tensors):
        src = tensors[name]
        if src.dtype == np.int8:
            a, dt = np.ascontiguousarray(src), "I8"
        else:
            a, dt = np.ascontiguousarray(src, dtype="<f4"), "F32"
        nbytes = a.nbytes
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + nbytes]}
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
    if os.path.exists(out) and os.listdir(out):
        raise ConvertError(f"{out} is not empty; outputs go to a fresh directory, never overwrite")
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
        "converter": {"tool": "python/ultra/tools/r1_to_safetensors.py", **source_identity(),
                      "torch": torch.__version__, "numpy": np.__version__},
    }
    apath = os.path.join(out, "arch.json")
    with open(apath + ".tmp", "w") as fh:
        json.dump(arch, fh, indent=1)
    os.replace(apath + ".tmp", apath)
    return arch


def _chunk_taps(ce, model, fs, tt, n, caches):
    """One chunk through the trainer's ChunkEncoder(MR) arithmetic, one block at a time
    (ChunkEncoder._run_blocks on single-layer cache slices computes exactly what the
    all-layer call computes: the mask and positions depend only on the input kv_len).
    Returns (taps, final encoder output)."""
    import torch
    import torch.nn.functional as F

    e = model.encoder
    c = e.cfg
    taps = []

    def run(blocks, x, pos0, k, v, kvl, conv):
        for li, blk in enumerate(blocks):
            x, *_ = ce._run_blocks([blk], x, pos0, k[li:li + 1], v[li:li + 1], kvl,
                                   conv[li:li + 1])
            taps.append(x[0])
        return x

    x = ce._stem(fs, tt, n)
    taps.append(x[0])
    if c.family == "dense":
        k, v, kvl, conv = caches
        x = run(e.blocks, x, tt, k, v, kvl, conv)
    else:
        k, v, kvl, conv, km, vm, kvm, convm = caches
        nl = c.lower_blocks
        x = run(e.blocks, x, tt, k[:nl], v[:nl], kvl, conv[:nl])
        skip = x
        b, _, d = x.shape
        xm = F.pad(x, (0, 0, 0, 1))[:, :2 * ((n + 1) // 2)]
        m = e.pair(xm.reshape(b, -1, 2 * d))
        m = run(e.mid_blocks, m, torch.div(tt, 2, rounding_mode="floor"), km, vm, kvm, convm)
        x = e.restore(m).reshape(b, -1, d)[:, :n] + skip
        x = run(e.final_blocks, x, tt, k[nl:], v[nl:], kvl, conv[nl:])
    x = e.final_norm(x)
    taps.append(x[0])
    taps.append(model.joiner.enc_proj(x)[0])
    return taps, x


def reference(ckpt: str, out: str, split: str, limit: int, start: int) -> dict:
    """Per-layer streaming outputs of the trainer's ChunkEncoder(MR) arithmetic (torch fp32)."""
    import torch

    from train.asr import ASRModel, ModelConfig
    from train.features import LogMel
    from train.infer import ChunkEncoder, ChunkEncoderMR, cache_shapes
    from train.model import stem_out_len
    from ultra.libri import load_split

    if os.path.exists(out):
        raise ConvertError(f"{out} exists; never overwrite")
    torch.set_num_threads(1)
    p = load_checkpoint(ckpt)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval()
    c = model.cfg.encoder
    mr = c.family == "multirate"
    ce = (ChunkEncoderMR if mr else ChunkEncoder)(model.encoder).eval()
    sh = cache_shapes(c)
    logmel = LogMel()
    utts = load_split(split, start + limit)[start:]
    tensors: dict[str, np.ndarray] = {}
    ids = []

    def z(key):
        L, *rest = sh[key]
        return torch.zeros(L, 1, *rest)

    with torch.no_grad():
        for ui, u in enumerate(utts):
            pcm = torch.from_numpy(u.pcm.astype(np.float32) / 32768.0)
            feats = model.cmvn(logmel(pcm))  # (F, 80) normalised
            total = int(stem_out_len(feats.shape[0]))
            caches = [z("k"), z("k"), torch.zeros(1, dtype=torch.long), z("conv")]
            if mr:
                caches += [z("km"), z("km"), torch.zeros(1, dtype=torch.long), z("convm")]
            layers: list[list[torch.Tensor]] = []
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
                ref = ce(fs, tt, *caches)
                taps, x = _chunk_taps(ce, model, fs, tt, n, caches)
                if not layers:
                    layers = [[] for _ in taps]
                for li, tp in enumerate(taps):
                    layers[li].append(tp)
                check.append(float((x - ref[0]).abs().max()))
                caches = list(ref[1:])
                t0 += n
            if max(check) != 0.0:
                raise ConvertError(f"recorded path differs from ChunkEncoder by {max(check)}")
            tensors[f"u{ui:03d}.feats"] = feats.numpy()
            for li, rows in enumerate(layers):
                tensors[f"u{ui:03d}.layer{li:02d}"] = torch.cat(rows, 0).numpy()
            ids.append(u.uid)
    nb = (c.lower_blocks + c.middle_blocks + c.final_blocks) if mr else c.n_blocks
    meta = {"schema": "ultra.r1-reference/v1", "checkpoint_sha256": p["_sha256"],
            "split": split, "start": str(start), "count": str(len(ids)), "ids": json.dumps(ids),
            "family": c.family,
            "layers": f"00 stem, 01..{nb:02d} blocks in execution order (multirate middle "
                      f"blocks at 12.5 Hz), {nb + 1:02d} final_norm, {nb + 2:02d} enc_proj",
            "source": json.dumps(source_identity())}
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
    tn = sub.add_parser("ternary")
    tn.add_argument("--bundle", required=True)
    tn.add_argument("--assets", required=True)
    tn.add_argument("--out", required=True)
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
    elif a.cmd == "ternary":
        from r1_ternary import ternary

        print(json.dumps(ternary(a.bundle, a.assets, a.out), indent=1))
    else:
        print(json.dumps(reference(a.checkpoint, a.out, a.split, a.limit, a.start), indent=1))


if __name__ == "__main__":
    main()
