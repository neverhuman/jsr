"""Function-preserving depth growth of a dense encoder (H53 capacity pilot, h53-trunk draft).

Turns a trained dense checkpoint (e.g. B0 = A1-49000, 16 blocks) into an --init-from seed for a
deeper dense model (e.g. 24 or 32 blocks, same width) that computes EXACTLY what the source computed:

* every original block, the stem, the final norm, CMVN, predictor, joiner, CTC head and the
  training-only simple heads are copied unchanged;
* each new block is inserted after an original block i. Its internal parameters (all LayerNorms,
  ff1.w1 / ff2.w1, att.qkv, conv.pw1 / dw / dnorm) are COPIED from block i, so their norms are on
  the trained scale (the qkv cap, 1.5x the step-100 row norms, then binds a new block exactly as it
  binds an old one), and its four residual output projections (ff1.w2, att.out, conv.pw2, ff2.w2:
  weight AND bias) are ZERO. With pre-norm modules and no post-block norm (post_block_norm=False,
  the S preset), such a block returns its input unchanged: x + 0.5*0 + 0 + 0 + 0.5*0 = x.
* the output projections get non-zero gradients at the first step (dL/dW2 = h^T dL/dy, h != 0), so
  the new blocks start learning at once (zero-init residual branches, as ReZero / Fixup / staged
  growth operators).

Refuses anything it cannot grow exactly: non-dense families, post-block norms, LayerScale.
The source file is never modified; the seed goes to a new path with a .sha256 marker, in the
same payload layout as train.prepare_seed (so train.trainer --init-from reads it).

  PYTHONPATH=<trainer worktree>/python/ultra python grow_depth.py --src <ckpt.pt> --out <seed.pt> \
      --blocks 24 [--weights model|ema]

The run that uses the seed must build the same config: --preset S --set encoder_blocks=24 (see
patches/0001-encoder-blocks-override.patch; check_init_compatible then passes unchanged).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import sys

import torch

# Residual output projections of a ConformerBlock (zeroed in a new block).
OUTPUT_PROJ = ("ff1.w2", "att.out", "conv.pw2", "ff2.w2")
_BLOCK = re.compile(r"^encoder\.blocks\.(\d+)\.(.+)$")


class GrowError(ValueError):
    pass


def insert_positions(n_src: int, n_dst: int) -> list[int]:
    """Original block index after which each new block goes, spread evenly over the depth:
    new block j (0-based, j < k = n_dst - n_src) follows original block floor((j + 1) n / k) - 1.
    16 -> 24: after 1, 3, ..., 15; 16 -> 32: after every block."""
    k = n_dst - n_src
    if n_src < 1 or k < 1 or k > n_src:
        raise GrowError(f"can grow {n_src} blocks to between {n_src + 1} and {2 * n_src}, not {n_dst}")
    return [(j + 1) * n_src // k - 1 for j in range(k)]


def layout(n_src: int, n_dst: int) -> list[tuple[int, bool]]:
    """Destination block list as (source block index, is_new) pairs, in depth order."""
    after = insert_positions(n_src, n_dst)
    out: list[tuple[int, bool]] = []
    for i in range(n_src):
        out.append((i, False))
        out.extend((i, True) for _ in range(after.count(i)))
    return out


def check_growable(enc) -> None:
    if enc.family != "dense":
        raise GrowError(f"family {enc.family!r}: only dense encoders are grown exactly")
    if enc.post_block_norm:
        raise GrowError("post_block_norm=True: a zero-output block would still apply a LayerNorm")
    if getattr(enc, "layer_scale_init", 0.0) > 0.0:
        raise GrowError("LayerScale present: not handled by this tool")


def grow_state(state: dict, n_src: int, n_dst: int) -> tuple[dict, list[tuple[int, bool]]]:
    """Grown state dict (new tensors; the input is not modified) and the block layout used."""
    blocks: dict[int, dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m = _BLOCK.match(k)
        if m:
            blocks.setdefault(int(m.group(1)), {})[m.group(2)] = v
        else:
            other[k] = v
    if sorted(blocks) != list(range(n_src)):
        raise GrowError(f"state has blocks {sorted(blocks)[:3]}..., expected 0..{n_src - 1}")
    plan = layout(n_src, n_dst)
    out = {k: v.detach().clone() for k, v in other.items()}
    for j, (i, new) in enumerate(plan):
        for name, v in blocks[i].items():
            t = v.detach().clone()
            if new and name.rsplit(".", 1)[0] in OUTPUT_PROJ:
                t.zero_()
            out[f"encoder.blocks.{j}.{name}"] = t
    return out, plan


def grown_config(src_cfg, n_dst: int):
    check_growable(src_cfg.encoder)
    return dataclasses.replace(src_cfg, encoder=dataclasses.replace(src_cfg.encoder, n_blocks=n_dst))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_seed(src: str, out: str, n_dst: int, weights: str = "model") -> dict:
    from train import checkpoint
    from train.asr import ASRModel, ModelConfig

    p = checkpoint.load(src, restore_rng=False)  # checks the source's .sha256 marker
    src_cfg = ModelConfig.from_dict(p["model_config"])
    n_src = src_cfg.encoder.n_blocks
    dst_cfg = grown_config(src_cfg, n_dst)
    if weights == "ema" and p.get("ema") is None:
        raise GrowError(f"{src} has no EMA weights")
    state, plan = grow_state(p["ema"] if weights == "ema" else p["model"], n_src, n_dst)
    model = ASRModel(dst_cfg)
    model.load_state_dict(state, strict=True)  # every key present, every shape right
    payload = {"model": model.state_dict(), "model_config": dst_cfg.to_dict(), "step": p.get("step", 0),
               "seed_from": {"path": os.path.abspath(src), "sha256": _sha256(src), "weights": weights,
                             "tool": "h53 grow_depth", "n_src": n_src, "n_dst": n_dst,
                             "layout": [[i, bool(n)] for i, n in plan],
                             "new_blocks": "internal params copied from the preceding original block; "
                                           "ff1.w2 / att.out / conv.pw2 / ff2.w2 zero (exact identity)"},
               "tokenizer_sha256": p.get("tokenizer_sha256")}
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    with open(out[:-3] + ".sha256", "w") as fh:
        fh.write(_sha256(out) + "\n")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, required=True, help="destination block count")
    ap.add_argument("--weights", default="model", choices=("model", "ema"))
    a = ap.parse_args(argv)
    if not a.out.endswith(".pt") or os.path.abspath(a.out) == os.path.abspath(a.src):
        ap.error("--out must be a new .pt path (the source is never overwritten)")
    pl = make_seed(a.src, a.out, a.blocks, a.weights)
    sf = pl["seed_from"]
    print(f"grown {sf['n_src']} -> {sf['n_dst']} blocks from {a.src} ({a.weights} weights); "
          f"seed {a.out} sha256 {_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
