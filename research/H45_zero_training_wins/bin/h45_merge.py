#!/usr/bin/env python3
"""h45-wins weight-space merges: uniform / weighted averages, post-hoc EMA, task arithmetic.

  python h45_merge.py --src <tree>/python/ultra --out-dir <dir> recipe.json [recipe.json ...]

A recipe is {"name": str, "terms": [{"ckpt": path, "key": "model"|"ema", "coef": float}, ...],
             "config_from": path (default: first term), "pred_ctx": int (optional widening),
             "left_context": int (optional encoder override), "note": str}
The result is sum_i coef_i * weights_i, computed in float64 per tensor and cast back to the source dtype.
Coefficients may be negative (task vectors theta_k - theta_base enter as +1 / -1 terms); a pure average has
coefficients summing to 1. Non-floating tensors must be identical in every term and are copied.
"pred_ctx": every term whose predictor context is shorter is widened exactly as train/recipe_opts.widen_state
widens a fork (zero columns for the older positions, so the widened weights compute what the source did).
The output is an eval-only snapshot (model, model_config, train_config, step, tokenizer_sha256, h45_recipe with
every input path and sha256) plus its .sha256 completion marker, so train.evaluate / train.export load it.
The checkpoint's own .sha256 marker is verified for every input (train.checkpoint.load).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("recipes", nargs="+")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.src)
    import torch

    from train import checkpoint
    from train.asr import ModelConfig

    os.makedirs(a.out_dir, exist_ok=True)
    cache: dict[str, dict] = {}

    def load(path):
        if path not in cache:
            p = checkpoint.load(path, restore_rng=False)
            with open(path[:-3] + ".sha256") as fh:
                p["_sha256"] = fh.read().strip()
            cache.clear() if len(cache) > 6 else None
            cache[path] = p
        return cache[path]

    def widen(state: dict, src_ctx: int, dst_ctx: int) -> dict:
        if src_ctx == dst_ctx:
            return state
        w = state["predictor.proj.weight"]
        emb = w.shape[1] // src_ctx
        pad = w.new_zeros(w.shape[0], (dst_ctx - src_ctx) * emb)
        return dict(state, **{"predictor.proj.weight": torch.cat([pad, w], dim=1)})

    for rpath in a.recipes:
        r = json.load(open(rpath))
        name = r["name"]
        out = os.path.join(a.out_dir, name + ".pt")
        if os.path.exists(out) and os.path.exists(out[:-3] + ".sha256"):
            print(json.dumps({"skip": name, "sha256": open(out[:-3] + ".sha256").read().strip()}), flush=True)
            continue
        terms = r["terms"]
        base = load(r.get("config_from") or terms[0]["ckpt"])
        mcfg = ModelConfig.from_dict(base["model_config"])
        dst_ctx = int(r.get("pred_ctx") or getattr(mcfg, "pred_ctx", 2))
        acc: dict | None = None
        keys = None
        dtypes: dict = {}
        prov = []
        for t in terms:
            p = load(t["ckpt"])
            key = t.get("key", "model")
            if p.get(key) is None:
                raise SystemExit(f"{t['ckpt']}: no {key!r} weights")
            tcfg = ModelConfig.from_dict(p["model_config"])
            src_ctx = int(getattr(tcfg, "pred_ctx", 2))
            if src_ctx > dst_ctx:
                raise SystemExit(f"{t['ckpt']}: pred_ctx {src_ctx} > target {dst_ctx}")
            sd = widen(p[key], src_ctx, dst_ctx)
            if keys is None:
                keys = list(sd)
                acc = {}
                dtypes = {k: v.dtype for k, v in sd.items()}
            elif sorted(sd) != sorted(keys):
                raise SystemExit(f"{t['ckpt']}: parameter names differ")
            c = float(t["coef"])
            for k in keys:
                v = sd[k]
                if torch.is_floating_point(v):
                    if k in acc and acc[k].shape != v.shape:
                        raise SystemExit(f"{t['ckpt']}: shape of {k} differs")
                    acc[k] = acc[k] + c * v.double() if k in acc else c * v.double()
                else:
                    if k in acc and not torch.equal(acc[k], v):
                        raise SystemExit(f"{t['ckpt']}: non-float tensor {k} differs")
                    acc[k] = v.clone()
            prov.append({"ckpt": os.path.abspath(t["ckpt"]), "sha256": p["_sha256"], "key": key, "coef": c,
                         "step": p.get("step")})
        model = {k: (acc[k].to(dtypes[k]) if torch.is_floating_point(acc[k]) else acc[k]) for k in keys}
        md = json.loads(json.dumps(base["model_config"]))
        if r.get("pred_ctx"):
            md["pred_ctx"] = dst_ctx
        if r.get("left_context") is not None:
            md["encoder"]["left_context"] = int(r["left_context"])
        ModelConfig.from_dict(md)  # validates
        payload = {"model": model, "model_config": md, "train_config": base.get("train_config"),
                   "step": base.get("step"), "tokenizer_sha256": base.get("tokenizer_sha256"),
                   "h45_recipe": {"name": name, "terms": prov, "pred_ctx": dst_ctx,
                                  "left_context": md["encoder"].get("left_context"),
                                  "coef_sum": sum(x["coef"] for x in prov), "note": r.get("note", ""),
                                  "recipe_file": os.path.abspath(rpath)}}
        tmp = out + ".tmp"
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        h = sha256_file(out)
        with open(out[:-3] + ".sha256", "w") as fh:
            fh.write(h + "\n")
        print(json.dumps({"name": name, "sha256": h, "coef_sum": payload["h45_recipe"]["coef_sum"],
                          "n_terms": len(prov)}), flush=True)


if __name__ == "__main__":
    main()
