#!/usr/bin/env python3
"""h45-wins Arm B diagnostics in weight space (not gated).

  h45_wdiag.py --src <tree>/python/ultra --base CK@key --pair NAME=CK@key,CK@key [--pair ...] --vec NAME=CK@key [...]

Prints, over all floating parameters (the cmvn buffers excluded): for each --vec v, the norm of (v - base);
for each pair (x, y) the cosine of (x - base, y - base), the distance ||x - y||, and ||x - y|| relative to ||y - base||.
Predictor projections of different context lengths are compared after widening the shorter one with zero columns
(recipe_opts.widen_state).
"""
from __future__ import annotations

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--vec", action="append", default=[])
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import torch

    from train import checkpoint

    cache = {}

    def load(spec):
        path, _, key = spec.partition("@")
        key = key or "model"
        if path not in cache:
            cache[path] = checkpoint.load(path, restore_rng=False)
        sd = cache[path][key]
        return {k: v.double() for k, v in sd.items() if torch.is_floating_point(v) and not k.startswith("cmvn.")}

    def widen(sd, ctx):
        w = sd["predictor.proj.weight"]
        if w.shape[1] == ctx:
            return sd
        pad = w.new_zeros(w.shape[0], ctx - w.shape[1])
        return dict(sd, **{"predictor.proj.weight": torch.cat([pad, w], dim=1)})

    base = load(a.base)

    def diff(x, y):
        ctx = max(x["predictor.proj.weight"].shape[1], y["predictor.proj.weight"].shape[1])
        x, y = widen(x, ctx), widen(y, ctx)
        return {k: x[k] - y[k] for k in x}

    def dot(u, v):
        tot = 0.0
        for k in u:
            x, y = u[k], v[k]
            if x.shape != y.shape:  # predictor projections of different context: zero-pad the older positions
                n = max(x.shape[1], y.shape[1])
                x = torch.cat([x.new_zeros(x.shape[0], n - x.shape[1]), x], dim=1)
                y = torch.cat([y.new_zeros(y.shape[0], n - y.shape[1]), y], dim=1)
            tot += float((x * y).sum())
        return tot

    out = {"base": a.base, "vec": {}, "pair": {}}
    for item in a.vec:
        name, spec = item.split("=", 1)
        d = diff(load(spec), base)
        out["vec"][name] = {"norm_vs_base": dot(d, d) ** 0.5}
    for item in a.pair:
        name, specs = item.split("=", 1)
        sx, sy = specs.split(",")
        x, y = load(sx), load(sy)
        dx, dy, dxy = diff(x, base), diff(y, base), diff(x, y)
        nx, ny, nxy = dot(dx, dx) ** 0.5, dot(dy, dy) ** 0.5, dot(dxy, dxy) ** 0.5
        out["pair"][name] = {"cos_vs_base": dot(dx, dy) / (nx * ny), "dist": nxy, "dist_rel_to_y": nxy / ny,
                             "norm_x_vs_base": nx, "norm_y_vs_base": ny}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
