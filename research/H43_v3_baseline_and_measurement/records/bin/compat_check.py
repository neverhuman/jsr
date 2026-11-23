#!/usr/bin/env python3
"""eval-lane: check trainer-lane's compat copy against its original checkpoint: identical tensors (model + ema, every key) and a
model_config that equals the original's minus only the keys listed in <run>.compat.json 'dropped', each holding the listed value.
  compat_check.py ORIGINAL.pt COMPAT.pt COMPAT.json"""
import json, sys, torch
o = torch.load(sys.argv[1], map_location="cpu", weights_only=False); c = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
info = json.load(open(sys.argv[3])); dropped = info.get("dropped", {})
bad = []
for part in ("model", "ema"):
    a, b = o.get(part), c.get(part)
    if (a is None) != (b is None): bad.append(f"{part}: present in one only"); continue
    if a is None: continue
    if sorted(a) != sorted(b): bad.append(f"{part}: key sets differ")
    else:
        for k in a:
            if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]): bad.append(f"{part}.{k} differs"); break
def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict): out.update(flat(v, f"{pre}{k}."))
        else: out[f"{pre}{k}"] = v
    return out
oc, cc = flat(o.get("model_config", {})), flat(c.get("model_config", {}))
extra = {k for k in oc if k not in cc}; added = {k for k in cc if k not in oc}
if added: bad.append(f"compat adds keys {sorted(added)}")
leaf = lambda k: k.rsplit(".", 1)[-1]
if sorted(leaf(k) for k in extra) != sorted(dropped): bad.append(f"dropped keys {sorted(extra)} != listed {sorted(dropped)}")
norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v
for k in extra:
    if norm(oc[k]) != norm(dropped.get(leaf(k))): bad.append(f"{k}: original {oc[k]!r} != listed {dropped.get(leaf(k))!r}")
for k in cc:
    if k in oc and norm(cc[k]) != norm(oc[k]): bad.append(f"config {k} changed")
print(json.dumps({"ok": not bad, "problems": bad, "dropped": {k: norm(oc[k]) for k in sorted(extra)}, "step": o.get("step"), "tensors_checked": sum(len(o[p]) for p in ("model", "ema") if o.get(p))}))
