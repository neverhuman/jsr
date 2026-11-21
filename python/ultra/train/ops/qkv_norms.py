"""Max (and per-block) attention qkv / att.norm weight norms of checkpoints, in step order."""
import glob, re, sys, torch
paths = sorted(sys.argv[1:], key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
for p in paths:
    sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
    q = {k.rsplit(".att.qkv.weight", 1)[0]: float(v.float().norm()) for k, v in sd.items() if k.endswith("att.qkv.weight")}
    g = {k.rsplit(".att.norm.weight", 1)[0]: float(v.float().norm()) for k, v in sd.items() if k.endswith("att.norm.weight")}
    kmax = max(q, key=q.get)
    print(f"{p.split('/')[-1]:22s} max_qkv {q[kmax]:7.1f} ({kmax})  median_qkv {sorted(q.values())[len(q)//2]:6.1f}  att.norm.w[{kmax}] {g[kmax]:6.1f}")
