"""Parity witness for sidecar packing (handoff to runtime-lane, lead 16:43:04Z): rebuild the first N global batches of a fresh run
TWO ways and compare them exactly (the row ids of every global group, in order):
  (A) the pin's way, reading every row's audio bytes (ParquetBatches._read_rows + _frames_after, as _load_window_sharded does);
  (B) from a per-row frame-count sidecar (dur_table.py's JSONL: {file, size, mtime_ns, groups: [[[id, n_frames], ...], ...]}).
Prints the number of batches compared and the first mismatch, if any.
  parity_groups.py <tables,comma> <ls_root> <base_weight> <N> <window_groups> <extra_dir> [...]   (from the tree's python/ultra)"""
import json
import sys
from fractions import Fraction

import numpy as np

from train.data import ParquetBatches, _round_up, pack

TABLES, LS, BW, N, WG = sys.argv[1].split(","), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
EXTRA = [(d, None) for d in sys.argv[6:]]


class _Feat:
    def __init__(self):
        from train.features import LogMel
        self.logmel = LogMel()


T = {}
for t in TABLES:
    for line in open(t):
        r = json.loads(line)
        T[r["file"]] = r["groups"]
src = ParquetBatches(["train-clean-100", "train-clean-360", "train-other-500"], _Feat(), 7919, 7800.0, 1728, 20.0,
                     window_groups=WG, parquet_root=LS, workers=1, speed_perturb=(0.9, 1.0, 1.1), extra=EXTRA, base_weight=BW)


def sidecar_frames(uid, n, epoch):  # = ParquetBatches._frames_after, from the sidecar's frame count
    if src.speed_perturb:
        f = src._speed(uid, epoch)
        if f != 1.0:
            r = Fraction(f).limit_denominator(100)
            n = -(-n * r.denominator // r.numerator)
    if n / 16000.0 > src.max_seconds:
        return None
    return src.feat.logmel.num_frames(n)


def groups_of(ids, frames, e, w):
    keep = [i for i, f in enumerate(frames) if f is not None]
    rng = np.random.default_rng([src.seed, e, w])
    dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
    if src.pad_frames > 1:
        dur = [_round_up(int(d * 100), src.pad_frames) / 100.0 for d in dur]
    g = pack(dur, src.budget_s, src.max_batch)
    rng.shuffle(g)
    return [[ids[keep[i]] for i in grp] for grp in g]


n, e, w = 0, 0, 0
while n < N:
    units = src._window_units(e, w)
    rows = [r for fi, rg in units for r in src._read_rows(fi, rg)]                       # (A) reads audio bytes
    ga = groups_of([r["id"] for r in rows], [src._frames_after(r, e) for r in rows], e, w)
    side = [(u, k) for fi, rg in units for u, k in T[src.files[fi]][rg]]                # (B) sidecar only
    gb = groups_of([u for u, _ in side], [sidecar_frames(u, k, e) for u, k in side], e, w)
    if ga != gb:
        bad = next(i for i, (x, y) in enumerate(zip(ga, gb)) if x != y) if len(ga) == len(gb) else min(len(ga), len(gb))
        print(json.dumps({"parity": False, "epoch": e, "window": w, "first_bad_group": bad, "batches_ok_before": n + bad}))
        sys.exit(1)
    n += len(ga); w += 1
    if w >= src._windows(e):
        e, w = e + 1, 0
print(json.dumps({"parity": True, "batches_compared": n, "windows": w, "base_weight": BW, "window_groups": WG}))
