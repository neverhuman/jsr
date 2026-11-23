"""H53 Amendment 6 per-batch share audit, table-driven (same result as share_audit.py, in seconds): replays the pin's
sampler over the first N global batches of a fresh run (Position 0, 0, 0) using per-row FLAC frame counts from
dur_table.py instead of re-reading audio. Window units, header frames after speed perturbation (> max_seconds dropped),
the seeded 5% jitter, pack() and the seeded shuffle are the pin's own code / formulas (ParquetBatches, 90b7f4d).
Targets: median Libri-Light share of a batch's audio 0.45-0.55, p95 <= 0.65.
  fast_audit.py <tables,comma> <ls_root> <base_weight> <N> <window_groups> <extra_dir> [...]   (from the tree's python/ultra)"""
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
        T[r["file"]] = r["groups"]  # a later line for the same file (re-scan) wins
src = ParquetBatches(["train-clean-100", "train-clean-360", "train-other-500"], _Feat(), 7919, 7800.0, 1728, 20.0,
                     window_groups=WG, parquet_root=LS, workers=1, speed_perturb=(0.9, 1.0, 1.1),
                     extra=EXTRA, base_weight=BW)
missing = [f for f in src.files if f not in T]
if missing:
    sys.exit(f"duration table lacks {len(missing)} files, e.g. {missing[0]}")


def frames_after(uid, n, epoch):  # = ParquetBatches._frames_after, from the table's header frame count
    if src.speed_perturb:
        f = src._speed(uid, epoch)
        if f != 1.0:
            r = Fraction(f).limit_denominator(100)
            n = -(-n * r.denominator // r.numerator)
    if n / 16000.0 > src.max_seconds:
        return None
    return src.feat.logmel.num_frames(n)


res, e, w, windows = [], 0, 0, 0
while len(res) < N:
    rows = []
    for fi, rg in src._window_units(e, w):
        ll = fi >= src._first_extra
        rows += [(u, n, ll) for u, n in T[src.files[fi]][rg]]
    frames = [frames_after(u, n, e) for u, n, _ in rows]
    keep = [i for i, f in enumerate(frames) if f is not None]
    rng = np.random.default_rng([src.seed, e, w])
    dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
    if src.pad_frames > 1:
        dur = [_round_up(int(d * 100), src.pad_frames) / 100.0 for d in dur]
    groups = pack(dur, src.budget_s, src.max_batch)
    rng.shuffle(groups)
    for g in groups:
        fr = sum(frames[keep[i]] for i in g)
        res.append((sum(frames[keep[i]] for i in g if rows[keep[i]][2]) / fr, fr))
    windows += 1
    w += 1
    if w >= src._windows(e):
        e, w = e + 1, 0
s = np.array([x for x, _ in res[:N]])
f = np.array([y for _, y in res[:N]])
print(json.dumps({"batches": int(len(s)), "windows_read": windows, "epochs_touched": e + 1, "base_weight": BW,
                  "window_groups": WG, "n_epoch_units": src.n_epoch_units, "windows_per_epoch0": src._windows(0),
                  "extra": [d for d, _ in EXTRA],
                  "ll_share_median": round(float(np.median(s)), 4), "ll_share_p95": round(float(np.percentile(s, 95)), 4),
                  "ll_share_p5": round(float(np.percentile(s, 5)), 4), "ll_share_mean": round(float(s.mean()), 4),
                  "ll_audio_share_pooled": round(float((s * f).sum() / f.sum()), 4),
                  "pass": bool(0.45 <= np.median(s) <= 0.55 and np.percentile(s, 95) <= 0.65)}))
