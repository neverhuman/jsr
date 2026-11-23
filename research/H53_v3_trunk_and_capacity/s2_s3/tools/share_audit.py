"""H53 Amendment 6 per-batch share audit for S2b: replay the deterministic sampler (seed, manifests, S2b's shape) over the
first N global batches and report the Libri-Light share of each batch's audio. Targets: median 0.45-0.55, p95 <= 0.65.
Global batches are rebuilt exactly as ParquetBatches._load_window_sharded builds them at the pin (90b7f4d): header-only
frames after speed perturbation, rows over max_seconds dropped, the seeded 5% jitter, pack(), the seeded shuffle.
(Fixed by the new trainer-lane at 10:5xZ: the predecessor's draft called a _header_groups method the pin does not have.)
  share_audit.py <ls_root> <base_weight> <N> <procs> <extra_dir> [<extra_dir> ...]      (run from the tree's python/ultra)
  base_weight 0 = natural share (S2a's mix)."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from train.data import ParquetBatches, pack, _round_up

LS, BW, N, P = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
EXTRA = [(d, None) for d in sys.argv[5:]]
SRC = None


class _Feat:
    def __init__(self):
        from train.features import LogMel
        self.logmel = LogMel()


def _init():
    global SRC
    SRC = ParquetBatches(["train-clean-100", "train-clean-360", "train-other-500"], _Feat(), 7919, 7800.0, 1728, 20.0,
                         window_groups=int(os.environ.get("WG", "96")), parquet_root=LS, workers=1, speed_perturb=(0.9, 1.0, 1.1),
                         extra=EXTRA, base_weight=BW)


def window_shares(ew):
    e, w = ew
    src = SRC
    rows, is_ll = [], []
    for fi, rg in src._window_units(e, w):
        rs = src._read_rows(fi, rg)
        rows += rs
        is_ll += [fi >= src._first_extra] * len(rs)
    frames = [src._frames_after(r, e) for r in rows]
    keep = [i for i, f in enumerate(frames) if f is not None]
    rng = np.random.default_rng([src.seed, e, w])
    dur = [frames[i] / 100.0 * (1.0 + 0.05 * rng.random()) for i in keep]
    if src.pad_frames > 1:
        dur = [_round_up(int(d * 100), src.pad_frames) / 100.0 for d in dur]
    groups = pack(dur, src.budget_s, src.max_batch)
    rng.shuffle(groups)
    out = []
    for g in groups:
        fr = [frames[keep[i]] for i in g]
        ll = sum(frames[keep[i]] for i in g if is_ll[keep[i]])
        out.append((ll / sum(fr), sum(fr)))
    return ew, out


if __name__ == "__main__":
    _init()
    nw0 = SRC._windows(0)
    todo = [(0, w) for w in range(nw0)] + [(1, w) for w in range(SRC._windows(1))]
    res, i = [], 0
    with Pool(P, initializer=_init) as pool:
        while len(res) < N and i < len(todo):
            chunk = todo[i:i + 2 * P]
            for _, s in sorted(pool.map(window_shares, chunk)):
                res.extend(s)
            i += len(chunk)
    s = np.array([x for x, _ in res[:N]])
    f = np.array([y for _, y in res[:N]])
    print(json.dumps({"batches": int(len(s)), "windows_read": i, "windows_per_epoch0": nw0, "base_weight": BW,
                      "extra": [d for d, _ in EXTRA], "n_epoch_units": SRC.n_epoch_units,
                      "ll_share_median": round(float(np.median(s)), 4), "ll_share_p95": round(float(np.percentile(s, 95)), 4),
                      "ll_share_p5": round(float(np.percentile(s, 5)), 4), "ll_share_mean": round(float(s.mean()), 4),
                      "ll_audio_share_pooled": round(float((s * f).sum() / f.sum()), 4), "window_groups": SRC.window_groups,
                      "pass": bool(0.45 <= np.median(s) <= 0.55 and np.percentile(s, 95) <= 0.65)}))
