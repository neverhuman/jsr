"""H51-PQ data prep: the 730-utt DEV panel (hash-verified) and a seeded TRAIN subset, as compact
npz files in RAM scratch. TRAIN audio is only used for activation traces / k-means calibration;
the DEV panel only for WER. Test sets are never opened.

usage: python -I prep_data.py <src_python_ultra_dir> <out_dir> <results_dir>
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import sys

import numpy as np

SRC, OUT, RES = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)

from train.evaluate import load_panel  # noqa: E402

PANEL = os.path.join(SRC, "..", "..", "docs", "ultra", "data", "dev_selection_panel.json")
PARQ = "/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all"
SEED = 51


def save(path, utts, extra):
    pcms = [u[3] for u in utts]
    offs = np.cumsum([0] + [len(p) for p in pcms]).astype(np.int64)
    np.savez(path, pcm=np.concatenate(pcms), offsets=offs, uids=np.array([u[0] for u in utts]),
             speakers=np.array([u[1] for u in utts]), texts=np.array([u[2] for u in utts]),
             splits=np.array([u[4] for u in utts]))
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    secs = float(offs[-1]) / 16000.0
    print(path, len(utts), "utts", round(secs / 3600, 3), "h", h[:16])
    return {"path": path, "n": len(utts), "hours": round(secs / 3600, 4), "sha256": h, **extra}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    rec = {}
    utts, digest = load_panel(PANEL)
    spec = json.load(open(PANEL))
    split_of = {uid: sp["split"] for sp in spec["splits"] for uid in sp["uids"]}
    pu = [(u.uid, u.speaker, u.text, u.pcm, split_of[u.uid]) for u in utts]
    rec["panel"] = save(os.path.join(OUT, "panel730.npz"), pu,
                        {"panel_manifest_sha256": digest, "panel_file": "docs/ultra/data/dev_selection_panel.json"})

    import pyarrow.parquet as pq
    import soundfile as sf

    rng = np.random.default_rng(SEED)
    tu, picks = [], []
    for sp in ("train.clean.100", "train.clean.360", "train.other.500"):
        files = sorted(glob.glob(os.path.join(PARQ, sp, "*.parquet")))
        idx = np.linspace(0, len(files) - 1, 5).round().astype(int)
        for fi in idx:
            pf = pq.ParquetFile(files[fi])
            rg = int(rng.integers(pf.metadata.num_row_groups))
            t = pf.read_row_group(rg, columns=["id", "speaker_id", "text", "audio"]).to_pylist()
            choose = sorted(rng.choice(len(t), size=min(20, len(t)), replace=False).tolist())
            for j in choose:
                row = t[j]
                pcm, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
                assert sr == 16000 and pcm.ndim == 1
                tu.append((row["id"], int(row["speaker_id"]), row["text"], pcm, sp))
            picks.append({"split": sp, "file": os.path.basename(files[fi]), "row_group": rg,
                          "rows": choose})
    rec["train"] = save(os.path.join(OUT, "train300.npz"), tu, {"seed": SEED, "picks": picks})
    rec["train_uids"] = [u[0] for u in tu]
    rec["train_speakers"] = sorted({u[1] for u in tu})
    with open(os.path.join(RES, "data_manifest.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
