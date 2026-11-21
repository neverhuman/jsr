"""Speaker-stratified DEV selection panel (PLAN_TO_V1 A1).

The first-400 dev-clean panel covers only 6 speakers, so its bootstrap intervals are fragile.
This panel takes the same number of utterances from every speaker in dev-clean and dev-other,
choosing within a speaker by sha256(seed + uid). Selection reads ids and speakers only, never
the reference texts. It is used for checkpoint and setting selection; PUBLIC-TEST is untouched.

    python -m tools.dev_panel --per-speaker 10 --out docs/ultra/data/dev_selection_panel.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

CACHE = "/mnt/xbabe3-external/jsr/cache/librispeech_pcm"
SEED = "ultra-dev-panel-v1"


def select(uids: list[str], speakers: list[int], per_speaker: int, seed: str = SEED) -> list[str]:
    """Up to `per_speaker` uids per speaker, ordered by sha256(seed + uid), then sorted."""
    by_spk: dict[int, list[str]] = defaultdict(list)
    for u, s in zip(uids, speakers):
        by_spk[int(s)].append(u)
    out: list[str] = []
    for s in sorted(by_spk):
        ranked = sorted(by_spk[s], key=lambda u: hashlib.sha256((seed + u).encode()).hexdigest())
        out.extend(ranked[:per_speaker])
    return sorted(out)


def split_panel(split: str, per_speaker: int, cache: str = CACHE) -> dict:
    path = os.path.join(cache, f"{split}.npz")
    with np.load(path, allow_pickle=False) as z:
        uids = [str(u) for u in z["uids"]]
        speakers = [int(s) for s in z["speakers"]]
    chosen = select(uids, speakers, per_speaker)
    spk = {u: s for u, s in zip(uids, speakers)}
    return {
        "split": split,
        "speakers": len(set(speakers)),
        "utterances": len(chosen),
        "uids": chosen,
        "speaker_of": {u: spk[u] for u in chosen},
        "cache_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-speaker", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args(argv)
    panel = {
        "schema": "ultra.dev-panel/v1",
        "seed": SEED,
        "per_speaker": a.per_speaker,
        "splits": [split_panel(s, a.per_speaker, a.cache) for s in ("dev-clean", "dev-other")],
    }
    body = json.dumps(panel, indent=1, sort_keys=True)
    panel["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(panel, f, indent=1, sort_keys=True)
    for s in panel["splits"]:
        print(f"{s['split']}: {s['utterances']} utts from {s['speakers']} speakers")
    print("manifest_sha256", panel["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
