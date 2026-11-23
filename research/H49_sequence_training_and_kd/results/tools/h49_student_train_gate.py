"""H49a recall gate, TRAIN half (h49-seqkd): the student's own beam-K oracle headroom on kd-cache's 20k
pilot units, with references read from the TRAIN parquet (id + text columns of the pilot row groups only),
so the gate does not wait for the R0 union pass. Canonical words, oracles before reference injection,
categories as frozen in h49_headroom.py.

  python h49_student_train_gate.py --student-dir DIR8[,DIR4,DIR16] --out PREFIX
Pre-registered gate (H49a stage 1): beam-4 oracle headroom >= 2 x spread (0.174 pp).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h49_headroom import (Categoriser, load_cmu, load_train_freq, matched_ref_positions, n_errors,  # noqa: E402
                          normalise_words)

PARQUET = "/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all"
SPLIT_DIRS = {"train-clean-100": "train.clean.100", "train-clean-360": "train.clean.360",
              "train-other-500": "train.other.500"}


def unit_refs(unit_files: list[str]) -> dict:
    """uid -> (split, reference text) for the pilot units named by kd-cache's student file names."""
    import pyarrow.parquet as pq

    refs = {}
    for name in unit_files:  # e.g. train-clean-360_0027_rg04-05_sp1.0.student.jsonl
        base = os.path.basename(name).split("_sp")[0]
        split, shard, rg = base.split("_")
        lo, hi = (int(x) for x in rg[2:].split("-"))
        files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))
        pf = pq.ParquetFile(files[int(shard)])
        for g in range(lo, hi):
            for r in pf.read_row_group(g, columns=["id", "text"]).to_pylist():
                refs[r["id"]] = (split, r["text"])
    return refs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--student-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spread-pp", type=float, default=0.087)
    ap.add_argument("--freq-cache", default="/dev/shm/h49-seqkd/train_word_freq.json")
    a = ap.parse_args(argv)
    lists, files8 = collections.defaultdict(dict), []
    for d in a.student_dir.split(","):
        beam = json.load(open(os.path.join(d, "STUDENT_META.json")))["beam"]
        fs = sorted(glob.glob(os.path.join(d, "*.student.jsonl")))
        if beam == 8:
            files8 = fs
        for p in fs:
            for line in open(p):
                r = json.loads(line)
                lists[r["uid"]].setdefault(beam, []).append((r["rank"], r["text"]))
    refs = unit_refs(files8)
    cat = Categoriser(load_cmu(), load_train_freq(cache=a.freq_cache))
    beams = sorted({b for v in lists.values() for b in v})
    tot = collections.defaultdict(collections.Counter)
    rec = collections.defaultdict(collections.Counter)
    n_utts = 0
    for uid, by_beam in lists.items():
        if uid not in refs or 8 not in by_beam:
            continue
        split, text = refs[uid]
        cond = "other" if "other" in split else "clean"
        ref = normalise_words(text)
        n_utts += 1
        one = sorted(by_beam[8])[0][1]
        sel = {"student_1best": " ".join(normalise_words(one))}
        cand_words = {}
        for b, l in by_beam.items():
            ws = [" ".join(normalise_words(t)) for _, t in sorted(l)]
            cand_words[b] = ws
            sel[f"b{b}_oracle"] = min(ws, key=lambda w: (n_errors(ref, w.split()), ws.index(w)))
        base_ev = cat.errors(ref, sel["student_1best"].split())
        for name, words in sel.items():
            ev = base_ev if name == "student_1best" else cat.errors(ref, words.split())
            for sl in ("all", cond):
                t = tot[(name, sl)]
                t["N"] += len(ref)
                for op, c, _ in ev:
                    t[op] += 1
                    t["cat:" + c] += 1
        miss = [(c, i) for op, c, i in base_ev if op in "SD"]
        if miss:
            for b, ws in cand_words.items():
                matched = set().union(*(matched_ref_positions(ref, w.split()) for w in ws))
                for c, i in miss:
                    for sl in ("all", cond):
                        rec[(b, sl)]["pos"] += 1
                        rec[(b, sl)]["hit"] += i in matched
                        rec[(b, sl)]["pos:" + c] += 1
                        rec[(b, sl)]["hit:" + c] += i in matched
    rep = {"utts": n_utts, "beams": beams, "spread_pp": a.spread_pp, "systems": {}, "recall": {}}
    for (name, sl), t in sorted(tot.items()):
        b = tot[("student_1best", sl)]
        e, be = t["S"] + t["D"] + t["I"], b["S"] + b["D"] + b["I"]
        cats = sorted(k[4:] for k in set(t) | set(b) if k.startswith("cat:"))
        rep["systems"].setdefault(name, {})[sl] = {
            "N": t["N"], "S": t["S"], "D": t["D"], "I": t["I"], "wer": round(100 * e / max(1, t["N"]), 4),
            "headroom_pp": round(100 * (be - e) / max(1, t["N"]), 4),
            "by_category_headroom_pp": {c: round(100 * (b["cat:" + c] - t["cat:" + c]) / max(1, t["N"]), 4) for c in cats}}
    for (bm, sl), r in sorted(rec.items()):
        cats = sorted(k[4:] for k in r if k.startswith("pos:"))
        rep["recall"].setdefault(f"b{bm}", {})[sl] = {
            "positions": r["pos"], "recall": round(r["hit"] / max(1, r["pos"]), 4),
            "by_category": {c: round(r["hit:" + c] / max(1, r["pos:" + c]), 4) for c in cats}}
    if "b4_oracle" in rep["systems"]:
        h = rep["systems"]["b4_oracle"]["all"]["headroom_pp"]
        rep["h49a_train_recall_gate"] = {"beam4_oracle_headroom_pp": h, "bar_pp": round(2 * a.spread_pp, 3),
                                         "verdict": "PASS" if h >= 2 * a.spread_pp else "FAIL"}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("utts", "beams")}))
    for name, v in sorted(rep["systems"].items()):
        print(name, {sl: (v[sl]["wer"], v[sl]["headroom_pp"]) for sl in v})
    print("recall", {k: v["all"]["recall"] for k, v in rep["recall"].items()})
    print("gate", rep.get("h49a_train_recall_gate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
