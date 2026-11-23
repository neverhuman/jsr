#!/usr/bin/env python3
"""h45-wins scoring helper around the pinned ultra-score.

  h45_score.py row <prefix> [--field hyp|beam_hyp] [--uids FILE] [--vs <prefix_b>] [--replicates N] [--out JSON]
      <prefix>.dev-clean.jsonl + <prefix>.dev-other.jsonl (train.evaluate --dump rows) -> canonical score.
      --uids restricts both systems to those utterance ids (e.g. a speaker half of a full-dev decode).
      --vs: paired comparison, system A = <prefix_b> (the comparator), system B = <prefix>; delta = B - A.
  h45_score.py rows <jsonl rows file> [--vs FILE] ...  (already {id,speaker,dataset,ref,hyp,duration_s})
Prints one JSON line: pooled / per-set WER (%), S/D/I, N, utterances, and for --vs the pooled delta (pp) with its
one-sided 95% bounds from ultra-score's speaker-cluster bootstrap (seed 20261122).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.environ.get("H45_SCORER", os.path.join(HERE, "ultra-score"))
SETS = ("dev-clean", "dev-other")


def load_rows(prefix: str, field: str, keep: set | None) -> list[dict]:
    rows = []
    for split in SETS:
        path = f"{prefix}.{split}.jsonl"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if keep is not None and r["uid"] not in keep:
                    continue
                rows.append({"id": r["uid"], "speaker": r["uid"].split("-")[0], "dataset": split,
                             "ref": r["ref"], "hyp": r[field], "duration_s": r["seconds"]})
    return rows


def summarise(rep: dict) -> dict:
    out = {}
    tot = {"S": 0, "D": 0, "I": 0, "N": 0, "utts": 0}
    for ds, d in rep["datasets"].items():
        c = d["counts"]
        out[ds] = {"wer": round(100 * d["wer"], 4), "S": c["substitutions"], "D": c["deletions"],
                   "I": c["insertions"], "N": c["ref_words"], "utts": d["utterances"]}
        tot["S"] += c["substitutions"]; tot["D"] += c["deletions"]; tot["I"] += c["insertions"]
        tot["N"] += c["ref_words"]; tot["utts"] += d["utterances"]
    out["pooled"] = {"wer": round(100 * rep["pooled_wer"], 4), **tot, "edits": tot["S"] + tot["D"] + tot["I"]}
    return out


def score(rows_a: list[dict], rows_b: list[dict] | None, replicates: int, md: str | None = None) -> dict:
    with tempfile.TemporaryDirectory(dir=os.environ.get("H45_TMP")) as td:
        pa = os.path.join(td, "a.jsonl")
        with open(pa, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows_a)
        cmd = [SCORER, pa]
        if rows_b is not None:
            pb = os.path.join(td, "b.jsonl")
            with open(pb, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows_b)
            cmd.append(pb)
        cmd += ["--seed", "20261122", "--replicates", str(replicates), "--alpha", "0.05"]
        if md:
            cmd += ["--md-out", md]
        res = subprocess.run(cmd, capture_output=True, check=True)
        return json.loads(res.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("row", "rows"))
    ap.add_argument("path")
    ap.add_argument("--field", default="hyp")
    ap.add_argument("--uids", default=None)
    ap.add_argument("--vs", default=None)
    ap.add_argument("--vs-field", default=None)
    ap.add_argument("--replicates", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    keep = None
    if a.uids:
        with open(a.uids) as fh:
            keep = {x.strip() for x in fh if x.strip()}
    if a.mode == "row":
        rb = load_rows(a.path, a.field, keep)
        ra = load_rows(a.vs, a.vs_field or a.field, keep) if a.vs else None
    else:
        rb = [json.loads(x) for x in open(a.path)]
        rb = [r for r in rb if keep is None or r["id"] in keep]
        ra = None
        if a.vs:
            ra = [json.loads(x) for x in open(a.vs)]
            ra = [r for r in ra if keep is None or r["id"] in keep]
    if ra is not None:
        ids_a, ids_b = sorted(r["id"] for r in ra), sorted(r["id"] for r in rb)
        if ids_a != ids_b:
            raise SystemExit(f"utterance sets differ: {len(ids_a)} vs {len(ids_b)}")
        rep = score(ra, rb, max(a.replicates, 1), a.md)
        out = {"system": a.path, "comparator": a.vs, "subset": a.uids and os.path.basename(a.uids),
               "b": summarise(rep["system_b"]), "a": summarise(rep["system_a"])}
        c = rep["comparison"]["pooled_delta"]
        out["delta_pp"] = round(100 * c["delta"], 4)
        out["delta_lower_pp"] = round(100 * c["lower"], 4)
        out["delta_upper_pp"] = round(100 * c["upper"], 4)
        out["replicates"] = c["replicates"]
        out["per_set_delta_pp"] = {ds: {"delta": round(100 * v["delta"], 4), "lower": round(100 * v["lower"], 4),
                                        "upper": round(100 * v["upper"], 4)}
                                   for ds, v in rep["comparison"]["per_dataset"].items()}
    else:
        rep = score(rb, None, 0, a.md)
        out = {"system": a.path, "subset": a.uids and os.path.basename(a.uids), **summarise(rep["system_a"])}
    line = json.dumps(out)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(rep) + "\n")
    print(line)


if __name__ == "__main__":
    main()
