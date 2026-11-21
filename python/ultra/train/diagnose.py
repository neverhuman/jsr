"""H12 error breakdown of `train.evaluate --dump` files (per utterance: uid, ref, hyp, seconds).

  python -m train.diagnose <dump.jsonl> [--hyp-key hyp|beam_hyp|ctc_hyp] [--compare A.jsonl B.jsonl]

Reports WER by utterance duration and by reference word position (first / interior / last word),
where each reference word's fate (correct, substituted, deleted) comes from one minimum-cost
alignment, plus insertions after the last reference word. --compare A B lists, for the same uids,
how many utterances got better/worse from A to B (e.g. greedy vs beam, chunk vs full context).
Same normalisation as ultra.wer (provisional; ultra-eval stays the scoring authority).
"""
from __future__ import annotations

import argparse
import json
import sys

from ultra.wer import corpus_wer, edit_counts, normalise

BINS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 1e9))


def align(ref: list[str], hyp: list[str]) -> tuple[list[str], int]:
    """Fate of each reference word ('C', 'S', 'D') on one minimum-cost alignment, and the number
    of insertions after the last reference word."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    fate, i, j, tail_ins = [""] * n, n, m, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            fate[i - 1] = "C" if ref[i - 1] == hyp[j - 1] else "S"
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            fate[i - 1] = "D"
            i -= 1
        else:
            if i == n:
                tail_ins += 1
            j -= 1
    return fate, tail_ins


def load(path: str) -> list[dict]:
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        if not {"uid", "ref", "hyp"} <= r.keys():
            raise ValueError(f"{path}: dump rows need uid, ref, hyp")
    return rows


def breakdown(rows: list[dict], key: str = "hyp") -> dict:
    out = {"overall": corpus_wer([r["ref"] for r in rows], [r[key] for r in rows]), "by_seconds": {}}
    if all("seconds" in r for r in rows):
        for lo, hi in BINS:
            sel = [r for r in rows if lo <= r["seconds"] < hi]
            if sel:
                w = corpus_wer([r["ref"] for r in sel], [r[key] for r in sel])
                out["by_seconds"][f"{lo}-{hi if hi < 1e9 else 'inf'}s"] = dict(w, utts=len(sel))
    pos = {p: {"words": 0, "S": 0, "D": 0} for p in ("first", "interior", "last")}
    tail_ins = 0
    for r in rows:
        ref, hyp = normalise(r["ref"]), normalise(r[key])
        if not ref:
            continue
        fate, ti = align(ref, hyp)
        tail_ins += ti
        for k, f in enumerate(fate):
            p = "last" if k == len(ref) - 1 else "first" if k == 0 else "interior"
            pos[p]["words"] += 1
            if f in "SD":
                pos[p][f] += 1
    for p in pos.values():
        p["err_pct"] = round(100.0 * (p["S"] + p["D"]) / max(p["words"], 1), 2)
        p["del_pct"] = round(100.0 * p["D"] / max(p["words"], 1), 2)
    out["by_position"] = pos
    out["insertions_after_last_ref_word"] = tail_ins
    return out


def compare(a: list[dict], b: list[dict], key_a: str = "hyp", key_b: str = "hyp") -> dict:
    ha = {r["uid"]: r for r in a}
    better = worse = same = 0
    for r in b:
        if r["uid"] not in ha:
            raise ValueError(f"uid {r['uid']} missing from the first dump")
        ref = normalise(r["ref"])
        ea = sum(edit_counts(ref, normalise(ha[r["uid"]][key_a])))
        eb = sum(edit_counts(ref, normalise(r[key_b])))
        better, worse, same = better + (eb < ea), worse + (eb > ea), same + (eb == ea)
    return {"utts": len(b), "better": better, "worse": worse, "same": same}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--hyp-key", default="hyp", choices=("hyp", "beam_hyp", "ctc_hyp"))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    a = ap.parse_args(argv)
    res = {"dump": a.dump, "hyp_key": a.hyp_key, **breakdown(load(a.dump), a.hyp_key)}
    if a.compare:
        res["compare"] = {"a": a.compare[0], "b": a.compare[1],
                          **compare(load(a.compare[0]), load(a.compare[1]))}
    json.dump(res, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
