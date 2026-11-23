#!/usr/bin/env python3
"""h45-wins Arm C: offline commit-pass probe ("provisional at 160 ms, final at 640 ms").

  h45_commit_probe.py --prov <prefix> --final <prefix> [--prov-field hyp] [--final-field hyp] --name NAME --out DIR
                      [--uids FILE]   (restrict to a speaker half)

<prefix>.dev-clean.jsonl / .dev-other.jsonl are train.evaluate --dump rows (uid, ref, hyp, ...). Each utterance stands
for one endpoint-delimited segment: the user sees the provisional text while speaking, the final text replaces it at
the endpoint. Words use eval-lane's port of the canonical normaliser and aligner (ultra-en-wer-norm/v1, parity-checked
against ultra-score counts); WER rows also come from the pinned ultra-score itself (paired bootstrap, final - prov).

Reports: WER prov / final; revision burden (segments whose final word sequence differs; word edits between provisional
and final per 100 reference words); per reference word, fixes (wrong -> right) and breaks (right -> wrong) plus the
insertion change; first- and last-word error rates; edits by reference length bin.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evallane_h43_latency import align, normalise_words  # noqa: E402

SCORER = os.path.join(HERE, "ultra-score")
SETS = ("dev-clean", "dev-other")
BINS = ((1, 4), (5, 9), (10, 19), (20, 10 ** 6))


def load(prefix, field, keep):
    out = {}
    for s in SETS:
        with open(f"{prefix}.{s}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if keep is None or r["uid"] in keep:
                    out[r["uid"]] = (s, r["ref"], r[field], r["seconds"])
    return out


def correct_mask(ops, n):
    ok = [False] * n
    ins = 0
    for op, i, j in ops:
        if op == "C":
            ok[i] = True
        elif op == "I":
            ins += 1
    return ok, ins


def scorer_pair(rows_a, rows_b):
    with tempfile.TemporaryDirectory(dir=os.environ.get("H45_TMP")) as td:
        pa, pb = os.path.join(td, "a.jsonl"), os.path.join(td, "b.jsonl")
        for p, rows in ((pa, rows_a), (pb, rows_b)):
            with open(p, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
        res = subprocess.run([SCORER, pa, pb, "--seed", "20261122", "--replicates", "10000", "--alpha", "0.05"],
                             capture_output=True, check=True)
        return json.loads(res.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prov", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--prov-field", default="hyp")
    ap.add_argument("--final-field", default="hyp")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uids", default=None)
    a = ap.parse_args()
    keep = None
    if a.uids:
        keep = {x.strip() for x in open(a.uids) if x.strip()}
    P = load(a.prov, a.prov_field, keep)
    F = load(a.final, a.final_field, keep)
    if sorted(P) != sorted(F):
        raise SystemExit("provisional and final cover different utterances")
    tot = {k: 0 for k in ("utts", "N", "e_prov", "e_final", "changed", "rev_edits", "fix", "brk", "ins_prov",
                          "ins_final", "first_n", "first_err_prov", "first_err_final", "last_n", "last_err_prov",
                          "last_err_final", "utt_better", "utt_worse")}
    bins = {f"{lo}-{hi if hi < 10**6 else 'inf'}": {"utts": 0, "N": 0, "e_prov": 0, "e_final": 0} for lo, hi in BINS}
    per = []
    rows_p, rows_f = [], []
    for uid in sorted(P):
        s, ref, hp, sec = P[uid]
        hf = F[uid][2]
        R, Hp, Hf = normalise_words(ref), normalise_words(hp), normalise_words(hf)
        op_p, op_f = align(R, Hp), align(R, Hf)
        ep = sum(o[0] != "C" for o in op_p)
        ef = sum(o[0] != "C" for o in op_f)
        okp, insp = correct_mask(op_p, len(R))
        okf, insf = correct_mask(op_f, len(R))
        rev = sum(o[0] != "C" for o in align(Hp, Hf))
        tot["utts"] += 1; tot["N"] += len(R); tot["e_prov"] += ep; tot["e_final"] += ef
        tot["changed"] += int(Hp != Hf); tot["rev_edits"] += rev
        tot["fix"] += sum((not x) and y for x, y in zip(okp, okf))
        tot["brk"] += sum(x and (not y) for x, y in zip(okp, okf))
        tot["ins_prov"] += insp; tot["ins_final"] += insf
        tot["utt_better"] += int(ef < ep); tot["utt_worse"] += int(ef > ep)
        if R:
            tot["first_n"] += 1; tot["first_err_prov"] += int(not okp[0]); tot["first_err_final"] += int(not okf[0])
            tot["last_n"] += 1; tot["last_err_prov"] += int(not okp[-1]); tot["last_err_final"] += int(not okf[-1])
        for (lo, hi), key in zip(BINS, bins):
            if lo <= len(R) <= hi:
                b = bins[key]; b["utts"] += 1; b["N"] += len(R); b["e_prov"] += ep; b["e_final"] += ef
        per.append({"uid": uid, "set": s, "n": len(R), "e_prov": ep, "e_final": ef, "rev_edits": rev,
                    "changed": Hp != Hf})
        spk = uid.split("-")[0]
        rows_p.append({"id": uid, "speaker": spk, "dataset": s, "ref": ref, "hyp": hp, "duration_s": sec})
        rows_f.append({"id": uid, "speaker": spk, "dataset": s, "ref": ref, "hyp": hf, "duration_s": sec})
    rep = scorer_pair(rows_p, rows_f)
    c = rep["comparison"]["pooled_delta"]
    N = tot["N"]
    summary = {
        "name": a.name, "subset": a.uids and os.path.basename(a.uids), "utterances": tot["utts"], "ref_words": N,
        "wer_prov": round(100 * rep["system_a"]["pooled_wer"], 4), "wer_final": round(100 * rep["system_b"]["pooled_wer"], 4),
        "delta_pp": round(100 * c["delta"], 4), "delta_lower_pp": round(100 * c["lower"], 4),
        "delta_upper_pp": round(100 * c["upper"], 4),
        "port_check_edits": {"prov": tot["e_prov"], "final": tot["e_final"]},
        "segments_changed_pct": round(100 * tot["changed"] / tot["utts"], 2),
        "revision_word_edits_per_100_words": round(100 * tot["rev_edits"] / N, 3),
        "fixes_per_100_words": round(100 * tot["fix"] / N, 3), "breaks_per_100_words": round(100 * tot["brk"] / N, 3),
        "insertions_prov": tot["ins_prov"], "insertions_final": tot["ins_final"],
        "fix_to_break_ratio": round(tot["fix"] / max(1, tot["brk"]), 3),
        "utts_better": tot["utt_better"], "utts_worse": tot["utt_worse"],
        "first_word_err_pct": {"prov": round(100 * tot["first_err_prov"] / tot["first_n"], 3),
                               "final": round(100 * tot["first_err_final"] / tot["first_n"], 3)},
        "last_word_err_pct": {"prov": round(100 * tot["last_err_prov"] / tot["last_n"], 3),
                              "final": round(100 * tot["last_err_final"] / tot["last_n"], 3)},
        "by_ref_length": {k: dict(v, wer_prov=round(100 * v["e_prov"] / max(1, v["N"]), 3),
                                  wer_final=round(100 * v["e_final"] / max(1, v["N"]), 3)) for k, v in bins.items()},
        "per_set": {ds: {"prov": round(100 * rep["system_a"]["datasets"][ds]["wer"], 4),
                         "final": round(100 * rep["system_b"]["datasets"][ds]["wer"], 4)} for ds in SETS},
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.name}.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(a.out, f"{a.name}.utts.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in per)
    with open(os.path.join(a.out, f"{a.name}.score.json"), "w") as fh:
        json.dump(rep, fh)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
