#!/usr/bin/env python3
"""h45-wins Arm D: CTC rescoring and word-level MBR over beam-8 N-best dumps (h45_decode.py --nbest --ctc-score).

  h45_rescore.py tune  --nbest <prefix> --uids S.txt --out DIR --name NAME     (grid on the selection half only)
  h45_rescore.py apply --nbest <prefix> --config cfg.json --out DIR --name NAME [--uids C.txt]

N-best rows: [text, transducer_logp, ctc_logp_or_null, n_ctc_tokens]. Words use eval-lane's parity-checked port of
ultra-en-wer-norm/v1; candidate edit counts against the reference are computed once, so the grid only re-ranks.
D1: score = transducer + lam * ctc + beta * words; a null (impossible) CTC path ranks last.
D2: MBR, posteriors p_i = softmax(score_i / T) with score = transducer (D2a) or the tuned D1 score (D2b); the chosen
    entry minimises sum_i p_i * word_edit_distance(candidate, entry_i).
'tune' writes <name>.grid.json (every grid point's S pooled WER) and <name>.config.json (the S-best of each family,
ties -> the simplest setting). 'apply' writes ultra-score input rows for the chosen hypotheses (and for the beam
1-best and the greedy hypothesis of the same decode) so the canonical scorer gives the reported numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evallane_h43_latency import align, normalise_words  # noqa: E402

SETS = ("dev-clean", "dev-other")
LAMS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
TEMPS = (0.5, 1.0, 2.0, 4.0)


def edits(r, h):
    return sum(o[0] != "C" for o in align(r, h))


def load(prefix, keep):
    utts = []
    for s in SETS:
        with open(f"{prefix}.{s}.jsonl") as fh:
            for line in fh:
                row = json.loads(line)
                if keep is not None and row["uid"] not in keep:
                    continue
                R = normalise_words(row["ref"])
                cands = []
                for text, rn, ct, nt in row["nbest"]:
                    W = normalise_words(text)
                    cands.append({"text": text, "rnnt": rn, "ctc": ct, "words": W, "nw": len(W), "e": edits(R, W)})
                utts.append({"uid": row["uid"], "set": s, "ref": row["ref"], "R": R, "seconds": row["seconds"],
                             "greedy": row["hyp"], "beam": row.get("beam_hyp"), "cands": cands,
                             "e_greedy": edits(R, normalise_words(row["hyp"]))})
    return utts


def d1_pick(u, lam, beta):
    best, bs = 0, -math.inf
    for i, c in enumerate(u["cands"]):
        if c["ctc"] is None and lam > 0:
            s = -1e30 + c["rnnt"]
        else:
            s = c["rnnt"] + lam * (c["ctc"] or 0.0) + beta * c["nw"]
        if s > bs:
            best, bs = i, s
    return best


def d1_score(c, lam, beta):
    if c["ctc"] is None and lam > 0:
        return -1e30
    return c["rnnt"] + lam * (c["ctc"] or 0.0) + beta * c["nw"]


def pairwise(u):
    if "D" not in u:
        n = len(u["cands"])
        D = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                D[i][j] = D[j][i] = edits(u["cands"][i]["words"], u["cands"][j]["words"])
        u["D"] = D
    return u["D"]


def mbr_pick(u, temp, lam, beta):
    sc = [d1_score(c, lam, beta) for c in u["cands"]]
    m = max(sc)
    w = [math.exp((s - m) / temp) if s > -1e29 else 0.0 for s in sc]
    z = sum(w) or 1.0
    p = [x / z for x in w]
    D = pairwise(u)
    n = len(sc)
    risk = [sum(p[i] * D[j][i] for i in range(n)) for j in range(n)]
    return min(range(n), key=lambda j: (risk[j], -sc[j]))


def total(utts, picks):
    e = sum(u["cands"][k]["e"] for u, k in zip(utts, picks))
    n = sum(len(u["R"]) for u in utts)
    return 100.0 * e / n, e, n


def rows_for(utts, texts):
    return [{"id": u["uid"], "speaker": u["uid"].split("-")[0], "dataset": u["set"], "ref": u["ref"], "hyp": t,
             "duration_s": u["seconds"]} for u, t in zip(utts, texts)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("tune", "apply"))
    ap.add_argument("--nbest", required=True)
    ap.add_argument("--uids", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    keep = None
    if a.uids:
        keep = {x.strip() for x in open(a.uids) if x.strip()}
    utts = load(a.nbest, keep)
    os.makedirs(a.out, exist_ok=True)
    n = sum(len(u["R"]) for u in utts)
    base = {"greedy": 100.0 * sum(u["e_greedy"] for u in utts) / n,
            "beam1": total(utts, [0] * len(utts))[0],
            "oracle": 100.0 * sum(min(c["e"] for c in u["cands"]) for u in utts) / n,
            "mean_nbest": sum(len(u["cands"]) for u in utts) / len(utts),
            "ctc_null_share": sum(c["ctc"] is None for u in utts for c in u["cands"]) /
            max(1, sum(len(u["cands"]) for u in utts))}
    if a.mode == "tune":
        grid = {"base": base, "d1": [], "d2a": [], "d2b": []}
        for lam in LAMS:
            for beta in BETAS:
                w = total(utts, [d1_pick(u, lam, beta) for u in utts])[0]
                grid["d1"].append({"lam": lam, "beta": beta, "wer": round(w, 4)})
        # ties -> smallest |beta| then smallest lam
        b1 = min(grid["d1"], key=lambda x: (x["wer"], abs(x["beta"]), x["lam"]))
        for t in TEMPS:
            w = total(utts, [mbr_pick(u, t, 0.0, 0.0) for u in utts])[0]
            grid["d2a"].append({"T": t, "wer": round(w, 4)})
            w = total(utts, [mbr_pick(u, t, b1["lam"], b1["beta"]) for u in utts])[0]
            grid["d2b"].append({"T": t, "lam": b1["lam"], "beta": b1["beta"], "wer": round(w, 4)})
        b2a = min(grid["d2a"], key=lambda x: (x["wer"], abs(x["T"] - 1.0)))
        b2b = min(grid["d2b"], key=lambda x: (x["wer"], abs(x["T"] - 1.0)))
        cfg = {"tuned_on": a.uids and os.path.basename(a.uids), "nbest": a.nbest, "d1": b1, "d2a": b2a, "d2b": b2b,
               "base": base}
        json.dump(grid, open(os.path.join(a.out, f"{a.name}.grid.json"), "w"), indent=1)
        json.dump(cfg, open(os.path.join(a.out, f"{a.name}.config.json"), "w"), indent=1)
        print(json.dumps(cfg))
    else:
        cfg = json.load(open(a.config))
        d1, d2a, d2b = cfg["d1"], cfg["d2a"], cfg["d2b"]
        sys_texts = {
            "greedy": [u["greedy"] for u in utts],
            "beam1": [u["cands"][0]["text"] for u in utts],
            "d1": [u["cands"][d1_pick(u, d1["lam"], d1["beta"])]["text"] for u in utts],
            "d2a": [u["cands"][mbr_pick(u, d2a["T"], 0.0, 0.0)]["text"] for u in utts],
            "d2b": [u["cands"][mbr_pick(u, d2b["T"], d2b["lam"], d2b["beta"])]["text"] for u in utts],
        }
        out = {"config": a.config, "subset": a.uids and os.path.basename(a.uids), "base": base, "port_wer": {}}
        for k, t in sys_texts.items():
            path = os.path.join(a.out, f"{a.name}.{k}.score-input.jsonl")
            with open(path, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows_for(utts, t))
            out["port_wer"][k] = round(100.0 * sum(edits(u["R"], normalise_words(x)) for u, x in zip(utts, t)) / n, 4)
        json.dump(out, open(os.path.join(a.out, f"{a.name}.apply.json"), "w"), indent=1)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
