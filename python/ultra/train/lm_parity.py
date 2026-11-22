"""H42: compare engine fusion runs (`ultra-engine-eval`, `ultra.engine-eval/v2`) with the Python
decodes of the same weights on the same panel utterances.

  python -m train.lm_parity --cache ENC_CACHE --engine-dir DIR --runs RUNS \
      --pairs nolm_b4=b4 M5_b4=M5_b4_l0.60_r0.75_p0.00 --out parity.json

Each pair is <engine run dir name>=<python run name>; the first pair is the no-LM baseline. Per
pair: WER (`ultra.wer`; dev-clean, dev-other, pooled) of both sides on the engine's utterances,
how many token sequences and normalised texts are identical, and for every later pair each side's
own change against its baseline. References come from the encoder cache index.
"""
from __future__ import annotations

import argparse
import json
import os

from train.lm_tune import count_errors, load_run, load_utts
from ultra.wer import normalise


def wer(rows: list[tuple[str, str, str]], refs: dict[str, str]) -> dict:
    """WER per split and pooled of (uid, split, hypothesis) rows."""
    sdi = count_errors([(refs[u], h) for u, _, h in rows])
    out, tot_e, tot_n = {}, 0, 0
    for split in sorted({s for _, s, _ in rows}):
        e = n = 0
        for (u, s, _), (a, b, c) in zip(rows, sdi, strict=True):
            if s == split:
                e, n = e + a + b + c, n + len(normalise(refs[u]))
        out[split] = round(100.0 * e / n, 3)
        tot_e, tot_n = tot_e + e, tot_n + n
    out["pooled"] = round(100.0 * tot_e / tot_n, 3)
    return out


def compare_pair(eng: dict, py: dict, refs: dict[str, str]) -> dict:
    rows = [(r["id"], r["split"], r["hyp"], r["tokens"]) for r in eng["rows"]]
    return {
        "utterances": len(rows),
        "engine_flags": {k: eng.get(k) for k in ("beam", "beam_max_symbols", "beam_margin",
                                                 "precision", "lm", "lm_weight", "lodr_weight",
                                                 "blank_penalty", "git_commit",
                                                 "executable_sha256", "weights_sha256")},
        "engine_reported_wer_pct": round(float(eng["wer_pct"]), 3),
        "engine_wer": wer([(u, s, h) for u, s, h, _ in rows], refs),
        "python_wer": wer([(u, s, py[u]["hyp"]) for u, s, _, _ in rows], refs),
        "identical_tokens": sum(t == py[u]["tokens"] for u, _, _, t in rows),
        "identical_text": sum(normalise(h) == normalise(py[u]["hyp"]) for u, _, h, _ in rows),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache", required=True, help="encoder cache (references)")
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    refs = {u["uid"]: u["ref"] for u in load_utts(a.cache, ["dev-clean", "dev-other"])}
    res: dict = {"exploratory": True, "pairs": {}}
    for pair in a.pairs:
        eng_name, py_name = pair.split("=", 1)
        with open(os.path.join(a.engine_dir, eng_name, "engine_eval.json")) as fh:
            eng = json.load(fh)
        py = load_run(a.runs, py_name, sorted({r["split"] for r in eng["rows"]}))
        res["pairs"][pair] = {"engine_run": eng_name, "python_run": py_name,
                              **compare_pair(eng, py, refs)}
    pairs = list(res["pairs"].values())
    for p in pairs[1:]:
        for side in ("engine", "python"):
            p[f"{side}_change_pp"] = {k: round(p[f"{side}_wer"][k] - pairs[0][f"{side}_wer"][k], 3)
                                      for k in p[f"{side}_wer"]}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
