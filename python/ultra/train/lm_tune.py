"""H42 tuning and statistics: speaker folds, held-out choice of decode weights, paired bootstrap,
N-best rescoring with a word n-gram.

  python -m train.lm_tune rescore --cache DIR --runs DIR --nbest b8_nbest --baseline b4 \
      --lm 4-gram.trie.bin --out rescore.json
  python -m train.lm_tune grid --tag t5 --lm tok5.bin --lodr lodr2.bin --out grid.json
  python -m train.lm_tune fusion --cache DIR --runs DIR --baseline b4 --grid grid.json \
      --out fusion.json

Protocol (research/H42_ngram_lm/README.md): within each dev set the speakers are sorted and dealt
alternately to folds A and B; decode weights are chosen on one fold (dev-clean + dev-other
together, fewest word errors, ties to the earliest config in preference order) and scored on the
other. The reported number is the held-out one. "tuned_on_itself" entries choose on all utterances
and are labelled as such. Scoring is `ultra.wer` (provisional scorer), dev sets only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from train.lm_decode import load_cache
from ultra.wer import edit_counts, normalise

SEED = 20261122
RESAMPLES = 10000
LN10 = math.log(10.0)


def load_utts(cache: str, splits) -> list[dict]:
    """Utterances of the encoder cache, in cache order, with split, fold and reference length."""
    out = []
    for split in splits:
        index, _ = load_cache(cache, split)
        fold = speaker_folds([u["speaker"] for u in index["utts"]])
        for u in index["utts"]:
            out.append({"uid": u["uid"], "split": split, "speaker": u["speaker"], "ref": u["ref"],
                        "nref": len(normalise(u["ref"])), "fold": fold[u["speaker"]],
                        "frames": u["frames"], "greedy": u["greedy"]})
    return out


def speaker_folds(speakers) -> dict[int, int]:
    """Speaker -> fold (0 = A, 1 = B): sorted speaker ids, even positions A, odd positions B."""
    return {s: i % 2 for i, s in enumerate(sorted(set(speakers)))}


def load_run(runs: str, name: str, splits, n_expected: int | None = None) -> dict[str, dict]:
    """uid -> row of one decode config, merged over its shard files."""
    rows: dict[str, dict] = {}
    for split in splits:
        files = sorted(glob.glob(os.path.join(runs, name, f"{split}.*of*.jsonl")))
        if not files:
            raise FileNotFoundError(f"no shard files for {name}/{split} under {runs}")
        want = int(files[0].rsplit("of", 1)[1].split(".")[0])
        if len(files) != want:
            raise ValueError(f"{name}/{split}: {len(files)} of {want} shards present")
        for f in files:
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows[r["uid"]] = r
    if n_expected is not None and len(rows) != n_expected:
        raise ValueError(f"{name}: {len(rows)} utterances, expected {n_expected}")
    return rows


def _errs(pair) -> tuple[int, int, int]:
    return edit_counts(normalise(pair[0]), normalise(pair[1]))


def count_errors(pairs, procs: int = 1) -> list[tuple[int, int, int]]:
    """(sub, del, ins) for each (reference, hypothesis) pair; identical pairs are computed once."""
    uniq = sorted(set(pairs))
    if procs > 1 and len(uniq) > 2000:
        with Pool(procs) as pool:
            vals = pool.map(_errs, uniq, chunksize=256)
    else:
        vals = [_errs(p) for p in uniq]
    table = dict(zip(uniq, vals, strict=True))
    return [table[p] for p in pairs]


def wer_row(sdi: np.ndarray, nref: np.ndarray, mask=None) -> dict:
    """WER summary of per-utterance (sub, del, ins) rows over the utterances in `mask`."""
    if mask is None:
        mask = np.ones(len(nref), dtype=bool)
    s, d, i = (int(x) for x in sdi[mask].sum(axis=0))
    n = int(nref[mask].sum())
    return {"wer": round(100.0 * (s + d + i) / max(n, 1), 3), "sub": s, "del": d, "ins": i,
            "ref_words": n, "utts": int(mask.sum())}


def choose(err: np.ndarray, mask: np.ndarray) -> int:
    """Index of the config with the fewest errors on `mask`; ties go to the earliest config, so
    callers order configs by preference (smallest LM weight, then smallest bonus)."""
    return int(np.argmin(err[:, mask].sum(axis=1)))


def held_out(err: np.ndarray, fold: np.ndarray, tune_mask=None) -> tuple[np.ndarray, list[int]]:
    """Per-utterance config index under 2-fold cross-validation: utterances of fold B get the
    config chosen on fold A and the reverse. `tune_mask` restricts the utterances that may be
    tuned on (e.g. one dev set); default all. Returns (config index per utterance, [config chosen
    on A, config chosen on B])."""
    if tune_mask is None:
        tune_mask = np.ones(len(fold), dtype=bool)
    picked = [choose(err, tune_mask & (fold == f)) for f in (0, 1)]
    return np.where(fold == 0, picked[1], picked[0]), picked


def bootstrap(arm: np.ndarray, base: np.ndarray, nref: np.ndarray, strata: np.ndarray,
              units: np.ndarray | None = None, resamples: int = RESAMPLES,
              seed: int = SEED) -> dict:
    """Paired bootstrap of WER(arm) - WER(base) in percentage points. Resampling is within each
    stratum (dev set); the statistic sums errors and reference words over the strata. `units`
    (e.g. speaker ids) switches from resampling utterances to resampling whole units."""
    rng = np.random.default_rng(seed)
    d_err = np.zeros(resamples)
    n_ref = np.zeros(resamples)
    for s in np.unique(strata):
        m = strata == s
        d, n = (arm[m] - base[m]).astype(np.float64), nref[m].astype(np.float64)
        if units is not None:
            _, inv = np.unique(units[m], return_inverse=True)
            d, n = np.bincount(inv, weights=d), np.bincount(inv, weights=n)
        idx = rng.integers(0, len(d), size=(resamples, len(d)))
        d_err += d[idx].sum(axis=1)
        n_ref += n[idx].sum(axis=1)
    diff = 100.0 * d_err / n_ref
    lo, hi = np.percentile(diff, [2.5, 97.5])
    point = 100.0 * float((arm - base).sum()) / float(nref.sum())
    return {"diff_pp": round(point, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "excludes_zero": bool(hi < 0 or lo > 0), "unit": "speaker" if units is not None
            else "utterance", "resamples": resamples, "seed": seed}


def compare(sdi_arm: np.ndarray, sdi_base: np.ndarray, utts: list[dict]) -> dict:
    """WER rows and paired bootstraps (utterance- and speaker-level) for each dev set and pooled."""
    nref = np.array([u["nref"] for u in utts])
    split = np.array([u["split"] for u in utts])
    spk = np.array([u["speaker"] for u in utts])
    ea, eb = sdi_arm.sum(axis=1), sdi_base.sum(axis=1)
    out = {}
    for name in [*sorted(set(split)), "pooled"]:
        m = np.ones(len(utts), dtype=bool) if name == "pooled" else split == name
        a, b = wer_row(sdi_arm, nref, m), wer_row(sdi_base, nref, m)
        out[name] = {"arm": a, "baseline": b,
                     "rel_change_pct": round(100.0 * (a["wer"] - b["wer"]) / max(b["wer"], 1e-9), 2),
                     "bootstrap": bootstrap(ea[m], eb[m], nref[m], split[m]),
                     "bootstrap_speaker": bootstrap(ea[m], eb[m], nref[m], split[m], spk[m])}
    return out


def cross_validate(sdi: np.ndarray, sdi_base: np.ndarray, utts: list[dict], labels: list) -> dict:
    """Everything reported for one family of configs: sdi (configs, utts, 3)."""
    fold = np.array([u["fold"] for u in utts])
    split = np.array([u["split"] for u in utts])
    err = sdi.sum(axis=2)
    pick, chosen = held_out(err, fold)
    rows = np.arange(len(utts))
    res = {"held_out": compare(sdi[pick, rows], sdi_base, utts),
           "chosen": {"tuned_on_A_scored_on_B": labels[chosen[0]],
                      "tuned_on_B_scored_on_A": labels[chosen[1]]},
           "configs": len(labels)}
    best = choose(err, np.ones(len(utts), dtype=bool))
    res["tuned_on_itself"] = {"config": labels[best], "label": "tuned on itself",
                              **compare(sdi[best], sdi_base, utts)}
    per_set = {}
    for s in sorted(set(split)):
        p, c = held_out(err, fold, split == s)
        per_set[s] = {"chosen": [labels[c[0]], labels[c[1]]],
                      **compare(sdi[p, rows], sdi_base, utts)[s]}
    res["held_out_per_set_tuning"] = per_set
    return res


def rescore(a) -> None:
    import kenlm

    utts = load_utts(a.cache, a.splits)
    base = load_run(a.runs, a.baseline, a.splits, len(utts))
    nb = load_run(a.runs, a.nbest, a.splits, len(utts))
    lm = kenlm.Model(a.lm)
    k = max(len(nb[u["uid"]]["nbest"]) for u in utts)
    n = len(utts)
    score = np.full((n, k), -np.inf)
    lmlp, words = np.zeros((n, k)), np.zeros((n, k))
    pairs, where = [], []
    for i, u in enumerate(utts):
        for j, (_, sc, text) in enumerate(nb[u["uid"]]["nbest"]):
            score[i, j] = sc
            lmlp[i, j] = lm.score(text, bos=True, eos=True) * LN10
            words[i, j] = len(text.split())
            pairs.append((u["ref"], text))
            where.append((i, j))
    sdi_c = np.zeros((n, k, 3), dtype=np.int64)
    for (i, j), e in zip(where, count_errors(pairs, a.procs), strict=True):
        sdi_c[i, j] = e
    sdi_base = np.array(count_errors([(u["ref"], base[u["uid"]]["hyp"]) for u in utts], a.procs))
    sdi_greedy = None
    nref = np.array([u["nref"] for u in utts])
    split = np.array([u["split"] for u in utts])
    grid = sorted(((w, b) for w in np.arange(a.w[0], a.w[1] + 1e-9, a.w[2])
                   for b in np.arange(a.b[0], a.b[1] + 1e-9, a.b[2])),
                  key=lambda wb: (round(wb[0], 6), abs(round(wb[1], 6)), wb[1]))
    rows = np.arange(n)
    sdi = np.zeros((len(grid), n, 3), dtype=np.int64)
    for g, (w, b) in enumerate(grid):
        best = np.argmax(score + w * lmlp + b * words, axis=1)
        sdi[g] = sdi_c[rows, best]
    labels = [{"w": round(float(w), 3), "b": round(float(b), 3)} for w, b in grid]
    err_c = np.where(np.isfinite(score), sdi_c.sum(axis=2), 1 << 30)
    oracle = sdi_c[rows, np.argmin(err_c, axis=1)]
    first = sdi_c[:, 0]
    res = {"arm": f"N-best rescoring of {a.nbest} with {os.path.basename(a.lm)}",
           "baseline": a.baseline, "nbest_mean_size": round(float(np.isfinite(score).sum()) / n, 2),
           "grid": {"w": a.w, "b": a.b}, "exploratory": True,
           **cross_validate(sdi, sdi_base, utts, labels), "nbest_first": {}, "oracle": {}}
    for name in [*sorted(set(split)), "pooled"]:
        m = np.ones(n, dtype=bool) if name == "pooled" else split == name
        res["nbest_first"][name] = wer_row(first, nref, m)
        res["oracle"][name] = wer_row(oracle, nref, m)
    del sdi_greedy
    _write(a.out, res)


def fusion(a) -> None:
    utts = load_utts(a.cache, a.splits)
    base = load_run(a.runs, a.baseline, a.splits, len(utts))
    with open(a.grid) as fh:
        grid = json.load(fh)
    sdi_base = np.array(count_errors([(u["ref"], base[u["uid"]]["hyp"]) for u in utts], a.procs))
    families: dict[str, list[dict]] = {}
    for cfg in grid:
        families.setdefault(cfg["family"], []).append(cfg)
    res = {"baseline": a.baseline, "exploratory": True, "families": {}}
    res["baseline_decode_seconds"] = round(sum(base[u["uid"]]["sec"] for u in utts), 1)
    for fam, cfgs in families.items():
        cfgs = sorted(cfgs, key=lambda c: (c.get("lm_weight", 0.0), c.get("lodr_weight", 0.0),
                                           c.get("blank_penalty", 0.0)))
        sdi = np.zeros((len(cfgs), len(utts), 3), dtype=np.int64)
        secs = []
        for g, cfg in enumerate(cfgs):
            run = load_run(a.runs, cfg["name"], a.splits, len(utts))
            sdi[g] = count_errors([(u["ref"], run[u["uid"]]["hyp"]) for u in utts], a.procs)
            secs.append(round(sum(run[u["uid"]]["sec"] for u in utts), 1))
        labels = [{k: c[k] for k in ("name", "lm_weight", "lodr_weight", "blank_penalty")
                   if k in c} for c in cfgs]
        res["families"][fam] = cross_validate(sdi, sdi_base, utts, labels)
        nref = np.array([u["nref"] for u in utts])
        res["families"][fam]["per_config_all_utts"] = [
            {**lab, "wer": wer_row(sdi[g], nref)["wer"], "decode_seconds": secs[g]}
            for g, lab in enumerate(labels)]
    _write(a.out, res)


# The pre-registered Phase 2 grids (research/H42_ngram_lm/README.md).
CTRL_PENALTY = (0.0, 0.25, 0.5, 1.0)
FUSION_LM = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
FUSION_PENALTY = (0.0, 0.25, 0.5, 1.0)
LODR_LM = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
LODR_RATIO = (0.25, 0.5, 0.75)
LODR_PENALTY = (0.0, 0.25, 0.5)


def make_grid(tag: str, lm: str, lodr: str | None, beam: int = 4, baseline: str = "b4",
              fusion_lm=FUSION_LM, fusion_penalty=FUSION_PENALTY, lodr_lm=LODR_LM,
              lodr_ratio=LODR_RATIO, lodr_penalty=LODR_PENALTY) -> list[dict]:
    """Decode configs of one token LM: the no-LM blank-penalty control, fusion, fusion + LODR.
    The baseline (penalty 0, no LM) is the first control config under its own name."""
    grid = [{"name": baseline if pen == 0 else f"ctrl_p{pen:.2f}", "family": "ctrl", "beam": beam,
             "blank_penalty": pen} for pen in CTRL_PENALTY]
    for w in fusion_lm:
        for pen in fusion_penalty:
            grid.append({"name": f"{tag}_l{w:.2f}_r0.00_p{pen:.2f}", "family": f"{tag}_fusion",
                         "beam": beam, "blank_penalty": pen, "lm": lm, "lm_weight": w})
    for w in lodr_lm if lodr else ():
        for ratio in lodr_ratio:
            for pen in lodr_penalty:
                grid.append({"name": f"{tag}_l{w:.2f}_r{ratio:.2f}_p{pen:.2f}",
                             "family": f"{tag}_fusion_lodr", "beam": beam, "blank_penalty": pen,
                             "lm": lm, "lm_weight": w, "lodr": lodr,
                             "lodr_weight": round(w * ratio, 6)})
    return grid


def grid(a) -> None:
    with open(a.out, "w") as fh:
        json.dump(make_grid(a.tag, a.lm, a.lodr), fh, indent=1)
    print(f"{a.out}: {len(make_grid(a.tag, a.lm, a.lodr))} configs")


def _write(path: str, res: dict) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps(res, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("rescore", rescore), ("fusion", fusion)):
        p = sub.add_parser(name)
        p.add_argument("--cache", required=True)
        p.add_argument("--runs", required=True)
        p.add_argument("--baseline", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
        p.add_argument("--procs", type=int, default=8)
        p.set_defaults(fn=fn)
        if name == "rescore":
            p.add_argument("--nbest", required=True)
            p.add_argument("--lm", required=True)
            p.add_argument("--w", type=float, nargs=3, default=[0.0, 1.0, 0.05])
            p.add_argument("--b", type=float, nargs=3, default=[-2.0, 4.0, 0.5])
        else:
            p.add_argument("--grid", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--tag", required=True, help="short name of the token LM, e.g. t5")
    g.add_argument("--lm", required=True)
    g.add_argument("--lodr", default=None)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=grid)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
