"""H46 tuning on eval-lane's fixed dev speaker split v2 (selection half / confirmation half).

  python -m train.h46_tune select --cache ENC --runs RUNS --grid GRID --split dev_split_v2.json \
      --out select_<name>.json
  python -m train.h46_tune confirm --cache ENC --runs RUNS --select select_<name>.json \
      --baseline b4 --prefix ref_ --split dev_split_v2.json --scorer ultra-score \
      --dump DIR --out confirm_<name>.json [--extra NAME=RUN ...]

  python -m train.h46_tune engine --cache ENC --runs RUNS --engine-dir DIR --split S \
      --scorer ultra-score --dump DIR --out engine_parity.json \
      --pairs nolm_b4=b4 M5ship_b4=ref_M5_b4_l0.60_r0.75_p0.00

`select` chooses, per family of the grid, the config with the fewest `ultra.wer` errors on the
SELECTION half only (ties to the earliest config in train.lm_tune's preference order: smallest LM
weight, then LODR weight, then blank penalty). It reads no confirmation-half hypothesis. Its JSON
has train.lm_tune's `families[fam]["chosen"]` layout, so `python -m train.lm_tune extend` runs the
pre-registered edge rule on the selection-half choice alone.

`confirm` scores each family's finalist (the run `<prefix><name>`, i.e. the reference re-decode)
against the no-LM comparator: the canonical `ultra-score` paired speaker-cluster bootstrap on the
confirmation half (the decision statistic) and on full dev (the reported endpoint), plus
train.lm_tune's utterance / speaker bootstraps on `ultra.wer` counts. `--extra` adds labelled rows
that are not chosen by the split (e.g. the transfer row at H42's ship weights).

`engine` compares Rust engine runs (`ultra-engine-eval` shard runs under <engine-dir>/<name>/*/
engine_eval.json, see train/ops/h46_engine_fulldev.sh) with Python runs of the same weights on the
same utterances: canonical WER of each side, each side's paired change against its own first pair
(the no-LM runs), identical token sequences and texts, and the engine's CPU-s per audio-s.

Run from python/ultra (imports train.lm_tune, ultra.wer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

from train.errors import LmEvalError
from train.lm_tune import compare, count_errors, dump_hyps, load_run, load_utts, wer_row

SEED = 20261122
REPLICATES = 10000
ALPHA = 0.025  # two-sided 95% percentile interval from ultra-score's one-sided bounds


def load_split(path: str) -> tuple[dict[str, str], str]:
    with open(path, "rb") as fh:
        raw = fh.read()
    d = json.loads(raw)
    half = {}
    for h in ("selection", "confirmation"):
        for uid in d["utterances"][h]:
            if uid in half:
                raise LmEvalError(f"{uid} is in both halves of {path}")
            half[uid] = h
    return half, hashlib.sha256(raw).hexdigest()


def masks(utts: list[dict], half: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    missing = [u["uid"] for u in utts if u["uid"] not in half]
    if missing or len(half) != len(utts):
        raise LmEvalError(f"split does not cover the cache exactly: {len(missing)} missing, "
                         f"{len(half)} in split, {len(utts)} in cache")
    sel = np.array([half[u["uid"]] == "selection" for u in utts])
    return sel, ~sel


def families_of(grid: list[dict]) -> dict[str, list[dict]]:
    fams: dict[str, list[dict]] = {}
    for cfg in grid:
        fams.setdefault(cfg["family"], []).append(cfg)
    return {f: sorted(c, key=lambda c: (c.get("lm_weight", 0.0), c.get("lodr_weight", 0.0),
                                        c.get("blank_penalty", 0.0)))
            for f, c in fams.items()}


def select(a) -> None:
    utts = load_utts(a.cache, a.splits)
    half, split_sha = load_split(a.split)
    sel, _ = masks(utts, half)
    sel_utts = [u for u, s in zip(utts, sel, strict=True) if s]
    nref = np.array([u["nref"] for u in sel_utts])
    with open(a.grid) as fh:
        grid = json.load(fh)
    res = {"protocol": "eval-lane dev split v2: choose on the selection half only",
           "split": a.split, "split_sha256": split_sha, "grid": a.grid, "exploratory": True,
           "selection_utterances": len(sel_utts), "families": {}}
    for fam, cfgs in families_of(grid).items():
        rows = []
        for cfg in cfgs:
            run = load_run(a.runs, cfg["name"], a.splits, len(utts))
            sdi = np.array(count_errors([(u["ref"], run[u["uid"]]["hyp"]) for u in sel_utts],
                                        a.procs))
            rows.append({"name": cfg["name"], **{k: cfg[k] for k in
                                                 ("lm_weight", "lodr_weight", "blank_penalty")
                                                 if k in cfg},
                         **wer_row(sdi, nref)})
        err = [r["sub"] + r["del"] + r["ins"] for r in rows]
        best = int(np.argmin(err))  # first minimum = earliest in preference order
        label = {k: rows[best][k] for k in ("name", "lm_weight", "lodr_weight", "blank_penalty")
                 if k in rows[best]}
        res["families"][fam] = {"chosen": {"tuned_on_selection": label},
                                "selection_half": rows, "configs": len(rows)}
    _write(a.out, res)


def ultra_score(scorer: str, base: str, arm: str) -> dict:
    out = subprocess.run([scorer, base, arm, "--seed", str(SEED), "--replicates", str(REPLICATES),
                          "--alpha", str(ALPHA)], check=True, capture_output=True, text=True)
    rep = json.loads(out.stdout)
    c = rep["comparison"]

    def counts(sysrep):
        tot = {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
        per = {}
        for ds, d in sysrep["datasets"].items():
            per[ds] = {"wer": round(100 * d["wer"], 3), **{k: d["counts"][k] for k in tot}}
            for k in tot:
                tot[k] += d["counts"][k]
        return {"pooled_wer": round(100 * sysrep["pooled_wer"], 3),
                "macro_wer": round(100 * sysrep["macro_wer"], 3), **tot,
                "errors": tot["substitutions"] + tot["deletions"] + tot["insertions"],
                "per_set": per}

    def boot(b):
        return {"delta_pp": round(100 * b["delta"], 3),
                "relative_pct": None if b["relative_delta"] is None
                else round(100 * b["relative_delta"], 2),
                "lo_pp": round(100 * b["lower"], 3), "hi_pp": round(100 * b["upper"], 3),
                "limited": b["limited"]}

    return {"normaliser_sha256": rep["normaliser_sha256"], "baseline": counts(rep["system_a"]),
            "arm": counts(rep["system_b"]), "pooled_delta": boot(c["pooled_delta"]),
            "macro_delta": boot(c["macro_delta"]),
            "per_set_delta": {k: boot(v) for k, v in c["per_dataset"].items()},
            "bootstrap": {"unit": "speaker", "replicates": REPLICATES, "seed": SEED,
                          "interval": "two-sided 95% (alpha 0.025 per side)"}}


def confirm(a) -> None:
    utts = load_utts(a.cache, a.splits)
    half, split_sha = load_split(a.split)
    sel, conf = masks(utts, half)
    with open(a.select) as fh:
        chosen = json.load(fh)
    base = load_run(a.runs, a.baseline, a.splits, len(utts))
    os.makedirs(a.dump, exist_ok=True)
    with open(a.scorer, "rb") as fh:
        scorer_sha = hashlib.sha256(fh.read()).hexdigest()
    res = {"protocol": "eval-lane dev split v2: finalist chosen on the selection half, scored on "
                       "the confirmation half (decision) and on full dev (endpoint)",
           "split_sha256": split_sha, "select": a.select, "baseline": a.baseline,
           "scorer": a.scorer, "scorer_sha256": scorer_sha, "exploratory": True, "rows": {}}
    rows = {fam: (f["chosen"]["tuned_on_selection"], a.prefix + f["chosen"]["tuned_on_selection"]
                  ["name"]) for fam, f in chosen["families"].items()}
    for extra in a.extra or []:
        label, run_name = extra.split("=", 1)
        rows[label] = ({"name": run_name, "label": "not chosen by the split"}, run_name)
    sdi_base = np.array(count_errors([(u["ref"], base[u["uid"]]["hyp"]) for u in utts], a.procs))
    for half_name, m in (("full", np.ones(len(utts), dtype=bool)), ("confirmation", conf),
                         ("selection", sel)):
        sub = [u for u, keep in zip(utts, m, strict=True) if keep]
        dump_hyps(os.path.join(a.dump, f"{a.baseline}.{half_name}.jsonl"), sub,
                  [base[u["uid"]]["hyp"] for u in sub])
    for fam, (label, run_name) in rows.items():
        if run_name == a.baseline:
            continue
        run = load_run(a.runs, run_name, a.splits, len(utts))
        sdi = np.array(count_errors([(u["ref"], run[u["uid"]]["hyp"]) for u in utts], a.procs))
        row = {"finalist": label, "run": run_name}
        for half_name, m in (("confirmation", conf), ("full", np.ones(len(utts), dtype=bool)),
                             ("selection", sel)):
            sub = [u for u, keep in zip(utts, m, strict=True) if keep]
            path = os.path.join(a.dump, f"{run_name}.{half_name}.jsonl")
            dump_hyps(path, sub, [run[u["uid"]]["hyp"] for u in sub])
            row[half_name] = {
                "canonical": ultra_score(a.scorer, os.path.join(
                    a.dump, f"{a.baseline}.{half_name}.jsonl"), path),
                "ultra_wer": compare(sdi[m], sdi_base[m], sub)["pooled"]}
        ctx = sum(r.get("lm_contexts", 0) for r in run.values())
        audio = sum(u["seconds"] for u in utts)
        row["cost"] = {"decode_seconds": round(sum(r["sec"] for r in run.values()), 1),
                       "baseline_decode_seconds": round(sum(r["sec"] for r in base.values()), 1),
                       "lm_contexts_per_audio_s": round(ctx / audio, 2) if ctx else None}
        row["identical_tokens_to_baseline"] = sum(run[u["uid"]]["tokens"] == base[u["uid"]]
                                                  ["tokens"] for u in utts)
        res["rows"][fam] = row
    _write(a.out, res)


def engine_rows(folder: str) -> tuple[dict[str, dict], dict]:
    """uid -> engine row, merged over the shard runs in `folder`, and summed run costs."""
    import glob

    rows: dict[str, dict] = {}
    cost = {"cpu_s": 0.0, "audio_s": 0.0, "peak_rss_mib": 0.0, "runs": 0}
    meta = {}
    for f in sorted(glob.glob(os.path.join(folder, "*", "engine_eval.json"))):
        with open(f) as fh:
            d = json.load(fh)
        for r in d["rows"]:
            if r["id"] in rows:
                raise LmEvalError(f"{r['id']} decoded twice under {folder}")
            rows[r["id"]] = r
        cost["cpu_s"] += d["cpu_s"]
        cost["audio_s"] += d["audio_s"]
        cost["peak_rss_mib"] = max(cost["peak_rss_mib"], d["peak_rss_mib"])
        cost["runs"] += 1
        meta = {k: d.get(k) for k in ("executable_sha256", "git_commit", "git_dirty", "isa", "host",
                                      "int8_prepared_sha256", "weights_sha256", "lm", "lm_weight",
                                      "lodr_weight", "blank_penalty", "beam", "streams",
                                      "packet_ms", "precision")}
    cost["cpu_s_per_audio_s"] = round(cost["cpu_s"] / max(cost["audio_s"], 1e-9), 4)
    return rows, {**meta, **cost}


def engine(a) -> None:
    utts = load_utts(a.cache, a.splits)
    half, split_sha = load_split(a.split)
    _, conf = masks(utts, half)
    os.makedirs(a.dump, exist_ok=True)
    with open(a.scorer, "rb") as fh:
        scorer_sha = hashlib.sha256(fh.read()).hexdigest()
    res = {"split_sha256": split_sha, "scorer_sha256": scorer_sha, "exploratory": True,
           "pairs": {}}
    first = None
    for pair in a.pairs:
        eng_name, py_name = pair.split("=", 1)
        erows, emeta = engine_rows(os.path.join(a.engine_dir, eng_name))
        if set(erows) != {u["uid"] for u in utts}:
            raise LmEvalError(f"{eng_name}: engine rows cover {len(erows)} utterances, the cache "
                              f"{len(utts)}")
        py = load_run(a.runs, py_name, a.splits, len(utts))
        out = {"engine_run": eng_name, "python_run": py_name, "engine": emeta,
               "identical_tokens": sum(erows[u["uid"]]["tokens"] == py[u["uid"]]["tokens"]
                                       for u in utts),
               "identical_text": sum(erows[u["uid"]]["hyp"] == py[u["uid"]]["hyp"]
                                     for u in utts),
               "utterances": len(utts)}
        for side, hyp in (("engine", lambda u: erows[u["uid"]]["hyp"]),
                          ("python", lambda u: py[u["uid"]]["hyp"])):
            for half_name, m in (("full", np.ones(len(utts), dtype=bool)), ("confirmation", conf)):
                sub = [u for u, keep in zip(utts, m, strict=True) if keep]
                path = os.path.join(a.dump, f"{side}.{eng_name}.{half_name}.jsonl")
                dump_hyps(path, sub, [hyp(u) for u in sub])
                if first is None:
                    rep = ultra_score(a.scorer, path, path)
                    out.setdefault(side, {})[half_name] = {"wer": rep["arm"]}
                else:
                    base = os.path.join(a.dump, f"{side}.{first}.{half_name}.jsonl")
                    rep = ultra_score(a.scorer, base, path)
                    out.setdefault(side, {})[half_name] = {
                        "wer": rep["arm"], "change_vs_" + first: rep["pooled_delta"],
                        "per_set_change": rep["per_set_delta"]}
        res["pairs"][pair] = out
        if first is None:
            first = eng_name
    _write(a.out, res)


def _write(path: str, res: dict) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps(res, indent=1)[:4000])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--grid", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--select", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--prefix", default="ref_")
    c.add_argument("--scorer", required=True)
    c.add_argument("--dump", required=True)
    c.add_argument("--extra", nargs="*", help="LABEL=RUN rows scored the same way")
    e = sub.add_parser("engine")
    e.add_argument("--engine-dir", required=True)
    e.add_argument("--pairs", nargs="+", required=True, help="<engine run>=<python run>; the "
                   "first pair is the no-LM comparator of the others")
    e.add_argument("--scorer", required=True)
    e.add_argument("--dump", required=True)
    for p in (s, c, e):
        p.add_argument("--cache", required=True)
        p.add_argument("--runs", required=True)
        p.add_argument("--split", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
        p.add_argument("--procs", type=int, default=8)
    s.set_defaults(fn=select)
    c.set_defaults(fn=confirm)
    e.set_defaults(fn=engine)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
