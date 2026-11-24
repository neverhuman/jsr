"""The measured next candidate (RC2): accuracy from its single registered confirmatory look.

RC2 is NOT the release; it missed registered gates (docs/ultra/releases/v3.1.0-rc2.md). This module reads the
pinned `ultra-score` receipts in raw/versus/v310 (paired, 730-utt panel and the unselected C half of DEV) and
raw/release_v31 (full-DEV reports, the Holm audit against shipped v3.0.0 outputs) and emits \\Cand* macros plus
data/table_candidate.tex. No timing here: speed claims wait for the quiet-host receipts.
"""

from __future__ import annotations

from pathlib import Path
import json

from .common import DATA, RAW, PaperDataError, fmt, sha256_file, thousands
from .records import Records
from .words import signed as _sg, tex

V = RAW / "versus" / "v310"
R = RAW / "release_v31"
# (key, label, chunk ms, compare with shipped v3.0.0 receipt or None)
MODES = (
    ("live_c4_greedy", "Live, greedy", 160, "shipped-v300-vs-rc2-live"),
    ("live_c4_lm", "Live + 34.5\\,MiB LM", 160, "shipped-v300-vs-rc2-live_c4_lm"),
    ("live_c8_lm", "Live + 34.5\\,MiB LM", 320, "shipped-v300-vs-rc2-live_c8_lm"),
    ("final_m5", "Final, M5 LM", 640, "shipped-v300-vs-rc2-final"),
    ("final_lowmem", "Final, 146.8\\,MiB LM", 640, None),
)
TAG = {"live_c4_greedy": "LiveGreedy", "live_c4_lm": "LiveLm", "live_c8_lm": "LiveLmEight",
       "final_m5": "Final", "final_lowmem": "FinalLowmem"}


def _load(root: Path, rel: Path) -> dict:
    p = root / rel
    if not p.is_file():
        raise PaperDataError(f"candidate: missing receipt {rel}")
    return json.loads(p.read_text(encoding="utf-8"))


def _pair(d: dict) -> dict:
    c = d["comparison"]["pooled_delta"]
    return {"a": 100 * d["system_a"]["pooled_wer"], "b": 100 * d["system_b"]["pooled_wer"],
            "delta": 100 * c["delta"], "lo": 100 * c["lower"], "hi": 100 * c["upper"],
            "utts": d["system_b"]["utterances"]}


def _edits(d: dict, side: str = "system_a") -> int:
    return sum(v["counts"]["substitutions"] + v["counts"]["deletions"] + v["counts"]["insertions"]
               for v in d[side]["datasets"].values())


def _bounds(p: dict) -> str:
    return f"{tex(_sg(p['delta'], 2))} [{tex(_sg(p['lo'], 2))}, {tex(_sg(p['hi'], 2))}]"


def add_candidate(root: Path, rec: Records, out: dict[str, str]) -> None:
    if not (root / V).is_dir():
        return
    rows = []
    for key, label, chunk, old in MODES:
        panel = _pair(_load(root, V / f"zip-vs-rc2-{key}.panel730.score.json"))
        chalf = _pair(_load(root, V / f"zip-vs-rc2-{key}.c_half.score.json"))
        full = _load(root, R / f"rc2-{key}.fulldev.score.json")
        if panel["utts"] != 730 or full["system_a"]["utterances"] != 5567:
            raise PaperDataError(f"candidate: {key} receipts are not the 730 panel / full DEV")
        vs_old = _pair(_load(root, V / f"{old}.panel730.score.json")) if old else None
        if vs_old and abs(vs_old["b"] - panel["b"]) > 1e-9:
            raise PaperDataError(f"candidate: {key} shipped-v3.0.0 pair is not on the same RC2 outputs")
        rows.append({"key": key, "label": label, "chunk": chunk, "panel": panel, "c": chalf,
                     "full": 100 * full["system_a"]["pooled_wer"], "full_edits": _edits(full),
                     "panel_edits": _edits(_load(root, V / f"zip-vs-rc2-{key}.panel730.score.json"), "system_b"),
                     "old": vs_old})
    holm = _load(root, R / "rc2-panel-holm-shipped-v300.json")
    claims = {c["name"]: c for c in holm["claims"]}
    hsrc = str(R / "rc2-panel-holm-shipped-v300.json")
    hdig = sha256_file(root / R / "rc2-panel-holm-shipped-v300.json")

    def put(rid, value, raw, unit, what, src, dig, workload):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what,
                workload=workload, workload_digest=dig, source=src)

    for r in rows:
        t = TAG[r["key"]]
        psrc = str(V / f"zip-vs-rc2-{r['key']}.panel730.score.json")
        pdig = sha256_file(root / psrc)
        csrc = str(V / f"zip-vs-rc2-{r['key']}.c_half.score.json")
        cdig = sha256_file(root / csrc)
        put(f"Cand{t}Wer", fmt(r["panel"]["b"], 2), r["panel"]["b"], "% WER", f"RC2 {r['key']} pooled panel WER", psrc, pdig, "730-utt DEV panel")
        put(f"Cand{t}Edits", str(r["panel_edits"]), r["panel_edits"], "edits", f"RC2 {r['key']} panel edits", psrc, pdig, "730-utt DEV panel")
        put(f"Cand{t}FullWer", fmt(r["full"], 2), r["full"], "% WER", f"RC2 {r['key']} full-DEV pooled WER (descriptive: contains the selection half)",
            str(R / f"rc2-{r['key']}.fulldev.score.json"), sha256_file(root / R / f"rc2-{r['key']}.fulldev.score.json"), "full LibriSpeech DEV (5,567 utt)")
        for which, p, src, dig, wl in (("Panel", r["panel"], psrc, pdig, "730-utt DEV panel"),
                                       ("C", r["c"], csrc, cdig, "DEV C half (2,785 utt, never used for selection)")):
            put(f"Cand{t}{which}Delta", tex(_sg(p["delta"], 2)), p["delta"], "pp WER", f"RC2 {r['key']} minus Zipformer INT8 ({which})", src, dig, wl)
            put(f"Cand{t}{which}Lo", tex(_sg(p["lo"], 2)), p["lo"], "pp WER", f"one-sided 95% lower bound ({which})", src, dig, wl)
            put(f"Cand{t}{which}Hi", tex(_sg(p["hi"], 2)), p["hi"], "pp WER", f"one-sided 95% upper bound ({which})", src, dig, wl)
        if r["old"]:
            osrc = str(V / f"{dict((k, o) for k, _, _, o in MODES)[r['key']]}.panel730.score.json")
            odig = sha256_file(root / osrc)
            put(f"Cand{t}OldDelta", tex(_sg(r["old"]["delta"], 2)), r["old"]["delta"], "pp WER",
                f"RC2 {r['key']} minus shipped v3.0.0 (same-family mode)", osrc, odig, "730-utt DEV panel")
            put(f"Cand{t}OldLo", tex(_sg(r["old"]["lo"], 2)), r["old"]["lo"], "pp WER", "one-sided 95% lower bound vs v3.0.0", osrc, odig, "730-utt DEV panel")
            put(f"Cand{t}OldHi", tex(_sg(r["old"]["hi"], 2)), r["old"]["hi"], "pp WER", "one-sided 95% upper bound vs v3.0.0", osrc, odig, "730-utt DEV panel")
    zc = rows[0]["c"]["a"]
    zrel = R / "zipformer_int8.fulldev.score.json"
    zfull = 100 * _load(root, zrel)["system_a"]["pooled_wer"]
    put("CandZipFullWer", fmt(zfull, 2), zfull, "% WER", "Zipformer INT8 full-DEV pooled WER", str(zrel), sha256_file(root / zrel), "full LibriSpeech DEV (5,567 utt)")
    put("CandZipCWer", fmt(zc, 2), zc, "% WER", "Zipformer INT8 on the DEV C half", str(V / "zip-vs-rc2-final_m5.c_half.score.json"),
        sha256_file(root / V / "zip-vs-rc2-final_m5.c_half.score.json"), "DEV C half (2,785 utt)")
    for rid, name in (("CandHolmNi", "live_versus_zipformer"), ("CandHolmFinal", "m5_versus_zipformer"),
                      ("CandHolmOld", "rc2_versus_v300")):
        c = claims[name]
        if c["reject"]:
            raise PaperDataError(f"candidate: Holm claim {name} now rejects; reword the candidate section")
        put(rid, fmt(c["holm_p"], 2), c["holm_p"], "p", f"Holm-adjusted p, {name}", hsrc, hdig, "730-utt DEV panel")
    put("CandNiMargin", fmt(claims["live_versus_zipformer"]["margin_pp"], 2), claims["live_versus_zipformer"]["margin_pp"], "pp",
        "registered non-inferiority margin", hsrc, hdig, "730-utt DEV panel")
    fin = next(r for r in rows if r["key"] == "final_m5")
    GATES = _load(root, R / "gates.json")
    gsrc, gdig = str(R / "gates.json"), sha256_file(root / R / "gates.json")
    for rid, v in (("CandGateLivePanel", GATES["live_panel_edits"]), ("CandGateFinalPanel", GATES["final_panel_edits"]),
                   ("CandGateFinalFull", GATES["final_fulldev_edits"])):
        put(rid, thousands(str(v)), v, "edits", "registered gate", gsrc, gdig, "gate")
    put("CandFinalFullEdits", thousands(str(fin["full_edits"])), fin["full_edits"], "edits", "RC2 final M5 full-DEV edits",
        str(R / "rc2-final_m5.fulldev.score.json"), sha256_file(root / R / "rc2-final_m5.fulldev.score.json"), "full LibriSpeech DEV (5,567 utt)")

    import csv
    lin = list(csv.DictReader((root / R / "lineage.csv").open(encoding="utf-8")))
    kh = sum(float(x["audio_h"]) for x in lin)
    put("CandTrainKh", thousands(fmt(kh, 0)), kh, "audio-h", "RC2 total logged training audio (7 hops; ~32 h unlogged excluded)",
        str(R / "lineage.csv"), sha256_file(root / R / "lineage.csv"), "training logs (train.jsonl audio_s sums)")
    tx = [r"\begin{table*}[t]", r"\centering",
          r"\caption{\textbf{Measured next candidate (RC2), single registered look.} Not the release: it missed its "
          r"registered gates. $\Delta$ is RC2 minus Zipformer INT8 (or minus shipped v3.0.0) on the same utterances, with "
          r"one-sided 95\% bounds from the pinned scorer; negative means fewer errors. The 730-utterance panel overlaps "
          r"the half of DEV used for selection; the C half (2,785 utterances) was never used. Full-DEV WER is descriptive. "
          r"Final rows run the 160\,ms live stream and commit with a 640\,ms chunk at reference utterance boundaries.}",
          r"\label{tab:candidate}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}lrrrlll@{}}", r"\toprule",
          r"Mode & Chunk (ms) & Panel WER & Full-DEV WER & $\Delta$ vs.\ Zip.\ (panel) & $\Delta$ vs.\ Zip.\ (C half) & $\Delta$ vs.\ v3.0.0 (panel) \\",
          r"\midrule"]
    for r in rows:
        tx.append(" & ".join([r["label"], str(r["chunk"]), fmt(r["panel"]["b"], 2), fmt(r["full"], 2),
                              _bounds(r["panel"]), _bounds(r["c"]), _bounds(r["old"]) if r["old"] else "---"]) + r" \\")
    tx += [r"\midrule", " & ".join(["Zipformer INT8", "320", fmt(rows[0]["panel"]["a"], 2), fmt(zfull, 2), "---", f"WER {fmt(zc, 2)}", "---"]) + r" \\",
           r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out["table_candidate.tex"] = "\n".join(tx) + "\n"
