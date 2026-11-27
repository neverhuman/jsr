"""v4.2.0 final pass (F3 soup, TRAIN-30% gated B48 re-search, exact H261 t2 rerank): accuracy from its frozen DEV look.

Reads the pinned `ultra-score` receipts in raw/versus/v420 (10,000 speaker-cluster replicates, seed 20261122): the
730-utterance panel and the unselected C half of DEV against Zipformer INT8 and against the v4.1.0 final outputs, the
selection S half against v4.1.0 final and against the research F3 B16 exact-H261 list (the gate's own effect), and full
DEV (S + C, descriptive). Emits \\Fin* macros. The release binary (isosyll v4.2.0) produced every v4.2.0 transcript;
live mode is unchanged from v4.1.0 (look41.py).
"""

from __future__ import annotations

from pathlib import Path
import json

from .common import PaperDataError, RAW, fmt, sha256_file, thousands
from .look41 import _wer
from .records import Records
from .words import signed as _sg, tex

V = RAW / "versus" / "v420"
SPLITS = {"panel730": ("Panel", 730), "c_half": ("C", 2785), "shalf": ("S", 2782)}
WORK = {"panel730": "DEV selection panel, 730 utt", "c_half": "DEV confirmation (C) half, 2,785 utt",
        "shalf": "DEV selection (S) half, 2,782 utt"}


def _load(root: Path, name: str) -> tuple[dict, str, str]:
    p = root / V / name
    if not p.is_file():
        raise PaperDataError(f"look42: missing receipt {V / name}")
    return json.loads(p.read_text(encoding="utf-8")), str(V / name), sha256_file(p)


def add_look42(root: Path, rec: Records) -> None:
    if not (root / V).is_dir():
        return

    def put(rid, value, raw, unit, what, src, dig, workload):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what, workload=workload,
                workload_digest=dig, source=src)

    for split, (stag, n) in SPLITS.items():
        work = WORK[split]
        pairs = [("Prev", f"v41-vs-v42-final.{split}.score.json", "v4.1.0 final")]
        if split != "shalf":
            pairs.append(("Zip", f"zip-vs-v42-final.{split}.score.json", "Zipformer INT8"))
        else:
            pairs.append(("Base", f"f3b16-vs-v42-final.{split}.score.json", "the ungated F3 B16 exact-H261 research list"))
        for who, name, label in pairs:
            d, src, dig = _load(root, name)
            a, b = _wer(d["system_a"]), _wer(d["system_b"])
            if b["utts"] != n or a["utts"] != n:
                raise PaperDataError(f"look42: {name} has {b['utts']} utterances, expected {n}")
            c = d["comparison"]["pooled_delta"]
            if who == "Prev":  # the v4.2.0 numbers themselves, once per split
                for k, key in (("wer", "Wer"), ("clean", "Clean"), ("other", "Other")):
                    put(f"Fin{stag}{key}", fmt(b[k], 2), b[k], "% WER", f"v4.2.0 final, {work}: pooled {k}", src, dig, work)
                    put(f"FinPrev{stag}{key}", fmt(a[k], 2), a[k], "% WER", f"v4.1.0 final, {work}: pooled {k}", src, dig, work)
                put(f"Fin{stag}Edits", thousands(str(b["edits"])), b["edits"], "edits", f"v4.2.0 final, {work}: edits", src, dig, work)
            if who == "Zip":
                put(f"Fin{stag}ZipClean", fmt(a["clean"], 2), a["clean"], "% WER", f"Zipformer INT8, {work}: pooled clean",
                    src, dig, work)
            if who == "Base":
                put(f"FinBase{stag}Wer", fmt(a["wer"], 2), a["wer"], "% WER", f"ungated F3 B16 exact H261 (research), {work}",
                    src, dig, work)
            for k, v in (("Delta", c["delta"]), ("Lo", c["lower"]), ("Hi", c["upper"])):
                put(f"Fin{stag}{who}{k}", tex(_sg(100 * v, 2)), 100 * v, "pp WER",
                    f"v4.2.0 final minus {label}, {work}, paired speaker-cluster bootstrap ({k})", src, dig, work)
    d, src, dig = _load(root, "v42-final.fulldev.score.json")
    f = _wer(d["system_a"])
    if f["utts"] != 5567:
        raise PaperDataError(f"look42: full DEV receipt has {f['utts']} utterances")
    for k, key in (("wer", "Wer"), ("clean", "Clean"), ("other", "Other")):
        put(f"FinFull{key}", fmt(f[k], 2), f[k], "% WER", f"v4.2.0 final, full DEV (S + C, descriptive): pooled {k}",
            src, dig, "LibriSpeech DEV, 5,567 utt")
    put("FinFullEdits", thousands(str(f["edits"])), f["edits"], "edits", "v4.2.0 final, full DEV (S + C, descriptive): edits",
        src, dig, "LibriSpeech DEV, 5,567 utt")


def add_final_pass(root: Path, rec: Records) -> None:
    """Constants of the v4.2.0 final pass from raw/release_v42/final-pass.json: soup weights, gate route share, cost budget."""
    rel = RAW / "release_v42" / "final-pass.json"
    p = root / rel
    if not p.is_file():
        return
    d, dig = json.loads(p.read_text(encoding="utf-8")), sha256_file(p)
    g = d["gate"]
    share = 100.0 * g["train_routed"] / g["train_utterances"]
    if round(share) != g["train_route_pct"]:
        raise PaperDataError(f"look42: gate routes {share:.2f}% of TRAIN, receipt says {g['train_route_pct']}%")

    def put(rid, value, raw, unit, what):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what,
                workload="none (configuration or descriptive constant)", workload_digest="n/a", source=f"{rel} sha256 {dig[:12]}")
    put("FinGateTrainPct", f"{share:.0f}", share, "% of TRAIN utterances", "v4.2.0 final: share of training utterances the gate margin routes to beam 48")
    put("FinGateWideBeam", str(g["wide_beam"]), g["wide_beam"], "beam width", "v4.2.0 final: re-search beam width of gated utterances")
    put("FinGateBudgetPct", str(d["cost_budget"]["gate_cpu_pct"]), d["cost_budget"]["gate_cpu_pct"], "% CPU", "budget set for the gate's added CPU time")
    put("FinSoupWeights", d["final_acoustic"]["weights_text"], None, "weights", "v4.2.0 final acoustic model: weight-average coefficients")
