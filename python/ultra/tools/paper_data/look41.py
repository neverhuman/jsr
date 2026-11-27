"""v4.1.0 (H247 step 450 weights, beam-12 compact-M5 live decoder): accuracy from its single frozen-finalist DEV look.

Reads the pinned `ultra-score` receipts in raw/versus/v410 (10,000 speaker-cluster replicates, seed 20261122):
the 730-utterance panel and the unselected C half of DEV against Zipformer INT8 and against the v4.0.0 outputs
(the v3.1.0 RC2 weights and decoder), the selection S half against v4.0.0, and full DEV (S + C, descriptive).
Emits \\Look* macros. The release binary (isosyll v4.1.0) produced every transcript.
"""

from __future__ import annotations

from pathlib import Path
import json

from .common import PaperDataError, RAW, fmt, sha256_file, thousands
from .records import Records
from .words import signed as _sg, tex

V = RAW / "versus" / "v410"
MODES = (("live", "Live"), ("final", "Final"))
SPLITS = {"panel730": ("Panel", 730), "c_half": ("C", 2785), "shalf": ("S", 2782)}


def _load(root: Path, name: str) -> tuple[dict, str, str]:
    p = root / V / name
    if not p.is_file():
        raise PaperDataError(f"look41: missing receipt {V / name}")
    return json.loads(p.read_text(encoding="utf-8")), str(V / name), sha256_file(p)


def _wer(side: dict) -> dict:
    ds = side["datasets"]
    return {"wer": 100 * side["pooled_wer"], "clean": 100 * ds["dev-clean"]["wer"], "other": 100 * ds["dev-other"]["wer"],
            "edits": sum(v["counts"]["substitutions"] + v["counts"]["deletions"] + v["counts"]["insertions"] for v in ds.values()),
            "words": sum(v["counts"]["ref_words"] for v in ds.values()), "utts": side["utterances"]}


def add_look41(root: Path, rec: Records) -> None:
    if not (root / V).is_dir():
        return

    def put(rid, value, raw, unit, what, src, dig, workload):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what, workload=workload,
                workload_digest=dig, source=src)

    for mode, tag in MODES:
        for split, (stag, n) in SPLITS.items():
            work = {"panel730": "DEV selection panel, 730 utt", "c_half": "DEV confirmation (C) half, 2,785 utt",
                    "shalf": "DEV selection (S) half, 2,782 utt"}[split]
            pairs = [("Prev", f"v4-vs-v41-{mode}.{split}.score.json", "v4.0.0")]
            if split != "shalf":
                pairs.append(("Zip", f"zip-vs-v41-{mode}.{split}.score.json", "Zipformer INT8"))
            for who, name, label in pairs:
                d, src, dig = _load(root, name)
                a, b = _wer(d["system_a"]), _wer(d["system_b"])
                if b["utts"] != n or a["utts"] != n:
                    raise PaperDataError(f"look41: {name} has {b['utts']} utterances, expected {n}")
                c = d["comparison"]["pooled_delta"]
                if who == "Prev":  # the v4.1.0 numbers themselves, once per split
                    for k, unit in (("wer", "% WER"), ("clean", "% WER"), ("other", "% WER")):
                        key = {"wer": "Wer", "clean": "Clean", "other": "Other"}[k]
                        put(f"Look{tag}{stag}{key}", fmt(b[k], 2), b[k], unit, f"v4.1.0 {mode}, {work}: pooled {k}", src, dig, work)
                        put(f"LookPrev{tag}{stag}{key}", fmt(a[k], 2), a[k], unit, f"v4.0.0 {mode}, {work}: pooled {k}", src, dig, work)
                    put(f"Look{tag}{stag}Edits", thousands(str(b["edits"])), b["edits"], "edits", f"v4.1.0 {mode}, {work}: edits", src, dig, work)
                for k, v in (("Delta", c["delta"]), ("Lo", c["lower"]), ("Hi", c["upper"])):
                    put(f"Look{tag}{stag}{who}{k}", tex(_sg(100 * v, 2)), 100 * v, "pp WER",
                        f"v4.1.0 {mode} minus {label}, {work}, paired speaker-cluster bootstrap ({k})", src, dig, work)
                if who == "Zip":
                    put(f"Look{tag}{stag}ZipClean", fmt(a["clean"], 2), a["clean"], "% WER", f"Zipformer INT8 dev-clean, {work}", src, dig, work)
        d, src, dig = _load(root, f"v41-{mode}.fulldev.score.json")
        f = _wer(d["system_a"])
        if f["utts"] != 5567:
            raise PaperDataError(f"look41: full DEV receipt for {mode} has {f['utts']} utterances")
        for k, key in (("wer", "Wer"), ("clean", "Clean"), ("other", "Other")):
            put(f"Look{tag}Full{key}", fmt(f[k], 2), f[k], "% WER", f"v4.1.0 {mode}, full DEV (S + C, descriptive): pooled {k}",
                src, dig, "LibriSpeech DEV, 5,567 utt")
        put(f"Look{tag}FullEdits", thousands(str(f["edits"])), f["edits"], "edits", f"v4.1.0 {mode}, full DEV (S + C, descriptive): edits",
            src, dig, "LibriSpeech DEV, 5,567 utt")
    put("LookCUtts", thousands("2785"), 2785, "utterances", "unselected C half of DEV", str(V), "dev_split_v2", "DEV C half")
