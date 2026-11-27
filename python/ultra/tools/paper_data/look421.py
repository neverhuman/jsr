"""v4.2.1 live mode (v4.1.0 live decoder + research H318 endpoint rerank): accuracy from its DEV look.

Reads the pinned `ultra-score` receipts in raw/versus/v421 (10,000 speaker-cluster replicates, seed 20261122): the
730-utterance panel and the unselected C half of DEV against Zipformer INT8 and against the v4.1.0 live outputs (v4.2.0
live is the same decoder and gives the same words), the selection S half against v4.1.0 live and against the research
H318 native list (the release binary's own replay of it), and full DEV (S + C, descriptive). Emits \\Liv* macros. The
release binary (isosyll v4.2.1) produced every v4.2.1 transcript.
"""

from __future__ import annotations

from pathlib import Path
import json

from .common import PaperDataError, RAW, fmt, sha256_file, thousands
from .look41 import _wer
from .records import Records
from .words import signed as _sg, tex

V = RAW / "versus" / "v421"
SPLITS = {"panel730": ("Panel", 730), "c_half": ("C", 2785), "shalf": ("S", 2782)}
WORK = {"panel730": "DEV selection panel, 730 utt", "c_half": "DEV confirmation (C) half, 2,785 utt",
        "shalf": "DEV selection (S) half, 2,782 utt"}


def _load(root: Path, name: str) -> tuple[dict, str, str]:
    p = root / V / name
    if not p.is_file():
        raise PaperDataError(f"look421: missing receipt {V / name}")
    return json.loads(p.read_text(encoding="utf-8")), str(V / name), sha256_file(p)


def add_look421(root: Path, rec: Records) -> None:
    if not (root / V).is_dir():
        return

    def put(rid, value, raw, unit, what, src, dig, workload):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what, workload=workload,
                workload_digest=dig, source=src)

    for split, (stag, n) in SPLITS.items():
        work = WORK[split]
        pairs = [("Prev", f"v41-vs-v421-live.{split}.score.json", "v4.1.0 live")]
        if split != "shalf":
            pairs.append(("Zip", f"zip-vs-v421-live.{split}.score.json", "Zipformer INT8"))
        else:
            pairs.append(("Res", f"h318-vs-v421-live.{split}.score.json", "the research H318 native list"))
        for who, name, label in pairs:
            d, src, dig = _load(root, name)
            a, b = _wer(d["system_a"]), _wer(d["system_b"])
            if b["utts"] != n or a["utts"] != n:
                raise PaperDataError(f"look421: {name} has {b['utts']} utterances, expected {n}")
            c = d["comparison"]["pooled_delta"]
            if who == "Prev":  # the v4.2.1 live numbers themselves, once per split
                for k, key in (("wer", "Wer"), ("clean", "Clean"), ("other", "Other")):
                    put(f"Liv{stag}{key}", fmt(b[k], 2), b[k], "% WER", f"v4.2.1 live, {work}: pooled {k}", src, dig, work)
                put(f"Liv{stag}Edits", thousands(str(b["edits"])), b["edits"], "edits", f"v4.2.1 live, {work}: edits", src, dig, work)
                put(f"LivPrev{stag}Edits", thousands(str(a["edits"])), a["edits"], "edits", f"v4.1.0 live, {work}: edits", src, dig, work)
            if who == "Zip":
                put(f"Liv{stag}ZipClean", fmt(a["clean"], 2), a["clean"], "% WER", f"Zipformer INT8, {work}: pooled clean",
                    src, dig, work)
            if who == "Res":
                put(f"LivRes{stag}Wer", fmt(a["wer"], 2), a["wer"], "% WER", f"research H318 native live list, {work}",
                    src, dig, work)
            for k, v in (("Delta", c["delta"]), ("Lo", c["lower"]), ("Hi", c["upper"])):
                put(f"Liv{stag}{who}{k}", tex(_sg(100 * v, 2)), 100 * v, "pp WER",
                    f"v4.2.1 live minus {label}, {work}, paired speaker-cluster bootstrap ({k})", src, dig, work)
    d, src, dig = _load(root, "v421-live.fulldev.score.json")
    f = _wer(d["system_a"])
    if f["utts"] != 5567:
        raise PaperDataError(f"look421: full DEV receipt has {f['utts']} utterances")
    for k, key in (("wer", "Wer"), ("clean", "Clean"), ("other", "Other")):
        put(f"LivFull{key}", fmt(f[k], 2), f[k], "% WER", f"v4.2.1 live, full DEV (S + C, descriptive): pooled {k}",
            src, dig, "LibriSpeech DEV, 5,567 utt")
    put("LivFullEdits", thousands(str(f["edits"])), f["edits"], "edits", "v4.2.1 live, full DEV (S + C, descriptive): edits",
        src, dig, "LibriSpeech DEV, 5,567 utt")
    add_endpoint421(root, rec)


def add_endpoint421(root: Path, rec: Records) -> None:
    """The shipped endpoint-rerank constants and the research cost estimate the release decision used."""
    rel = RAW / "release_v421" / "live-endpoint.json"
    p = root / rel
    if not p.is_file():
        raise PaperDataError(f"look421: missing {rel}")
    d, dig = json.loads(p.read_text(encoding="utf-8")), sha256_file(p)
    rc = d["research_cost"]
    cpu, p95 = max(rc["process_cpu_pct"]), max(rc["endpoint_p95_ms"])
    work = "LibriSpeech DEV (research H244/H314/H318)"
    for rid, value, raw, unit, what in (
        ("LivSkipMargin", fmt(d["skip_margin"], 2), d["skip_margin"], "beam score",
         "v4.2.1 live endpoint rerank: top-1 minus top-2 beam score below which the rerank runs"),
        ("LivRoutedPct", str(d["dev_routed_pct"]), d["dev_routed_pct"], "%",
         "share of DEV utterance ends routed to the endpoint rerank in research"),
        ("LivResCpuPct", tex("+" + fmt(cpu, 0)), cpu, "%",
         "research paced ARM timing (H314, larger of two ABBA pairs): added process CPU"),
        ("LivResEndpointPNinetyFive", fmt(p95, 1), p95, "ms",
         "research paced ARM timing (H314, larger of two ABBA pairs): added endpoint p95"),
    ):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what, workload=work,
                workload_digest=dig, source=str(rel))
