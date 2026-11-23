"""Receipts of the always-on tests, read into plain numbers. Every file is under docs/paper/data/raw/ and
pinned in raw/MANIFEST.txt; a missing receipt returns None, and the sentence built on it says "pending".

capacity    energy's paced capacity test: N live streams on one core (TIMING entry 4b, quiet host)
silence     eval-lane's ten 1-hour non-speech blocks, and the pauses of the whole endless stream
stream      that first hour as one continuous stream against the same utterances decoded one by one,
            with no reset and with reset-with-replay (eval-lane's paired reports, its own scorer)
endless     runtime-lane's two whole endless runs: with reset-with-replay (the always-on configuration) and without
int8        FP32 PyTorch against the INT8 engine on full dev (eval-lane's paired report)
same_words  one checkpoint's panel transcripts on Zen 2 (AVX2), Xeon (AVX-512) and Apple M3 (NEON)
m3          bench-cross's one-core Apple M3 Ultra row of that checkpoint
"""

from __future__ import annotations

from pathlib import Path
import json
import math

from .common import RAW, PaperDataError, sha256_file

AO = RAW / "alwayson"
CAPACITY = RAW / "energy/derived/capacity-test-entry4b.jsonl"
CAP_ARMS = {"h22ctl-79800-ema-int8": "paced-h22ctl", "r0-int8": "paced-zipformer"}  # variant: arm stem
FLAT_PCT = 5.0  # a p95 lag whose spread over N stays within this share of its lowest value is "flat"
REPLAY = {"t_s": 1.5, "r_s": 1.0, "report": AO / "stream/replay/p.isolated-v-T1500R1000.json",
          "summary": AO / "stream/replay/hour1-replay-t1500_r1000.summary.json"}
WARM_S = 10.0  # memory samples before this much audio include the model load and the first chunks
ENDLESS = {run: {k: AO / f"endless/endless27h.{run}.{v}" for k, v in (("run", "run.json"), ("summary", "scored.summary.json"),
                                                                     ("rss", "rss.jsonl"))}
           for run in ("t0", "t1500_r1000")}  # without a reset; with reset-with-replay, the always-on configuration
INT8 = {"a1-49000-ema-int8": "A1-49000-ema", "a1-20000-ema-int8": "A1-20000-ema",
        "h22ctl-79800-ema-int8": "H22ctl-79800-ema"}  # variant: report
XISA = (("Zen 2 (AVX2)", RAW / "crossisa/xbabe0-h22ctl-entry2.predictions.jsonl"),
        ("Xeon (AVX-512)", RAW / "opponents/xeon8358/jsr-h22ctl-int8-t1/predictions.6ac85d6ff9dc.jsonl"),
        ("Apple M3 (NEON)", RAW / "opponents/m3ultra/jsr-h22ctl-int8-enginetool-t1/predictions.6ac85d6ff9dc.jsonl"))
XISA_VARIANT = "h22ctl-79800-ema-int8"
M3 = (RAW / "opponents/rows/m3ultra-bench-cross.jsonl", "h22ctl-79800-ema-int8-enginetool-m3ultra-t1")


def _json(root: Path, rel: Path):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _jsonl(root: Path, rel: Path) -> list[dict]:
    return [json.loads(x) for x in (root / rel).read_text(encoding="utf-8").splitlines() if x.strip()]


def _have(root: Path, *rels: Path) -> bool:
    return all((root / r).is_file() for r in rels)


def _src(root: Path, *rels: Path) -> dict:
    return {"source": "; ".join(str(r) for r in rels), "digest": sha256_file(root / rels[0])}


def _pooled(root: Path, rel: Path) -> dict:
    """A paired report's pooled WER of system A and B, B - A and its bootstrap bounds, in percent."""
    c = _json(root, rel)["comparison"]["pooled_delta"]
    return {k: 100.0 * c[k] for k in ("observed_a", "observed_b", "delta", "lower", "upper")}


def capacity(root: Path) -> dict | None:
    if not _have(root, CAPACITY):
        return None
    arms = _jsonl(root, CAPACITY)
    out = {"systems": {}, **_src(root, CAPACITY)}
    for variant, stem in CAP_ARMS.items():
        mine = sorted((a for a in arms if a["arm"].startswith(stem + "-n")), key=lambda a: a["streams"])
        if not mine:
            raise PaperDataError(f"{CAPACITY}: no arm {stem}-n*")
        top, p95 = mine[-1], [a["lag_p95_ms"] for a in mine]
        out["systems"][variant] = {
            "n": top["streams"], "n_min": mine[0]["streams"], "lo": min(p95), "hi": max(p95),
            "flat": max(p95) - min(p95) <= FLAT_PCT / 100.0 * min(p95), "w": top["watts_per_stream_core"],
            "gross": top["gross_watts_per_stream_core"], "ceiling": 1.0 / top["cpu_s_per_audio_s_per_stream"],
            "busy": 100.0 * top["core_busy_fraction"], "arms": mine}
    return out


def poisson_upper(k: int, hours: float, conf: float = 0.95) -> float:
    """One-sided upper bound on an event rate per hour after k events in `hours` (Poisson, exact by bisection)."""
    def cdf(lam: float) -> float:  # P(X <= k)
        term = total = math.exp(-lam)
        for i in range(1, k + 1):
            term *= lam / i
            total += term
        return total
    lo, hi = 0.0, 10.0 * (k + 3)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if cdf(mid) > 1.0 - conf else (lo, mid)
    return hi / hours


def silence(root: Path) -> dict | None:
    """The ten 1-hour non-speech blocks, and the pauses of the whole endless stream (always-on configuration)."""
    man, full = AO / "nonspeech/MANIFEST.json", ENDLESS["t1500_r1000"]["summary"]
    blocks = sorted(p for p in (root / AO / "nonspeech/h22ctl").glob("*") if p.is_dir()) if _have(root, man) else []
    if not blocks or not _have(root, full):
        return None
    files = _json(root, man)["files"]
    if sorted(files) != sorted(f"{b.name}.wav" for b in blocks):
        raise PaperDataError(f"{AO}/nonspeech: decoded blocks {[b.name for b in blocks]} differ from the manifest")
    words = sum(len(json.loads(x)["hyp"].split()) for b in blocks
                for x in (b / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())
    summ = _json(root, full)
    gaps, quiet = summ["gap_false_words_total"], summ["gap_false_words"]["silence"]
    return {"blocks": len(blocks), "hours": sum(f["seconds"] for f in files.values()) / 3600.0, "words": words,
            "levels": sorted({f["level_dbfs"] for f in files.values() if f["level_dbfs"] is not None}),
            "gap_words": gaps["words"], "gap_hours": gaps["hours"], "silence_words": quiet["words"],
            "silence_hours": quiet["hours"], **_src(root, man, full),
            "bound": poisson_upper(words, sum(f["seconds"] for f in files.values()) / 3600.0),
            "gap_bound": poisson_upper(gaps["words"], gaps["hours"])}


def stream(root: Path) -> dict | None:
    pen, diag, t0 = (AO / "stream/hour1/isolated-v-stream.json", AO / "stream/hour1/p.isolated-v-diagA.json",
                     AO / "stream/replay/p.isolated-v-T0.json")
    if not _have(root, pen, diag, t0, REPLAY["report"], REPLAY["summary"], AO / "stream/hour1/CUT.json"):
        return None
    cut, rep = _json(root, AO / "stream/hour1/CUT.json"), _json(root, REPLAY["summary"])
    return {"utts": cut["utterances"], "minutes": cut["seconds"] / 60.0, "penalty": _pooled(root, pen),
            "left": _pooled(root, diag), "t0": _pooled(root, t0), "replay": _pooled(root, REPLAY["report"]),
            "replay_gap_words": rep["gap_false_words_total"]["words"], "t_s": REPLAY["t_s"], "r_s": REPLAY["r_s"],
            **_src(root, pen, diag, t0, REPLAY["report"], REPLAY["summary"])}


def _run(root: Path, files: dict) -> dict:
    r, s = _json(root, files["run"]), _json(root, files["summary"])
    warm = next(x["rss_mb"] for x in _jsonl(root, files["rss"]) if x["audio_s"] >= WARM_S)
    hours = [s["wer_per_hour"][k] for k in sorted(s["wer_per_hour"], key=int)]
    half = len(hours) // 2

    def wer(part: list[dict]) -> float:
        return sum(h["pooled_wer"] * h["utts"] for h in part) / sum(h["utts"] for h in part)

    return {"hours": r["audio_h"], "start": r["rss_start_mb"], "warm": warm, "peak": r["rss_peak_mb"],
            "end": r["rss_end_mb"], "resets": r["endpoint_resets"], "cpu": r["cpu_s_per_audio_s"], "utts": s["utterances"],
            "words": s["lag_ms"]["all"]["n"], "lag50": s["lag_ms"]["all"]["p50"], "lag95": s["lag_ms"]["all"]["p95"],
            "wer": s["pooled_wer"], "first": wer(hours[:half]), "second": wer(hours[half:]),
            "gap_words": s["gap_false_words_total"]["words"], **_src(root, *files.values())}


def endless(root: Path) -> dict | None:
    """The whole endless stream with reset-with-replay (the always-on configuration), and without a reset (`off`):
    memory, word lag, WER by half, CPU (runtime-lane's runs, eval-lane's scorer)."""
    if not all(_have(root, *f.values()) for f in ENDLESS.values()):
        return None
    on, off = _run(root, ENDLESS["t1500_r1000"]), _run(root, ENDLESS["t0"])
    return {**on, "off": off, "t_s": REPLAY["t_s"], "r_s": REPLAY["r_s"], "source": f"{on['source']}; {off['source']}"}


def int8(root: Path, variant: str) -> dict | None:
    rel = RAW / f"export/{INT8.get(variant, variant)}.fulldev.pytorch-v-engine.json"
    if variant not in INT8 or not _have(root, rel):
        return None
    return {"utts": _json(root, rel)["system_a"]["utterances"], **_pooled(root, rel), **_src(root, rel)}


def same_words(root: Path) -> dict | None:
    if not _have(root, *(p for _, p in XISA)):
        return None
    hyps = [{r["id"]: r["hyp"] for r in _jsonl(root, p)} for _, p in XISA]
    if any(set(h) != set(hyps[0]) for h in hyps):
        raise PaperDataError("cross-ISA transcripts cover different utterances")
    same = sum(all(h[i] == hyps[0][i] for h in hyps) for i in hyps[0])
    return {"same": same, "total": len(hyps[0]), "isas": [n for n, _ in XISA], **_src(root, *(p for _, p in XISA))}


def m3(root: Path) -> dict | None:
    if not _have(root, M3[0]):
        return None
    row = next((r for r in _jsonl(root, M3[0]) if r["id"] == M3[1]), None)
    if row is None:
        raise PaperDataError(f"{M3[0]}: no row {M3[1]}")
    return {"speed": 1.0 / row["cpu_s_per_audio_s"], "cpu": row["cpu_s_per_audio_s"], "hardware": row["hardware"],
            **_src(root, M3[0])}
