"""Release-model (v3.0.0 RC1-EMA) full-DEV accuracy, panel WER, INT8 parity, commit delay, and power macros.

Reads SHA-256-pinned receipts under docs/paper/data/raw/release_v3/ and
docs/paper/data/raw/opponents/bench-open-release/joins/ so every release number in the paper and
README comes from a pinned receipt and swaps cleanly in Stage 2 (v3.1).
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import RAW, fmt, sha256_file, thousands
from .records import Records
from .words import signed as _sg, tex

REL_DIR = RAW / "release_v3"
JOINS = RAW / "opponents/bench-open-release/joins"
OPP = RAW / "opponents"
P73_AUDIO_S = 597.089875


def _score_pcts(data: dict) -> tuple[float, float, float, int]:
    sa = data["system_a"]
    dc = 100.0 * sa["datasets"]["dev-clean"]["wer"]
    do = 100.0 * sa["datasets"]["dev-other"]["wer"]
    c1, c2 = sa["datasets"]["dev-clean"]["counts"], sa["datasets"]["dev-other"]["counts"]
    ed = (c1["substitutions"] + c1["deletions"] + c1["insertions"]
          + c2["substitutions"] + c2["deletions"] + c2["insertions"])
    rw = c1["ref_words"] + c2["ref_words"]
    return dc, do, 100.0 * ed / rw, sa["utterances"]


def _json(root: Path, rel: Path):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _jsonl(root: Path, rel: Path) -> list[dict]:
    return [json.loads(ln) for ln in (root / rel).read_text(encoding="utf-8").splitlines() if ln.strip()]


def add_release(root: Path, rec: Records) -> dict:
    """Emit \\Rel* macros for our single winning release artifact (v3.0.0 RC1-EMA INT8) plus other-host comparators."""
    ver_p = REL_DIR / "rc1_release_row_verify.json"
    par_p = REL_DIR / "rc1_int8_vs_pytorch.json"
    pwer_p = REL_DIR / "14p_wer.json"
    j14p_p = JOINS / "14p.jsonl"
    ver = _json(root, ver_p)
    par = _json(root, par_p)
    pwer = _json(root, pwer_p)
    j14p = _jsonl(root, j14p_p)

    full_wl = "LibriSpeech full DEV (5,567 utt: 2,703 dev-clean + 2,864 dev-other)"
    p_wl = "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)"
    ver_dig = sha256_file(root / ver_p)
    par_dig = sha256_file(root / par_p)
    pwer_dig = sha256_file(root / pwer_p)
    p14_dig = sha256_file(root / j14p_p)

    def put(rid: str, value: str, unit: str, conf: str, wl: str, dig: str, src: Path, raw=None) -> None:
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=conf,
                workload=wl, workload_digest=dig, source=str(src))

    # 730-utt DEV panel WER for v3.0.0 live (provisional text of 14p) from eval-lane's canonical 14p_wer.json
    lv = pwer["evidence_fields"]["live_provisional"]
    put("RelPanelLivePooled", fmt(lv["wer_pct"], 2), "% WER",
        "ULTRA v3.0.0 live (c=4 greedy INT8), 730-utt DEV panel pooled", p_wl, pwer_dig, pwer_p, raw=lv["wer_pct"])
    put("RelPanelLiveClean", fmt(lv["wer_clean_pct"], 2), "% WER",
        "ULTRA v3.0.0 live (c=4 greedy INT8), 730-utt DEV panel dev-clean", p_wl, pwer_dig, pwer_p,
        raw=lv["wer_clean_pct"])
    put("RelPanelLiveOther", fmt(lv["wer_other_pct"], 2), "% WER",
        "ULTRA v3.0.0 live (c=4 greedy INT8), 730-utt DEV panel dev-other", p_wl, pwer_dig, pwer_p,
        raw=lv["wer_other_pct"])

    # Full-DEV final (c=16 commit + M5/LODR beam-16)
    put("RelFullFinalPooled", fmt(ver["pooled_wer_pct"], 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) final pass (c=16 + M5/LODR beam-16), full DEV pooled", full_wl, ver_dig, ver_p,
        raw=ver["pooled_wer_pct"])
    put("RelFullFinalClean", fmt(ver["splits_wer_pct"]["dev-clean"], 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) final pass, full dev-clean", full_wl, ver_dig, ver_p,
        raw=ver["splits_wer_pct"]["dev-clean"])
    put("RelFullFinalOther", fmt(ver["splits_wer_pct"]["dev-other"], 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) final pass, full dev-other", full_wl, ver_dig, ver_p,
        raw=ver["splits_wer_pct"]["dev-other"])
    put("RelFullFinalEdits", thousands(str(ver["edits"])), "edits",
        "ULTRA v3.0.0 (RC1-EMA) final pass total word edits on full DEV", full_wl, ver_dig, ver_p,
        raw=ver["edits"])

    # Full-DEV live (c=4 greedy INT8) and INT8-vs-FP32 parity
    eng = par["system_b"]
    clean_l = 100.0 * eng["datasets"]["dev-clean"]["wer"]
    other_l = 100.0 * eng["datasets"]["dev-other"]["wer"]
    pool_l = 100.0 * eng["pooled_wer"]
    put("RelFullLivePooled", fmt(pool_l, 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) live stream (c=4 greedy INT8), full DEV pooled", full_wl, par_dig, par_p, raw=pool_l)
    put("RelFullLiveClean", fmt(clean_l, 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) live stream (c=4 greedy INT8), full dev-clean", full_wl, par_dig, par_p, raw=clean_l)
    put("RelFullLiveOther", fmt(other_l, 2), "% WER",
        "ULTRA v3.0.0 (RC1-EMA) live stream (c=4 greedy INT8), full dev-other", full_wl, par_dig, par_p, raw=other_l)
    put("RelFullUtts", thousands(str(eng["utterances"])), "utterances",
        "Full LibriSpeech DEV utterance count", full_wl, par_dig, par_p, raw=eng["utterances"])
    cd = par["comparison"]["pooled_delta"]
    d_pp, lo_pp, hi_pp = 100.0 * cd["delta"], 100.0 * cd["lower"], 100.0 * cd["upper"]
    put("RelIntEightDelta", tex(_sg(d_pp, 3)), "pp WER",
        "ULTRA v3.0.0 INT8 minus FP32 PyTorch pooled WER on full DEV", full_wl, par_dig, par_p, raw=d_pp)
    put("RelIntEightLo", tex(_sg(lo_pp, 3)), "pp WER",
        "ULTRA v3.0.0 INT8 minus FP32 95% lower bound on full DEV", full_wl, par_dig, par_p, raw=lo_pp)
    put("RelIntEightHi", tex(_sg(hi_pp, 3)), "pp WER",
        "ULTRA v3.0.0 INT8 minus FP32 95% upper bound on full DEV", full_wl, par_dig, par_p, raw=hi_pp)

    # 14p quiet-host commit-pass delay, revision rate, and package watts
    p50 = statistics.median(r["final_text_delay_ms"]["p50"] for r in j14p)
    p95 = statistics.median(r["final_text_delay_ms"]["p95"] for r in j14p)
    rev = statistics.median(r["revised_pct"] for r in j14p)
    pkg = statistics.median(r["watts"]["per_stream_package_w"] for r in j14p)
    put("RelCommitDelayPfifty", fmt(p50, 0), "ms",
        "ULTRA v3.0.0 commit-pass final-text delay after endpoint, median (14p)", p_wl, p14_dig, j14p_p, raw=p50)
    put("RelCommitDelayPninetyFive", fmt(p95, 0), "ms",
        "ULTRA v3.0.0 commit-pass final-text delay after endpoint, p95 (14p)", p_wl, p14_dig, j14p_p, raw=p95)
    put("RelCommitRevisedPct", fmt(rev, 1), "%",
        "Share of panel utterances revised by the commit pass (14p)", p_wl, p14_dig, j14p_p, raw=rev)
    put("RelCommitPkgWatts", fmt(pkg, 2), "W",
        "ULTRA v3.0.0 commit-pass package watts per stream, median of 3 (14p)", p_wl, p14_dig, j14p_p, raw=pkg)

    # Other-host complete-panel comparator (included in complete-panel WER rank, excluded from H1 speed/frontier)
    extras: dict[str, list[dict]] = {"complete": [], "subset": [], "capacity": []}
    j14s_p = JOINS / "14s.jsonl"
    if (root / j14s_p).is_file():
        s14 = _jsonl(root, j14s_p)
        s14_dig = sha256_file(root / j14s_p)
        s14_wl = "paced120.wav: one continuous 128 s stream of 22 DEV clips (paced 1x, 20 ms packets)"
        tag_map = {1: "One", 4: "Four", 8: "Eight", 16: "Sixteen"}
        for r in s14:
            n = int(r["streams"])
            tag = tag_map.get(n)
            if not tag:
                continue
            busy = 100.0 * r["core_busy_fraction"]
            put(f"RelCap{tag}Busy", fmt(busy, 1), "% of one core",
                f"ULTRA v3.0.0 live paced 1x, N={n} streams on one quiet Zen 2 core (14s)", s14_wl, s14_dig, j14s_p, raw=busy)
            put(f"RelCap{tag}NetW", fmt(r["watts_per_stream_core"], 3), "W/stream (net core)",
                f"ULTRA v3.0.0 live paced 1x, N={n} net core W/stream (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["watts_per_stream_core"])
            put(f"RelCap{tag}GrossW", fmt(r["gross_watts_per_stream_core"], 3), "W/stream (gross core)",
                f"ULTRA v3.0.0 live paced 1x, N={n} gross core W/stream (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["gross_watts_per_stream_core"])
            put(f"RelCap{tag}LagPfifty", fmt(r["lag_p50_ms"], 0), "ms",
                f"ULTRA v3.0.0 live paced 1x, N={n} stable-word lag p50 (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["lag_p50_ms"])
            put(f"RelCap{tag}LagPninetyFive", fmt(r["lag_p95_ms"], 0), "ms",
                f"ULTRA v3.0.0 live paced 1x, N={n} stable-word lag p95 (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["lag_p95_ms"])
            extras["capacity"].append({
                "streams": n, "busy_pct": busy, "w_net": r["watts_per_stream_core"],
                "w_gross": r["gross_watts_per_stream_core"], "lag_p50": r["lag_p50_ms"], "lag_p95": r["lag_p95_ms"],
            })
    pu_score_p = OPP / "m3ultra/parakeet-unified-stream160-w8/round-wer/score.json"
    pu_meas_p = OPP / "xeon8358/parakeet-unified-stream160-t1-panel73/round-1/measurement.json"
    if (root / pu_score_p).is_file() and (root / pu_meas_p).is_file():
        dc, do, po, utts = _score_pcts(_json(root, pu_score_p))
        xm = _json(root, pu_meas_p)
        xcpu = (xm["cpu_seconds_user"] + xm["cpu_seconds_sys"]) / P73_AUDIO_S
        rss_mb = xm["peak_rss_bytes"] / 1e6
        pu_dig = sha256_file(root / pu_score_p)
        xm_dig = sha256_file(root / pu_meas_p)
        put("RelParakeetUnifiedClean", fmt(dc, 2), "% WER",
            "Parakeet unified en q8 (stream 160 ms), 730-utt DEV panel dev-clean (M3 Ultra untimed WER)",
            p_wl, pu_dig, pu_score_p, raw=dc)
        put("RelParakeetUnifiedOther", fmt(do, 2), "% WER",
            "Parakeet unified en q8 (stream 160 ms), 730-utt DEV panel dev-other (M3 Ultra untimed WER)",
            p_wl, pu_dig, pu_score_p, raw=do)
        put("RelParakeetUnifiedWer", fmt(po, 2), "% WER",
            "Parakeet unified en q8 (stream 160 ms), 730-utt DEV panel pooled (M3 Ultra untimed WER)",
            p_wl, pu_dig, pu_score_p, raw=po)
        put("RelParakeetUnifiedRss", fmt(rss_mb, 0), "MB",
            "Parakeet unified en q8 (stream 160 ms) peak RSS on Xeon 8358", p_wl, xm_dig, pu_meas_p, raw=rss_mb)
        put("RelParakeetUnifiedXeonCpu", fmt(xcpu, 1), "CPU-s/audio-s",
            "Parakeet unified en q8 (stream 160 ms) single-core CPU-s/audio-s on Xeon 8358 (panel73)",
            "panel73 (73 utt)", xm_dig, pu_meas_p, raw=xcpu)
        extras["complete"].append({
            "id": "parakeet-unified-stream160-other-host",
            "variant": "parakeet-unified-stream160",
            "system": "Parakeet unified en q8 (stream 160 ms)",
            "text": "Parakeet unified en q8 (stream 160\\,ms)",
            "runtime_short": "NeMo-Speech.cpp",
            "cite": ["nemospeechcpp"],
            "mode": "live streaming",
            "wer_clean": dc, "wer_other": do, "wer": po, "utterances": utts,
            "rss_mb": rss_mb, "xeon_cpu": xcpu,
            "source": str(pu_score_p), "digest": pu_dig,
        })

    # Subset-WER comparators (other hardware; disclosed honestly, not ranked)
    subset_specs = (
        ("RelSubsetParakeetOfflineWer", "Parakeet unified (whole clip)", "panel73", "M3 Ultra",
         OPP / "m3ultra/parakeet-unified-offline-t1-panel73/round-1/score.json"),
        ("RelSubsetNemotronOfflineWer", "Nemotron labs unified (whole clip)", "panel73", "Xeon 8358",
         OPP / "xeon8358/nemotron-labs-unified-offline-t1-panel73/round-1/score.json"),
        ("RelSubsetNemotronStreamWer", "Nemotron labs unified (stream 160 ms)", "panel10", "Xeon 8358",
         OPP / "xeon8358/nemotron-labs-unified-stream160-t1-panel10/round-1/score.json"),
    )
    sub_parts = []
    for rid, label, panel_name, hw, rel in subset_specs:
        if (root / rel).is_file():
            dc, do, po, utts = _score_pcts(_json(root, rel))
            dig = sha256_file(root / rel)
            put(rid, fmt(po, 2), "% WER", f"{label}, {panel_name} ({utts} utt, {hw}, not ranked)",
                f"{panel_name} ({utts} utt)", dig, rel, raw=po)
            extras["subset"].append({"system": label, "panel": panel_name, "hw": hw, "wer": po, "utts": utts})
            sub_parts.append(f"{label} ({fmt(po, 2)}\\% on {panel_name}, {hw})")
    if sub_parts:
        rec.add("RelSubsetNote",
                "Also measured, not ranked (subset WER, other hardware): " + "; ".join(sub_parts) + ".",
                unit="sentence", label="reported",
                configuration="subset-WER comparators measured on other hardware, disclosed without ranking",
                workload="panel73 / panel10 subsets", workload_digest="mixed-subsets",
                source="docs/paper/data/raw/opponents/{m3ultra,xeon8358}/**/score.json")
    else:
        rec.add("RelSubsetNote", "\\relax", unit="sentence", label="reported",
                configuration="no subset-WER comparators present", workload="none",
                workload_digest="none", source="none")
    return extras
