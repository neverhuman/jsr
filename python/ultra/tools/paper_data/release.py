"""Release-model (v3.0.0 RC1-EMA) full-DEV accuracy, panel WER, INT8 parity, commit delay, and power macros.

Reads SHA-256-pinned receipts under docs/paper/data/raw/release_v3/ and
docs/paper/data/raw/opponents/bench-open-release/joins/ so every release number in the paper and
README comes from a pinned receipt and swaps cleanly in Stage 2 (v3.1).
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import RAW, PaperDataError, fmt, sha256_file, thousands
from .evidence import presentation
from .records import Records
from .words import signed as _sg, tex

# Which receipts carry the release numbers; systems.json "release_receipts" overrides any key (v3.0.0 defaults).
RECEIPTS_DEFAULT = {"dir": "release_v3", "verify": "rc1_release_row_verify.json", "parity": "rc1_int8_vs_pytorch.json",
                    "panel": "14p_wer.json", "join_commit": "14p", "join_capacity": "14s", "isa_prefix": "v300",
                    "label": "Isosyll v3.0.0"}
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
    cfg = {**RECEIPTS_DEFAULT, **presentation(root).get("release_receipts", {})}
    rel_dir = RAW / cfg["dir"]
    ver_p = rel_dir / cfg["verify"]
    par_p = rel_dir / cfg["parity"]
    pwer_p = rel_dir / cfg["panel"]
    j14p_p = JOINS / f"{cfg['join_commit']}.jsonl"
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

    isa_dir = RAW / "exact_isa"
    isa_summary = isa_dir / f"{cfg['isa_prefix']}-exact-isa.summary.json"
    isa_proof = ""
    if (root / isa_summary).is_file():
        evidence = _json(root, isa_summary)
        groups = {
            "live": ("zen2-avx2", "xeon8358-avx2", "xeon8358-avx512vnni", "m3-neon"),
            "final": ("zen2-avx2", "xeon8358-avx2", "m3-neon"),
        }
        panel_ids = set()
        for mode, paths in groups.items():
            rows = [
                _jsonl(root, isa_dir / f"{cfg['isa_prefix']}-{mode}.{path}.predictions.jsonl")
                for path in paths
            ]
            ids = {row["id"] for row in rows[0]}
            if (len(ids) != 730 or len(rows[0]) != 730
                    or any(candidate != rows[0] for candidate in rows[1:])
                    or (panel_ids and ids != panel_ids)):
                raise PaperDataError(f"invalid release text identity witness for {mode}")
            panel_ids = ids
        if evidence["workload_digest"] != pwer["workload_digest"]:
            raise PaperDataError("release text identity witness uses a different waveform panel")
        isa_proof = ("Live transcripts are identical across tested AVX2, AVX-512 VNNI and NEON kernels and "
                     "batch sizes. Final transcripts match between the tested x86 and ARM CPUs.")
        put("RelIsaProof", isa_proof, "sentence", "release SIMD and batch text identity",
            p_wl, evidence["workload_digest"], isa_summary)

    # 730-utt DEV panel WER for v3.0.0 live (provisional text of 14p) from eval-lane's canonical 14p_wer.json
    lv = pwer["evidence_fields"]["live_provisional"]
    put("RelPanelLivePooled", fmt(lv["wer_pct"], 2), "% WER",
        cfg["label"] + " live (c=4 greedy INT8), 730-utt DEV panel pooled", p_wl, pwer_dig, pwer_p, raw=lv["wer_pct"])
    put("RelPanelLiveClean", fmt(lv["wer_clean_pct"], 2), "% WER",
        cfg["label"] + " live (c=4 greedy INT8), 730-utt DEV panel dev-clean", p_wl, pwer_dig, pwer_p,
        raw=lv["wer_clean_pct"])
    put("RelPanelLiveOther", fmt(lv["wer_other_pct"], 2), "% WER",
        cfg["label"] + " live (c=4 greedy INT8), 730-utt DEV panel dev-other", p_wl, pwer_dig, pwer_p,
        raw=lv["wer_other_pct"])

    # Full-DEV final (c=16 commit + M5/LODR beam-16)
    put("RelFullFinalPooled", fmt(ver["pooled_wer_pct"], 2), "% WER",
        cfg["label"] + " final pass (c=16 + M5/LODR beam-16), full DEV pooled", full_wl, ver_dig, ver_p,
        raw=ver["pooled_wer_pct"])
    put("RelFullFinalClean", fmt(ver["splits_wer_pct"]["dev-clean"], 2), "% WER",
        cfg["label"] + " final pass, full dev-clean", full_wl, ver_dig, ver_p,
        raw=ver["splits_wer_pct"]["dev-clean"])
    put("RelFullFinalOther", fmt(ver["splits_wer_pct"]["dev-other"], 2), "% WER",
        cfg["label"] + " final pass, full dev-other", full_wl, ver_dig, ver_p,
        raw=ver["splits_wer_pct"]["dev-other"])
    put("RelFullFinalEdits", thousands(str(ver["edits"])), "edits",
        cfg["label"] + " final pass total word edits on full DEV", full_wl, ver_dig, ver_p,
        raw=ver["edits"])

    # Full-DEV live (c=4 greedy INT8) and INT8-vs-FP32 parity
    eng = par["system_b"]
    clean_l = 100.0 * eng["datasets"]["dev-clean"]["wer"]
    other_l = 100.0 * eng["datasets"]["dev-other"]["wer"]
    pool_l = 100.0 * eng["pooled_wer"]
    put("RelFullLivePooled", fmt(pool_l, 2), "% WER",
        cfg["label"] + " live stream (c=4 greedy INT8), full DEV pooled", full_wl, par_dig, par_p, raw=pool_l)
    put("RelFullLiveClean", fmt(clean_l, 2), "% WER",
        cfg["label"] + " live stream (c=4 greedy INT8), full dev-clean", full_wl, par_dig, par_p, raw=clean_l)
    put("RelFullLiveOther", fmt(other_l, 2), "% WER",
        cfg["label"] + " live stream (c=4 greedy INT8), full dev-other", full_wl, par_dig, par_p, raw=other_l)
    put("RelFullUtts", thousands(str(eng["utterances"])), "utterances",
        "Full LibriSpeech DEV utterance count", full_wl, par_dig, par_p, raw=eng["utterances"])
    cd = par["comparison"]["pooled_delta"]
    d_pp, lo_pp, hi_pp = 100.0 * cd["delta"], 100.0 * cd["lower"], 100.0 * cd["upper"]
    put("RelIntEightDelta", tex(_sg(d_pp, 3)), "pp WER",
        cfg["label"] + " INT8 minus FP32 PyTorch pooled WER on full DEV", full_wl, par_dig, par_p, raw=d_pp)
    put("RelIntEightLo", tex(_sg(lo_pp, 3)), "pp WER",
        cfg["label"] + " INT8 minus FP32 95% lower bound on full DEV", full_wl, par_dig, par_p, raw=lo_pp)
    put("RelIntEightHi", tex(_sg(hi_pp, 3)), "pp WER",
        cfg["label"] + " INT8 minus FP32 95% upper bound on full DEV", full_wl, par_dig, par_p, raw=hi_pp)

    # 14p quiet-host commit-pass delay, revision rate, and package watts
    p50 = statistics.median(r["final_text_delay_ms"]["p50"] for r in j14p)
    p95 = statistics.median(r["final_text_delay_ms"]["p95"] for r in j14p)
    rev = statistics.median(r["revised_pct"] for r in j14p)
    pkg = statistics.median(r["watts"]["per_stream_package_w"] for r in j14p)
    put("RelCommitDelayPfifty", fmt(p50, 0), "ms",
        cfg["label"] + " commit-pass final-text delay after endpoint, median (14p)", p_wl, p14_dig, j14p_p, raw=p50)
    put("RelCommitDelayPninetyFive", fmt(p95, 0), "ms",
        cfg["label"] + " commit-pass final-text delay after endpoint, p95 (14p)", p_wl, p14_dig, j14p_p, raw=p95)
    put("RelCommitRevisedPct", fmt(rev, 1), "%",
        "Share of panel utterances revised by the commit pass (14p)", p_wl, p14_dig, j14p_p, raw=rev)
    put("RelCommitPkgWatts", fmt(pkg, 2), "W",
        cfg["label"] + " commit-pass package watts per stream, median of 3 (14p)", p_wl, p14_dig, j14p_p, raw=pkg)

    # Other-host complete-panel comparator (included in complete-panel WER rank, excluded from H1 speed/frontier)
    extras: dict[str, list[dict]] = {"complete": [], "subset": [], "capacity": []}
    j14s_p = JOINS / f"{cfg['join_capacity']}.jsonl"
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
                cfg["label"] + f" live paced 1x, N={n} streams on one quiet Zen 2 core (14s)", s14_wl, s14_dig, j14s_p, raw=busy)
            put(f"RelCap{tag}NetW", fmt(r["watts_per_stream_core"], 3), "W/stream (net core)",
                cfg["label"] + f" live paced 1x, N={n} net core W/stream (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["watts_per_stream_core"])
            put(f"RelCap{tag}GrossW", fmt(r["gross_watts_per_stream_core"], 3), "W/stream (gross core)",
                cfg["label"] + f" live paced 1x, N={n} gross core W/stream (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["gross_watts_per_stream_core"])
            put(f"RelCap{tag}LagPfifty", fmt(r["lag_p50_ms"], 0), "ms",
                cfg["label"] + f" live paced 1x, N={n} stable-word lag p50 (14s)", s14_wl, s14_dig, j14s_p,
                raw=r["lag_p50_ms"])
            put(f"RelCap{tag}LagPninetyFive", fmt(r["lag_p95_ms"], 0), "ms",
                cfg["label"] + f" live paced 1x, N={n} stable-word lag p95 (14s)", s14_wl, s14_dig, j14s_p,
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

    # The 14u follow-up includes two whole-clip WER receipts on the exact full panel.
    full_specs = (
        ("RelParakeetWhole", "parakeet-unified-offline", "Parakeet unified (whole clip)"),
        ("RelNemotronWhole", "nemotron-labs-unified-offline", "Nemotron Labs unified (whole clip)"),
    )
    full_wer_labels = set()
    for prefix, variant, label in full_specs:
        directory = OPP / "zen2" / f"{variant}-w8" / "round-wer"
        score_path = directory / "score.json"
        if not (root / score_path).is_file():
            continue
        raw = _json(root, score_path)
        dc, do, pooled, utts = _score_pcts(raw)
        waveform_pins = {
            line.split()[0] for line in (root / directory / "workload.sha256").read_text().splitlines() if line.strip()
        }
        if (utts != 730 or waveform_pins != {pwer["workload_digest"]}
                or raw["normaliser_sha256"] != pwer["scorer"]["normaliser_sha256"]):
            raise PaperDataError(f"full-panel WER receipt does not match the release panel: {score_path}")
        digest = sha256_file(root / score_path)
        for suffix, split, value in (("Clean", "dev-clean", dc), ("Other", "dev-other", do), ("Wer", "pooled", pooled)):
            put(prefix + suffix, fmt(value, 2), "% WER", f"{label}, {split}, 730-utterance WER only",
                p_wl, digest, score_path, raw=value)
        extras["complete"].append({
            "id": variant, "variant": variant, "system": label, "runtime_short": "NeMo-Speech.cpp",
            "mode": "offline (whole clip)", "wer_clean": dc, "wer_other": do, "wer": pooled,
            "metrics_prefix": prefix, "source": str(score_path), "digest": digest,
        })
        full_wer_labels.add(label)

    # Subset-WER comparators (other hardware; disclosed honestly, not ranked)
    subset_specs = (
        ("RelSubsetParakeetOfflineWer", "Parakeet unified (whole clip)", "panel73", "M3 Ultra",
         OPP / "m3ultra/parakeet-unified-offline-t1-panel73/round-1/score.json"),
        ("RelSubsetNemotronOfflineWer", "Nemotron Labs unified (whole clip)", "panel73", "Xeon 8358",
         OPP / "xeon8358/nemotron-labs-unified-offline-t1-panel73/round-1/score.json"),
        ("RelSubsetNemotronStreamWer", "Nemotron labs unified (stream 160 ms)", "panel10", "Xeon 8358",
         OPP / "xeon8358/nemotron-labs-unified-stream160-t1-panel10/round-1/score.json"),
    )
    sub_parts = []
    for rid, label, panel_name, hw, rel in subset_specs:
        if label not in full_wer_labels and (root / rel).is_file():
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
    extras["isa_proof"] = isa_proof
    return extras
