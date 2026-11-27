"""Evidence row of the v4.2.1 live mode from the R10 quiet-host paired timing and the v4.2.1 DEV look.

Inputs (all pinned in raw/MANIFEST.txt):
  raw/release_v421/timing_interleaved/block{1,2,3}/<arm>.json        one cold full-panel pass per block per arm, same quiet core
  raw/release_v421/timing_interleaved/block{1,2,3}/<arm>.out.jsonl   the isosyll arms' own per-file output (text, wall_s)
  raw/versus/v421/zip-vs-v421-live.panel730.score.json               DEV WER of the same 730-utterance panel (paired with Zipformer INT8)
R10 repeats the R8/R9 protocol on one pinned core of the quiet host: four arms on the full 730-utterance panel in three
counterbalanced blocks (the published isosyll v4.2.0 binary and the v4.2.1 glibc build, both in live mode, and the R8
anchors, the v4.0.0 live +LM default and Zipformer INT8). The v4.2.0 live arm prices the H318 endpoint rerank and is reported
as within-block ratios (`cost`), never as a row of its own; v4.2.0 live is the v4.1.0 live decoder and keeps the v4.1.0 row
(v41_rows.py) as history. The added endpoint delay is the per-file paired difference of `wall_s` (v4.2.1 minus v4.2.0, same
block, same file; the first file carries the lazy model load and is dropped), pooled over the blocks where both arms are
core-clean. Every block-1 anchor must land within 5% of its R6 value or the campaign is void (PREREG.md). CPU and wall time
are medians of all three blocks; energy and peak RSS use only the arm's core-clean blocks; ratios are medians of within-block
ratios. `python -m tools.paper_data.v421_rows` rewrites data/evidence/xbabe0-v421-release.jsonl; a test checks the committed
file against these receipts.
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import DATA, RAW, PaperDataError, sha256_file

R = RAW / "release_v421"
T = R / "timing_interleaved"
VS = RAW / "versus" / "v421"
BLOCKS = ("block1", "block2", "block3")
AUDIO_S = 5259.5599375
DIGEST = "010aa985e8c677d4daf43561dd9600d1ad27429494e88b74c177825a8cbc3963"
SESSION = "2026-11-27 xbabe0 R10 paired isosyll v4.2.1 live"
WEIGHTS_BYTES = 59558368  # weights.int8.safetensors of the live bundle (H247 step 450; unchanged since v4.1.0)
HARDWARE = ("AMD Ryzen Threadripper PRO 3995WX; one pinned core (CPU 16, its SMT sibling kept idle), "
            "arms interleaved in three counterbalanced blocks, each arm alone on the measured core")
ANCHOR_R6 = {"anchor730-live": 0.328, "anchor730-zip": 0.573}  # R6 core J per audio-second (raw/release_v31)
ANCHOR_TOL = 0.05
ZIP = "anchor730-zip"
# arm -> (variant, system, configuration, paired WER receipt)
ARMS = {
    "v421-live-730": ("v421-live", "ULTRA v4.2.1 (live, +LM)",
                      "isosyll v4.2.1 live: v4.1.0 live decoder (H247 step 450 INT8, beam 12, compact M5 LM, 160 ms chunks) "
                      "plus the H318 endpoint rerank (frozen H261 content-word t2 reranker, fitLive weights, top1-top2 margin below 3.75)",
                      "zip-vs-v421-live.panel730.score.json"),
}
PREV = "v42-live-730"  # cost control: the published isosyll v4.2.0 binary in live mode


def _load(root: Path, rel: Path) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _med(values: list[float]) -> float:
    return statistics.median(values)


def anchors(root: Path) -> dict:
    """Anchor check (PREREG.md): block 1's anchors within 5% of R6, core-clean; every block's deviation is recorded."""
    out = {}
    for arm, r6 in ANCHOR_R6.items():
        recs = [_load(root, T / b / f"{arm}.json") for b in BLOCKS]
        dev = [100.0 * (r["joules_per_audio_s"] / r6 - 1.0) for r in recs]
        clean = [r["clean_repetitions"]["core"] == 1 for r in recs]
        ok = abs(dev[0]) <= 100 * ANCHOR_TOL and clean[0]
        out[arm] = {"r6": r6, "blocks": [r["joules_per_audio_s"] for r in recs], "deviation_pct": [round(d, 2) for d in dev],
                    "core_clean": clean, "verdict": "VALID" if ok else "VOID"}
    return out


def cost(root: Path, recs: list[dict]) -> dict:
    """Within-block ratios of the v4.2.1 live arm to the v4.2.0 live arm (both-clean blocks for energy and RSS)."""
    other = [_load(root, T / b / f"{PREV}.json") for b in BLOCKS]
    both = [i for i in range(len(BLOCKS)) if recs[i]["clean_repetitions"]["core"] == 1 and other[i]["clean_repetitions"]["core"] == 1]
    if len(both) < 2:
        raise PaperDataError(f"R10 {PREV}: fewer than two blocks where it and v4.2.1 live are core-clean ({both})")
    return {"vs_v42_live": {
        "cpu": _med([recs[i]["cpu_s_per_audio_s"] / other[i]["cpu_s_per_audio_s"] for i in range(len(BLOCKS))]),
        "cpu_blocks": [recs[i]["cpu_s_per_audio_s"] / other[i]["cpu_s_per_audio_s"] for i in range(len(BLOCKS))],
        "energy": _med([recs[i]["joules_per_audio_s"] / other[i]["joules_per_audio_s"] for i in both]),
        "rss": _med([recs[i]["reps"][0]["peak_rss_bytes"] / other[i]["reps"][0]["peak_rss_bytes"] for i in both]),
        "other_cpu_s_per_audio_s": _med([r["cpu_s_per_audio_s"] for r in other]),
        "other_joules_per_audio_s": _med([other[i]["joules_per_audio_s"] for i in both]),
        "other_peak_rss_bytes": int(_med([other[i]["reps"][0]["peak_rss_bytes"] for i in both])),
        "energy_blocks": [i + 1 for i in both]}}


def _outs(root: Path, block: str, arm: str) -> list[dict]:
    p = root / T / block / f"{arm}.out.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _pct(values: list[float], q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def delay(root: Path, recs: list[dict]) -> dict:
    """Added endpoint delay: per-file wall_s of v4.2.1 live minus v4.2.0 live, same block and file, first file dropped."""
    other = [_load(root, T / b / f"{PREV}.json") for b in BLOCKS]
    both = [i for i in range(len(BLOCKS)) if recs[i]["clean_repetitions"]["core"] == 1 and other[i]["clean_repetitions"]["core"] == 1]
    diffs, new, old = [], [], []
    for i in both:
        a, b = _outs(root, BLOCKS[i], ARMS_KEY), _outs(root, BLOCKS[i], PREV)
        if [x["file"] for x in a] != [x["file"] for x in b] or len(a) != 730:
            raise PaperDataError(f"R10 {BLOCKS[i]}: the two live arms' outputs do not list the same 730 files")
        for x, y in list(zip(a, b))[1:]:
            diffs.append(1000.0 * (x["wall_s"] - y["wall_s"]))
            new.append(1000.0 * x["wall_s"])
            old.append(1000.0 * y["wall_s"])
    return {"blocks": [i + 1 for i in both], "files": len(diffs),
            "added_ms_p50": round(_pct(diffs, 0.5), 3), "added_ms_p95": round(_pct(diffs, 0.95), 3),
            "added_ms_mean": round(sum(diffs) / len(diffs), 3),
            "file_ms_p95_v421": round(_pct(new, 0.95), 3), "file_ms_p95_v42": round(_pct(old, 0.95), 3)}


ARMS_KEY = "v421-live-730"


def rows(root: Path) -> list[dict]:
    check = anchors(root)
    void = [a for a, c in check.items() if c["verdict"] != "VALID"]
    if void:
        raise PaperDataError(f"R10 campaign is void: anchors outside +-5% of R6 or core-unclean: {void}")
    zrecs = [_load(root, T / b / f"{ZIP}.json") for b in BLOCKS]
    out: list[dict] = []
    for arm, (variant, system, config, paired) in ARMS.items():
        recs = [_load(root, T / b / f"{arm}.json") for b in BLOCKS]
        clean = [i for i, r in enumerate(recs) if r["clean_repetitions"]["core"] == 1]
        if len(clean) < 2:
            raise PaperDataError(f"R10 {arm}: fewer than two core-clean blocks ({clean})")
        both = [i for i in clean if zrecs[i]["clean_repetitions"]["core"] == 1]
        if len(both) < 2:
            raise PaperDataError(f"R10 {arm}: fewer than two blocks where it and Zipformer INT8 are core-clean ({both})")
        reps = [r["reps"][0] for r in recs]
        cpu_all = [r["cpu_s_per_audio_s"] for r in recs]
        wall_all = [x["wall_seconds"] / AUDIO_S for x in reps]
        joules = [r["joules_per_audio_s"] for r in recs]
        rss = [reps[i]["peak_rss_bytes"] for i in clean]
        cj = [joules[i] for i in clean]
        score = _load(root, VS / paired)["system_b"]
        if score["utterances"] != 730:
            raise PaperDataError(f"{paired}: expected 730 utterances, got {score['utterances']}")
        ds = score["datasets"]
        stem = f"{variant}-zen2-t1"
        ssha = sha256_file(root / T / BLOCKS[0] / f"{arm}.json")
        out.append({
            "id": stem, "variant": variant, "system": system, "runtime": "isosyll v4.2.1 glibc build ca4c230b (x86_64; the published binary is built from the same code)",
            "role": "ours", "class": "streaming", "machine": "zen2", "host": "xbabe0", "hardware": HARDWARE, "threads": 1,
            "parallel": "intra_op", "configuration": config, "arm": arm,
            "workload": "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)", "workload_digest": DIGEST,
            "audio_seconds": AUDIO_S, "timing_panel": "full", "scope": "cold_process_tree_launch_to_exit",
            # Timing is the median of all three blocks (schema: >= 3 repetitions); an unclean block can only be its max.
            "cpu_s_per_audio_s": _med(cpu_all), "cpu_reps": cpu_all, "wall_s_per_audio_s": _med(wall_all), "wall_reps": wall_all,
            "core_clean_blocks": [i + 1 for i in clean],
            "peak_rss_bytes": int(_med(rss)), "artifact_bytes": WEIGHTS_BYTES,
            "n_busy_cores_reps": [1, 1, 1], "label": "exploratory", "session": SESSION,
            "source": f"docs/paper/data/raw/release_v421/timing_interleaved/block*/{arm}.json",
            "receipt_sha256": ssha,
            "wer_pct": round(100.0 * score["pooled_wer"], 4), "wer_clean_pct": round(100.0 * ds["dev-clean"]["wer"], 4),
            "wer_other_pct": round(100.0 * ds["dev-other"]["wer"], 4),
            "wer_source": f"docs/paper/data/raw/versus/v421/{paired} system_b",
            "xrt_median_audited": round(_med([1.0 / c for c in cpu_all]), 4),
            "paired_vs_anchor": {
                "speed": _med([zrecs[i]["cpu_s_per_audio_s"] / recs[i]["cpu_s_per_audio_s"] for i in both]),
                "energy": _med([recs[i]["joules_per_audio_s"] / zrecs[i]["joules_per_audio_s"] for i in both]),
                "rss": _med([recs[i]["reps"][0]["peak_rss_bytes"] / zrecs[i]["reps"][0]["peak_rss_bytes"] for i in both]),
                "blocks": [i + 1 for i in both]},
            "cost": cost(root, recs), "endpoint_delay": delay(root, recs),
        })
        flags = sorted({f for r in recs for f in r["flags"]})
        out.append({
            "kind": "energy", "row": stem, "host": "xbabe0", "energy_method": "core_msr_active",
            "joules_per_audio_s": _med(cj), "watts_per_stream": _med(cj), "energy_reps": joules,
            "energy_active_w": _med([reps[i]["core"]["joules_per_cpu_s"] for i in clean]), "energy_receipt_sha256": ssha,
            "energy_host": "xbabe0",
            "energy_tool": "jsr.energy-bench/v1 (AMD core energy MSR, idle-subtracted, exact phase boundaries)",
            "energy_flags": flags, "session": SESSION, "kwh_per_stream_year": round(_med(cj) * 8.76, 3),
            "energy_receipts": [f"{b}/{arm}.json" for b in BLOCKS],
            "energy_note": "median of the core-clean blocks' core-register joules per audio-second; block-1 anchors (v3.1.0 live +LM, "
                           "Zipformer INT8) within 5% of R6 (pre-registered); per-block deviations in the r7_energy row",
        })
        mid = min(clean, key=lambda i: (abs(joules[i] - _med(cj)), i))
        out.append({  # the R7-protocol value for Figure 1 and Table 1: same core, same windows, full panel (R10)
            "kind": "r7_energy", "id": stem, "protocol": "R10", "panel": "p730", "anchor_check": "VALID",
            "anchors": check, "energy_method": "core_idle_subtracted", "energy_j_per_audio_s": _med(cj),
            "n_clean": len(clean), "blocks": joules, "clean_blocks": [i + 1 for i in clean],
            "block_core_clean": [r["clean_repetitions"]["core"] == 1 for r in recs],
            "block_spread_pct": round(100.0 * (max(cj) - min(cj)) / _med(cj), 2),
            "cpu_s_per_audio_s": _med(cpu_all), "class": "streaming", "host": "xbabe0", "machine": "zen2", "label": "exploratory",
            "receipt": f"{BLOCKS[mid]}/{arm}.json", "receipt_sha256": sha256_file(root / T / BLOCKS[mid] / f"{arm}.json"),
        })
    return out


def text(root: Path) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows(root))


if __name__ == "__main__":
    from .common import REPO
    (REPO / DATA / "evidence" / "xbabe0-v421-release.jsonl").write_text(text(REPO), encoding="utf-8")


def add_cost421(root: Path, rec) -> None:
    """\\LivCost* / \\LivDelay* macros: what the H318 endpoint rerank costs over v4.2.0 live (R10)."""
    if not (root / T).is_dir():
        return
    from .words import signed as _sg, tex
    from .common import fmt
    row = next(r for r in rows(root) if r.get("variant") == "v421-live" and "cost" in r)
    src = f"docs/paper/data/raw/release_v421/timing_interleaved/block*/{{{ARMS_KEY},{PREV}}}.json"
    work = "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)"
    c = row["cost"]["vs_v42_live"]
    for k, name, unit in (("cpu", "CpuPct", "CPU-s per audio-s"), ("energy", "EnergyPct", "core J per audio-s"),
                          ("rss", "RssPct", "peak RSS")):
        pct = 100.0 * (c[k] - 1.0)
        rec.add(f"LivCost{name}", tex(_sg(pct, 1)), raw=pct, unit="%", label="exploratory",
                configuration=f"v4.2.1 live vs v4.2.0 live: median within-block ratio of {unit} minus 1 (R10)",
                workload=work, workload_digest=DIGEST, source=src)
    mb = (row["peak_rss_bytes"] - c["other_peak_rss_bytes"]) / 1e6
    rec.add("LivCostRssMb", fmt(mb, 0), raw=mb, unit="MB", label="exploratory",
            configuration="v4.2.1 live minus v4.2.0 live: peak RSS, medians of the both-clean blocks (R10)",
            workload=work, workload_digest=DIGEST, source=src)
    d = row["endpoint_delay"]
    osrc = f"docs/paper/data/raw/release_v421/timing_interleaved/block*/{{{ARMS_KEY},{PREV}}}.out.jsonl"
    for k, name in (("added_ms_p50", "PFifty"), ("added_ms_p95", "PNinetyFive"), ("added_ms_mean", "Mean")):
        rec.add(f"LivDelay{name}", fmt(d[k], 1), raw=d[k], unit="ms", label="exploratory",
                configuration=f"v4.2.1 live minus v4.2.0 live, per-file wall time on one pinned core ({k}, {d['files']} paired files, R10)",
                workload=work, workload_digest=DIGEST, source=osrc)
