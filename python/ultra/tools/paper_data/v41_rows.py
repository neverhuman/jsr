"""Evidence rows of the v4.1.0 release from the R8 quiet-host paired timing and the v4.1.0 DEV look.

Inputs (all pinned in raw/MANIFEST.txt):
  raw/release_v41/timing_interleaved/block{1,2,3}/<arm>.json   one cold full-panel pass per block per arm, same quiet core
  raw/versus/v410/zip-vs-v41-<mode>.panel730.score.json         DEV WER of the same 730-utterance panel (paired with Zipformer INT8)
R8 repeats the R6/R7 protocol on one pinned core of the quiet host: four arms on the full 730-utterance panel in three
counterbalanced blocks (v4.1.0 live +LM, v4.1.0 final, and the v3.1.0 live +LM and Zipformer INT8 anchors). Every
block-1 anchors must land within 5% of their R6 values or the campaign is void (the pre-registered rule, PREREG.md);
later blocks' anchor deviations and core cleanliness are recorded with the rows. CPU and wall time are medians of all three
blocks (the schema's three repetitions); energy and peak RSS use only the arm's core-clean blocks (a block whose arm
stayed core-unclean after its one rerun is kept in the receipts and block lists but left out of those medians); ratios
to Zipformer INT8 are medians of within-block ratios over the blocks where both arms are core-clean. `python -m tools.paper_data.v41_rows` rewrites data/evidence/xbabe0-v41-release.jsonl; a test
checks the committed file against these receipts.
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import DATA, RAW, PaperDataError, sha256_file

R = RAW / "release_v41"
T = R / "timing_interleaved"
VS = RAW / "versus" / "v410"
BLOCKS = ("block1", "block2", "block3")
AUDIO_S = 5259.5599375
DIGEST = "010aa985e8c677d4daf43561dd9600d1ad27429494e88b74c177825a8cbc3963"
SESSION = "2026-11-27 xbabe0 R8 paired 3bd85e9"
WEIGHTS_BYTES = 59558368  # weights.int8.safetensors of the v4.1.0 bundle (same shape as v3.1.0)
HARDWARE = ("AMD Ryzen Threadripper PRO 3995WX; one pinned core (CPU 16, its SMT sibling kept idle), "
            "arms interleaved in three counterbalanced blocks, each arm alone on the measured core")
ANCHOR_R6 = {"anchor730-live": 0.328, "anchor730-zip": 0.573}  # R6 core J per audio-second (raw/release_v31)
ANCHOR_TOL = 0.05
ZIP = "anchor730-zip"
# arm -> (variant, system, configuration, paired WER receipt)
ARMS = {
    "v41-live-lm-730": ("v41-live-lm", "ULTRA v4.1.0 (live, +LM)",
                        "v4.1.0 (H247 step 450), prepared INT8, beam 12 (margin 4) with the 243 MiB compact M5 LM, "
                        "blank penalty 0.24, chunk 4 = 160 ms", "zip-vs-v41-live.panel730.score.json"),
    "v41-final-730": ("v41-final", "ULTRA v4.1.0 (final, seg.)",
                      "v4.1.0, prepared INT8, 160 ms live stream + utterance-segmented commit pass, M5 LM, beam 16",
                      "zip-vs-v41-final.panel730.score.json"),
}


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


def rows(root: Path) -> list[dict]:
    check = anchors(root)
    void = [a for a, c in check.items() if c["verdict"] != "VALID"]
    if void:
        raise PaperDataError(f"R8 campaign is void: anchors outside +-5% of R6 or core-unclean: {void}")
    zrecs = [_load(root, T / b / f"{ZIP}.json") for b in BLOCKS]
    out: list[dict] = []
    for arm, (variant, system, config, paired) in ARMS.items():
        recs = [_load(root, T / b / f"{arm}.json") for b in BLOCKS]
        clean = [i for i, r in enumerate(recs) if r["clean_repetitions"]["core"] == 1]
        if len(clean) < 2:
            raise PaperDataError(f"R8 {arm}: fewer than two core-clean blocks ({clean})")
        both = [i for i in clean if zrecs[i]["clean_repetitions"]["core"] == 1]
        if len(both) < 2:
            raise PaperDataError(f"R8 {arm}: fewer than two blocks where it and Zipformer INT8 are core-clean ({both})")
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
            "id": stem, "variant": variant, "system": system, "runtime": "ultra-cold-wav 3bd85e9e (frozen)",
            "role": "ours", "class": "streaming", "machine": "zen2", "host": "xbabe0", "hardware": HARDWARE, "threads": 1,
            "parallel": "intra_op", "configuration": config, "arm": arm,
            "workload": "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)", "workload_digest": DIGEST,
            "audio_seconds": AUDIO_S, "timing_panel": "full", "scope": "cold_process_tree_launch_to_exit",
            # Timing is the median of all three blocks (schema: >= 3 repetitions); an unclean block can only be its max.
            "cpu_s_per_audio_s": _med(cpu_all), "cpu_reps": cpu_all, "wall_s_per_audio_s": _med(wall_all), "wall_reps": wall_all,
            "core_clean_blocks": [i + 1 for i in clean],
            "peak_rss_bytes": int(_med(rss)), "artifact_bytes": WEIGHTS_BYTES,
            "n_busy_cores_reps": [1, 1, 1], "label": "exploratory", "session": SESSION,
            "source": f"docs/paper/data/raw/release_v41/timing_interleaved/block*/{arm}.json",
            "receipt_sha256": ssha,
            "wer_pct": round(100.0 * score["pooled_wer"], 4), "wer_clean_pct": round(100.0 * ds["dev-clean"]["wer"], 4),
            "wer_other_pct": round(100.0 * ds["dev-other"]["wer"], 4),
            "wer_source": f"docs/paper/data/raw/versus/v410/{paired} system_b",
            "xrt_median_audited": round(_med([1.0 / c for c in cpu_all]), 4),
            "paired_vs_anchor": {
                "speed": _med([zrecs[i]["cpu_s_per_audio_s"] / recs[i]["cpu_s_per_audio_s"] for i in both]),
                "energy": _med([recs[i]["joules_per_audio_s"] / zrecs[i]["joules_per_audio_s"] for i in both]),
                "rss": _med([recs[i]["reps"][0]["peak_rss_bytes"] / zrecs[i]["reps"][0]["peak_rss_bytes"] for i in both]),
                "blocks": [i + 1 for i in both]},
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
        out.append({  # the R7-protocol value for Figure 1 and Table 1: same core, same windows, full panel (R8)
            "kind": "r7_energy", "id": stem, "protocol": "R8", "panel": "p730", "anchor_check": "VALID",
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
    (REPO / DATA / "evidence" / "xbabe0-v41-release.jsonl").write_text(text(REPO), encoding="utf-8")
