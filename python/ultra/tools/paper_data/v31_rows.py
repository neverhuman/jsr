"""Evidence rows of the v3.1.0 (RC2) release from the R6 quiet-host interleaved timing and the confirmatory look.

Inputs (all pinned in raw/MANIFEST.txt):
  raw/release_v31/timing_interleaved/block{1,2,3}/<arm>.json   one cold pass per block per arm, same quiet core
  raw/release_v31/timing_interleaved/summary_report.json       the audited per-arm medians
  raw/release_v31/confirmatory_look_summary.json               DEV WER of the same 730-utterance panel
The five arms run in three counterbalanced blocks on one pinned core of the quiet host, so each arm has three
repetitions and Zipformer INT8 is timed in the same blocks. `python -m tools.paper_data.v31_rows` rewrites
data/evidence/xbabe0-v31-release.jsonl; a test checks the committed file against these receipts.
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import DATA, RAW, sha256_file

R = RAW / "release_v31"
T = R / "timing_interleaved"
LOOK = R / "confirmatory_look_summary.json"
BLOCKS = ("block1", "block2", "block3")
AUDIO_S = 5259.5599375
DIGEST = "010aa985e8c677d4daf43561dd9600d1ad27429494e88b74c177825a8cbc3963"
SESSION = "2026-11-24 xbabe0 R6 interleaved 3bd85e9"
RC2_WEIGHTS_BYTES = 59558368  # weights.int8.safetensors, raw/release_v31/policies.json model_bundle.files
ZIP_BYTES = 72654782          # the Zipformer INT8 model dir, as in the earlier Zipformer row
HARDWARE = ("AMD Ryzen Threadripper PRO 3995WX; one pinned core (CPU 16, its SMT sibling kept idle), "
            "arms interleaved in three counterbalanced blocks, each arm alone on the measured core")
# arm -> (variant, role, class, system, configuration, WER key in confirmatory_look_summary.json)
ARMS = {
    "live_c4_greedy": ("rc2-live-greedy", "ours", "streaming", "ULTRA v3.1.0 (live, greedy)",
                       "RC2 (REGctl step 43,980), prepared INT8, greedy, chunk 4 = 160 ms", "live_c4_greedy"),
    "live_c4_lm": ("rc2-live-lm", "ours", "streaming", "ULTRA v3.1.0 (live, +LM)",
                   "RC2, prepared INT8, beam 4 with the 34.5 MiB LM, chunk 4 = 160 ms", "live_c4_lm"),
    "live_c8_lm": ("rc2-live-lm320", "ours", "streaming", "ULTRA v3.1.0 (live 320 ms, +LM)",
                   "RC2, prepared INT8, beam 4 with the 34.5 MiB LM, chunk 8 = 320 ms", "live_c8_lm"),
    "final_m5": ("rc2-final", "ours", "streaming", "ULTRA v3.1.0 (final, seg.)",
                 "RC2, prepared INT8, 160 ms live stream + utterance-segmented commit pass, M5 LM, beam 16", "final_m5"),
    "zipformer_anchor": ("r0-int8-r6", "comparator", "streaming", "Zipformer INT8 (same blocks)",
                         "INT8; greedy; chunk 320 ms, right context 0 ms; timed in the same blocks as RC2", None),
}


def _load(root: Path, rel: Path) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _med(values: list[float]) -> float:
    return statistics.median(values)


def _paired(arm: str, recs: dict[str, list[dict]]) -> dict:
    """Median over the three blocks of the within-block ratio to the Zipformer anchor of the same block."""
    z = recs["zipformer_anchor"]
    r = recs[arm]
    return {"speed": _med([zc["cpu_s_per_audio_s"] / a["cpu_s_per_audio_s"] for a, zc in zip(r, z)]),
            "energy": _med([a["joules_per_audio_s"] / zc["joules_per_audio_s"] for a, zc in zip(r, z)]),
            "rss": _med([a["reps"][0]["peak_rss_bytes"] / zc["reps"][0]["peak_rss_bytes"] for a, zc in zip(r, z)])}


def rows(root: Path) -> list[dict]:
    summary = _load(root, T / "summary_report.json")
    every = {arm: [_load(root, T / b / f"{arm}.json") for b in BLOCKS] for arm in ARMS}
    look = _load(root, LOOK)
    ssha = sha256_file(root / T / "summary_report.json")
    out: list[dict] = []
    for arm, (variant, role, cls, system, config, mode) in ARMS.items():
        recs = [_load(root, T / b / f"{arm}.json") for b in BLOCKS]
        reps = [r["reps"][0] for r in recs]
        cpu = [r["cpu_s_per_audio_s"] for r in recs]
        wall = [x["wall_seconds"] / AUDIO_S for x in reps]
        rss = [x["peak_rss_bytes"] for x in reps]
        joules = [r["joules_per_audio_s"] for r in recs]
        med = summary["arm_medians_across_clean_blocks"][arm]
        wer = (look["modes"][mode]["panel730"]["metrics"] if mode else look["baselines"]["panel730"]["zipformer_int8"])
        wer = {"wer_pct": wer["wer_pct"], "clean_pct": wer["clean_pct"], "other_pct": wer["other_pct"]}
        stem = f"{variant}-zen2-t1"
        row = {
            "id": stem, "variant": variant, "system": system, "runtime": "ultra-cold-wav 3bd85e9e (frozen)"
            if role == "ours" else "sherpa-onnx 2.0.2 / ultra-cold-wav 3bd85e9e",
            "role": role, "class": cls, "machine": "zen2", "host": "xbabe0", "hardware": HARDWARE, "threads": 1,
            "parallel": "intra_op", "configuration": config, "arm": arm,
            "workload": "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)", "workload_digest": DIGEST,
            "audio_seconds": AUDIO_S, "timing_panel": "full", "scope": "cold_process_tree_launch_to_exit",
            "cpu_s_per_audio_s": _med(cpu), "cpu_reps": cpu, "wall_s_per_audio_s": _med(wall), "wall_reps": wall,
            "peak_rss_bytes": int(_med(rss)), "artifact_bytes": RC2_WEIGHTS_BYTES if role == "ours" else ZIP_BYTES,
            "n_busy_cores_reps": [1, 1, 1], "label": "exploratory", "session": SESSION,
            "source": f"docs/paper/data/raw/release_v31/timing_interleaved/summary_report.json arm={arm}",
            "receipt_sha256": ssha,
            "wer_pct": wer["wer_pct"], "wer_clean_pct": wer["clean_pct"], "wer_other_pct": wer["other_pct"],
            "wer_source": "docs/paper/data/raw/release_v31/confirmatory_look_summary.json "
                          + (f"modes.{mode}.panel730.metrics" if mode else "baselines.panel730.zipformer_int8"),
            "xrt_median_audited": med["xrt_median"],
        }
        if arm != "zipformer_anchor":
            row["paired_vs_anchor"] = _paired(arm, every)
        out.append(row)
        flags = sorted({f for r in recs for f in r["flags"]})
        out.append({
            "kind": "energy", "row": stem, "host": "xbabe0", "energy_method": "core_msr_active",
            "joules_per_audio_s": _med(joules), "watts_per_stream": _med(joules), "energy_reps": joules,
            "energy_active_w": med["core_w_median"], "energy_receipt_sha256": ssha, "energy_host": "xbabe0",
            "energy_tool": "jsr.energy-bench/v1 (AMD core energy MSR, idle-subtracted, exact phase boundaries)",
            "energy_flags": flags, "session": SESSION, "kwh_per_stream_year": round(_med(joules) * 8.76, 3),
            "energy_receipts": [f"{b}/{arm}.json" for b in BLOCKS],
            "energy_note": "median of the three blocks' core-register joules per audio-second; core energy clean "
                           "in every block (package energy was not clean and is not used)",
        })
    return out


def text(root: Path) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows(root))


if __name__ == "__main__":
    from .common import REPO
    (REPO / DATA / "evidence" / "xbabe0-v31-release.jsonl").write_text(text(REPO), encoding="utf-8")
