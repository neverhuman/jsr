#!/usr/bin/env python3
"""Summarise the R7 matched energy campaign.

    r7_summarize.py RECEIPT_ROOT [--evidence docs/paper/data/evidence/xbabe0-r7-energy.jsonl]

RECEIPT_ROOT holds block1/, block2/, block3/ (one energy_bench receipt per arm, named <arm id>.json; anchors
are anchor730-live.json and anchor730-zip.json in block1/). Writes RECEIPT_ROOT/summary.json and, with
--evidence, one JSON row per arm keyed by the id used in data/cold_dev.csv. The metric is the receipt's
joules_per_audio_s: (core energy - idle power x wall time) / audio seconds, the R6 definition.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

R6 = {"rc2-live-lm-zen2-t1": 0.328, "r0-int8-r6-zen2-t1": 0.573}
TOL = 0.05
# arm key used in the receipt file names -> id of data/cold_dev.csv
IDS = {
    "rc2-live-lm": "rc2-live-lm-zen2-t1",
    "rc2-final": "rc2-final-zen2-t1",
    "r0-int8-r6": "r0-int8-r6-zen2-t1",
    "moonshine-tiny": "moonshine-tiny-zen2-t1",
    "moonshine-small": "moonshine-small-zen2-t1-xbabe3",
    "nemotron-en-560": "nemotron-en-560-zen2-t1",
    "nemotron-en-160": "nemotron-en-160-zen2-t1-xbabe3",
    "nemotron-35": "nemotron-35-zen2-t1-xbabe3",
    "parakeet-tdt-v3-int8": "parakeet-tdt-v3-int8-zen2-t1",
    "parakeet-v3-onnxasr-int8": "parakeet-v3-onnxasr-int8-zen2-t1",
    "vosk-small": "vosk-small-zen2-t1",
    "vosk-large": "vosk-large-zen2-t1",
    "whispercpp-stream-tiny": "whispercpp-stream-tiny-zen2-t1",
    "whispercpp-stream-base": "whispercpp-stream-base-zen2-t1",
}


def load(root, block, key):
    """(parsed receipt, path, sha256): the hash is of the exact bytes that were parsed, from one read."""
    path = root / f"block{block}" / f"{key}.json"
    if not path.exists():
        return None, None, None
    raw = path.read_bytes()
    return json.loads(raw), path, hashlib.sha256(raw).hexdigest()


def check_manifest(root, keys):
    """Read each receipt once, check its hash against the staged selection manifest (r7_stage.py) and return
    the verified snapshot {(block, key): (receipt, path, sha256)}. Nothing else may read the receipts."""
    sel_path = root / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} is missing: stage the receipts with r7_stage.py first")
    sel = json.loads(sel_path.read_bytes())
    snap, bad = {}, []
    for b, key in keys:
        d, p, h = load(root, b, key)
        if d is None:
            continue
        want = sel.get(f"b{b}-{key}", {}).get("used_sha256")
        if want != h:
            bad.append(f"{p.relative_to(root)}: manifest {want}, bytes read {h}")
        snap[(b, key)] = (d, p, h)
    if bad:
        raise SystemExit("receipts do not match selection.json:\n  " + "\n  ".join(bad))
    return snap


def row_of(d):
    rep = d["reps"][0]
    return {
        "j_per_audio_s": d["joules_per_audio_s"],
        "cpu_s_per_audio_s": d["cpu_s_per_audio_s"],
        "wall_s_per_audio_s": rep["wall_seconds"] / rep["audio_seconds"],
        "idle_core_w": rep["core"]["idle_watts"],
        "idle_pkg_w": rep["package"]["idle_watts"],
        "mean_mhz": rep.get("mean_mhz"),
        "audio_seconds": rep["audio_seconds"],
        "core_clean": d["clean_repetitions"]["core"] == 1,
        "flags": [f for f in d["flags"] if not f.startswith("no_clean_package") and "package" not in f],
    }


def write_atomic(path, text):
    """Temp file in the target directory, then rename: a reader never sees a half-written evidence file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--cold-dev", type=Path, default=Path(__file__).resolve().parents[4] / "docs" / "paper" / "data" / "cold_dev.csv")
    args = ap.parse_args()
    root = args.root
    snap = check_manifest(root, [(b, k) for k in IDS for b in (1, 2, 3)] + [(1, "anchor730-live"), (1, "anchor730-zip")])
    meta = {r["id"]: r for r in csv.DictReader(args.cold_dev.open())}
    summary, rows = {}, []
    for key, cid in IDS.items():
        blocks, paths = [], []
        for b in (1, 2, 3):
            d, p, h = snap.get((b, key), (None, None, None))
            if d is None:
                continue
            r = row_of(d)
            r["block"] = b
            r["sha256"] = h
            blocks.append(r)
            paths.append(p)
        if not blocks:
            continue
        # Colour rule: only core-clean blocks count; an arm with none stays grey (no evidence row).
        keep = [i for i, r in enumerate(blocks) if r["core_clean"]]
        if not keep:
            summary[key] = {"grey": True, "n_clean": 0, "id": cid,
                            "disturbed_blocks_j_per_audio_s": [r["j_per_audio_s"] for r in blocks],
                            "note": "no core-clean attempt; disturbed values stay in the receipts only"}
            continue
        blocks_all, paths_all = blocks, paths
        blocks = [blocks_all[i] for i in keep]
        paths = [paths_all[i] for i in keep]
        med = statistics.median(r["j_per_audio_s"] for r in blocks)
        cpu = statistics.median(r["cpu_s_per_audio_s"] for r in blocks)
        mid = min(range(len(blocks)), key=lambda i: abs(blocks[i]["j_per_audio_s"] - med))
        vals = [r["j_per_audio_s"] for r in blocks]
        entry = {
            "j_per_audio_s_median": round(med, 6),
            "blocks": vals,
            "n_clean": len(blocks),
            "clean_blocks": [r["block"] for r in blocks],
            "n_blocks_run": len(blocks_all),
            "block_spread_pct": round(100 * (max(vals) - min(vals)) / med, 2) if med else None,
            "cpu_s_per_audio_s": round(cpu, 6),
            "speed_x": round(1 / cpu, 3),
            "wall_speed_x": round(1 / statistics.median(r["wall_s_per_audio_s"] for r in blocks), 3),
            "panel": "p73",
            "audio_seconds": blocks[0]["audio_seconds"],
            "idle_w": round(statistics.median(r["idle_core_w"] for r in blocks), 4),
            "idle_pkg_w": round(statistics.median(r["idle_pkg_w"] for r in blocks), 2),
            "block_core_clean": [r["core_clean"] for r in blocks],
            "block_flags": [r["flags"] for r in blocks],
            "id": cid,
        }
        summary[key] = entry
        rel = [str(p.relative_to(root)) for p in paths]
        m = meta.get(cid, {})
        rows.append({
            "id": cid, "variant": m.get("variant"), "short": m.get("short"), "role": m.get("role"),
            "class": m.get("class"), "machine": "zen2", "host": "xbabe0", "threads": 1,
            "series": m.get("series"), "panel": "p73", "protocol": "R7",
            "energy_j_per_audio_s": entry["j_per_audio_s_median"], "energy_method": "core_idle_subtracted",
            "kind": "r7_energy", "blocks": vals, "n_clean": len(blocks),
            "clean_blocks": [r["block"] for r in blocks], "block_spread_pct": entry["block_spread_pct"],
            "cpu_s_per_audio_s": entry["cpu_s_per_audio_s"], "speed_x": entry["speed_x"],
            "idle_core_w": entry["idle_w"], "block_core_clean": entry["block_core_clean"],
            "block_flags": entry["block_flags"],
            "receipt": rel[mid], "receipt_sha256": blocks[mid]["sha256"],
            "receipts": [{"path": r, "sha256": b["sha256"]} for r, b in zip(rel, blocks)],
            "label": "exploratory",
        })
    anchors = {}
    for key, cid, name in (("anchor730-live", "rc2-live-lm-zen2-t1", "live"), ("anchor730-zip", "r0-int8-r6-zen2-t1", "zip")):
        d, p, h = snap.get((1, key), (None, None, None))
        if d is None:
            continue
        r = row_of(d)
        ref = R6[cid]
        anchors[name] = {"j_per_audio_s": r["j_per_audio_s"], "r6": ref,
                         "deviation_pct": round(100 * (r["j_per_audio_s"] / ref - 1), 2),
                         "within_5pct": abs(r["j_per_audio_s"] / ref - 1) <= TOL,
                         "cpu_s_per_audio_s": r["cpu_s_per_audio_s"], "speed_x": round(1 / r["cpu_s_per_audio_s"], 3),
                         "core_clean": r["core_clean"], "flags": r["flags"], "panel": "full730",
                         "receipt": str(p.relative_to(root)), "receipt_sha256": h}
    missing = [f"block{b}/{key}.json" for key in IDS for b in (1, 2, 3) if (b, key) not in snap]
    problems = [f"missing receipt {m}" for m in missing]
    for name in ("live", "zip"):
        a = anchors.get(name)
        if a is None:
            problems.append(f"anchor {name} missing")
            continue
        if not a["core_clean"]:
            problems.append(f"anchor {name} is not core-clean")
        if not a["within_5pct"]:
            problems.append(f"anchor {name} is {a['deviation_pct']}% from R6 (band 5%)")
    verdict = "VALID" if not problems else "VOID"
    out = {"protocol": "R7", "validity": verdict, "anchors": anchors, "arms": summary, "problems": problems}
    (root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    if args.evidence and verdict == "VALID":
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit("duplicate target ids in evidence")
        for r in rows:
            v = r["energy_j_per_audio_s"]
            if not (isinstance(v, float) and math.isfinite(v) and v > 0):
                raise SystemExit(f"bad energy value for {r['id']}: {v!r}")
        for r in rows:
            r["anchor_check"] = verdict
        write_atomic(args.evidence, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(verdict, json.dumps(anchors, indent=1))
    for k, v in summary.items():
        if v.get("grey"):
            print(f"{k:28s} GREY (no clean block)")
            continue
        print(f"{k:28s} {v['j_per_audio_s_median']:8.4f} {v['blocks']} spread {v['block_spread_pct']}% speed {v['speed_x']}x")
    if verdict != "VALID":
        for p in problems:
            print("VOID:", p, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
