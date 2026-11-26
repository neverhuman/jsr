#!/usr/bin/env python3
"""Stage the R7 receipts.

    r7_stage.py RAW_DIR RECEIPT_ROOT [--sums FILE] [--clear-owned]

RAW_DIR is a copy of runs/r7 from the benchmark host: b<N>-<arm>.json plus b<N>-<arm>.unclean<K>.json and
b<N>-<arm>.failed*.json attempts. Output: RECEIPT_ROOT/attempts/ (EVERY attempt, clean, unclean and failed,
byte-identical, copied before any selection), RECEIPT_ROOT/block<N>/<arm>.json (the attempt used) and
RECEIPT_ROOT/selection.json (per attempt: sha256 of the exact bytes that were parsed, status, disturbance).
A group with only failed attempts is kept in attempts/ and recorded in selection.json with used=null.

Rule (fixed before reading any selection outcome, it looks at disturbance, never at the energy value):
a core-clean attempt beats an unclean one; among clean attempts the latest; among unclean attempts the one
with the smallest sibling_busy_fraction_run + elsewhere_busy_cpus_run / 64.

Custody: the sha256 of each attempt is taken from the bytes that are then parsed. When SHA256SUMS of the
retained receipt archive exists (RAW_DIR/SHA256SUMS or --sums) every attempt must match it, and every listed
b*.json must be present. A RECEIPT_ROOT that already holds staged files is refused unless --clear-owned
removes exactly the files this tool writes (attempts/, block<N>/, selection.json).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def split_name(name):
    stem = name[:-5]
    if ".failed" in stem:
        return stem.split(".failed")[0], "failed" + stem.split(".failed")[1]
    if ".unclean" in stem:
        return stem.split(".unclean")[0], "unclean" + stem.split(".unclean")[1]
    return stem, "final"


def read_sums(path):
    sums = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[Path(name.strip().lstrip("*")).name] = digest
    return sums


def disturbance(d):
    r = d["reps"][0]["busy"]
    return r["sibling_busy_fraction_run"] + r["elsewhere_busy_cpus_run"] / 64


def owned(root):
    return [root / "selection.json", root / "attempts", *sorted(root.glob("block[123]"))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--sums", type=Path)
    ap.add_argument("--clear-owned", action="store_true")
    args = ap.parse_args(argv)
    raw, root = args.raw, args.root

    present = [p for p in owned(root) if p.exists()]
    if present:
        if not args.clear_owned:
            raise SystemExit(f"{root} already holds staged files ({', '.join(p.name for p in present)}); "
                             "pass --clear-owned to remove exactly those")
        for p in present:
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    files = sorted(raw.glob("b[123]-*.json"))
    sums_path = args.sums or raw / "SHA256SUMS"
    sums = read_sums(sums_path) if sums_path.is_file() else None
    data = {p.name: p.read_bytes() for p in files}  # one read: these bytes are hashed, parsed and copied
    digest = {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}
    if sums is not None:
        for n, want in sums.items():
            if n.startswith(("b1-", "b2-", "b3-")) and n.endswith(".json") and n not in data:
                raise SystemExit(f"{n} is listed in {sums_path.name} but absent from {raw}")
        for n, got in digest.items():
            if n not in sums:
                raise SystemExit(f"{n} is not listed in {sums_path.name}")
            if sums[n] != got:
                raise SystemExit(f"sha256 mismatch for {n}: archive {sums[n]}, bytes read {got}")

    # custody first: every attempt is retained before any selection looks at it
    (root / "attempts").mkdir(parents=True, exist_ok=True)
    for n, b in data.items():
        (root / "attempts" / n).write_bytes(b)

    groups = {}
    for n in data:
        base, tag = split_name(n)
        groups.setdefault(base, []).append((tag, n))

    selection = {}
    for base, items in sorted(groups.items()):
        blk, arm = base.split("-", 1)
        attempts, cands = [], []
        for tag, n in items:
            entry = {"file": n, "tag": tag, "sha256": digest[n], "status": "ok", "core_clean": None, "disturbance": None}
            try:
                d = json.loads(data[n])
                if tag.startswith("failed") or "nonzero_exit" in d["flags"]:
                    entry["status"] = "failed"
                else:
                    entry["core_clean"] = d["clean_repetitions"]["core"] == 1
                    entry["disturbance"] = round(disturbance(d), 4)
                    cands.append((entry["core_clean"], tag, n, disturbance(d)))
            except (ValueError, KeyError, IndexError, TypeError):
                entry["status"] = "failed"
            attempts.append(entry)
        use = None
        if cands:
            clean = [c for c in cands if c[0]]
            if clean:
                # latest clean attempt: "final" is the last one written, otherwise the highest unclean number is older
                use = next((c for c in clean if c[1] == "final"), clean[-1])
            else:
                use = min(cands, key=lambda c: c[3])
            out = root / ("block" + blk[1:])
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{arm}.json").write_bytes(data[use[2]])
        selection[base] = {"used": use[1] if use else None, "used_file": use[2] if use else None,
                           "used_sha256": digest[use[2]] if use else None,
                           "core_clean": use[0] if use else False,
                           "disturbance": round(use[3], 4) if use else None, "attempts": attempts}
    (root / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    staged = [v for v in selection.values() if v["used"] is not None]
    print(len(staged), "receipts staged;", sum(1 for v in staged if not v["core_clean"]), "unclean used;",
          len(selection) - len(staged), "groups with no usable attempt;", len(data), "attempts retained")


if __name__ == "__main__":
    sys.exit(main())
