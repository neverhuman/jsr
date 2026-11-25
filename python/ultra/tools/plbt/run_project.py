"""plbt/v1 step 2: book-text labels for rows that are already cut (H61, route A).

For every row (id, src, t0, t1, text) of the given parquet shards (H52 trainer-ready shards; only
these five columns are read, never the audio) look up the Libriheavy cuts of `src` within MARGIN
seconds of the row, project the row's transcript onto their book text (tools/plbt/project.py), apply
H52's text filters to the new label, and write into --out (which must not exist):

    plbt_v1.<name>.tsv           uid<TAB>label    kept rows, sorted by uid   -> TrainConfig.label_map (T1)
    r0.rows_of_plbt_v1.<name>.tsv uid<TAB>text    the same uids with the shard's own text         (T0)
    audit.<name>.jsonl           one record per row: decision and counts (no text)
    FILTER_REPORT.json           rules, parameters, counts, hours, input and output sha256s

Filters on the label (H52 section 5): H3 word rate within [0.6, 4.6] words/s over t1 - t0; H4 LibriVox
boilerplate; H5 an identical label at most 3 times (first by sha256(uid)). S1 (R0 confidence) is not
applied: with book text R0's confidence marks hard audio, not a bad label.

usage (from python/ultra): python -m tools.plbt.run_project --index lh.sqlite --out DIR --name small SHARD_DIR [SHARD_DIR ...]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3

from tools.plbt.lh_index import rec_key
from tools.plbt.norm import CONVENTIONS, book_words, label_words
from tools.plbt.project import PARAMS, Cut, project

MARGIN = 2.0  # seconds of book on each side of the row
RULES = {"H3_min_wps": 0.6, "H3_max_wps": 4.6, "H5_max_copies": 3, "MARGIN": MARGIN,
         "H4_pattern": r"\bLIB[A-Z]*\s?VO[CX][A-Z]*\b|RECORDINGS? (?:IS|ARE) IN THE PUBLIC DOMAIN|TO VOLUNTEER PLEASE VISIT"}
H4 = re.compile(RULES["H4_pattern"])
COLUMNS = ["id", "src", "t0", "t1", "text"]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cuts_for(db: sqlite3.Connection, src: str, t0: float, t1: float) -> list[Cut]:
    rows = db.execute("SELECT start, end, book FROM cuts WHERE rec = ? AND end > ? AND start < ? ORDER BY start",
                      (rec_key(src), t0 - MARGIN, t1 + MARGIN)).fetchall()
    return [Cut(s, e, book_words(b)) for s, e, b in rows]


def label_row(db: sqlite3.Connection, row: dict) -> tuple[str, list[str], dict]:
    """(decision, label words, audit record) for one row."""
    t0, t1 = float(row["t0"]), float(row["t1"])
    res = project(label_words(row["text"] or ""), cuts_for(db, row["src"], t0, t1), t0, t1)
    decision = res.decision
    if decision == "keep":
        rate = len(res.label) / max(t1 - t0, 1e-3)
        if not RULES["H3_min_wps"] <= rate <= RULES["H3_max_wps"]:
            decision = "H3_rate"
        elif H4.search(" ".join(res.label)):
            decision = "H4_librivox"
    return decision, res.label, {"id": row["id"], "decision": decision, "dur": round(t1 - t0, 3), **res.audit()}


def run(index: str, out: str, name: str, shard_dirs: list[str]) -> dict:
    import pyarrow.parquet as pq

    os.makedirs(out, exist_ok=False)
    db = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    kept: dict[str, tuple[str, str, float]] = {}
    counts: dict[str, int] = {}
    hours = {"rows": 0.0}
    inputs = []
    with open(os.path.join(out, f"audit.{name}.jsonl"), "w", encoding="utf-8") as audit:
        for d in shard_dirs:
            for path in sorted(glob.glob(os.path.join(d, "*.parquet"))):
                t = pq.read_table(path, columns=COLUMNS).to_pylist()
                inputs.append({"file": os.path.join(os.path.basename(d.rstrip("/")), os.path.basename(path)),
                               "sha256": _sha256_file(path), "rows": len(t)})
                for row in t:
                    decision, label, rec = label_row(db, row)
                    hours["rows"] += rec["dur"] / 3600.0
                    if decision == "keep":
                        if row["id"] in kept:
                            raise ValueError(f"duplicate row id {row['id']!r}")
                        kept[row["id"]] = (" ".join(label), " ".join(label_words(row["text"])), rec["dur"])
                    else:
                        counts[decision] = counts.get(decision, 0) + 1
                    audit.write(json.dumps(rec, sort_keys=True) + "\n")
    groups: dict[str, list[str]] = {}
    for uid, (label, _, _) in kept.items():
        groups.setdefault(label, []).append(uid)
    for ids in groups.values():  # H5
        for uid in sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[RULES["H5_max_copies"]:]:
            del kept[uid]
            counts["H5_duplicate"] = counts.get("H5_duplicate", 0) + 1
    files = {}
    for fname, col in ((f"plbt_v1.{name}.tsv", 0), (f"r0.rows_of_plbt_v1.{name}.tsv", 1)):
        path = os.path.join(out, fname)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for uid in sorted(kept):
                fh.write(f"{uid}\t{kept[uid][col]}\n")
        files[fname] = _sha256_file(path)
    changed = sum(a != b for a, b, _ in kept.values())
    report = {"conventions": CONVENTIONS, "name": name, "rules": RULES, "project": PARAMS,
              "index": {"file": os.path.basename(index), "sha256": _sha256_file(index),
                        "stats": json.loads(db.execute("SELECT value FROM meta WHERE key='stats'").fetchone()[0])},
              "inputs": inputs, "rows": sum(i["rows"] for i in inputs), "kept": len(kept), "drops": counts,
              "rows_hours": round(hours["rows"], 2), "kept_hours": round(sum(v[2] for v in kept.values()) / 3600.0, 2),
              "label_differs_from_input": changed, "files": files}
    with open(os.path.join(out, "FILTER_REPORT.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args(argv)
    rep = run(a.index, a.out, a.name, a.shards)
    print(json.dumps({k: rep[k] for k in ("rows", "kept", "drops", "label_differs_from_input", "files")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
