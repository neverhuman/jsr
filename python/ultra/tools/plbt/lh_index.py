"""plbt/v1 step 1: index the Libriheavy manifests by recording (H61).

Reads `libriheavy_cuts_{small,medium,large}.jsonl.gz` (the cased manifests published by
k2-fsa/libriheavy, Apache-2.0) and writes one sqlite file:

    cuts(rec TEXT, start REAL, end REAL, book TEXT, asr TEXT)   index on (rec, start)
    meta(key TEXT PRIMARY KEY, value TEXT)                      manifest sha256s, counts, conventions

`rec` is the Libri-Light member path without ".flac" ("small/100/<book dir>/<file>"), which is what
H52 rows carry in `src`. `book` is `supervisions[0].custom.texts[0]` (the book text with case and
punctuation), `asr` is `texts[1]` (Libriheavy's own transcript, upper-case). Nothing is executed and
no audio is touched. A line that does not have this shape stops the run: the schema is checked, not
assumed.

usage (from python/ultra): python -m tools.plbt.lh_index --out lh.sqlite libriheavy_cuts_small.jsonl.gz [more manifests]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3

from tools.plbt.norm import CONVENTIONS

PREFIXES = ("download/librilight/", "librilight/")


def rec_key(path: str) -> str:
    """A Libriheavy recording id / source path or an H52 `src` member name -> the shared key."""
    p = path.strip().lstrip("./")
    for pre in PREFIXES:
        if p.startswith(pre):
            p = p[len(pre):]
    return p[:-5] if p.endswith(".flac") else p


def parse_cut(line: str) -> tuple[str, float, float, str, str]:
    c = json.loads(line)
    sup = c["supervisions"]
    if len(sup) != 1:
        raise ValueError(f"cut {c.get('id')!r}: expected one supervision, got {len(sup)}")
    texts = sup[0]["custom"]["texts"]
    if len(texts) < 2 or not isinstance(texts[0], str) or not isinstance(texts[1], str):
        raise ValueError(f"cut {c.get('id')!r}: custom.texts is not [book text, transcript]")
    start = float(c["start"])
    return rec_key(c["recording"]["id"]), start, start + float(c["duration"]), texts[0], texts[1]


def build(out: str, manifests: list[str], batch: int = 50_000) -> dict:
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists; never overwrite an index")
    db = sqlite3.connect(out)
    db.execute("CREATE TABLE cuts(rec TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL, "
               "book TEXT NOT NULL, asr TEXT NOT NULL)")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    stats = {"conventions": CONVENTIONS, "manifests": [], "cuts": 0, "hours": 0.0}
    for path in manifests:
        h, n, secs, rows = hashlib.sha256(), 0, 0.0, []
        with open(path, "rb") as raw:
            for block in iter(lambda: raw.read(1 << 20), b""):
                h.update(block)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = parse_cut(line)
                rows.append(row)
                n, secs = n + 1, secs + (row[2] - row[1])
                if len(rows) >= batch:
                    db.executemany("INSERT INTO cuts VALUES (?,?,?,?,?)", rows)
                    rows = []
        db.executemany("INSERT INTO cuts VALUES (?,?,?,?,?)", rows)
        stats["manifests"].append({"file": os.path.basename(path), "sha256": h.hexdigest(), "cuts": n,
                                   "hours": round(secs / 3600.0, 2)})
        stats["cuts"], stats["hours"] = stats["cuts"] + n, round(stats["hours"] + secs / 3600.0, 2)
    db.execute("CREATE INDEX cuts_rec ON cuts(rec, start)")
    stats["recordings"] = db.execute("SELECT COUNT(DISTINCT rec) FROM cuts").fetchone()[0]
    db.execute("INSERT INTO meta VALUES ('stats', ?)", (json.dumps(stats, sort_keys=True),))
    db.commit()
    db.close()
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("manifests", nargs="+")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, a.manifests), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
