"""CLI: python -m tools.paper_data [--check] [--write-manifest] [--publish] [--out DIR]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __doc__ as PACKAGE_DOC
from . import build, publish_figure
from .common import DATA, REPO, write_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=PACKAGE_DOC.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / DATA, help="output directory")
    ap.add_argument("--check", action="store_true", help="compare against --out instead of writing")
    ap.add_argument("--write-manifest", action="store_true", help="rebuild raw/MANIFEST.txt from disk")
    ap.add_argument("--publish", action="store_true",
                    help="copy the Figure 1 block into README.md and stamp docs/paper/money.png (after exporting it)")
    args = ap.parse_args(argv)
    if args.write_manifest:
        write_manifest(REPO)
    if args.publish:
        publish_figure(REPO)
        return 0
    files, missing = build(REPO)
    if args.check:
        bad = [n for n, t in files.items() if not (args.out / n).is_file() or (args.out / n).read_text(encoding="utf-8") != t]
        for n in bad:
            print(f"stale: {n}", file=sys.stderr)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for n, t in files.items():
            (args.out / n).write_text(t, encoding="utf-8", newline="\n")
        bad = []
    for m in missing:
        print(f"MISSING required value: {m}", file=sys.stderr)
    for line in (files["records.jsonl"].splitlines()):
        r = json.loads(line)
        if r.get("pending"):
            print(f"PENDING (prints red ?? in the PDF): {r['id']}: {r['pending']}", file=sys.stderr)
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
