"""Check TRAIN label selection against a sealed, ID-only Libriheavy roster."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from tools.plbt.lh_index import rec_key
from train.label_map import LabelMap


def _speaker_book(path: str) -> tuple[str, str]:
    parts = rec_key(path).split("/")
    if len(parts) != 4 or not all(parts) or not parts[1].isdigit():
        raise ValueError("expected tier/speaker/book/recording Libri-Light IDs")
    return parts[1], parts[2].casefold()


def audit(sealed_ids: Path, sealed_sha256: str, access_log: Path, labels: LabelMap,
          shards: list[Path]) -> dict:
    import pyarrow.parquet as pq

    with access_log.open("a") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.write(f"{datetime.now(timezone.utc).isoformat()} PLBT V2 roster check. "
                  f"ID-only read, SHA-256 {sealed_sha256}.\n")
    raw = sealed_ids.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("sealed roster SHA-256 differs from its registration")
    roster = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    if roster.fieldnames != ["split", "cut_id", "speaker", "start", "duration"]:
        raise ValueError("expected the registered ID-only Libriheavy roster schema")
    speakers, books, cuts = set(), set(), set()
    for row in roster:
        speaker, book = _speaker_book(row["cut_id"])
        if speaker != row["speaker"] or row["cut_id"] in cuts:
            raise ValueError("sealed cut ID or speaker disagrees with the roster")
        cuts.add(row["cut_id"])
        speakers.add(speaker)
        books.add(book)
    if not cuts or not labels.texts or not shards:
        raise ValueError("the roster, TRAIN label map and TRAIN shards must be nonempty")

    checked, speaker_hits, book_hits = set(), set(), set()
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "src"]):
            for row in batch.to_pylist():
                uid = row["id"]
                if uid not in labels.texts:
                    continue
                if uid in checked:
                    raise ValueError("duplicate TRAIN UID in the audited shard set")
                checked.add(uid)
                speaker, book = _speaker_book(row["src"])
                if speaker in speakers:
                    speaker_hits.add(uid)
                if book in books:
                    book_hits.add(uid)
    missing = len(labels.texts) - len(checked)
    return {
        "schema": "ultra.plbt-d7-roster/v1",
        "sealed_ids_sha256": sealed_sha256,
        "labels_sha256": labels.sha256,
        "sealed_cuts": len(cuts),
        "sealed_speakers": len(speakers),
        "sealed_book_directories": len(books),
        "checked_training_rows": len(checked),
        "missing_training_rows": missing,
        "speaker_overlap_rows": len(speaker_hits),
        "book_directory_overlap_rows": len(book_hits),
        "pass": not (missing or speaker_hits or book_hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-ids", required=True, type=Path)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--access-log", required=True, type=Path)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--label-map-sha256", required=True)
    parser.add_argument("train_shards", nargs="+", type=Path)
    args = parser.parse_args()
    labels = LabelMap.load(args.label_map, args.label_map_sha256, only=True)
    shards = [p for path in args.train_shards
              for p in (sorted(path.glob("*.parquet")) if path.is_dir() else [path])]
    result = audit(args.sealed_ids, args.sealed_sha256, args.access_log, labels, shards)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
