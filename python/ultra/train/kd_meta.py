"""Write a teacher cache's identity side file (`KD_META.json`, format ultra.kd-cache/v1).

  python -m train.kd_meta <cache_dir> --tokenizer <bpe.model> --teacher "<teacher description>"

It records the sha256 of the tokenizer the cached token ids belong to and of every shard, checked
against the producer's SHA256SUMS. `TeacherStore(..., tokenizer_sha256=...)` refuses a cache whose
side file names another tokenizer or does not list or match a shard.
"""
from __future__ import annotations

import argparse
import json

from train.kd import write_meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--tokenizer", required=True, help="the SentencePiece model the teacher emitted")
    ap.add_argument("--teacher", required=True, help="teacher description, e.g. its export and decode mode")
    a = ap.parse_args(argv)
    meta = write_meta(a.cache_dir, a.tokenizer, a.teacher)
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"} | {"shards": len(meta["shards"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
