#!/usr/bin/env python3
"""h45-wins: run train.lm_decode beam unchanged on a uid subset of an encoder cache (e.g. the selection half).

  python h45_lmbeam.py --uids S.txt -- <train.lm_decode beam arguments>

The only change is a filter on the cache index: utterances whose uid is not in --uids are dropped from
index["utts"] before the decode loop (offsets and the frame array are untouched, so every kept utterance sees exactly
the frames it would see in a full run). Shards (--shard i/n) then split the kept utterances.
"""
from __future__ import annotations

import sys


def main():
    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--uids":
        raise SystemExit(__doc__)
    uids_file = argv[1]
    rest = argv[argv.index("--") + 1:]
    keep = {x.strip() for x in open(uids_file) if x.strip()}
    import os

    sys.path.insert(0, os.getcwd())  # run from the python/ultra tree, like `python -m train.lm_decode`
    from train import lm_decode

    orig = lm_decode.load_cache

    def filtered(cache, split):
        index, frames = orig(cache, split)
        index = dict(index, utts=[u for u in index["utts"] if u["uid"] in keep])
        return index, frames

    lm_decode.load_cache = filtered
    lm_decode.main(rest)


if __name__ == "__main__":
    main()
