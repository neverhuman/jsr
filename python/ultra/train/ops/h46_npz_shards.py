"""H46: cut a cached dev split (`<split>.npz`: pcm, offsets, uids, speakers, texts) into n shard
files for parallel `ultra-engine-eval --npz` runs. Shard i holds utterances i, i+n, i+2n, ... in
the original order, written as `<out>/s<i>/<split>.npz` so the engine keeps the split name (it
takes it from the file stem). Arrays keep their dtypes; pcm and offsets are re-based per shard.

  cd python/ultra && python -m train.ops.h46_npz_shards <split.npz> <out dir> <n>

Prints one JSON line per shard with its utterance count and sha256. Dev splits only.
"""
import hashlib
import json
import os
import sys

import numpy as np

from train.errors import LmEvalError


def main(src: str, out: str, n: int) -> None:
    split = os.path.basename(src)[:-4]
    if not split.startswith("dev-"):
        raise LmEvalError(f"{src}: only dev splits are sharded (test sets are sealed)")
    z = np.load(src)
    offs, pcm = z["offsets"], z["pcm"]
    uids, spk, texts = z["uids"], z["speakers"], z["texts"]
    for i in range(n):
        idx = np.arange(i, len(uids), n)
        lens = offs[idx + 1] - offs[idx]
        new_offs = np.concatenate([[0], np.cumsum(lens)]).astype(offs.dtype)
        new_pcm = np.concatenate([pcm[offs[j]:offs[j + 1]] for j in idx]).astype(pcm.dtype)
        d = os.path.join(out, f"s{i:02d}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{split}.npz")
        tmp = path + ".tmp.npz"
        np.savez(tmp, pcm=new_pcm, offsets=new_offs, uids=uids[idx], speakers=spk[idx],
                 texts=texts[idx])
        os.replace(tmp, path)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        print(json.dumps({"shard": path, "utterances": int(len(idx)), "sha256": digest}),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
