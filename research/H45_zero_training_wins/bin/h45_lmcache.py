#!/usr/bin/env python3
"""h45-wins: write an `ultra.lm-enc-cache/v1` encoder cache (train.lm_decode's format) on a GPU at any chunk, so
train.lm_decode beam can run the LM decode on the commit pass's final encoder output (chunk 16).

  python h45_lmcache.py --src <tree>/python/ultra --jsr-data DIR --checkpoint CKPT [--weights ema] --chunk 16 --out DIR

Exactly train.lm_decode.encode's procedure (model loaded by lm_decode._load_model, utterances ordered by feature
frames, stable, batches of --batch, joiner.enc_proj frames + greedy tokens per utterance) except that the encoder
runs on --device with model.encoder.cfg.chunk set to --chunk, and one process writes <split>.enc_p.npy +
<split>.index.json directly in the layout lm_decode.merge produces. Dev splits only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--jsr-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-fraction", type=float, default=0.30)
    ap.add_argument("--feat-cache", default=None, help="h45_decode.py feature cache dir (full-split entries): "
                    "reuse its featurised examples (same order: feature frames, stable) instead of featurising")
    a = ap.parse_args()
    os.environ["JSR_DATA"] = a.jsr_data
    sys.path.insert(0, a.src)
    import numpy as np
    import torch

    from train.data import Featurizer, collate
    from train.evaluate import check_split
    from train.lm_decode import CACHE_SCHEMA, _load_model
    from train.transducer import greedy_decode
    from ultra.libri import load_split

    torch.set_num_threads(2)
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)
    step, model, tok, tok_c, sha = _load_model(a.checkpoint, a.weights)
    model.encoder.cfg = dataclasses.replace(model.encoder.cfg, chunk=a.chunk)
    model.to(a.device)
    feat = Featurizer(tok, tok_c)
    os.makedirs(a.out, exist_ok=True)
    import glob
    import hashlib

    def _fsha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    for split in a.splits:
        check_split(split)
        if os.path.exists(os.path.join(a.out, f"{split}.index.json")):
            print(json.dumps({"skip": split, "chunk": a.chunk}), flush=True)
            continue
        cached = None
        utts = order = None
        if a.feat_cache:
            from train.data import Example

            key_src = "|".join([split, "all", _fsha(tok.path), _fsha(tok_c.path),
                                _fsha(os.path.join(a.src, "train", "features.py")),
                                _fsha(os.path.join(a.src, "train", "data.py"))])
            cpath = os.path.join(a.feat_cache, hashlib.sha256(key_src.encode()).hexdigest()[:16] + f".{split}.pt")
            if os.path.exists(cpath):
                # the cache holds the h45_decode examples in train.evaluate's order (feature frames, stable), which is
                # lm_decode.encode's order too; libri.load_split is skipped (it re-reads the npz arrays per utterance)
                z = torch.load(cpath, map_location="cpu", weights_only=False)
                cached = [Example(*row) for row in z["rows"]]
        if cached is None:
            utts = load_split(split, None)
            order = sorted(range(len(utts)), key=lambda i: feat.logmel.num_frames(len(utts[i].pcm)))
        rows, frames, off, t0 = [], [], 0, time.time()
        n_utts = len(cached) if cached is not None else len(order)
        for bi in range((n_utts + a.batch - 1) // a.batch):
            if cached is not None:
                exs = cached[bi * a.batch:(bi + 1) * a.batch]
                spk = [int(e.uid.split("-")[0]) for e in exs]
            else:
                idx = order[bi * a.batch:(bi + 1) * a.batch]
                exs = [feat.example(utts[i].uid, utts[i].text, utts[i].pcm) for i in idx]
                spk = [utts[i].speaker for i in idx]
            b = collate(exs).to(a.device)
            with torch.no_grad():
                enc, lens = model.encode(b.feats, b.feat_lens, None)
                enc_p = model.joiner.enc_proj(enc.float())
                greedy, _ = greedy_decode(model.predictor, model.joiner, enc_p, lens, a.max_symbols)
            ep = enc_p.float().cpu()
            for j, uid in enumerate(b.uids):
                n = int(lens[j])
                frames.append(ep[j, :n].numpy().astype(np.float32))
                rows.append({"uid": uid, "speaker": spk[j], "ref": b.texts[j], "offset": off,
                             "frames": n, "seconds": float(b.feat_lens[j]) / 100.0, "greedy": greedy[j]})
                off += n
        np.save(os.path.join(a.out, f"{split}.enc_p.npy"), np.concatenate(frames))
        index = {"schema": CACHE_SCHEMA, "split": split, "checkpoint_sha256": sha, "step": step,
                 "weights": a.weights, "chunk": model.encoder.cfg.chunk,
                 "left_context": model.encoder.cfg.left_context, "batch": a.batch, "order": "length",
                 "max_symbols": a.max_symbols, "tokenizer_sha256": tok.sha256, "utterances": n_utts,
                 "frames": off, "utts": rows, "encode_cpu_seconds": round(time.time() - t0, 1),
                 "h45_note": f"GPU encode ({a.device}) by h45_lmcache.py at chunk {a.chunk}"}
        tmp = os.path.join(a.out, f"{split}.index.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(index, fh)
        os.replace(tmp, os.path.join(a.out, f"{split}.index.json"))
        print(json.dumps({"split": split, "chunk": a.chunk, "utts": len(rows), "frames": off,
                          "seconds": index["encode_cpu_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
