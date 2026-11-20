"""Greedy streaming-decode WER of a checkpoint (provisional scorer: ultra.wer).

  python -m train.evaluate --run-dir <run> --split dev-clean --limit 200 [--max-symbols 8]
  python -m train.evaluate --run-dir <run> --overfit 16 --split train-clean-100

Decoding uses the trained chunk mask, so each output depends only on audio up to the end of its
own chunk (the causality test in tests/test_train_model.py), as in streaming.
"""
from __future__ import annotations

import argparse
import json

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.data import Featurizer, collate
from train.tokenizer import Tokenizer, tokenizer_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--show", type=int, default=0, help="print N ref/hyp pairs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    p = checkpoint.load(a.checkpoint or checkpoint.latest(a.run_dir), restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval()
    tok = Tokenizer(tokenizer_path(4096))
    feat = Featurizer(tok, Tokenizer(tokenizer_path(1024)))
    exs = sorted((feat.example(u.uid, u.text, u.pcm) for u in load_split(a.split, a.limit)),
                 key=lambda e: e.seconds)
    refs, hyps, forced, exact = [], [], 0, 0
    for i in range(0, len(exs), 8):
        b = collate(exs[i:i + 8])
        toks, f = model.greedy(b.feats, b.feat_lens, max_symbols=a.max_symbols)
        forced += f
        for r, t in zip(b.texts, toks, strict=True):
            h = tok.decode(t)
            refs.append(r)
            hyps.append(h)
            exact += int(r.lower().split() == h.split())
    w = corpus_wer(refs, hyps)
    out = {"step": p["step"], "split": a.split, "n": len(refs), "max_symbols": a.max_symbols,
           "forced_advances": forced, "exact_utts": exact, **w}
    print(json.dumps(out))
    for r, h in list(zip(refs, hyps, strict=True))[:a.show]:
        print("REF", r.lower())
        print("HYP", h)


if __name__ == "__main__":
    main()
