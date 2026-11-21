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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit-all", action="store_true", help="evaluate the whole split")
    ap.add_argument("--dump", default=None, help="write per-utterance JSON lines (uid, ref, hyp)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    p = checkpoint.load(a.checkpoint or checkpoint.latest(a.run_dir), restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval().to(a.device)
    tok = Tokenizer(tokenizer_path(4096))
    tok_c = Tokenizer(tokenizer_path(1024))
    feat = Featurizer(tok, tok_c)
    exs = sorted((feat.example(u.uid, u.text, u.pcm) for u in load_split(a.split, None if a.limit_all else a.limit)),
                 key=lambda e: e.seconds)
    refs, hyps, ctc_hyps, uids, forced, exact, ctc_exact = [], [], [], [], 0, 0, 0
    for i in range(0, len(exs), 8):
        b = collate(exs[i:i + 8]).to(a.device)
        toks, f = model.greedy(b.feats, b.feat_lens, max_symbols=a.max_symbols)
        forced += f
        for t in model.ctc_greedy(b.feats, b.feat_lens):
            ctc_hyps.append(tok_c.decode(t))
        uids.extend(b.uids)
        for r, t in zip(b.texts, toks, strict=True):
            h = tok.decode(t)
            refs.append(r)
            hyps.append(h)
            exact += int(r.lower().split() == h.split())
    w = corpus_wer(refs, hyps)
    wc = corpus_wer(refs, ctc_hyps)
    ctc_exact = sum(r.lower().split() == h.split() for r, h in zip(refs, ctc_hyps, strict=True))
    out = {"step": p["step"], "split": a.split, "n": len(refs), "max_symbols": a.max_symbols,
           "forced_advances": forced, "exact_utts": exact, **w,
           "ctc_wer": wc["wer"], "ctc_exact_utts": ctc_exact}
    if a.dump:
        with open(a.dump, "w") as fh:
            for u, r, h, c in zip(uids, refs, hyps, ctc_hyps, strict=True):
                fh.write(json.dumps({"uid": u, "ref": r, "hyp": h, "ctc_hyp": c}) + "\n")
        out["dump"] = a.dump
    print(json.dumps(out))
    for r, h in list(zip(refs, hyps, strict=True))[:a.show]:
        print("REF", r.lower())
        print("HYP", h)


if __name__ == "__main__":
    main()
