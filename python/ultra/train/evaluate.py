"""Greedy streaming-decode WER of a checkpoint (provisional scorer: ultra.wer).

  python -m train.evaluate --run-dir <run> --split dev-clean --limit 200 [--max-symbols 8]
  python -m train.evaluate --run-dir <run> --overfit 16 --split train-clean-100

Decoding uses the trained chunk mask, so each output depends only on audio up to the end of its
own chunk (the causality test in tests/test_train_model.py), as in streaming.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.beam import modified_beam_search
from train.data import Featurizer, collate
from train.tokenizer import Tokenizer, tokenizer_path
from train.transducer import greedy_decode


PANEL_SCHEMA = "ultra.dev-panel/v1"


def check_split(split: str) -> str:
    """Refuse held-out test splits on the plain --split path (as the panel loader does): model
    and checkpoint choices are made on dev only, so this tool never scores test-* sets."""
    if split.startswith("test-"):
        raise ValueError(f"split {split!r} refused: test-* splits are held out and never scored here")
    return split


def panel_utts(path: str, only: str | None = None):
    """Utterances of a verified panel; see load_panel."""
    return load_panel(path, only)[0]


def load_panel(path: str, only: str | None = None):
    """(utterances, manifest_sha256) of an `ultra.dev-panel/v1` panel. Utterances of an `ultra.dev-panel/v1` panel (python/ultra/tools/dev_panel.py), verified:
    the manifest sha256 is recomputed exactly as the tool writes it (sha256 of the panel without
    `manifest_sha256`, json.dumps(indent=1, sort_keys=True)); each split must be a `dev-*` split
    whose cache file hashes to the recorded `cache_sha256`; every uid must exist in its cache."""
    import hashlib

    from ultra import libri

    with open(path) as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict) or spec.get("schema") != PANEL_SCHEMA:
        raise ValueError(f"panel schema must be {PANEL_SCHEMA}")
    body = {k: v for k, v in spec.items() if k != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest()
    if digest != spec.get("manifest_sha256"):
        raise ValueError(f"panel manifest_sha256 mismatch: {digest} != {spec.get('manifest_sha256')}")
    out = []
    for sp in spec["splits"]:
        split = sp["split"]
        if not split.startswith("dev-"):
            raise ValueError(f"panel split {split!r} refused: only dev-* splits may select")
        if only is not None and split != only:
            continue
        cache = os.path.join(libri.CACHE, f"{split}.npz")
        with open(cache, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != sp["cache_sha256"]:
            raise ValueError(f"{split} cache sha256 {got} != panel {sp['cache_sha256']}")
        ids = set(sp["uids"])
        chosen = [u for u in libri.load_split(split) if u.uid in ids]
        if len(chosen) != len(ids):
            raise ValueError(f"{split}: {len(ids) - len(chosen)} panel uids missing from the cache")
        out.extend(chosen)
    if only is not None and not out:
        raise ValueError(f"panel has no split {only!r}")
    return out, digest


def panel_record(path: str, only: str | None, manifest_sha256: str) -> dict:
    """Identity of the panel an evaluation used, for its output record: the file name alone does
    not pin the contents, the verified manifest sha256 does."""
    return {"file": os.path.basename(path), "split": only or "all",
            "manifest_sha256": manifest_sha256}


def main(argv=None) -> None:
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
    ap.add_argument("--panel", default=None,
                    help="ultra.dev-panel/v1 panel to score instead of --split/--limit "
                         "(manifest and cache hashes verified; dev-* splits only)")
    ap.add_argument("--panel-split", default=None, help="score only this split of the panel")
    ap.add_argument("--ema", action="store_true", help="score the checkpoint's EMA weights")
    ap.add_argument("--batch", type=int, default=8, help="utterances per forward pass")
    ap.add_argument("--beam", type=int, default=0,
                    help=">0: also decode with modified beam search of this width (diagnostic)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="override the attention chunk (frames); 0 = the trained value")
    ap.add_argument("--left-context", type=int, default=-1,
                    help="override the attention left context (frames); -1 = the trained value")
    ap.add_argument("--full-context", action="store_true",
                    help="unrestricted attention (whole utterance, incl. future frames): measures "
                         "how much a causally trained model gains from future context it never saw")
    ap.add_argument("--order", default="length", choices=("length", "shuffled"),
                    help="length: sort by duration (least padding); shuffled: fixed-seed random "
                         "order (most padding; a padding-sensitivity check)")
    a = ap.parse_args(argv)
    if a.panel is None:
        check_split(a.split)
    torch.set_num_threads(a.threads)
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    p = checkpoint.load(a.checkpoint or checkpoint.latest(a.run_dir), restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    if a.ema and p.get("ema") is None:
        raise ValueError("--ema: the checkpoint holds no EMA weights (ema_decay was 0)")
    model.load_state_dict(p["ema"] if a.ema else p["model"])
    model.eval().to(a.device)
    ecfg = model.encoder.cfg
    if a.full_context:
        a.chunk, a.left_context = 1 << 16, 1 << 16
    if a.left_context >= 0:
        import dataclasses

        ecfg = model.encoder.cfg = dataclasses.replace(ecfg, left_context=a.left_context)
    chunk = a.chunk or None
    mcfg = ModelConfig.from_dict(p["model_config"])  # heads sized to the run's tokenizers
    tok = Tokenizer(tokenizer_path(mcfg.vocab_rnnt - 1))
    tok_c = Tokenizer(tokenizer_path(mcfg.vocab_ctc - 1))
    feat = Featurizer(tok, tok_c)
    panel_sha = None
    if a.panel:
        utts, panel_sha = load_panel(a.panel, a.panel_split)
    else:
        utts = load_split(a.split, None if a.limit_all else a.limit)
    if a.batch < 1:
        raise ValueError("--batch must be >= 1")
    exs = sorted((feat.example(u.uid, u.text, u.pcm) for u in utts), key=lambda e: e.seconds)
    if a.order == "shuffled":
        import random

        random.Random(0).shuffle(exs)
    refs, hyps, ctc_hyps, uids, forced, exact, ctc_exact = [], [], [], [], 0, 0, 0
    beam_hyps, secs = [], []
    for i in range(0, len(exs), a.batch):
        b = collate(exs[i:i + a.batch]).to(a.device)
        with torch.no_grad():
            enc, lens = model.encode(b.feats, b.feat_lens, chunk)
            enc_p = model.joiner.enc_proj(enc.float())
            toks, f = greedy_decode(model.predictor, model.joiner, enc_p, lens, a.max_symbols)
            if a.beam:
                beam_hyps.extend(tok.decode(t) for t in modified_beam_search(
                    model.predictor, model.joiner, enc_p, lens, a.beam))
            best = model.ctc_head(enc.float()).argmax(dim=-1)
        forced += f
        for j in range(best.shape[0]):
            seq = best[j, :int(lens[j])].tolist()
            ctc_hyps.append(tok_c.decode(
                [k for n, k in enumerate(seq) if k != 0 and (n == 0 or seq[n - 1] != k)]))
        secs.extend(float(x) / 100.0 for x in b.feat_lens)
        uids.extend(b.uids)
        for r, t in zip(b.texts, toks, strict=True):
            h = tok.decode(t)
            refs.append(r)
            hyps.append(h)
            exact += int(r.lower().split() == h.split())
    w = corpus_wer(refs, hyps)
    wc = corpus_wer(refs, ctc_hyps)
    ctc_exact = sum(r.lower().split() == h.split() for r, h in zip(refs, ctc_hyps, strict=True))
    out = {"step": p["step"], "weights": "ema" if a.ema else "model",
           "split": f"panel:{a.panel_split or 'all'}:{os.path.basename(a.panel)}" if a.panel
           else a.split,
           "n": len(refs), "max_symbols": a.max_symbols, "batch": a.batch, "order": a.order,
           "forced_advances": forced, "exact_utts": exact, **w,
           "ctc_wer": wc["wer"], "ctc_exact_utts": ctc_exact,
           "chunk": chunk or ecfg.chunk, "left_context": ecfg.left_context,
           "full_context": a.full_context}
    if a.beam:
        out["beam"] = a.beam
        out["beam_wer"] = corpus_wer(refs, beam_hyps)
    if a.panel:
        out["panel"] = panel_record(a.panel, a.panel_split, panel_sha)
    if a.dump:
        with open(a.dump, "w") as fh:
            for n, (u, r, h, c) in enumerate(zip(uids, refs, hyps, ctc_hyps, strict=True)):
                row = {"uid": u, "ref": r, "hyp": h, "ctc_hyp": c, "seconds": secs[n]}
                if a.beam:
                    row["beam_hyp"] = beam_hyps[n]
                fh.write(json.dumps(row) + "\n")
        out["dump"] = a.dump
    print(json.dumps(out))
    for r, h in list(zip(refs, hyps, strict=True))[:a.show]:
        print("REF", r.lower())
        print("HYP", h)


if __name__ == "__main__":
    main()
