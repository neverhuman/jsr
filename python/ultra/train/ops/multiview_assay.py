"""Zero-GPU Multi-View Decoding WER Assay (CPU-only, DEV splits only).

Evaluates free / decode-time accuracy levers on any converged checkpoint without GPU hours:
  1. `baseline`: standard 0 ms streaming beam-K.
  2. `pad20ms`: prepend 20 ms (320 samples) silence pad to counter H12 utterance-start deletions.
  3. `stagger_consensus`: fuse N-best beams from 0 ms + 10 ms + 20 ms phase-shifted views.
  4. `tempo_consensus`: fuse N-best beams from 0.95x + 1.00x + 1.05x tempo-perturbed views.

Refuses held-out `test-*` splits via `train.evaluate.check_split`.
"""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction

import numpy as np
import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.beam import modified_beam_search
from train.data import Featurizer, collate
from train.evaluate import check_split, load_panel
from train.tokenizer import Tokenizer, tokenizer_path
from ultra.wer import corpus_wer, edit_counts, normalise


def pad_pcm(pcm: np.ndarray, prepend_ms: float, sr: int = 16000) -> np.ndarray:
    """Prepend `prepend_ms` of exact zero silence samples to int16 PCM."""
    n = int(round(prepend_ms * sr / 1000.0))
    if n <= 0:
        return pcm
    return np.concatenate([np.zeros(n, dtype=pcm.dtype), pcm])


def tempo_pcm(pcm: np.ndarray, factor: float) -> np.ndarray:
    """Resample int16 PCM by tempo factor `factor` (e.g. 0.95 or 1.05)."""
    if abs(factor - 1.0) < 1e-6:
        return pcm
    from scipy.signal import resample_poly

    r = Fraction(factor).limit_denominator(100)
    y = resample_poly(pcm.astype(np.float32), r.denominator, r.numerator)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def fuse_nbest_views(views_nbest: list[list[tuple[str, float]]]) -> str:
    """Fuse multiple decoded N-best views `[(text, log_prob), ...]` via log-posterior consensus.

    Within each view, candidate scores are softmax-normalized into view posteriors so no view's
    raw frame-count scale dominates; posteriors are averaged across views to pick the consensus
    transcript (ties broken deterministically by view-0 order).
    """
    votes: dict[str, float] = {}
    order: list[str] = []
    for cands in views_nbest:
        if not cands:
            continue
        mx = max(sc for _, sc in cands)
        ws = [float(np.exp(sc - mx)) for _, sc in cands]
        z = sum(ws) or 1.0
        for (txt, _), w in zip(cands, ws, strict=True):
            if txt not in votes:
                order.append(txt)
                votes[txt] = 0.0
            votes[txt] += w / z
    if not order:
        return ""
    return max(order, key=lambda t: votes[t])


def _decode_view(model: ASRModel, feat: Featurizer, tok: Tokenizer, utts, transform,
                 beam: int, batch_size: int, device: str, chunk: int | None):
    exs = [feat.example(u.uid, u.text, transform(u.pcm)) for u in utts]
    idx_order = sorted(range(len(exs)), key=lambda i: exs[i].seconds)
    nbest_out: list[list[tuple[str, float]]] = [[] for _ in utts]
    top1_out: list[str] = ["" for _ in utts]
    for pos in range(0, len(idx_order), batch_size):
        chunk_idx = idx_order[pos:pos + batch_size]
        b = collate([exs[i] for i in chunk_idx]).to(device)
        with torch.no_grad():
            enc, lens = model.encode(b.feats, b.feat_lens, chunk)
            enc_p = model.joiner.enc_proj(enc.float())
            nb: list[list[tuple[list[int], float]]] = []
            best = modified_beam_search(model.predictor, model.joiner, enc_p, lens,
                                        beam=beam, nbest=nb)
        for oi, toks, cands in zip(chunk_idx, best, nb, strict=True):
            top1_out[oi] = tok.decode(toks)
            nbest_out[oi] = [(tok.decode(t), sc) for t, sc in cands]
    return top1_out, nbest_out


def _first_word_deletions(refs: list[str], hyps: list[str]) -> int:
    """Count utterances where the very first reference word is missing from the start of hyp."""
    n = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalise(r), normalise(h)
        if rw and (not hw or hw[0] != rw[0]):
            if edit_counts(rw[:1], hw[:1])[1] > 0 or (hw and rw[0] not in hw[:2]):
                n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--panel-split", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--tempo", action="store_true", help="also run 0.95/1.0/1.05 tempo ensemble")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    a = ap.parse_args(argv)
    if a.panel is None:
        check_split(a.split)
    torch.set_num_threads(a.threads)

    from ultra.libri import load_split

    p = checkpoint.load(a.checkpoint, restore_rng=False)
    mcfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(mcfg)
    model.load_state_dict(p["ema"] if a.ema and p.get("ema") else p["model"])
    model.eval().to(a.device)

    tok_override = (p.get("train_config") or {}).get("tokenizer") or ""
    if tok_override and not os.path.exists(tok_override):
        from train.tokenizer import TOKDIR

        tok_override = os.path.join(TOKDIR, os.path.basename(tok_override))
    tok = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_rnnt - 1))
    tok_c = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_ctc - 1))
    feat = Featurizer(tok, tok_c)
    utts = load_panel(a.panel, a.panel_split)[0] if a.panel else load_split(a.split, a.limit)
    refs = [u.text for u in utts]

    h0, nb0 = _decode_view(model, feat, tok, utts, lambda x: x, a.beam, a.batch, a.device, None)
    h10, nb10 = _decode_view(model, feat, tok, utts, lambda x: pad_pcm(x, 10.0),
                             a.beam, a.batch, a.device, None)
    h20, nb20 = _decode_view(model, feat, tok, utts, lambda x: pad_pcm(x, 20.0),
                             a.beam, a.batch, a.device, None)
    h_stag = [fuse_nbest_views([nb0[i], nb10[i], nb20[i]]) for i in range(len(utts))]

    res = {
        "checkpoint": os.path.abspath(a.checkpoint),
        "step": p["step"],
        "n": len(utts),
        "beam": a.beam,
        "views": {
            "baseline": {**corpus_wer(refs, h0), "start_err": _first_word_deletions(refs, h0)},
            "pad10ms": {**corpus_wer(refs, h10), "start_err": _first_word_deletions(refs, h10)},
            "pad20ms": {**corpus_wer(refs, h20), "start_err": _first_word_deletions(refs, h20)},
            "stagger_consensus": {
                **corpus_wer(refs, h_stag),
                "start_err": _first_word_deletions(refs, h_stag),
            },
        },
    }
    if a.tempo:
        _, nb95 = _decode_view(model, feat, tok, utts, lambda x: tempo_pcm(x, 0.95),
                               a.beam, a.batch, a.device, None)
        _, nb105 = _decode_view(model, feat, tok, utts, lambda x: tempo_pcm(x, 1.05),
                                a.beam, a.batch, a.device, None)
        h_tempo = [fuse_nbest_views([nb0[i], nb95[i], nb105[i]]) for i in range(len(utts))]
        res["views"]["tempo_consensus"] = {
            **corpus_wer(refs, h_tempo),
            "start_err": _first_word_deletions(refs, h_tempo),
        }
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
