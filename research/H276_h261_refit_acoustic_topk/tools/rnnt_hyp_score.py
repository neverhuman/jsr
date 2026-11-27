#!/usr/bin/env python
"""Full RNN-T forward-algorithm log-likelihood log P(tokens | audio) for every n-best hypothesis.

Reuses python/ultra/train (ASRModel, LogMel, joiner_logprobs, rnnt_nll) by import; nothing is
reimplemented. Per hypothesis the score is -rnnt_nll(...), summed over all alignments (float64 lattice
recursion), not the best path and not length-normalised.

Chunk modes (--modes, comma list):
  full   Encoder chunk=0: unrestricted attention over the whole utterance (non-streaming context).
  cN     Encoder chunk=N (25 Hz encoder frames; c32 = 1.28 s). This is the same block mask the model
         is trained with (`--chunk-modes 16,32,32,32` for H255): frame t sees keys in
         [floor(t/N)*N - left_context, floor(t/N)*N + N); left_context=128. The streaming engine
         decodes chunk-by-chunk with caches that unroll exactly this mask (train/infer.py docstring),
         so c32 here is the training-time equivalent of the chunk-32 decode, evaluated over the whole
         lattice (not a streaming beam). Audio is expected to already carry the 320 ms zero tail.
Token convention: nbest tokens are sentencepiece id + 1 (Tokenizer.encode), blank = 0; they are fed
to the predictor/joiner unchanged. Checked against bpe.model decode on the first rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra")
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.features import LogMel, pcm_to_float  # noqa: E402
from train.transducer import StatelessPredictor, joiner_logprobs, rnnt_nll  # noqa: E402

MAXK = 16


def load_model(path, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(p["model_config"])
    m = ASRModel(cfg)
    sd = p["ema"] if "ema" in p else p["model"]
    m.load_state_dict(sd)
    return m.to(device).eval(), cfg


def mode_chunk(name):
    if name in ("full", "offline"):
        return 0
    if name.startswith("c"):
        return int(name[1:])
    raise ValueError(f"bad mode {name}")


@torch.no_grad()
def score_utt(model, enc_p, T, hyps, device, budget=1 << 24):
    """Same value as score_utt_ref, but the joiner is evaluated once per distinct (predictor context,
    next-token) cell instead of once per hypothesis cell: the stateless predictor (context 2) makes
    the joiner output a function of (t, last two tokens), and n-best hypotheses share most contexts.
    The alignment sum itself is the repo's rnnt_nll on the assembled blank/label log-prob lattices."""
    if model.joiner.act != "tanh":
        raise NotImplementedError("joiner_act other than tanh")
    K = len(hyps)
    lens = torch.tensor([len(h) for h in hyps], device=device)
    U = int(lens.max())
    C = model.predictor.context_len
    tgt = torch.zeros(K, U, dtype=torch.long, device=device)
    for i, h in enumerate(hyps):
        if h:
            tgt[i, :len(h)] = torch.tensor(h, device=device)
    ctx = StatelessPredictor.contexts(tgt, C)  # (K, U+1, C)
    V = model.joiner.out.weight.shape[0]
    key = torch.zeros(K, U + 1, dtype=torch.long, device=device)
    for c in range(C):
        key = key * V + ctx[..., c]
    ukey, inv = torch.unique(key.flatten(), return_inverse=True)  # distinct contexts
    inv = inv.view(K, U + 1)
    uctx = torch.stack([(ukey // V ** (C - 1 - c)) % V for c in range(C)], dim=-1)
    pred_u = model.predictor(uctx).float()  # (Nu, J)
    Nu = pred_u.shape[0]
    valid = torch.arange(U, device=device)[None] < lens[:, None]  # (K, U) cells with a label
    pkey = torch.where(valid, inv[:, :U] * V + tgt, torch.zeros_like(tgt))
    upk, pinv = torch.unique(pkey.flatten(), return_inverse=True)  # distinct (context, token)
    pinv = pinv.view(K, U)
    p_nu, p_tok = upk // V, upk % V
    w, b = model.joiner.out.weight, model.joiner.out.bias
    tc = max(1, min(T, budget // max(1, Nu * V)))
    lpb_u = enc_p.new_empty(T, Nu)
    lpl_p = enc_p.new_empty(T, upk.shape[0])
    for t0 in range(0, T, tc):
        t1 = min(T, t0 + tc)
        h = torch.tanh(enc_p[0, t0:t1, None, :] + pred_u[None])
        z = torch.matmul(h, w.t()) + b  # (n, Nu, V)
        lse = torch.logsumexp(z, dim=-1)
        lpb_u[t0:t1] = z[..., 0] - lse
        lpl_p[t0:t1] = z[:, p_nu, p_tok] - lse[:, p_nu]
    lpb = lpb_u[:, inv].permute(1, 0, 2).contiguous()  # (K, T, U+1)
    lpl = lpl_p[:, pinv].permute(1, 0, 2).contiguous()  # (K, T, U)
    t_lens = torch.full((K,), T, dtype=torch.long, device=device)
    return (-rnnt_nll(lpb, lpl, t_lens, lens)).cpu().numpy()


@torch.no_grad()
def score_utt_ref(model, enc_p, T, hyps, device):
    """Reference: enc_p (1,T,J) joiner-projected encoder output; hyps list of token lists -> (K,) log-lik."""
    K = len(hyps)
    lens = torch.tensor([len(h) for h in hyps], device=device)
    U = int(lens.max())
    tgt = torch.zeros(K, U, dtype=torch.long, device=device)
    for i, h in enumerate(hyps):
        if h:
            tgt[i, :len(h)] = torch.tensor(h, device=device)
    pred_p = model.predictor(StatelessPredictor.contexts(tgt, model.predictor.context_len)).float()
    lpb, lpl = joiner_logprobs(model.joiner, enc_p.expand(K, -1, -1).contiguous(), pred_p, tgt,
                               budget_elems=1 << 24)
    t_lens = torch.full((K,), T, dtype=torch.long, device=device)
    return (-rnnt_nll(lpb, lpl, t_lens, lens)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nbest", required=True)
    ap.add_argument("--pcm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", default="full,c32")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4, help="utterances per encoder batch (length-sorted)")
    ap.add_argument("--bpe", default="", help="optional bpe.model to verify token convention")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    modes = a.modes.split(",")

    rows = []
    with open(a.nbest) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if a.limit and len(rows) >= a.limit:
                break
    z = np.load(a.pcm)
    pcm, offs, uids = z["pcm"], z["offsets"], [str(u) for u in z["uids"]]
    idx = {u: i for i, u in enumerate(uids)}
    if a.bpe:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=a.bpe)
        bad = sum(sp.decode([t - 1 for t in r["nbest"][0][0]]).upper() != r["nbest"][0][2].upper()
                  for r in rows[:20])
        print(f"token-convention check (sp id+1): {bad}/20 top-1 decode mismatches", flush=True)

    model, cfg = load_model(a.ckpt, dev)
    logmel = LogMel().to(dev)
    owner, pos, ntok = [], [], []
    for r, row in enumerate(rows):
        for k in range(min(MAXK, len(row["nbest"]))):
            owner.append(r); pos.append(k); ntok.append(len(row["nbest"][k][0]))
    base = {}
    off = 0
    for r, row in enumerate(rows):
        base[r] = off
        off += min(MAXK, len(row["nbest"]))
    res = {m: np.full(len(owner), np.nan, dtype=np.float64) for m in modes}

    feats, nfr = [], []
    order = []
    for r, row in enumerate(rows):
        i = idx[str(row["uid"])]
        w = pcm_to_float(pcm[offs[i]:offs[i + 1]])
        order.append((int(offs[i + 1] - offs[i]), r, w))
    order.sort(key=lambda x: x[0])
    t0 = time.time()
    done = 0
    for b0 in range(0, len(order), a.batch):
        grp = order[b0:b0 + a.batch]
        fs = [logmel(w.to(dev)) for _, _, w in grp]
        fl = torch.tensor([f.shape[0] for f in fs], device=dev)
        x = torch.zeros(len(fs), int(fl.max()), fs[0].shape[1], device=dev)
        for i, f in enumerate(fs):
            x[i, :f.shape[0]] = f
        for m in modes:
            with torch.no_grad():
                enc, el = model.encode(x, fl, mode_chunk(m))
                enc_p = model.joiner.enc_proj(enc.float())
            for i, (_, r, _) in enumerate(grp):
                T = int(el[i])
                hy = [h[0] for h in rows[r]["nbest"][:MAXK]]
                sc = score_utt(model, enc_p[i:i + 1, :T], T, hy, dev)
                res[m][base[r]:base[r] + len(hy)] = sc
        done += len(grp)
        if (b0 // a.batch) % 20 == 0:
            print(f"{done}/{len(order)} utts {time.time() - t0:.0f}s", flush=True)
    out = {"owner": np.array(owner, dtype=np.int32), "pos": np.array(pos, dtype=np.int16),
           "n_tok": np.array(ntok, dtype=np.int32)}
    for m in modes:
        out["am_" + m] = res[m].astype(np.float32)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(owner)} hyps, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
