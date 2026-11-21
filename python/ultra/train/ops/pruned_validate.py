"""H13 prep: validate k2's pruned RNN-T loss against our exact loss (CPU, no training).

  python train/ops/pruned_validate.py --run-dir <dense run> --checkpoint <model.pt> [--n 16]

1. Exactness, synthetic: random enc/pred projections and targets with mixed lengths.
   - k2.rnnt_loss (full lattice, rnnt_type="regular") must equal our rnnt_nll(full_lattice_logprobs).
   - k2.rnnt_loss_pruned with s_range = max U + 1 (no pruning) must also equal it.
   - Checks both values and gradients w.r.t. enc_p, pred_p, W_out and b_out.
2. Approximation, real: D2 encoder/predictor outputs on dev-clean utterances with reference
   targets. Prune ranges come from the exact lattice's occupation gradients, which is the best a
   trained simple joiner could give, so this is a lower bound on pruning error. Reports the loss
   gap and the gradient cosine at s_range 3, 5, 8, plus the band mass left outside.
3. Speed: forward+backward time of our chunked exact loss vs pruned s_range=5 at a realistic
   batch shape (CPU, indicative only; the GPU ratio is what matters for H13).
Writes one JSON object to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import k2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from train.transducer import (BLANK, Joiner, full_lattice_logprobs,  # noqa: E402
                              joiner_logprobs, rnnt_nll)


def boundary(t_lens, u_lens):
    b = torch.zeros(len(t_lens), 4, dtype=torch.int64)
    b[:, 2], b[:, 3] = u_lens, t_lens
    return b


def ours(joiner, enc_p, pred_p, tgt, t_lens, u_lens, chunked=False):
    f = joiner_logprobs if chunked else full_lattice_logprobs
    lpb, lpl = f(joiner, enc_p, pred_p, tgt)
    return rnnt_nll(lpb, lpl, t_lens, u_lens)


def k2_full(joiner, enc_p, pred_p, tgt, t_lens, u_lens):
    logits = joiner.logits(enc_p[:, :, None, :], pred_p[:, None, :, :])
    return k2.rnnt_loss(logits=logits, symbols=tgt, termination_symbol=BLANK,
                        boundary=boundary(t_lens, u_lens), rnnt_type="regular", reduction="none")


def occupation_ranges(joiner, enc_p, pred_p, tgt, t_lens, u_lens, s_range):
    with torch.no_grad():
        logits = joiner.logits(enc_p[:, :, None, :], pred_p[:, None, :, :])
        bd = boundary(t_lens, u_lens)
        px, py = k2.get_rnnt_logprobs_joint(logits=logits, symbols=tgt, termination_symbol=BLANK,
                                            boundary=bd, rnnt_type="regular")
        _, (px_grad, py_grad) = k2.mutual_information_recursion(px, py, bd, return_grad=True)
    return k2.get_rnnt_prune_ranges(px_grad=px_grad, py_grad=py_grad, boundary=bd,
                                    s_range=s_range), px_grad, py_grad


def k2_pruned(joiner, enc_p, pred_p, tgt, t_lens, u_lens, ranges):
    am, lm = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
    logits = joiner.out(torch.tanh(am + lm)) if joiner.act == "tanh" else joiner.logits(am, lm)
    return k2.rnnt_loss_pruned(logits=logits, symbols=tgt, ranges=ranges,
                               termination_symbol=BLANK, boundary=boundary(t_lens, u_lens),
                               rnnt_type="regular", reduction="none")


def grads(loss, tensors):
    return torch.autograd.grad(loss.sum(), tensors)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(),
                                                       dim=0))


def exactness(seed=0):
    torch.manual_seed(seed)
    B, T, U, J, V = 4, 23, 7, 16, 13
    joiner = Joiner(32, J, V).double()
    enc_p = joiner.enc_proj(torch.randn(B, T, 32, dtype=torch.float64)).detach().requires_grad_()
    pred_p = torch.randn(B, U + 1, J, dtype=torch.float64, requires_grad=True)
    tgt = torch.randint(1, V, (B, U))
    t_lens, u_lens = torch.tensor([23, 19, 11, 23]), torch.tensor([7, 5, 3, 0])
    params = [enc_p, pred_p, joiner.out.weight, joiner.out.bias]
    ref = ours(joiner, enc_p, pred_p, tgt, t_lens, u_lens)
    g_ref = grads(ref, params)
    full = k2_full(joiner, enc_p, pred_p, tgt, t_lens, u_lens)
    g_full = grads(full, params)
    ranges, _, _ = occupation_ranges(joiner, enc_p, pred_p, tgt, t_lens, u_lens, U + 1)
    pr = k2_pruned(joiner, enc_p, pred_p, tgt, t_lens, u_lens, ranges)
    g_pr = grads(pr, params)
    return {"ref": ref.tolist(),
            "k2_full_max_abs_diff": float((full - ref).abs().max()),
            "k2_full_grad_max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(g_full, g_ref)),
            "k2_pruned_fullrange_max_abs_diff": float((pr - ref).abs().max()),
            "k2_pruned_fullrange_grad_max_abs_diff": max(float((a - b).abs().max())
                                                         for a, b in zip(g_pr, g_ref))}


def real_batch(run_dir, ckpt, n):
    from train import checkpoint
    from train.asr import ASRModel, ModelConfig
    from train.data import Featurizer, collate
    from train.tokenizer import Tokenizer, tokenizer_path
    from train.transducer import StatelessPredictor
    from ultra.libri import load_split

    p = checkpoint.load(ckpt, restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval()
    tok, tok_c = Tokenizer(tokenizer_path(4096)), Tokenizer(tokenizer_path(1024))
    feat = Featurizer(tok, tok_c)
    exs = [feat.example(u.uid, u.text, u.pcm) for u in load_split("dev-clean", n)]
    b = collate(exs)
    with torch.no_grad():
        enc, t_lens = model.encode(b.feats, b.feat_lens)
        enc_p = model.joiner.enc_proj(enc.float())
        pred_p = model.predictor(StatelessPredictor.contexts(b.rnnt)).float()
    return model.joiner, enc_p, pred_p, b.rnnt, t_lens, b.rnnt_lens


def approximation(joiner, enc_p, pred_p, tgt, t_lens, u_lens):
    enc_p = enc_p.clone().requires_grad_()
    pred_p = pred_p.clone().requires_grad_()
    params = [enc_p, pred_p, joiner.out.weight, joiner.out.bias]
    ref = ours(joiner, enc_p, pred_p, tgt, t_lens, u_lens, chunked=True)
    g_ref = grads(ref, params)
    out = {"utts": len(t_lens), "mean_T": float(t_lens.float().mean()),
           "mean_U": float(u_lens.float().mean()), "ref_loss_sum": float(ref.sum()), "s_range": {}}
    for s in (3, 5, 8):
        ranges, px_grad, py_grad = occupation_ranges(joiner, enc_p, pred_p, tgt, t_lens, u_lens, s)
        pr = k2_pruned(joiner, enc_p, pred_p, tgt, t_lens, u_lens, ranges)
        g = grads(pr, params)
        out["s_range"][s] = {
            "loss_sum": float(pr.sum()),
            "rel_gap": float((pr.sum() - ref.sum()) / ref.sum()),
            "max_utt_rel_gap": float(((pr - ref) / ref).abs().max()),
            "grad_cos": {n: round(cos(a, b), 6) for n, a, b in
                         zip(("enc_p", "pred_p", "W_out", "b_out"), g, g_ref)}}
    return out


def speed(joiner, B=8, T=250, U=60, reps=2):
    torch.manual_seed(1)
    J = joiner.out.weight.shape[1]
    enc_p = torch.randn(B, T, J, requires_grad=True)
    pred_p = torch.randn(B, U + 1, J, requires_grad=True)
    tgt = torch.randint(1, joiner.out.weight.shape[0], (B, U))
    t_lens, u_lens = torch.full((B,), T), torch.full((B,), U)
    params = [enc_p, pred_p, joiner.out.weight, joiner.out.bias]

    def timed(f):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            grads(f(), params)
            best = min(best, time.perf_counter() - t0)
        return best
    exact = timed(lambda: ours(joiner, enc_p, pred_p, tgt, t_lens, u_lens, chunked=True))
    ranges, _, _ = occupation_ranges(joiner, enc_p, pred_p, tgt, t_lens, u_lens, 5)
    pruned = timed(lambda: k2_pruned(joiner, enc_p, pred_p, tgt, t_lens, u_lens, ranges))
    return {"shape": [B, T, U, J, joiner.out.weight.shape[0]], "exact_s": round(exact, 3),
            "pruned_s5_s": round(pruned, 3), "speedup": round(exact / pruned, 2),
            "note": "pruned excludes computing the prune ranges (a cheap simple-joiner loss in training)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.n < 1:
        ap.error("--n must be >= 1")
    torch.set_num_threads(a.threads)
    res = {"k2": getattr(k2, "__version__", "unknown"), "torch": torch.__version__,
           "exactness": exactness()}
    joiner, *batch = real_batch(a.run_dir, a.checkpoint, a.n)
    res["approximation_dev_clean"] = approximation(joiner, *batch)
    res["speed_cpu"] = speed(joiner)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
