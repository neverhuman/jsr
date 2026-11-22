"""Student-own N-best Minimum Word Error Rate (MWER / L_risk) discriminative fine-tuning.

Unlike frame-locked teacher KD (which suffers from (t, u_t) alignment mismatch across architectures),
student N-best MWER optimizes expected word edit distance directly over the student's own beam-K
hypotheses using its exact RNN-T log-likelihood:

    p_i = softmax(s_i / tau),   r_i = W(y_i, y*) - mean_j W(y_j, y*)
    L_mwer = sum_i p_i * r_i
    L_total = (1 - lam) * L_rnnt + lam * L_mwer

Zero inference overhead, zero teacher alignment mismatch, and compatible with every ASRModel preset.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.beam import modified_beam_search
from train.data import Featurizer, ParquetBatches, Position
from train.tokenizer import Tokenizer, tokenizer_path
from train.transducer import StatelessPredictor, joiner_logprobs, rnnt_nll
from ultra.wer import edit_counts, normalise


def _word_errors(ref_text: str, hyp_text: str) -> float:
    return float(sum(edit_counts(normalise(ref_text), normalise(hyp_text))))


def _pred_contexts(model: ASRModel, tgt: torch.Tensor) -> torch.Tensor:
    k_ctx = int(getattr(model.cfg, "pred_ctx", 2))
    return StatelessPredictor.contexts(tgt) if k_ctx == 2 else StatelessPredictor.contexts(tgt, k_ctx)


def _score_candidates(model: ASRModel, enc_p_1: torch.Tensor, enc_len_1: int,
                      cands: list[list[int]]) -> torch.Tensor:
    """Differentiable exact RNN-T log-likelihood for `K` candidate token lists of one utterance."""
    k = len(cands)
    max_u = max((len(c) for c in cands), default=0)
    dev = enc_p_1.device
    k_ctx = int(getattr(model.cfg, "pred_ctx", 2))
    if max_u == 0:
        # Empty hypothesis: sum of blank log-probs along u=0.
        pred_ctx = torch.zeros(1, 1, k_ctx, dtype=torch.long, device=dev)
        pred_p = model.predictor(pred_ctx).float()  # (1, 1, J)
        logits = model.joiner.logits(enc_p_1[:, :enc_len_1, None, :], pred_p[:, None, :, :])
        lp = torch.log_softmax(logits.float(), dim=-1)
        return lp[0, :, 0, 0].sum().expand(k)

    tgt = torch.zeros(k, max_u, dtype=torch.long, device=dev)
    u_lens = torch.zeros(k, dtype=torch.long, device=dev)
    for i, seq in enumerate(cands):
        if seq:
            tgt[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=dev)
        u_lens[i] = len(seq)
    t_lens = torch.full((k,), enc_len_1, dtype=torch.long, device=dev)
    enc_k = enc_p_1[:, :enc_len_1].expand(k, -1, -1).contiguous()
    pred_p = model.predictor(_pred_contexts(model, tgt)).float()
    lpb, lpl = joiner_logprobs(model.joiner, enc_k, pred_p, tgt)
    nll = rnnt_nll(lpb, lpl, t_lens, u_lens)
    return -nll.float()  # log-likelihood s_i


def mwer_loss(model: ASRModel, feats: torch.Tensor, feat_lens: torch.Tensor,
              rnnt_tgt: torch.Tensor, rnnt_lens: torch.Tensor, texts: list[str], tok,
              beam: int = 4, lam: float = 0.5, tau: float = 1.0,
              chunk: int | None = None) -> dict:
    """Compute `(1 - lam) * L_rnnt + lam * L_mwer` over one batch with shared encoder forward."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    enc, enc_lens = model.encode(feats, feat_lens, chunk)
    enc = enc.float()
    with torch.autocast(device_type=enc.device.type, enabled=False):
        enc_p = model.joiner.enc_proj(enc).float()
        # Extract student's own beam-K N-best hypotheses under no_grad.
        nbest: list[list[tuple[list[int], float]]] = []
        with torch.no_grad():
            modified_beam_search(model.predictor, model.joiner, enc_p.detach(), enc_lens,
                                 beam=beam, nbest=nbest)
        # Differentiable standard RNN-T NLL on ground truth.
        pred_gt = model.predictor(_pred_contexts(model, rnnt_tgt)).float()
        lpb, lpl = joiner_logprobs(model.joiner, enc_p, pred_gt, rnnt_tgt)
        gt_nll = rnnt_nll(lpb, lpl, enc_lens, rnnt_lens)
        rnnt = gt_nll.sum() / max(1, int(rnnt_lens.sum()))

        risks: list[torch.Tensor] = []
        mean_wers: list[float] = []
        for b in range(feats.shape[0]):
            gt_seq = rnnt_tgt[b, :int(rnnt_lens[b])].tolist()
            seen: set[tuple[int, ...]] = set()
            cands: list[list[int]] = []
            for seq, _ in nbest[b]:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    cands.append(seq)
            if tuple(gt_seq) not in seen:
                cands.append(gt_seq)
            if len(cands) < 2:
                continue
            errs = torch.tensor(
                [_word_errors(texts[b], tok.decode(c)) for c in cands],
                dtype=torch.float32,
                device=enc.device,
            )
            mean_w = errs.mean()
            rel_err = errs - mean_w
            mean_wers.append(float(mean_w.item()))
            if float(rel_err.abs().max()) == 0.0:
                continue
            scores = _score_candidates(model, enc_p[b:b + 1], int(enc_lens[b]), cands)
            probs = torch.softmax(scores / tau, dim=0)
            risks.append((probs * rel_err).sum())

        mwer = torch.stack(risks).mean() if risks else enc_p.new_zeros(())
        total = (1.0 - lam) * rnnt + lam * mwer
    return {
        "loss": total,
        "rnnt": rnnt,
        "mwer": mwer,
        "active_utts": len(risks),
        "mean_cand_errors": sum(mean_wers) / max(1, len(mean_wers)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    ap.add_argument("--splits", nargs="+", default=["train-clean-100"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--budget-s", type=float, default=120.0)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)

    p = checkpoint.load(a.checkpoint, restore_rng=False)
    mcfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(mcfg)
    model.load_state_dict(p.get("ema") or p["model"])
    model.to(a.device).train()

    tok_override = (p.get("train_config") or {}).get("tokenizer") or ""
    tok_r = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_rnnt - 1))
    tok_c = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_ctc - 1))
    src = ParquetBatches(list(a.splits), Featurizer(tok_r, tok_c), a.seed,
                         budget_s=a.budget_s, max_batch=a.max_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    it = src.iterate(Position())
    for step in range(1, a.steps + 1):
        _, b = next(it)
        b = b.to(a.device)
        opt.zero_grad(set_to_none=True)
        out = mwer_loss(model, b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.texts, tok_r,
                        beam=a.beam, lam=a.lam, tau=a.tau)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if step == 1 or step % 25 == 0 or step == a.steps:
            print(json.dumps({"step": step, "loss": round(float(out["loss"]), 4),
                              "rnnt": round(float(out["rnnt"]), 4),
                              "mwer": round(float(out["mwer"]), 4),
                              "active": out["active_utts"]}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    payload = dict(p, model=model.state_dict(), ema=None, mwer_steps=a.steps)
    torch.save(payload, a.out)
    with open(a.out[:-3] + ".sha256", "w") as fh:  # the completion marker checkpoint.load requires
        fh.write(checkpoint._sha256(a.out) + "\n")
    print(json.dumps({"saved": a.out, "steps": a.steps}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
