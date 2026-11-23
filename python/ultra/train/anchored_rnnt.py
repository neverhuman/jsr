"""H48c: MFA-anchored latency loss, the tilted valid-path RNN-T objective on k2's lattices.

    L_kappa = -log sum_{pi in A(y)} exp{ s(pi) - kappa * C(pi) }

A(y) is k2's exact "regular" RNN-T valid graph: blank (t, u) -> (t + 1, u), reference label
(t, u) -> (t, u + 1), no label at t = T, terminal (T, U). s(pi) is the model's path log-probability
(plus k2's optional uniform delay penalty, which stays exactly as k2 applies it). The late-emission
cost is additive over the label arcs, so it is one extra term on k2's `px` (the label log-probs):

    C(pi) = sum_u c_u(t_u(pi)),   c_u(t) = min(cap, max(0, t - a_u - margin))

where t_u(pi) is the encoder frame at which path pi emits token u and a_u is the token's anchor:
the first 25 Hz encoder frame whose stem window reaches the MFA end of the word holding token u
(`train.mfa_anchors.evidence_frame`). Tokens without an anchor (a_u < 0) cost 0.

Properties (tests/test_train_anchored_rnnt.py):
  * kappa = 0, or no anchors, is k2's own objective bit for bit, with the same batch normalisation;
  * c >= 0 and c = 0 for every t <= a_u + margin, so L_kappa = L_0 + R with R >= 0, and emitting
    before the acoustic evidence is never rewarded: the tilt only moves posterior mass away from
    frames later than the anchor + margin (the uniform k2 delay penalty, by contrast, rewards every
    earlier frame, also those before the word has been heard);
  * the costs are constants (no gradient flows into them); the gradient is k2's occupancy
    gradient under the tilted posterior q(pi) ~ p(pi) exp(-kappa C(pi)), and
    dL/dkappa = E_q[C], d^2L/dkappa^2 = -Var_q(C).

Only training changes; inference and exported graphs are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorLoss:
    """Settings of the anchored tilt (TrainConfig: anchor_kappa / anchor_margin / anchor_cap /
    anchor_simple). kappa is in natural-log units per frame (40 ms) of lateness per token."""
    kappa: float = 0.0
    margin: int = 1
    cap: float = 12.0
    simple: bool = True  # also tilt k2's smoothed (simple-joiner) loss, which picks the band

    def __post_init__(self):
        if self.kappa < 0 or self.margin < 0 or self.cap <= 0:
            raise ValueError(f"bad anchor loss settings {self}")

    @property
    def active(self) -> bool:
        return self.kappa > 0.0


def late_cost(anchor: torch.Tensor, frames: int, margin: int, cap: float,
              dtype=torch.float32) -> torch.Tensor:
    """(B, S, frames) cost c_u(t) = min(cap, max(0, t - a_u - margin)) of emitting token u at
    frame t; rows / tokens with anchor < 0 (no MFA timing, padding) cost 0. No gradient."""
    if anchor.dim() != 2:
        raise ValueError(f"anchor must be (B, S), got {tuple(anchor.shape)}")
    a = anchor.to(torch.long)
    t = torch.arange(frames, device=a.device, dtype=torch.long)
    late = (t[None, None, :] - (a[:, :, None] + int(margin))).to(dtype)
    cost = late.clamp(min=0.0, max=float(cap))
    return torch.where(a[:, :, None] >= 0, cost, torch.zeros((), dtype=dtype, device=a.device))


def _k2_delay_penalty(px: torch.Tensor, boundary: torch.Tensor, delay_penalty: float) -> None:
    """k2's uniform delay penalty for rnnt_type="regular", applied in place exactly as
    k2.rnnt_loss_smoothed / rnnt_loss_pruned do (same ops, same order: bit-identical)."""
    if delay_penalty > 0.0:
        B, S, T0 = px.shape
        if boundary is None:
            offset = torch.tensor((T0 - 1 - 1) / 2, dtype=px.dtype, device=px.device).expand(B, 1, 1)
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = offset.reshape(B, 1, 1) - torch.arange(T0, device=px.device).reshape(1, 1, T0)
        penalty = penalty * delay_penalty
        px += penalty.to(px.dtype)


def tilt_px(px: torch.Tensor, cost: torch.Tensor | None, kappa: float) -> torch.Tensor:
    """px - kappa * cost (out of place). -inf entries (no label at t = T, outside the band) stay
    -inf because the cost is finite everywhere."""
    if cost is None or kappa == 0.0:
        return px
    if cost.shape != px.shape:
        raise ValueError(f"cost {tuple(cost.shape)} must match px {tuple(px.shape)}")
    return px - (kappa * cost.detach()).to(px.dtype)


def rnnt_loss_smoothed_tilted(lm, am, symbols, termination_symbol, lm_only_scale, am_only_scale,
                              boundary, delay_penalty: float = 0.0, cost=None, kappa: float = 0.0,
                              return_grad: bool = True):
    """k2.rnnt_loss_smoothed(reduction="none", rnnt_type="regular") with the tilt on px.
    Returns (per-sequence loss, (px_grad, py_grad)) when return_grad, as k2 does; the occupancy
    gradients are then those of the tilted posterior, so the pruning band follows it."""
    import k2

    px, py = k2.get_rnnt_logprobs_smoothed(lm=lm, am=am, symbols=symbols,
                                           termination_symbol=termination_symbol,
                                           lm_only_scale=lm_only_scale,
                                           am_only_scale=am_only_scale, boundary=boundary,
                                           rnnt_type="regular")
    _k2_delay_penalty(px, boundary, delay_penalty)
    px = tilt_px(px, cost, kappa)
    out = k2.mutual_information_recursion(px=px, py=py, boundary=boundary, return_grad=return_grad)
    if return_grad:
        return -out[0], out[1]
    return -out


def rnnt_loss_pruned_tilted(logits, symbols, ranges, termination_symbol, boundary,
                            delay_penalty: float = 0.0, cost=None, kappa: float = 0.0,
                            return_plain: bool = False):
    """k2.rnnt_loss_pruned(reduction="none", rnnt_type="regular") with the tilt on px; returns
    the per-sequence loss, or (loss, untilted loss without gradient) when return_plain: the
    untilted score reuses the same px / py, so it costs one extra recursion only."""
    import k2

    px, py = k2.get_rnnt_logprobs_pruned(logits=logits, symbols=symbols, ranges=ranges,
                                         termination_symbol=termination_symbol, boundary=boundary,
                                         rnnt_type="regular")
    _k2_delay_penalty(px, boundary, delay_penalty)
    loss = -k2.mutual_information_recursion(px=tilt_px(px, cost, kappa), py=py, boundary=boundary)
    if not return_plain:
        return loss
    with torch.no_grad():
        plain = -k2.mutual_information_recursion(px=px.detach(), py=py.detach(), boundary=boundary)
    return loss, plain


def anchored_pruned_rnnt(model, enc, enc_p, pred_p, tgt, enc_lens, tgt_lens, anchor: torch.Tensor,
                         spec: AnchorLoss, simple_detach: bool = False, zloss: bool = False,
                         delay_penalty: float = 0.0):
    """ASRModel._pruned_rnnt with the anchored tilt: (tilted pruned per-sequence NLL, tilted or
    plain simple loss, band z-loss or None, diagnostics). `anchor` is (B, S) encoder frames, -1
    where a token has no MFA anchor. Same k2 calls, arguments and order as the untilted path."""
    import k2

    from train.transducer import BLANK

    c = model.cfg
    bd = torch.zeros(tgt.shape[0], 4, dtype=torch.int64, device=enc.device)
    bd[:, 2], bd[:, 3] = tgt_lens, enc_lens
    symbols = tgt.long()
    S, T = symbols.shape[1], enc.shape[1]
    if anchor.shape != symbols.shape:
        raise ValueError(f"anchor {tuple(anchor.shape)} must match targets {tuple(symbols.shape)}")
    cost = late_cost(anchor.to(enc.device), T + 1, spec.margin, spec.cap)  # (B, S, T + 1)
    lm = model.simple_lm(pred_p.detach() if simple_detach else pred_p).float()
    am = model.simple_am(enc.detach() if simple_detach else enc).float()
    simple, (px_grad, py_grad) = rnnt_loss_smoothed_tilted(
        lm=lm, am=am, symbols=symbols, termination_symbol=BLANK, lm_only_scale=c.lm_only_scale,
        am_only_scale=c.am_only_scale, boundary=bd, delay_penalty=delay_penalty,
        cost=cost if spec.simple else None, kappa=spec.kappa if spec.simple else 0.0,
        return_grad=True)
    ranges = k2.get_rnnt_prune_ranges(px_grad=px_grad, py_grad=py_grad, boundary=bd,
                                      s_range=min(c.prune_range, S + 1))
    am_p, lm_p = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
    logits = model.joiner.logits(am_p, lm_p).float()
    nll, plain = rnnt_loss_pruned_tilted(logits=logits, symbols=symbols, ranges=ranges,
                                         termination_symbol=BLANK, boundary=bd,
                                         delay_penalty=delay_penalty, cost=cost, kappa=spec.kappa,
                                         return_plain=True)
    with torch.no_grad():  # diagnostics only: plain = the untilted pruned NLL on the same band
        valid = torch.arange(S, device=enc.device)[None] < tgt_lens[:, None]
        diag = {"anchor_risk": (nll.detach() - plain).sum(),  # R = L_kappa - L_0 >= 0 (summed)
                "anchor_tokens": int(((anchor.to(enc.device) >= 0) & valid).sum())}
    zl = None
    if zloss:
        from train.optim.zloss import band_zloss

        zl = band_zloss(logits, enc_lens)
    return nll, simple, zl, diag


def anchor_kwargs(cfg, batch) -> dict:
    """The trainer's loss kwargs for H48c: empty (unchanged call) unless the anchored tilt is on
    (cfg.anchor_kappa > 0) and the batch carries anchors."""
    if cfg.anchor_kappa <= 0 or batch.anchor is None:
        return {}
    return {"anchor": batch.anchor, "anchor_spec": AnchorLoss(cfg.anchor_kappa, cfg.anchor_margin,
                                                              cfg.anchor_cap, cfg.anchor_simple)}


def load_anchors(cfg, tok_rnnt):
    """H48c MFA anchor store for the loader, or None when the anchored latency loss is off."""
    if cfg.anchor_kappa > 0 and not cfg.mfa_anchors:
        raise ValueError("anchor_kappa > 0 needs mfa_anchors (the train MFA side file)")
    if cfg.anchor_kappa > 0 and cfg.rnnt_loss != "pruned":
        raise ValueError("the MFA-anchored latency loss needs rnnt_loss=pruned (k2)")
    if not cfg.mfa_anchors or cfg.anchor_kappa <= 0:
        return None
    import json

    from train.mfa_anchors import MfaAnchors

    store = MfaAnchors(cfg.mfa_anchors, tok_rnnt)
    print(json.dumps({"kind": "mfa_anchors", "path": cfg.mfa_anchors, "sha256": store.sha256,
                      "utterances": len(store), "speeds": store.speeds}), flush=True)
    return store
