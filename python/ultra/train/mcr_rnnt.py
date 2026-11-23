"""Wave 2B (MCR-RNNT): Full-context chunk=0 arm + symmetric lattice mode consistency (T3).

Computes symmetric full-vocab KL consistency over the shared pruned band (B, T, prune_range)
selected by the full-context (chunk=0) arm between streaming (c_s > 0) and full-context (c_l = 0)
joiner log-probs, with exact cell-count denominator for DDP global normalization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from train.transducer import BLANK, full_lattice_logprobs, rnnt_nll


def symmetric_band_kl(
    logits_stream: torch.Tensor,
    logits_full: torch.Tensor,
    enc_lens: torch.Tensor,
    *,
    stop_grad: bool = True,
) -> tuple[torch.Tensor, int]:
    """Symmetric full-vocab KL over shared band logits (B, T, R, V), masked by enc_lens.

    Returns ``(mean_kl, n_cells)`` where ``n_cells = sum(enc_lens) * R`` is the exact valid
    lattice-cell denominator for single-process and DDP global normalization.
    """
    if logits_stream.shape != logits_full.shape or logits_stream.dim() != 4:
        raise ValueError(f"expected matching 4D band logits (B,T,R,V), got "
                         f"{tuple(logits_stream.shape)} vs {tuple(logits_full.shape)}")
    B, T, R, _ = logits_stream.shape
    lp_s = F.log_softmax(logits_stream.float(), dim=-1)
    lp_f = F.log_softmax(logits_full.float(), dim=-1)
    p_s, p_f = lp_s.exp(), lp_f.exp()
    if stop_grad:
        kl_sf = (p_f.detach() * (lp_f.detach() - lp_s)).sum(-1)
        kl_fs = (p_s.detach() * (lp_s.detach() - lp_f)).sum(-1)
    else:
        kl_sf = (p_f * (lp_f - lp_s)).sum(-1)
        kl_fs = (p_s * (lp_s - lp_f)).sum(-1)
    kl = 0.5 * (kl_sf + kl_fs)  # (B, T, R)
    valid = (torch.arange(T, device=lp_s.device)[None] < enc_lens[:, None])  # (B, T)
    n_cells = int(valid.sum()) * R
    if n_cells == 0:
        return kl.sum() * 0.0, 0
    return (kl * valid[:, :, None]).sum() / n_cells, n_cells


def select_shared_band(
    model,
    enc_full: torch.Tensor,
    pred_p: torch.Tensor,
    tgt: torch.Tensor,
    enc_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
    *,
    simple_detach: bool = False,
    delay_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the shared k2 pruning band (B, T, R) from the full-context (chunk=0) arm."""
    import k2

    c = model.cfg
    dp = {"delay_penalty": delay_penalty} if delay_penalty else {}
    bd = torch.zeros(tgt.shape[0], 4, dtype=torch.int64, device=enc_full.device)
    bd[:, 2], bd[:, 3] = tgt_lens, enc_lens
    symbols = tgt.long()
    simple, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
        lm=model.simple_lm(pred_p.detach() if simple_detach else pred_p).float(),
        am=model.simple_am(enc_full.detach() if simple_detach else enc_full).float(),
        symbols=symbols,
        termination_symbol=BLANK,
        lm_only_scale=c.lm_only_scale,
        am_only_scale=c.am_only_scale,
        boundary=bd,
        rnnt_type="regular",
        reduction="none",
        return_grad=True,
        **dp,
    )
    ranges = k2.get_rnnt_prune_ranges(
        px_grad=px_grad, py_grad=py_grad, boundary=bd,
        s_range=min(c.prune_range, symbols.shape[1] + 1),
    )
    return simple, ranges, bd


def mcr_pruned_rnnt(
    model,
    enc_p_stream: torch.Tensor,
    enc_full: torch.Tensor,
    enc_p_full: torch.Tensor,
    pred_p: torch.Tensor,
    tgt: torch.Tensor,
    enc_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
    *,
    simple_detach: bool = False,
    zloss: bool = False,
    delay_penalty: float = 0.0,
    stop_grad: bool = True,
) -> dict[str, torch.Tensor | int | None]:
    """Pruned MCR-RNNT: shared band from chunk=0 arm, both arms' NLL + symmetric band KL."""
    import k2

    dp = {"delay_penalty": delay_penalty} if delay_penalty else {}
    symbols = tgt.long()
    simple, ranges, bd = select_shared_band(
        model, enc_full, pred_p, symbols, enc_lens, tgt_lens,
        simple_detach=simple_detach, delay_penalty=delay_penalty,
    )
    am_full, lm_shared = k2.do_rnnt_pruning(am=enc_p_full, lm=pred_p, ranges=ranges)
    am_stream, _ = k2.do_rnnt_pruning(am=enc_p_stream, lm=pred_p, ranges=ranges)
    logits_full = model.joiner.logits(am_full, lm_shared).float()
    logits_stream = model.joiner.logits(am_stream, lm_shared).float()
    nll_full = k2.rnnt_loss_pruned(
        logits=logits_full, symbols=symbols, ranges=ranges, termination_symbol=BLANK,
        boundary=bd, rnnt_type="regular", reduction="none", **dp,
    )
    nll_stream = k2.rnnt_loss_pruned(
        logits=logits_stream, symbols=symbols, ranges=ranges, termination_symbol=BLANK,
        boundary=bd, rnnt_type="regular", reduction="none", **dp,
    )
    mcr_kl, mcr_cells = symmetric_band_kl(logits_stream, logits_full, enc_lens, stop_grad=stop_grad)
    zl = None
    if zloss:
        from train.optim.zloss import band_zloss

        zl = 0.5 * (band_zloss(logits_stream, enc_lens) + band_zloss(logits_full, enc_lens))
    return {"nll_stream": nll_stream, "nll_full": nll_full, "simple": simple,
            "mcr_kl": mcr_kl, "mcr_cells": mcr_cells, "zloss": zl, "ranges": ranges,
            "logits_stream": logits_stream, "logits_full": logits_full}


def mcr_exact_rnnt(
    model,
    enc_p_stream: torch.Tensor,
    enc_p_full: torch.Tensor,
    pred_p: torch.Tensor,
    tgt: torch.Tensor,
    enc_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
    *,
    stop_grad: bool = True,
) -> dict[str, torch.Tensor | int | None]:
    """Exact full-lattice MCR-RNNT (for rnnt_loss='exact' and small-fixture tests)."""
    logits_stream = model.joiner.logits(enc_p_stream[:, :, None, :], pred_p[:, None, :, :]).float()
    logits_full = model.joiner.logits(enc_p_full[:, :, None, :], pred_p[:, None, :, :]).float()
    lpb_s, lpl_s = full_lattice_logprobs(model.joiner, enc_p_stream, pred_p, tgt)
    lpb_f, lpl_f = full_lattice_logprobs(model.joiner, enc_p_full, pred_p, tgt)
    nll_stream = rnnt_nll(lpb_s, lpl_s, enc_lens, tgt_lens)
    nll_full = rnnt_nll(lpb_f, lpl_f, enc_lens, tgt_lens)
    mcr_kl, mcr_cells = symmetric_band_kl(logits_stream, logits_full, enc_lens, stop_grad=stop_grad)
    return {"nll_stream": nll_stream, "nll_full": nll_full, "simple": None,
            "mcr_kl": mcr_kl, "mcr_cells": mcr_cells, "zloss": None, "ranges": None,
            "logits_stream": logits_stream, "logits_full": logits_full}


def mcr_forward(
    model,
    x: torch.Tensor,
    feat_lens: torch.Tensor,
    enc_p_stream: torch.Tensor,
    pred_p: torch.Tensor,
    rnnt_tgt: torch.Tensor,
    enc_lens: torch.Tensor,
    rnnt_lens: torch.Tensor,
    *,
    ctx: torch.Tensor | None = None,
    enc_lens0: torch.Tensor | None = None,
    simple_detach: bool = False,
    zloss: bool = False,
    delay_penalty: float = 0.0,
    stop_grad: bool = True,
) -> dict[str, torch.Tensor | int | None]:
    """Run the paired chunk=0 full-context arm and compute shared-band MCR-RNNT outputs."""
    enc_full, full_lens = model.encoder(x, feat_lens, chunk=0)
    enc_full = enc_full.float()
    if ctx is not None:
        from train.stream_ctx import shift_rows

        enc_full, full_lens = shift_rows(enc_full, enc_lens0 if enc_lens0 is not None else full_lens, ctx)
    enc_p_full = model.joiner.enc_proj(enc_full).float()
    with torch.autocast(device_type=enc_full.device.type, enabled=False):
        if model.cfg.rnnt_loss == "pruned":
            return mcr_pruned_rnnt(
                model, enc_p_stream, enc_full, enc_p_full, pred_p, rnnt_tgt, enc_lens, rnnt_lens,
                simple_detach=simple_detach, zloss=zloss, delay_penalty=delay_penalty, stop_grad=stop_grad,
            )
        return mcr_exact_rnnt(
            model, enc_p_stream, enc_p_full, pred_p, rnnt_tgt, enc_lens, rnnt_lens, stop_grad=stop_grad,
        )
