"""Wave 2C (ETM-KD): Timing-free occupancy-pooled non-blank knowledge distillation.

Computes detached student non-blank posterior occupancy:
  w_u(t) = occ_label(t, u) / sum_{t'} occ_label(t', u)
from k2.mutual_information_recursion(..., return_grad=True) (zero gradient into blank),
and evaluates both bound (mean_u sum_t w_u(t) KL(q_u || pi_{t,u})) and exact
(mean_u KL(q_u || sum_t w_u(t) pi_{t,u})) non-blank KL losses with token-level validity masking.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from train.transducer import BLANK, rnnt_nll


def _lcs_matches(hyp: list[int], ref: list[int]) -> list[tuple[int, int]]:
    """Exact LCS alignment returning sorted (hyp_idx, ref_idx) pairs of matching tokens."""
    nh, nr = len(hyp), len(ref)
    if nh == 0 or nr == 0:
        return []
    dp = [[0] * (nr + 1) for _ in range(nh + 1)]
    for i in range(nh - 1, -1, -1):
        hi, row, nxt = hyp[i], dp[i], dp[i + 1]
        for j in range(nr - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if hi == ref[j] else (nxt[j] if nxt[j] >= row[j + 1] else row[j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < nh and j < nr:
        if hyp[i] == ref[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def extract_teacher_token_targets(
    emitted: np.ndarray | list[int],
    teacher_probs: torch.Tensor,
    ref_tokens: list[int] | np.ndarray,
    *,
    shift_r0_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract timing-free token-indexed non-blank teacher distributions with token-level masking.

    Args:
      emitted: (Tt,) teacher emitted token id per frame (0 = blank, >0 = emitted token).
      teacher_probs: (Tt, V) full-vocab or (Tt, V-1) non-blank probabilities per teacher frame.
      ref_tokens: (S,) student reference token ids (blank-shifted, >= 1).
      shift_r0_ids: if True, converts R0 non-blank piece id p -> student id p + 1 before aligning.

    Returns ``(q_tokens (S, V_nb), valid_mask (S,))``. Completely independent of teacher frame
    timestamps (shifting emitted + teacher_probs by t -> t + delta produces identical outputs)
    and retains every matching token even when the teacher makes errors elsewhere in the utterance.
    """
    em = np.asarray(emitted, dtype=np.int64)
    ref = [int(x) for x in ref_tokens]
    S = len(ref)
    # Strip blank column if full-vocab (caller passes V >= 2 with col 0 = blank); renormalize non-blank.
    probs = teacher_probs.float()
    emit_frames = [int(t) for t in np.flatnonzero(em > 0)]
    hyp = [int(em[t]) + (1 if shift_r0_ids else 0) for t in emit_frames]
    V_nb = probs.shape[-1]
    q_out = probs.new_zeros((S, V_nb))
    valid = torch.zeros(S, dtype=torch.bool, device=probs.device)
    for hi, ri in _lcs_matches(hyp, ref):
        p = probs[emit_frames[hi]].clamp(min=0.0)
        s = p.sum()
        if float(s) > 0.0:
            q_out[ri] = p / s
            valid[ri] = True
    return q_out, valid


def _normalize_occupancy(occ: torch.Tensor) -> torch.Tensor:
    occ = occ.detach().float().clamp(min=0.0)
    denom = occ.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0.0, occ / denom.clamp(min=1e-12), torch.zeros_like(occ))


@torch.no_grad()
def label_occupancy_k2(px: torch.Tensor, py: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """Detached student non-blank posterior occupancy w_u(t) of shape (B, S, T) via k2."""
    import k2

    T = px.shape[2] - 1
    _, (px_grad, _) = k2.mutual_information_recursion(
        px=px.detach().float(), py=py.detach().float(), boundary=boundary, return_grad=True,
    )
    return _normalize_occupancy(px_grad[:, :, :T])


@torch.no_grad()
def label_occupancy_pruned(
    logits: torch.Tensor,
    symbols: torch.Tensor,
    ranges: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    """Detached student non-blank posterior occupancy w_u(t) (B, S, T) from k2 pruned logits."""
    import k2

    px, py = k2.get_rnnt_logprobs_pruned(
        logits=logits.detach().float(),
        symbols=symbols.long(),
        ranges=ranges,
        termination_symbol=BLANK,
        boundary=boundary,
        rnnt_type="regular",
    )
    return label_occupancy_k2(px, py, boundary)


def label_occupancy_exact(
    lpb: torch.Tensor,
    lpl: torch.Tensor,
    t_lens: torch.Tensor,
    u_lens: torch.Tensor,
) -> torch.Tensor:
    """Detached student non-blank posterior occupancy w_u(t) (B, S, T) for exact full lattice."""
    with torch.enable_grad():
        lpb_d = lpb.detach().double()
        lpl_d = lpl.detach().double().requires_grad_(True)
        nll = rnnt_nll(lpb_d, lpl_d, t_lens, u_lens).sum()
        (g_lpl,) = torch.autograd.grad(nll, lpl_d)
    # -d(nll)/d(lpl[b, t, u]) = P(emit token u at frame t | audio, y) >= 0
    occ = (-g_lpl).detach().float().transpose(1, 2)  # (B, S, T)
    B, S, T = occ.shape
    u_ok = torch.arange(S, device=occ.device)[None] < u_lens[:, None]
    t_ok = torch.arange(T, device=occ.device)[None] < t_lens[:, None]
    return _normalize_occupancy(occ * (u_ok[:, :, None] & t_ok[:, None, :]))


def _prep_teacher_nonblank(teacher_probs: torch.Tensor, v_nb: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = teacher_probs.detach().float()
    if q.shape[-1] == v_nb + 1:
        q = q[..., 1:]  # strip blank column so teacher distribution lives strictly on non-blank tokens
    if q.shape[-1] != v_nb:
        raise ValueError(f"teacher_probs last dim {q.shape[-1]} != non-blank vocab {v_nb}")
    q = q.clamp(min=0.0)
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    log_q = torch.where(q > 0.0, torch.log(q.clamp(min=1e-12)), torch.zeros_like(q))
    return q, log_q


def etm_kd_loss(
    joiner_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    occupancy: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str = "exact",
) -> tuple[torch.Tensor, int]:
    """Timing-free occupancy-pooled non-blank ETM-KD loss on full lattice logits (B, T, >=S, V).

    Slices ``joiner_logits[..., 1:]`` before softmax so blank (index 0) receives identically zero
    gradient. Supports ``mode='bound'`` (mean_u sum_t w_u(t) KL(q_u || pi_{t,u})) and
    ``mode='exact'`` (mean_u KL(q_u || sum_t w_u(t) pi_{t,u})). Returns ``(loss, n_valid_tokens)``.
    """
    if mode not in ("bound", "exact"):
        raise ValueError(f"unknown etm_kd mode {mode!r}; expected 'bound' or 'exact'")
    B, S, T = occupancy.shape
    z_nb = joiner_logits[:, :T, :S, 1:].float()  # (B, T, S, V-1); index 0 (blank) excluded!
    v_nb = z_nb.shape[-1]
    q, log_q = _prep_teacher_nonblank(teacher_probs, v_nb)
    w = _normalize_occupancy(occupancy)  # detached (B, S, T)
    mask = valid_mask.to(torch.bool) & (w.sum(dim=-1) > 0.0)
    n_tok = int(mask.sum())
    if n_tok == 0:
        return z_nb.sum() * 0.0, 0
    if mode == "bound":
        log_pi = F.log_softmax(z_nb, dim=-1).permute(0, 2, 1, 3)  # (B, S, T, V-1)
        kl_ut = (q[:, :, None, :] * (log_q[:, :, None, :] - log_pi)).sum(-1)  # (B, S, T)
        tok_kl = (w * kl_ut).sum(-1)  # (B, S)
    else:
        pi = F.softmax(z_nb, dim=-1).permute(0, 2, 1, 3)  # (B, S, T, V-1)
        pi_bar = torch.einsum("bst,bstv->bsv", w, pi)  # (B, S, V-1)
        tok_kl = (q * (log_q - torch.log(pi_bar.clamp(min=1e-12)))).sum(-1)  # (B, S)
    return (tok_kl * mask).sum() / n_tok, n_tok


def etm_kd_pruned_loss(
    pruned_logits: torch.Tensor,
    ranges: torch.Tensor,
    teacher_probs: torch.Tensor,
    occupancy: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str = "exact",
) -> tuple[torch.Tensor, int]:
    """Memory-bounded ETM-KD directly on k2 pruned band logits (B, T, R, V).

    Zero gradient flows into blank (index 0) or detached occupancy weights ``w``.
    """
    if mode not in ("bound", "exact"):
        raise ValueError(f"unknown etm_kd mode {mode!r}; expected 'bound' or 'exact'")
    B, T, R, V = pruned_logits.shape
    S = teacher_probs.shape[1]
    z_nb = pruned_logits[..., 1:].float()  # (B, T, R, V-1); index 0 (blank) excluded!
    v_nb = V - 1
    q, log_q = _prep_teacher_nonblank(teacher_probs, v_nb)
    w_full = _normalize_occupancy(occupancy)  # (B, S, T)
    u_idx = ranges.long()  # (B, T, R) in [0, S]
    in_s = u_idx < S
    u_safe = u_idx.clamp(max=max(0, S - 1))
    b_idx = torch.arange(B, device=z_nb.device)[:, None, None]
    t_idx = torch.arange(T, device=z_nb.device)[None, :, None]
    w_cell = torch.where(in_s, w_full[b_idx, u_safe, t_idx], torch.zeros_like(u_safe, dtype=torch.float32))
    # Renormalize w_cell per (b, u) over the pruned band so weights sum to 1 inside the band.
    u_flat = u_safe.reshape(B, T * R)
    w_flat = w_cell.reshape(B, T * R)
    w_sum = torch.zeros(B, S, device=z_nb.device, dtype=torch.float32).scatter_add_(1, u_flat, w_flat)
    denom = torch.gather(w_sum, 1, u_flat).clamp(min=1e-12)
    w_norm = torch.where(in_s.reshape(B, T * R) & (denom > 0.0), w_flat / denom, torch.zeros_like(w_flat))
    mask = valid_mask.to(torch.bool) & (w_sum > 0.0)
    n_tok = int(mask.sum())
    if n_tok == 0:
        return z_nb.sum() * 0.0, 0
    if mode == "bound":
        log_pi = F.log_softmax(z_nb, dim=-1).reshape(B, T * R, v_nb)
        q_c = torch.gather(q, 1, u_flat[..., None].expand(-1, -1, v_nb))
        lq_c = torch.gather(log_q, 1, u_flat[..., None].expand(-1, -1, v_nb))
        kl_c = (q_c * (lq_c - log_pi)).sum(-1)  # (B, T*R)
        tok_kl = torch.zeros(B, S, device=z_nb.device, dtype=torch.float32).scatter_add_(1, u_flat, w_norm * kl_c)
    else:
        pi = F.softmax(z_nb, dim=-1).reshape(B, T * R, v_nb)
        pi_bar = torch.zeros(B, S, v_nb, device=z_nb.device, dtype=torch.float32)
        pi_bar.scatter_add_(1, u_flat[..., None].expand(-1, -1, v_nb), w_norm[..., None] * pi)
        tok_kl = (q * (log_q - torch.log(pi_bar.clamp(min=1e-12)))).sum(-1)
    return (tok_kl * mask).sum() / n_tok, n_tok
