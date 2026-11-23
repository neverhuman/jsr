"""Global loss composition across data-parallel ranks (H44 training fixes).

Every objective is normalised by its own global denominator, so the averaged gradient equals one
process on the concatenated batch for the linear terms:
- transducer, CTC and InterCTC: tokens;
- frame KD: teacher frames used;
- the per-frame means (joiner z-loss, CR-CTC consistency KL, activation balancer): valid encoder
  frames, reported by the model as `frames`.
Each rank's term is scaled by world * n_rank / n_total. A rank without support for a term gets
weight 0 but still joins every collective. The balancer is a nonlinear statistic of its batch, so
for it this gives a frame-weighted mean of the per-rank values, not the single-process value.
"""
from __future__ import annotations

import torch

FRAME_MEAN_TERMS = (("zloss", "zloss"), ("cr_ctc", "cr_ctc_weight"), ("balancer", "balancer_weight"))


def global_loss(out: dict, cfg, model_cfg, world: int, device, pruned_scale: float,
                active: bool = True, reduce_sum=None):
    """(loss to back-propagate on this rank, out with global per-token report values)."""
    if reduce_sum is None:
        from train.dist import all_reduce_sum as reduce_sum
    w, a = world, float(active)
    nr, nc = float(out["rnnt_tokens"]), float(out["ctc_tokens"])
    nk, nf = float(out.get("kd_frames", 0)), float(out.get("frames", 0))
    has_mcr = getattr(cfg, "mcr_weight", 0.0) > 0 and "mcr_kl" in out
    has_etm = getattr(cfg, "etm_kd_weight", 0.0) > 0 and "etm_kd" in out
    if has_mcr or has_etm:
        nm, ne = float(out.get("mcr_cells", 0)), float(out.get("etm_tokens", 0))
        tr, tc, tk, tf, tm, te = reduce_sum([nr * a, nc * a, nk * a, nf * a, nm * a, ne * a], device)
        fm = w * a * nm / max(1.0, tm)
        fe = w * a * ne / max(1.0, te)
    else:
        tr, tc, tk, tf = reduce_sum([nr * a, nc * a, nk * a, nf * a], device)
        fm = fe = 0.0
    fr = w * a * max(1.0, nr) / max(1.0, tr)
    fc = w * a * max(1.0, nc) / max(1.0, tc)
    fk = w * a * nk / max(1.0, tk)
    ff = w * a * nf / max(1.0, tf)
    if "rnnt_simple" in out:
        rn = pruned_scale * out["rnnt"] + model_cfg.simple_loss_scale * out["rnnt_simple"]
    else:
        rn = out["rnnt"]
    loss = fr * rn + model_cfg.ctc_weight * fc * out["ctc"]
    if "kd" in out:
        loss = loss + cfg.kd_weight * fk * out["kd"]
    if has_mcr:
        loss = loss + cfg.mcr_weight * fm * out["mcr_kl"]
    if has_etm:
        loss = loss + cfg.etm_kd_weight * fe * out["etm_kd"]
    for name, weight in FRAME_MEAN_TERMS:
        if name in out:
            loss = loss + getattr(cfg, weight) * ff * out[name]
    if "inter_ctc" in out:
        loss = loss + cfg.inter_ctc_weight * fc * out["inter_ctc"]
    if cfg.kd_weight > 0 and "kd" not in out:
        out["kd"] = out["rnnt"].new_zeros(())
    keys = (["rnnt", "ctc"] + (["rnnt_simple"] if "rnnt_simple" in out else [])
            + (["kd"] if cfg.kd_weight > 0 else [])
            + (["mcr_kl"] if has_mcr else []) + (["etm_kd"] if has_etm else []))
    scale = {"rnnt": fr, "ctc": fc, "rnnt_simple": fr, "kd": fk, "mcr_kl": fm, "etm_kd": fe}
    vals = reduce_sum([float(out[k]) * scale[k] / w for k in keys] + [float(loss) / w], device)
    out = dict(out, **{k: torch.tensor(v) for k, v in zip(keys, vals[:-1], strict=True)})
    return loss, dict(out, loss_global=vals[-1])
