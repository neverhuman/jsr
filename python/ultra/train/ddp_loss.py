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
    has_mcr = getattr(cfg, "mcr_weight", 0.0) > 0
    has_etm = getattr(cfg, "etm_kd_weight", 0.0) > 0
    has_inter = cfg.inter_ctc_weight > 0
    if has_mcr and "mcr_kl" not in out:
        out["mcr_kl"] = out["rnnt"] * 0.0
    if has_etm and "etm_kd" not in out:
        out["etm_kd"] = out["rnnt"] * 0.0
    extra = {}
    if has_mcr:
        extra["mcr"] = float(out.get("mcr_cells", 0))
    if has_etm:
        extra["etm"] = float(out.get("etm_tokens", 0))
    if has_inter:
        extra["inter"] = float(out.get("inter_ctc_tokens", 0))
    tr, tc, tk, tf, *totals = reduce_sum([x * a for x in (nr, nc, nk, nf, *extra.values())], device)
    factors = {k: w * a * n / max(1.0, total)
               for (k, n), total in zip(extra.items(), totals, strict=True)}
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
        loss = loss + cfg.mcr_weight * factors["mcr"] * out["mcr_kl"]
    if has_etm:
        loss = loss + cfg.etm_kd_weight * factors["etm"] * out["etm_kd"]
    for name, weight in FRAME_MEAN_TERMS:
        if name in out:
            loss = loss + getattr(cfg, weight) * ff * out[name]
    if "inter_ctc" in out:
        loss = loss + cfg.inter_ctc_weight * factors["inter"] * out["inter_ctc"]
    if cfg.kd_weight > 0 and "kd" not in out:
        out["kd"] = out["rnnt"].new_zeros(())
    keys = (["rnnt", "ctc"] + (["rnnt_simple"] if "rnnt_simple" in out else [])
            + (["kd"] if cfg.kd_weight > 0 else [])
            + (["mcr_kl"] if has_mcr else []) + (["etm_kd"] if has_etm else [])
            + (["inter_ctc"] if has_inter and getattr(cfg, "seal_data_identity", False) else []))
    scale = {"rnnt": fr, "ctc": fc, "rnnt_simple": fr, "kd": fk,
             "mcr_kl": factors.get("mcr"), "etm_kd": factors.get("etm"), "inter_ctc": factors.get("inter")}
    vals = reduce_sum([float(out[k]) * scale[k] / w for k in keys] + [float(loss) / w], device)
    out = dict(out, **{k: torch.tensor(v) for k, v in zip(keys, vals[:-1], strict=True)})
    return loss, dict(out, loss_global=vals[-1], has_supervision=bool(tr or tc))
