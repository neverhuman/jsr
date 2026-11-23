"""Recipe options that change the model config without changing the weights a fork starts from
(H41 arms 16-18): stateless-predictor context, attention left context, SpecAugment strength.

All default to 0 = the preset / built-in value, so a run that does not set them is unchanged.
"""
from __future__ import annotations

import dataclasses

import torch

from train.asr import ModelConfig, SpecAugConfig


def model_overrides(base: ModelConfig, cfg) -> ModelConfig:
    """The preset with trained chunk, left context, stochastic depth, predictor ctx and text tables."""
    enc = base.encoder
    if cfg.chunk:  # the exported config must carry the trained attention chunk (H14), not the preset's
        enc = dataclasses.replace(enc, chunk=cfg.chunk)
    if cfg.left_context:
        enc = dataclasses.replace(enc, left_context=cfg.left_context)
    if getattr(cfg, "layer_drop", 0.0):
        enc = dataclasses.replace(enc, layer_drop=float(cfg.layer_drop))
    base = dataclasses.replace(base, encoder=enc)
    if cfg.pred_ctx:
        base = dataclasses.replace(base, pred_ctx=cfg.pred_ctx)
    if getattr(cfg, "text_tables_dim", 0):
        base = dataclasses.replace(base, text_tables_dim=int(cfg.text_tables_dim))
    return base


def specaug(cfg) -> SpecAugConfig:
    d = SpecAugConfig()
    return SpecAugConfig(enabled=cfg.specaug,
                         time_masks_per_10s=cfg.specaug_time_masks or d.time_masks_per_10s,
                         time_width=cfg.specaug_time_width or d.time_width,
                         time_warp=getattr(cfg, "specaug_time_warp", 0))


def add_wave2_args(ap) -> None:
    """Optional Wave 2 CLI flags (all default-off; mirror TrainConfig --set keys)."""
    ap.add_argument("--mcr-weight", type=float, default=None)
    ap.add_argument("--etm-kd-weight", type=float, default=None)
    ap.add_argument("--etm-kd-mode", default=None, choices=("exact", "bound"))
    ap.add_argument("--loss-trunc-q", type=float, default=None)
    ap.add_argument("--loss-trunc-warmup", type=int, default=None)
    ap.add_argument("--loss-trunc-mode", default=None, choices=("drop", "cauchy", "tukey"))
    ap.add_argument("--specaug-time-warp", type=int, default=None)
    ap.add_argument("--bpe-dropout-p", type=float, default=None)
    ap.add_argument("--bpe-dropout-alpha", type=float, default=None)


def apply_wave2_args(cfg, args) -> None:
    for name in ("mcr_weight", "etm_kd_weight", "etm_kd_mode", "loss_trunc_q", "loss_trunc_warmup",
                 "loss_trunc_mode", "specaug_time_warp", "bpe_dropout_p", "bpe_dropout_alpha"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)


def make_loss_truncator(cfg):
    if getattr(cfg, "loss_trunc_q", 0.0) <= 0.0:
        return None
    from train.loss_trunc import LossTruncator

    return LossTruncator(cfg.loss_trunc_q, cfg.loss_trunc_warmup, cfg.loss_trunc_mode)


def wave2_loss_kwargs(cfg, truncator, step: int) -> dict:
    kw: dict = {}
    if getattr(cfg, "mcr_weight", 0.0) > 0.0:
        kw["mcr_weight"] = cfg.mcr_weight
    if truncator is not None:
        kw["loss_trunc"], kw["loss_trunc_step"] = truncator, step
    if getattr(cfg, "etm_kd_weight", 0.0) > 0.0:
        kw["etm_kd_weight"], kw["etm_kd_mode"] = cfg.etm_kd_weight, cfg.etm_kd_mode
    return kw


def check_init_compatible(src: ModelConfig, dst: ModelConfig) -> None:
    """A checkpoint may seed a run that differs only in mask geometry (chunk, left context),
    eval-identity stochastic depth (`layer_drop`), longer predictor context (`widen_state`), or
    zero-initialised Factor D `text_tables` (`src.text_tables_dim == 0 -> dst > 0`)."""
    from train.errors import CheckpointError

    tables_ok = (src.text_tables_dim == dst.text_tables_dim) or (src.text_tables_dim == 0 and dst.text_tables_dim > 0)
    s = dataclasses.replace(src, qat="none", qat_stem="none", pred_ctx=dst.pred_ctx,
                            text_tables_dim=dst.text_tables_dim,
                            encoder=dataclasses.replace(src.encoder, chunk=dst.encoder.chunk,
                                                        left_context=dst.encoder.left_context,
                                                        layer_drop=dst.encoder.layer_drop))
    if not tables_ok or s != dataclasses.replace(dst, qat="none", qat_stem="none") or src.pred_ctx > dst.pred_ctx:
        raise CheckpointError("--init-from checkpoint has a different model config")


def widen_state(state: dict, src: ModelConfig, dst: ModelConfig) -> dict:
    """Predictor projection for a longer context: zero columns for the older positions (the input
    is oldest .. newest), so the widened model computes exactly what the source did."""
    if src.pred_ctx == dst.pred_ctx:
        return state
    w = state["predictor.proj.weight"]
    emb = w.shape[1] // src.pred_ctx
    pad = w.new_zeros(w.shape[0], (dst.pred_ctx - src.pred_ctx) * emb)
    return dict(state, **{"predictor.proj.weight": torch.cat([pad, w], dim=1)})
