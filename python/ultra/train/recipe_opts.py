"""Recipe options that change the model config without changing the weights a fork starts from
(H41 arms 16-18): stateless-predictor context, attention left context, SpecAugment strength.

All default to 0 = the preset / built-in value, so a run that does not set them is unchanged.
"""
from __future__ import annotations

import dataclasses

import torch

from train.asr import ModelConfig, SpecAugConfig


def model_overrides(base: ModelConfig, cfg) -> ModelConfig:
    """The preset with the trained attention chunk, left context and predictor context."""
    enc = base.encoder
    if cfg.chunk:  # the exported config must carry the trained attention chunk (H14), not the preset's
        enc = dataclasses.replace(enc, chunk=cfg.chunk)
    if cfg.left_context:
        enc = dataclasses.replace(enc, left_context=cfg.left_context)
    base = dataclasses.replace(base, encoder=enc)
    return dataclasses.replace(base, pred_ctx=cfg.pred_ctx) if cfg.pred_ctx else base


def specaug(cfg) -> SpecAugConfig:
    d = SpecAugConfig()
    return SpecAugConfig(enabled=cfg.specaug,
                         time_masks_per_10s=cfg.specaug_time_masks or d.time_masks_per_10s,
                         time_width=cfg.specaug_time_width or d.time_width)


def check_init_compatible(src: ModelConfig, dst: ModelConfig) -> None:
    """A checkpoint may seed a run that differs only in mask geometry (chunk, left context) or in a
    longer predictor context (widened function-preservingly by widen_state); anything else is a
    different model."""
    from train.errors import CheckpointError

    s = dataclasses.replace(src, qat="none", qat_stem="none", pred_ctx=dst.pred_ctx,
                            encoder=dataclasses.replace(src.encoder, chunk=dst.encoder.chunk,
                                                        left_context=dst.encoder.left_context))
    if s != dataclasses.replace(dst, qat="none", qat_stem="none") or src.pred_ctx > dst.pred_ctx:
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
