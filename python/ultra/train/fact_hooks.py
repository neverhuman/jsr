"""H57 FACT training hooks (Factors B / D / E): isolated from `trainer.py` so trainer stays compact.

All features are strictly default-OFF and perform zero RNG draws / zero tensor ops when disabled.
"""
from __future__ import annotations

import torch

from train.errors import CheckpointError, TextTablesError
from train.fact_aug import FactAugConfig
from train.recipe_opts import check_init_compatible, widen_state


def fact_aug_from_cfg(cfg) -> FactAugConfig | None:
    """Construct validated `FactAugConfig` from `TrainConfig`, or `None` when all Factor E flags are 0."""
    aug = FactAugConfig(
        gain_p=float(getattr(cfg, "gain_aug_p", 0.0)),
        gain_min_db=float(getattr(cfg, "gain_min_db", -35.0)),
        gain_max_db=float(getattr(cfg, "gain_max_db", 6.0)),
        noise_p=float(getattr(cfg, "noise_aug_p", 0.0)),
        noise_dir=str(getattr(cfg, "noise_dir", "")),
        noise_snr_min_db=float(getattr(cfg, "noise_snr_min_db", 5.0)),
        noise_snr_max_db=float(getattr(cfg, "noise_snr_max_db", 25.0)),
        rir_p=float(getattr(cfg, "rir_aug_p", 0.0)),
        rir_dir=str(getattr(cfg, "rir_dir", "")),
    )
    if not aug.active:
        return None
    aug.validate(getattr(cfg, "splits", ()))
    return aug


def load_init_state(model, ckpt_state: dict, src_cfg, dst_cfg) -> None:
    """Warm-start `model` from `--init-from`, handling `pred_ctx` widening and zero-init `text_tables`."""
    check_init_compatible(src_cfg, dst_cfg)
    widened = widen_state(ckpt_state, src_cfg, dst_cfg)
    if getattr(src_cfg, "text_tables_dim", 0) == 0 and getattr(dst_cfg, "text_tables_dim", 0) > 0:
        missing, unexpected = model.load_state_dict(widened, strict=False)
        if unexpected or any(not k.startswith("predictor.text.") for k in missing):
            raise CheckpointError(
                f"--init-from into text_tables model had unexpected/non-table keys: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return
    model.load_state_dict(widened, strict=True)


class FactTextHelper:
    """Factor D (`+text`) per-step text-only loss helper over `train.text_tables`.

    Inactive (`build` returns `None`) whenever `cfg.text_loss_weight <= 0`. When active, computes
    `e_null` (mean `joiner.enc_proj` over valid frames) in eval mode without disturbing the sampler
    cursor, draws `(ctx, tgt)` pairs from `TextStream`, and routes gradients strictly to
    `predictor.text.*` (joiner and dense predictor stay detached).
    """

    def __init__(self, model, cfg, data_rank: int = 0):
        from train.text_tables import TextPredictor, TextStream

        if not isinstance(model.predictor, TextPredictor):
            raise TextTablesError(
                "text_loss_weight > 0 requires text_tables_dim > 0 (TextPredictor)"
            )
        if not cfg.text_stream_path:
            raise TextTablesError("text_loss_weight > 0 requires text_stream_path")
        self.weight = float(cfg.text_loss_weight)
        self.batch_size = max(1, int(getattr(cfg, "text_batch_size", 2048)))
        self.enull_batches = max(1, int(getattr(cfg, "text_enull_batches", 4)))
        self.stream = TextStream(
            cfg.text_stream_path,
            width=model.predictor.context_len,
            seed=int(cfg.seed) + 97 * int(data_rank),
        )
        from train.data_identity import file_sha256

        self.stream_sha256 = file_sha256(cfg.text_stream_path)
        self._enull_sum: torch.Tensor | None = None
        self._enull_frames: int = 0
        self._enull_seen: int = 0
        self.e_null: torch.Tensor | None = None

    def restart_state(self) -> dict:
        return {
            "settings": [self.stream_sha256, self.weight, self.batch_size, self.enull_batches],
            "rng": self.stream.gen.get_state(),
            "sum": self._enull_sum.cpu() if self._enull_sum is not None else None,
            "frames": self._enull_frames, "seen": self._enull_seen,
        }

    def restore_restart_state(self, state: dict, device) -> None:
        if state["settings"] != [self.stream_sha256, self.weight, self.batch_size, self.enull_batches]:
            raise CheckpointError("objective restart text stream or settings changed")
        self.stream.gen.set_state(state["rng"])
        self._enull_frames, self._enull_seen = state["frames"], state["seen"]
        self._enull_sum = state["sum"].to(device) if state["sum"] is not None else None
        self.e_null = (self._enull_sum / max(self._enull_frames, 1)
                       if self._enull_sum is not None else None)

    @classmethod
    def build(cls, model, cfg, data_rank: int = 0) -> FactTextHelper | None:
        if float(getattr(cfg, "text_loss_weight", 0.0)) <= 0.0:
            return None
        return cls(model, cfg, data_rank=data_rank)

    @torch.no_grad()
    def _update_enull(self, model, batch) -> torch.Tensor:
        if self._enull_seen < self.enull_batches:
            was = model.training
            model.eval()
            enc, lens = model.encode(batch.feats, batch.feat_lens, None)
            e = model.joiner.enc_proj(enc.float())
            mask = torch.arange(e.shape[1], device=e.device)[None] < lens[:, None]
            s = e[mask].sum(0)
            n = int(mask.sum())
            self._enull_sum = s if self._enull_sum is None else self._enull_sum + s
            self._enull_frames += n
            self._enull_seen += 1
            self.e_null = self._enull_sum / max(self._enull_frames, 1)
            model.train(was)
        assert self.e_null is not None
        return self.e_null

    def compute_loss(self, model, batch, device) -> tuple[torch.Tensor, float]:
        """Return `(weighted_text_loss, raw_text_loss_float)` for the current training step."""
        from train.text_tables import text_loss

        e_null = self._update_enull(model, batch).to(device)
        ctx, tgt = self.stream.batch(self.batch_size)
        if tgt.numel() == 0:
            zero = e_null.new_zeros(())
            return zero, 0.0
        ctx = ctx.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            tl = text_loss(model.predictor, model.joiner, ctx, tgt, e_null)
        return self.weight * tl, float(tl.detach())

    def step(self, model, batch, device, loss: torch.Tensor, out: dict, world: int) -> torch.Tensor:
        """Compute weighted text loss, update `out` (including DDP `loss_global`), return new loss."""
        tl_w, tl_raw = self.compute_loss(model, batch, device)
        out["text_loss"] = tl_raw
        if "loss_global" in out:
            from train.dist import all_reduce_sum

            out["loss_global"] = float(out["loss_global"]) + all_reduce_sum([float(tl_w.detach())], device)[0] / world
        return loss + tl_w
