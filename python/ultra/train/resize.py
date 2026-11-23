"""Resume-time resizes (H53 Amendment 2 follow-up, for S2+ and cooldowns).

A run restarted with a new learning rate or batch shape (a world-size change under the trunk's LR rule) keeps its
optimizer moments, EMA and data position, but:
- the scheduler's base LR (and each param group's initial_lr) follows the configured lr, scaled per group so
  groups keep their ratios. LambdaLR's state_dict carries base_lrs, so before this a resumed run silently kept
  the checkpoint's LR whatever --set lr said;
- with a new budget_s / max_batch the saved batch index points into the old packing of the current window, so
  that window is finished at the checkpoint's shape (ParquetBatches.carry_shape) and the new shape starts at the
  next window: nothing is skipped and nothing repeats.
"""
from __future__ import annotations


def checkpoint_carry_shape(trainer) -> tuple[float, int] | None:
    carry = getattr(trainer.source, "carry_shape", None)
    if carry is not None and carry[1] == (trainer.pos.epoch, trainer.pos.window):
        return carry[0]
    return None


def apply_resize(trainer, saved_cfg: dict, carried_shape: tuple[float, int] | None = None) -> dict:
    """Apply the configured LR and batch shape after `trainer` restored a checkpoint whose TrainConfig was
    `saved_cfg`. The active carried shape overrides the checkpoint's configured shape."""
    cfg, out = trainer.cfg, {}
    old_lr = float(saved_cfg.get("lr", cfg.lr))
    if abs(old_lr - cfg.lr) > 1e-12 * max(abs(cfg.lr), 1e-12):
        sched = trainer.sched
        # scale per group (keeps group ratios); a zero checkpoint lr has no ratio to keep (v3-reviewer)
        sched.base_lrs = [b * cfg.lr / old_lr if old_lr else cfg.lr for b in sched.base_lrs]
        for g, base, lam in zip(trainer.opt.param_groups, sched.base_lrs, sched.lr_lambdas, strict=True):
            g["initial_lr"] = base
            g["lr"] = base * lam(sched.last_epoch)
        sched._last_lr = [g["lr"] for g in trainer.opt.param_groups]
        out["lr"] = [old_lr, cfg.lr]
    old_shape = carried_shape
    if old_shape is None:
        old_shape = (float(saved_cfg.get("budget_s", cfg.budget_s)), int(saved_cfg.get("max_batch", cfg.max_batch)))
    pos, src = trainer.pos, trainer.source
    if old_shape != (float(cfg.budget_s), int(cfg.max_batch)) and pos.batch > 0 and hasattr(src, "carry_shape"):
        # finish this window at the checkpoint's shape (its packing is what Position.batch indexes); the new
        # shape starts with the next window: nothing skipped, nothing repeated
        src.carry_shape = (old_shape, (pos.epoch, pos.window))
        b, u = window_remainder(src, pos.epoch, pos.window, pos.batch)
        out["carry_shape"] = {"window": [pos.epoch, pos.window], "budget_s": old_shape[0], "max_batch": old_shape[1],
                              "remaining_batches": b, "remaining_utterances": u}
    return out


def window_remainder(source, epoch: int, window: int, batch: int) -> tuple[int, int]:
    """(batches, utterances) of window (epoch, window) after `batch`, in the loader's own header-only packing of
    that window (its carried shape, if any). (0, 0) when the window was consumed."""
    dropped_conf = source.dropped_conf
    try:
        rows = [r for fi, rg in source._window_units(epoch, window) for r in source._read_rows(fi, rg)]
    finally:
        source.dropped_conf = dropped_conf
    rest = source._header_groups(rows, epoch, window)[batch:]
    return len(rest), sum(len(g) for g in rest)
