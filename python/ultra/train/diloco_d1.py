"""D1 island boundary gating & exact round/anchor resume validation for DiLoCo (`trainer.py`)."""
from __future__ import annotations

import hmac
from typing import Mapping

from train.errors import CheckpointError


def make_diloco_state(
    trainer, round_idx: int, anchor_sha256: str, data_rank: int, data_world: int, every: int
) -> dict:
    """Build durable checkpoint metadata pinning exact `(round, anchor_sha256, data_rank, data_world, step)`."""
    return {
        "round": int(round_idx),
        "anchor_sha256": str(anchor_sha256),
        "data_rank": int(data_rank),
        "data_world": int(data_world),
        "diloco_every": int(every),
        "step": int(trainer.step),
    }


def validate_diloco_resume(trainer, payload: Mapping) -> None:
    """Refuse mid-round, unaligned, or shard-mismatched checkpoints when DiLoCo is active."""
    state = payload.get("diloco")
    if not isinstance(state, Mapping):
        raise CheckpointError("DiLoCo resume requires a post-anchor checkpoint with 'diloco' metadata")
    every = int(trainer.cfg.diloco_every)
    if int(state.get("diloco_every", -1)) != every:
        raise CheckpointError(
            f"DiLoCo resume diloco_every mismatch: ckpt={state.get('diloco_every')} != cfg={every}"
        )
    if int(state.get("step", -1)) != int(trainer.step):
        raise CheckpointError(
            f"DiLoCo resume mid-round checkpoint refused: ckpt step={trainer.step} != anchor step={state.get('step')}"
        )
    if int(state.get("data_rank", -1)) != int(trainer.data_rank) or int(state.get("data_world", -1)) != int(trainer.data_world):
        raise CheckpointError(
            f"DiLoCo resume shard mismatch: ckpt=({state.get('data_rank')},{state.get('data_world')}) "
            f"!= cfg=({trainer.data_rank},{trainer.data_world})"
        )
    if int(state.get("round", -1)) < 0 or not str(state.get("anchor_sha256", "")):
        raise CheckpointError("DiLoCo resume checkpoint missing valid round or anchor_sha256")


def verify_hub_resume_alignment(trainer, hub_round: int, hub_anchor_sha256: str) -> None:
    """Ensure a resumed DiLoCo island's checkpoint matches the hub's exact `(round, anchor_sha256)`."""
    saved = getattr(trainer, "_diloco_state", None)
    if not isinstance(saved, Mapping):
        return
    saved_round = int(saved.get("round", -1))
    saved_sha = str(saved.get("anchor_sha256", ""))
    if int(hub_round) != saved_round or not saved_sha or not hmac.compare_digest(saved_sha, str(hub_anchor_sha256)):
        raise CheckpointError(
            f"DiLoCo resume hub mismatch: ckpt=(round={saved_round}, sha={saved_sha[:12]}) "
            f"!= hub=(round={hub_round}, sha={str(hub_anchor_sha256)[:12]})"
        )


class DiLoCoBoundaryGate:
    """Defers eval, weight snapshots, and checkpoints strictly to post-anchor outer-round boundaries."""

    def __init__(self) -> None:
        self._pending_eval = False
        self._pending_epoch_end = False
        self._pending_weights = False
        self._pending_ckpt = False

    def allow_eval(self, step: int, eval_every: int, epoch_end: bool, synced_now: bool) -> tuple[bool, bool]:
        if epoch_end:
            self._pending_eval = True
            self._pending_epoch_end = True
        if eval_every and step % eval_every == 0:
            self._pending_eval = True
        if not synced_now or not self._pending_eval:
            return False, False
        ep = self._pending_epoch_end
        self._pending_eval = self._pending_epoch_end = False
        return True, ep

    def allow_weights(self, step: int, weights_every: int, synced_now: bool) -> bool:
        if weights_every and step % weights_every == 0:
            self._pending_weights = True
        if not synced_now or not self._pending_weights:
            return False
        self._pending_weights = False
        return True

    def allow_ckpt(self, due_now: bool, synced_now: bool) -> bool:
        if due_now:
            self._pending_ckpt = True
        if not synced_now or not self._pending_ckpt:
            return False
        self._pending_ckpt = False
        return True
