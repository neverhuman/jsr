"""Typed trainer errors. Every failure that stops a run names its cause."""
from __future__ import annotations


class TrainError(Exception):
    """Base class for trainer failures."""


class DataError(TrainError):
    """Corpus, manifest or tokenizer input is missing or malformed."""


class CheckpointError(TrainError):
    """A checkpoint is incomplete, corrupt, or does not match the run."""


class NonFiniteLossError(TrainError):
    """A loss or gradient became non-finite; the run is paused with diagnostics."""

    def __init__(self, step: int, detail: str, diag_path: str):
        super().__init__(f"non-finite at step {step}: {detail} (diagnostics: {diag_path})")
        self.step = step
        self.detail = detail
        self.diag_path = diag_path


class DivergenceError(TrainError):
    """Training diverged (loss trend or dev WER); the run is stopped with diagnostics."""

    def __init__(self, step: int, detail: str, diag_path: str):
        super().__init__(f"divergence at step {step}: {detail} (diagnostics: {diag_path})")
        self.step = step
        self.detail = detail
        self.diag_path = diag_path


class ShapeError(TrainError):
    """Tensor shapes violate a model or loss contract."""


class OptimConfigError(TrainError, ValueError):
    """An optimiser, LR-schedule or gradient-clipping option is invalid or misused (train/optim).

    purpose: refuse an H23 optimiser configuration before a run starts, instead of training with
    a silently different optimiser.
    reason: unknown `optimizer` / `grad_clip_mode`, an out-of-range knob (lookahead k/alpha,
    clip_window, lr), an unsupported combination (lookahead + schedulefree_adamw), or a
    Schedule-Free step taken while its averaged weights are swapped in.
    common_fixes: use one of train.optim.OPTIMIZERS / train.optim.clip.MODES; fix the `--set`
    value; leave `lookahead_k=0` with schedulefree_adamw; call the step outside `eval_weights`.
    docs_url: research/H23_optimizer_bakeoff/README.md
    repair_hint: cd python/ultra && python -m pytest -q tests/test_train_optim.py
    """

    def __init__(self, reason: str):
        hint = "rerun: cd python/ultra && python -m pytest -q tests/test_train_optim.py"
        super().__init__(f"{reason} (see research/H23_optimizer_bakeoff/README.md; {hint})")
        self.reason = reason
        self.docs_url = "research/H23_optimizer_bakeoff/README.md"
        self.repair_hint = hint
