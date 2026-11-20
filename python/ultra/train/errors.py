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


class ShapeError(TrainError):
    """Tensor shapes violate a model or loss contract."""
