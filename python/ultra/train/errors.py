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


class LmEvalError(TrainError, ValueError):
    """An H46 language-model input is inconsistent (train.h46_tune, train.nlm, train/ops/h46_*).

    purpose: stop a "+LM" tuning, scoring, LM-training or export step before it reports a number
    from mismatched inputs (a dev split that does not cover the encoder cache, a decode copy that
    does not answer for the named source checkpoint, engine shards that do not cover dev, a
    neural LM without the runtime exporter).
    reason: the split file lists an utterance twice or not exactly the cache's utterances; a
    derived checkpoint records another source sha256 or non-EMA weights; an engine run decodes
    an utterance twice or misses one; a test split was passed to the shard tool;
    `train.ffnn_lm_export` (runtime-lane's PR #79) is not importable; a token stream for the
    n-gram tools (h46_ngram_counts, h46_u16_to_text) holds a token >= 501 or does not end with a
    sentence break.
    common_fixes: pass eval-lane's `dev_split_v2.json` for the full-dev cache; re-cut the decode
    copy from the derived file of the named checkpoint; rerun the missing engine shards; use dev
    splits only; merge or check out PR #79 before `train.nlm export`; rebuild the stream with
    `train.nlm tokens` (it writes a 0 after every sentence).
    docs_url: research/H46_lm_and_text/README.md
    repair_hint: cd python/ultra && python -m pytest -q tests/test_h46_tune.py
    """

    def __init__(self, reason: str):
        hint = "rerun: cd python/ultra && python -m pytest -q tests/test_h46_tune.py"
        super().__init__(f"{reason} (see research/H46_lm_and_text/README.md; {hint})")
        self.reason = reason
        self.docs_url = "research/H46_lm_and_text/README.md"
        self.repair_hint = hint


class TextTablesError(TrainError, ValueError):
    """An H46 "+text" predictor-table input does not match (train.text_tables).

    purpose: refuse to start a +text run from a state or a text stream that would silently train
    something other than the pre-registered arm (research/H46_lm_and_text, amendment 3).
    reason: `TextPredictor.load_plain` got a state that is not a plain stateless predictor (keys
    other than the new `text.*` ones missing, or unexpected keys); the text stream holds a token
    >= 501 or does not end with a sentence break.
    common_fixes: pass the B0 checkpoint's `predictor.*` state (strip the `predictor.` prefix);
    rebuild the stream with `python -m train.nlm tokens`.
    docs_url: research/H46_lm_and_text/README.md
    repair_hint: cd python/ultra && python -m pytest -q tests/test_text_tables.py
    """

    def __init__(self, reason: str):
        hint = "rerun: cd python/ultra && python -m pytest -q tests/test_text_tables.py"
        super().__init__(f"{reason} (see research/H46_lm_and_text/README.md; {hint})")
        self.reason = reason
        self.docs_url = "research/H46_lm_and_text/README.md"
        self.repair_hint = hint
