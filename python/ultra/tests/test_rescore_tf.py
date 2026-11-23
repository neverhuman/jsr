"""Torch-free tests for H55 Wave 1B-2 full-sentence L2R/R2L Transformer rescorer."""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pytest

ULTRA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ULTRA_ROOT not in sys.path:
    sys.path.insert(0, ULTRA_ROOT)

from train.rescore_tf import (  # noqa: E402
    DEFAULT_GPU_DUTY_CAP,
    MAX_RESCORER_VRAM_FRAC,
    PRESETS,
    VOCAB,
    build_sentence_positions_and_mask,
    compute_duty_sleep_s,
    should_yield_to_cotenant,
)


def test_rescore_tf_is_torch_free_at_import_and_narrow_exceptions():
    src_path = os.path.join(ULTRA_ROOT, "train", "rescore_tf.py")
    src = open(src_path, encoding="utf-8").read()
    assert len(src.splitlines()) <= 300
    tree = ast.parse(src, filename=src_path)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "torch" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "torch"
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None
            assert not (isinstance(node.type, ast.Name) and node.type.id == "Exception")


def test_presets_vram_and_duty_guards():
    assert VOCAB == 501
    assert MAX_RESCORER_VRAM_FRAC <= 0.45
    assert DEFAULT_GPU_DUTY_CAP <= 0.75
    assert PRESETS["3.5m"]["ctx"] == 128
    assert PRESETS["3.5m"]["n_layers"] == 4
    assert PRESETS["3.5m"]["d_model"] == 256
    assert abs(compute_duty_sleep_s(0.06, 0.75) - 0.02) < 1e-9
    assert compute_duty_sleep_s(0.06, 1.0) == 0.0
    for bad in (0.0, -0.5, 1.1, float("nan")):
        with pytest.raises(ValueError):
            compute_duty_sleep_s(0.05, bad)


def test_sentence_positions_and_causal_isolation_mask():
    x = np.array([[9, 8, 0, 5, 6, 0, 7, 4]], dtype=np.int64)
    pos, mask, valid = build_sentence_positions_and_mask(x)
    assert pos[0].tolist() == [0, 1, 0, 1, 2, 0, 1, 2]
    assert valid[0].tolist() == [False, False, True, True, True, True, True, True]
    assert mask[0, 5, 5] and not mask[0, 5, 4] and not mask[0, 6, 3]
    assert mask[0, 4, 2] and mask[0, 4, 3] and mask[0, 4, 4]
    assert not mask[0, 4, 1] and not mask[0, 4, 5]


def test_r2l_reversal_preserves_sentence_boundaries():
    raw = np.array([[0, 10, 20, 30, 0, 40, 50, 0]], dtype=np.int64)
    rev = raw[:, ::-1].copy()
    x_rev, y_rev = rev[:, :-1], rev[:, 1:]
    pos, mask, valid = build_sentence_positions_and_mask(x_rev)
    assert x_rev[0].tolist() == [0, 50, 40, 0, 30, 20, 10]
    assert y_rev[0].tolist() == [50, 40, 0, 30, 20, 10, 0]
    assert pos[0].tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert valid[0].tolist() == [True, True, True, True, True, True, True]
    assert not mask[0, 4, 1] and mask[0, 4, 3] and mask[0, 4, 4]


def test_cotenant_yield_rule(monkeypatch):
    import subprocess
    import types

    def fake_run(pid_out: str, gpu_out: str):
        def _run(cmd, **_kw):
            out = pid_out if "--query-compute-apps=pid" in cmd else gpu_out
            return types.SimpleNamespace(returncode=0, stdout=out)
        return _run

    # Idle cotenant (0% util, 22 GB free) -> do NOT yield
    monkeypatch.setattr(subprocess, "run", fake_run("100\n200\n", "0, 22500\n"))
    assert should_yield_to_cotenant(my_pid=100) is False
    # Active cotenant (>20% util) -> yield
    monkeypatch.setattr(subprocess, "run", fake_run("100\n200\n", "45, 22500\n"))
    assert should_yield_to_cotenant(my_pid=100) is True
    # Low free VRAM (<10.3 GiB) with cotenant -> yield
    monkeypatch.setattr(subprocess, "run", fake_run("100\n200\n", "5, 9000\n"))
    assert should_yield_to_cotenant(my_pid=100) is True
    # Solo (only own PID) -> never yield even at high util
    monkeypatch.setattr(subprocess, "run", fake_run("100\n", "80, 9000\n"))
    assert should_yield_to_cotenant(my_pid=100) is False
