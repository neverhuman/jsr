"""The research record (research/H*/) keeps results, never models.

Owner convention (2026-11-23): each wave folder holds its pre-registration, configs and results; checkpoints,
weights, audio and large logs stay on the data drive and are referenced by path and sha256 in a MANIFEST.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WEIGHT_SUFFIXES = {".pt", ".ckpt", ".safetensors", ".onnx", ".ulm", ".bin", ".npz", ".pth"}
MAX_BYTES = 5 * 1024 * 1024


def _tracked_research_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--", "research"], cwd=REPO, check=True, capture_output=True,
                         text=True).stdout.splitlines()
    assert out, "research/ has no tracked files: run this test from a git checkout"
    return [REPO / line for line in out]


def test_research_holds_no_model_weights():
    weights = [str(p.relative_to(REPO)) for p in _tracked_research_files() if p.suffix.lower() in WEIGHT_SUFFIXES]
    assert not weights, f"model weights committed under research/ (keep them on the data drive): {weights}"


def test_research_files_stay_small():
    big = [f"{p.relative_to(REPO)} ({p.stat().st_size} B)" for p in _tracked_research_files()
           if p.is_file() and p.stat().st_size > MAX_BYTES]
    assert not big, f"files over 5 MB under research/ (summaries only): {big}"


def test_weight_suffix_check_catches_a_checkpoint_name():
    assert Path("research/H99_x/results/model.safetensors").suffix in WEIGHT_SUFFIXES
    assert Path("research/H99_x/results/summary.json").suffix not in WEIGHT_SUFFIXES
