"""Fail-closed export guards for side-feature checkpoints (Spec 1 §4.4)."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.r1_to_safetensors import SCHEMA, ConvertError, convert, sha256_file  # noqa: E402
from train.asr import PRESETS, ASRModel  # noqa: E402
from train.export import ExportError, export_bundle  # noqa: E402


def _save_ckpt(path, payload: dict) -> str:
    torch.save(payload, path)
    marker = str(path)[: -len(".pt")] + ".sha256"
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"{sha256_file(str(path))}\n")
    return str(path)


@pytest.mark.parametrize(
    ("tag", "enc_patch", "extra_weights"),
    [
        ("side_mode_aux", {"side_mode": "aux"}, {}),
        ("side_mode_env_film", {"side_mode": "env_film"}, {}),
        ("side_film_dim", {"side_film_dim": 16}, {}),
        ("encoder_side_weight", {}, {"encoder.side.proj.weight": torch.zeros(4, 4)}),
        ("side_film_weight", {}, {"encoder.stem.side_film.weight": torch.zeros(4, 4)}),
    ],
)
def test_r1_to_safetensors_and_export_refuse_side_features(tmp_path, tag, enc_patch, extra_weights):
    cfg = PRESETS["TINY"].to_dict()
    model = ASRModel(PRESETS["TINY"]).eval()
    sd = dict(model.state_dict())
    sd.update(extra_weights)
    patched_cfg = dict(cfg, encoder={**cfg["encoder"], **enc_patch})

    ckpt = _save_ckpt(tmp_path / f"{tag}.pt", {"model": sd, "model_config": patched_cfg, "step": 100})

    with pytest.raises(ConvertError, match="side_mode / side_film_dim"):
        convert(ckpt, str(tmp_path / f"st_{tag}"), bundle=None)
    with pytest.raises(ExportError, match="side_mode / side_film_dim"):
        export_bundle(ckpt, str(tmp_path / f"onnx_{tag}"), int8=False)


def test_r1_to_safetensors_allows_clean_checkpoint_without_side_features(tmp_path):
    cfg = PRESETS["TINY"].to_dict()
    model = ASRModel(PRESETS["TINY"]).eval()
    # Explicit empty side_mode and zero side_film_dim must not trip the guard.
    clean_cfg = dict(cfg, encoder={**cfg["encoder"], "side_mode": "", "side_film_dim": 0})
    ckpt = _save_ckpt(tmp_path / "clean.pt", {"model": model.state_dict(), "model_config": clean_cfg, "step": 42})

    arch = convert(ckpt, str(tmp_path / "st_clean"), bundle=None)
    assert arch["schema"] == SCHEMA == "ultra.r1-weights/v1"
    assert (tmp_path / "st_clean" / "weights.safetensors").is_file()
    assert (tmp_path / "st_clean" / "arch.json").is_file()
