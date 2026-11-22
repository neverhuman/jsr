"""Prepare a verified weights-only seed for a fresh trainer run.

Both run configs must identify the actual SentencePiece models. An exact-to-pruned
transfer preserves all recognizer tensors and initializes the training-only simple
joiner from a linearization of the trained joiner around zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.errors import CheckpointError


def prepare(source: Path, source_config: Path, target_config: Path, out: Path) -> Path:
    source_bytes = source_config.read_bytes()
    target_bytes = target_config.read_bytes()
    source_run = json.loads(source_bytes)
    target_run = json.loads(target_bytes)
    tokenizers = source_run.get("tokenizers", {})
    if (
        tokenizers != target_run.get("tokenizers")
        or any(
            not isinstance(tokenizers.get(head), str)
            or len(tokenizers[head]) != 64
            or any(c not in "0123456789abcdef" for c in tokenizers[head])
            for head in ("rnnt", "ctc")
        )
    ):
        raise CheckpointError("both RNN-T and CTC tokenizer digests must match")

    payload = checkpoint.load(str(source), restore_rng=False)
    origin = ModelConfig.from_dict(payload["model_config"])
    target = ModelConfig.from_dict(target_run["model"])
    if origin != ModelConfig.from_dict(source_run["model"]):
        raise CheckpointError("source run config does not match the checkpoint")
    if payload.get("tokenizer_sha256") != tokenizers["rnnt"]:
        raise CheckpointError("source RNN-T tokenizer does not match the checkpoint")
    if payload.get("tokenizers", tokenizers) != tokenizers:
        raise CheckpointError("source tokenizer metadata does not match the checkpoint")

    weights = dict(payload["model"])
    method = "copy"
    if origin != target:
        exact_to_pruned = (
            origin.rnnt_loss == "exact"
            and target.rnnt_loss == "pruned"
            and replace(
                origin, rnnt_loss=target.rnnt_loss, prune_range=target.prune_range,
                simple_loss_scale=target.simple_loss_scale, lm_only_scale=target.lm_only_scale,
                am_only_scale=target.am_only_scale,
            ) == target
            and origin.joiner_act == "tanh"
        )
        if not exact_to_pruned:
            raise CheckpointError("incompatible model configuration for weight transfer")
        # W*tanh(A*x + a + y) + b ≈ (W*A)*x + W*a + W*y + b.
        projection = weights["joiner.enc_proj.weight"]
        bias = weights["joiner.enc_proj.bias"]
        output = weights["joiner.out.weight"]
        half_bias = weights["joiner.out.bias"] / 2
        weights.update({
            "simple_am.weight": output @ projection,
            "simple_am.bias": output @ bias + half_bias,
            "simple_lm.weight": output.clone(),
            "simple_lm.bias": half_bias,
        })
        method = "linearized-joiner"

    # A meta model validates names and shapes without allocating random full-size weights.
    with torch.device("meta"):
        model = ASRModel(target)
    model.load_state_dict(weights, strict=True, assign=True)
    out.mkdir(parents=True)
    return Path(checkpoint.save_weights(str(out), payload["step"], {
        "model": weights,
        "model_config": target.to_dict(),
        "tokenizer_sha256": tokenizers["rnnt"],
        "tokenizers": tokenizers,
        "seed": {
            "method": method,
            "weights_sha256": source.with_suffix(".sha256").read_text().strip(),
            "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "target_config_sha256": hashlib.sha256(target_bytes).hexdigest(),
        },
    }))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-config", type=Path, required=True)
    ap.add_argument("--target-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new seed directory")
    a = ap.parse_args()
    torch.set_num_threads(1)
    print(prepare(a.source, a.source_config, a.target_config, a.out))


if __name__ == "__main__":
    main()
