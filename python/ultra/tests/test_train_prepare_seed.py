import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import checkpoint  # noqa: E402
from train.asr import ASRModel, PRESETS  # noqa: E402
from train.errors import CheckpointError  # noqa: E402
from train.ops.prepare_seed import prepare  # noqa: E402


def test_prepare_seed_preserves_recognizer_and_validates_both_tokenizers(tmp_path):
    exact = ASRModel(PRESETS["TINY"])
    target = replace(exact.cfg, rnnt_loss="pruned")
    tokenizers = {"rnnt": "a" * 64, "ctc": "b" * 64}
    source = Path(checkpoint.save_weights(str(tmp_path), 25, {
        "model": exact.state_dict(),
        "model_config": exact.cfg.to_dict(),
        "tokenizer_sha256": tokenizers["rnnt"],
    }))
    source_config = tmp_path / "source.json"
    target_config = tmp_path / "target.json"
    source_config.write_text(json.dumps({"model": exact.cfg.to_dict(), "tokenizers": tokenizers}))
    target_config.write_text(json.dumps({
        "model": target.to_dict(), "tokenizers": dict(tokenizers, ctc="c" * 64),
    }))
    out = tmp_path / "seed"
    with pytest.raises(CheckpointError, match="tokenizer digests"):
        prepare(source, source_config, target_config, out)
    assert not out.exists()

    target_config.write_text(json.dumps({"model": target.to_dict(), "tokenizers": tokenizers}))
    seed = checkpoint.load(str(prepare(source, source_config, target_config, out)), restore_rng=False)
    assert seed["step"] == 25
    assert seed["seed"]["method"] == "linearized-joiner"
    assert "optimizer" not in seed
    for key, value in exact.state_dict().items():
        assert torch.equal(value, seed["model"][key]), key

    pruned = ASRModel(target)
    pruned.load_state_dict(seed["model"])
    projection = exact.joiner.enc_proj
    output = exact.joiner.out
    enc = torch.randn(2, projection.in_features)
    pred = torch.randn(2, projection.out_features)
    torch.testing.assert_close(
        pruned.simple_am(enc) + pruned.simple_lm(pred), output(projection(enc) + pred),
    )
