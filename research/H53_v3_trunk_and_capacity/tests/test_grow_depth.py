"""Witness tests for tools/grow_depth.py (H53 capacity pilot).

Run with the trainer worktree's code on the path (read-only import, nothing is modified there):
  PYTHONPATH=/home/ubuntu/jsr/.agent/worktrees/trainer/python/ultra:/mnt/xbabe3-external/jsr/h53-trunk/tools \
      python -m pytest -q /mnt/xbabe3-external/jsr/h53-trunk/tests/test_grow_depth.py
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from grow_depth import (OUTPUT_PROJ, GrowError, check_growable, grow_state, grown_config,
                        insert_positions, layout)
from train.asr import PRESETS, ASRModel
from train.infer import ChunkEncoder, cache_shapes
from train.model import EncoderConfig

torch.set_num_threads(2)


def _randomise(model: torch.nn.Module, seed: int) -> None:
    """Non-trivial values everywhere (LayerNorm gains / biases too), so equality is meaningful."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.05 if p.ndim >= 2 else 0.2)
                    + (1.0 if name.endswith("norm.weight") else 0.0))


def _small(n_blocks: int = 4) -> object:
    s = PRESETS["S"]
    enc = dataclasses.replace(s.encoder, d_model=64, n_heads=4, n_blocks=n_blocks, stem_channels=16,
                              left_context=16, dropout=0.0)
    return dataclasses.replace(s, encoder=enc, vocab_rnnt=33, vocab_ctc=33, pred_emb=32, joiner=48,
                               rnnt_loss="pruned")


def _pair(cfg, n_dst: int, seed: int = 0):
    src = ASRModel(cfg)
    _randomise(src, seed)
    state, plan = grow_state(src.state_dict(), cfg.encoder.n_blocks, n_dst)
    dst = ASRModel(grown_config(cfg, n_dst))
    dst.load_state_dict(state, strict=True)
    return src.eval(), dst.eval(), plan


def test_positions_and_layout():
    assert insert_positions(16, 24) == [1, 3, 5, 7, 9, 11, 13, 15]
    assert insert_positions(16, 32) == list(range(16))
    assert insert_positions(16, 20) == [3, 7, 11, 15]
    lay = layout(16, 24)
    assert len(lay) == 24 and [i for i, n in lay if not n] == list(range(16))
    assert lay[:4] == [(0, False), (1, False), (1, True), (2, False)]
    for bad in (16, 33, 0):
        with pytest.raises(GrowError):
            insert_positions(16, bad)


def test_refuses_what_it_cannot_grow_exactly():
    with pytest.raises(GrowError):
        check_growable(PRESETS["S_MR"].encoder)
    with pytest.raises(GrowError):
        check_growable(EncoderConfig(post_block_norm=True))
    if "V2_S" in PRESETS:  # not on A1L's older tree (d5f231e)
        with pytest.raises(GrowError):
            check_growable(PRESETS["V2_S"].encoder)


@pytest.mark.parametrize("n_dst", [5, 6, 8])
def test_small_offline_and_chunked_outputs_identical(n_dst):
    cfg = _small(4)
    src, dst, _ = _pair(cfg, n_dst, seed=n_dst)
    g = torch.Generator().manual_seed(7)
    feats = torch.randn(3, 161, 80, generator=g)
    lens = torch.tensor([161, 120, 97])
    with torch.no_grad():
        for chunk in (4, 8, 16, 0):
            a, la = src.encode(feats, lens, chunk=chunk)
            b, lb = dst.encode(feats, lens, chunk=chunk)
            assert torch.equal(la, lb)
            assert torch.equal(a, b), f"chunk {chunk}: max diff {(a - b).abs().max()}"


def test_new_blocks_are_copies_with_zero_outputs():
    cfg = _small(4)
    src, dst, plan = _pair(cfg, 8, seed=3)
    s, d = src.state_dict(), dst.state_dict()
    for j, (i, new) in enumerate(plan):
        for k, v in s.items():
            pre = f"encoder.blocks.{i}."
            if not k.startswith(pre):
                continue
            name = k[len(pre):]
            got = d[f"encoder.blocks.{j}.{name}"]
            if new and name.rsplit(".", 1)[0] in OUTPUT_PROJ:
                assert torch.count_nonzero(got) == 0
            else:
                assert torch.equal(got, v)
    for k in s:
        if not k.startswith("encoder.blocks."):
            assert torch.equal(s[k], d[k])


def test_new_blocks_learn_at_the_first_step():
    """Zero output projections get gradients at once; the copied internals get none until they move."""
    cfg = _small(4)
    _, dst, plan = _pair(cfg, 6, seed=5)
    dst.train()
    feats = torch.randn(2, 81, 80, generator=torch.Generator().manual_seed(1))
    out, _ = dst.encode(feats, torch.tensor([81, 70]), chunk=4)
    out.pow(2).mean().backward()
    for j, (_, new) in enumerate(plan):
        blk = dst.encoder.blocks[j]
        if new:
            assert blk.ff2.w2.weight.grad.abs().sum() > 0
            assert blk.att.out.weight.grad.abs().sum() > 0
            assert blk.ff1.w1.weight.grad.abs().sum() == 0


def _feats_for_chunk(feats: torch.Tensor, t0: int, n: int) -> torch.Tensor:
    """Rows 4 t0 - 6 .. 4 (t0 + n) - 4 inclusive (4 n + 3 rows), zero rows for indices < 0."""
    lo, hi = 4 * t0 - 6, 4 * (t0 + n) - 4
    rows = [feats[:, r] if 0 <= r < feats.shape[1] else torch.zeros_like(feats[:, 0])
            for r in range(lo, hi + 1)]
    return torch.stack(rows, dim=1)


def test_streaming_chunk_encoder_identical():
    """The streaming path (train.infer.ChunkEncoder, the exporter's reference) agrees too, with
    caches sized by n_blocks."""
    cfg = _small(4)
    src, dst, _ = _pair(cfg, 8, seed=11)
    feats = src.cmvn(torch.randn(1, 4 * 12 + 3, 80, generator=torch.Generator().manual_seed(2)))
    outs = []
    for m in (src, dst):
        ce = ChunkEncoder(m.encoder).eval()
        c = m.encoder.cfg
        sh = cache_shapes(c)
        k = torch.zeros((sh["k"][0], 1) + sh["k"][1:])
        v = torch.zeros_like(k)
        conv = torch.zeros((sh["conv"][0], 1) + sh["conv"][1:])
        kv_len = torch.zeros(1, dtype=torch.long)
        ys = []
        with torch.no_grad():
            for t0 in range(0, 12, c.chunk):
                x = _feats_for_chunk(feats, t0, c.chunk)
                y, k, v, kv_len, conv = ce(x, torch.tensor([t0]), k, v, kv_len, conv)
                ys.append(y)
        outs.append(torch.cat(ys, dim=1))
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("n_dst", [24, 32])
def test_real_s_preset_grows_exactly(n_dst):
    """The B0 shape itself (dense S, d 384, 16 blocks) at 24 / 32 blocks: same outputs, expected
    parameter counts (S-24 84.0 M, S-32 111.3 M with the training-only simple heads)."""
    cfg = dataclasses.replace(PRESETS["S"], vocab_rnnt=501, vocab_ctc=501, rnnt_loss="pruned")
    src, dst, _ = _pair(cfg, n_dst, seed=n_dst)
    want = {24: 83_998_932, 32: 111_262_932}[n_dst]
    assert sum(p.numel() for p in dst.parameters()) == want
    feats = torch.randn(2, 4 * 40 + 1, 80, generator=torch.Generator().manual_seed(9))
    lens = torch.tensor([161, 133])
    with torch.no_grad():
        for chunk in (4, 16):
            a, _ = src.encode(feats, lens, chunk=chunk)
            b, _ = dst.encode(feats, lens, chunk=chunk)
            assert torch.equal(a, b)


def test_make_seed_roundtrip_through_checkpoint_format(tmp_path):
    """The CLI path: a trainer-format checkpoint (with .sha256 marker) -> grown seed (+ marker) that
    train.checkpoint.load accepts, whose config is the source config with n_blocks changed, and whose
    weights load strictly into the grown model and reproduce the source outputs."""
    import os

    from grow_depth import make_seed
    from train import checkpoint
    from train.asr import ModelConfig

    cfg = _small(4)
    src = ASRModel(cfg)
    _randomise(src, 21)
    ema = {k: v.clone() * 0.5 for k, v in src.state_dict().items()}
    path = str(tmp_path / "ckpt-00001000.pt")
    torch.save({"model": src.state_dict(), "ema": ema, "model_config": cfg.to_dict(), "step": 1000,
                "tokenizer_sha256": "abc"}, path)
    with open(path[:-3] + ".sha256", "w") as fh:
        import hashlib
        fh.write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "\n")
    out = str(tmp_path / "seed_grow6.pt")
    make_seed(path, out, 6, weights="model")
    assert os.path.exists(out[:-3] + ".sha256")
    p = checkpoint.load(out, restore_rng=False)
    got_cfg = ModelConfig.from_dict(p["model_config"])
    assert got_cfg == dataclasses.replace(cfg, encoder=dataclasses.replace(cfg.encoder, n_blocks=6))
    assert p["seed_from"]["n_src"] == 4 and p["seed_from"]["n_dst"] == 6 and p["step"] == 1000
    dst = ASRModel(got_cfg)
    dst.load_state_dict(p["model"], strict=True)
    feats = torch.randn(1, 81, 80, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a, _ = src.eval().encode(feats, torch.tensor([81]), chunk=4)
        b, _ = dst.eval().encode(feats, torch.tensor([81]), chunk=4)
    assert torch.equal(a, b)
    out_ema = str(tmp_path / "seed_grow6_ema.pt")
    make_seed(path, out_ema, 6, weights="ema")
    pe = checkpoint.load(out_ema, restore_rng=False)
    assert torch.equal(pe["model"]["encoder.stem.out.weight"], ema["encoder.stem.out.weight"])
