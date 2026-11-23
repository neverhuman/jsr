"""Wave 2 objective, augmentation, causality, and 4-step default-off bit-exactness tests."""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS, ASRModel, SpecAugConfig, spec_augment  # noqa: E402
from train.data import Example, FixedBatches, ParquetBatches  # noqa: E402
from train.ddp_loss import global_loss  # noqa: E402
from train.etm_kd import (  # noqa: E402
    etm_kd_loss,
    etm_kd_pruned_loss,
    extract_teacher_token_targets,
    label_occupancy_exact,
    label_occupancy_pruned,
)
from train.features import time_warp  # noqa: E402
from train.loss_trunc import LossTruncator, truncate_utterance_losses  # noqa: E402
from train.mcr_rnnt import symmetric_band_kl  # noqa: E402
from train.model import Encoder, EncoderConfig  # noqa: E402
from train.recipe_opts import add_wave2_args, apply_wave2_args  # noqa: E402
from train.trainer import TrainConfig, Trainer  # noqa: E402
from train.transducer import full_lattice_logprobs  # noqa: E402


@pytest.mark.parametrize("family,chunk_pos", [("dense", 4), ("multirate", 4), ("unet3", 4)])
def test_chunk_zero_bidirectional_and_positive_chunk_strict_causality(family, chunk_pos):
    torch.manual_seed(11)
    kw = dict(d_model=32, n_heads=4, stem_channels=8, conv_kernel=5, chunk=chunk_pos,
              left_context=8, dropout=0.0, post_block_norm=False, family=family)
    if family == "dense":
        kw["n_blocks"] = 2
    elif family == "multirate":
        kw.update(lower_blocks=1, middle_blocks=1, final_blocks=1)
    else:
        kw.update(lower_blocks=1, middle_blocks=1, deep_blocks=1, up_blocks=1, final_blocks=1)
    enc = Encoder(EncoderConfig(**kw)).eval()
    T = 128
    x = torch.randn(1, T, 80)
    lens = torch.tensor([T])
    y = x.clone()
    y[:, 4 * chunk_pos:] += 5.0  # perturb strictly after first chunk boundary
    with torch.no_grad():
        base_s, _ = enc(x, lens, chunk=chunk_pos)
        pert_s, _ = enc(y, lens, chunk=chunk_pos)
        base_f, n_out = enc(x, lens, chunk=0)
        pert_f, _ = enc(y, lens, chunk=0)
        padded_f, _ = enc(torch.cat([x, torch.randn(1, 24, 80) * 10.0], dim=1), lens, chunk=0)
    # Streaming chunk > 0: strictly causal (future perturbation leaves earlier chunk identical).
    assert torch.equal(pert_s[:, :chunk_pos], base_s[:, :chunk_pos])
    # Full-context chunk == 0: bidirectional (future perturbation reaches frame 0), padding clean.
    assert not torch.allclose(pert_f[:, :chunk_pos], base_f[:, :chunk_pos])
    assert torch.allclose(padded_f[:, :int(n_out[0])], base_f, atol=1e-5)
    with pytest.raises(ValueError):
        enc(x, lens, chunk=-1)


def test_mcr_rnnt_symmetry_gradient_flow_and_ddp_denominator():
    torch.manual_seed(12)
    ls = torch.randn(2, 10, 3, 9, requires_grad=True)
    lf = torch.randn(2, 10, 3, 9, requires_grad=True)
    lens = torch.tensor([10, 6])
    k_sf, n1 = symmetric_band_kl(ls, lf, lens)
    k_fs, n2 = symmetric_band_kl(lf, ls, lens)
    assert n1 == n2 == (10 + 6) * 3
    assert torch.allclose(k_sf, k_fs, atol=1e-6)
    assert float(symmetric_band_kl(ls.detach(), ls.detach(), lens)[0]) == 0.0
    k_sf.backward()
    assert ls.grad is not None and lf.grad is not None
    assert float(ls.grad[1, 6:].abs().max()) == 0.0 and float(lf.grad[0, :10].abs().sum()) > 0.0

    # End-to-end pruned k2 MCR-RNNT (chunk=0 shared band) + exact MCR-RNNT + DDP normalization
    k2 = pytest.importorskip("k2")
    assert k2 is not None
    cfg_p = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned", prune_range=3)
    m = ASRModel(cfg_p)
    feats = torch.randn(2, 64, 80)
    fl = torch.tensor([64, 48])
    rt, rl = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([3, 2])
    ct, cl = torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1])
    out = m.losses(feats, fl, rt, rl, ct, cl, chunk=4, mcr_weight=0.25)
    assert "mcr_kl" in out and out["mcr_cells"] == int(out["frames"]) * 3
    out["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

    # DDP global denominator check for mcr_kl & etm_kd
    r0 = {"rnnt": torch.tensor(2.0), "ctc": torch.tensor(1.0), "rnnt_tokens": 10, "ctc_tokens": 10,
          "frames": 20, "mcr_kl": torch.tensor(0.4), "mcr_cells": 60, "etm_kd": torch.tensor(0.8), "etm_tokens": 8}
    r1 = {"rnnt": torch.tensor(2.0), "ctc": torch.tensor(1.0), "rnnt_tokens": 10, "ctc_tokens": 10,
          "frames": 10, "mcr_kl": torch.tensor(0.1), "mcr_cells": 30, "etm_kd": torch.tensor(0.2), "etm_tokens": 2}
    tc = TrainConfig(mcr_weight=0.5, etm_kd_weight=0.3)

    def red(vals, _dev):
        return [a + b for a, b in zip(vals, other_vals, strict=True)]

    other_vals = [10.0, 10.0, 0.0, 10.0, 30.0, 2.0]
    g0, o0 = global_loss(r0, tc, PRESETS["TINY"], world=2, device="cpu", pruned_scale=1.0,
                         reduce_sum=lambda v, d: [v[i] + other_vals[i] for i in range(len(v))] if len(v) == 6 else v)
    assert math.isclose(float(o0["mcr_kl"]) * 2, (0.4 * 60) / 90 * 2, rel_tol=1e-5)
    assert math.isclose(float(o0["etm_kd"]) * 2, (0.8 * 8) / 10 * 2, rel_tol=1e-5)
    assert torch.isfinite(g0)


def test_etm_kd_timing_invariance_zero_blank_grad_and_token_masking():
    torch.manual_seed(13)
    # 1) Token-level validity masking retains >90% even with 1 substitution in a 20-token utterance
    ref = [((i % 7) + 1) for i in range(20)]
    emitted = [0, 0]
    for idx, tok in enumerate(ref):
        emitted.extend([0, 9 if idx == 7 else tok])  # 1 corrupted token at pos 7
    emitted.extend([0, 0])
    Tt, V = len(emitted), 11
    t_probs = torch.softmax(torch.randn(Tt, V - 1), dim=-1)
    q0, valid0 = extract_teacher_token_targets(emitted, t_probs, ref)
    assert int(valid0.sum()) == 19 and float(valid0.float().mean()) >= 0.90 and not bool(valid0[7])

    # 2) Exact invariance to teacher timing shift t -> t + 3
    em_shift = [0, 0, 0] + emitted
    tp_shift = torch.cat([torch.softmax(torch.randn(3, V - 1), dim=-1), t_probs], dim=0)
    q_s, valid_s = extract_teacher_token_targets(em_shift, tp_shift, ref)
    assert torch.equal(valid0, valid_s) and torch.equal(q0, q_s)

    # 3) Exact vs bound ETM-KD (Jensen: 0 <= exact <= bound) + identically ZERO blank gradient!
    k2 = pytest.importorskip("k2")
    B, T, S, V = 2, 10, 4, 9
    m = ASRModel(dataclasses.replace(PRESETS["TINY"], vocab_rnnt=V, rnnt_loss="pruned", prune_range=3))
    J = m.cfg.joiner
    enc_p = torch.randn(B, T, J)
    pred_p = torch.randn(B, S + 1, J)
    tgt = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 0]], dtype=torch.long)
    t_lens, u_lens = torch.tensor([10, 8]), torch.tensor([4, 3])
    bd = torch.tensor([[0, 0, 4, 10], [0, 0, 3, 8]], dtype=torch.int64)
    z_full = m.joiner.logits(enc_p[:, :, None, :], pred_p[:, None, :, :]).detach().requires_grad_(True)
    lpb, lpl = full_lattice_logprobs(m.joiner, enc_p, pred_p, tgt)
    occ_ex = label_occupancy_exact(lpb, lpl, t_lens, u_lens)
    q_batch = torch.softmax(torch.randn(B, S, V - 1), dim=-1)
    vmask = torch.tensor([[True, True, False, True], [True, True, True, False]])
    le, ne = etm_kd_loss(z_full, q_batch, occ_ex, vmask, mode="exact")
    lb, nb = etm_kd_loss(z_full, q_batch, occ_ex, vmask, mode="bound")
    assert ne == nb == 6 and 0.0 <= float(le) <= float(lb) + 1e-6
    le.backward()
    assert z_full.grad is not None
    assert float(z_full.grad[..., 0].abs().max()) == 0.0  # ZERO gradient into blank!
    assert float(z_full.grad[..., 1:].abs().sum()) > 0.0

    # Verify end-to-end ETM-KD loss equality under teacher timing shift t -> t + 3
    q_b0 = q0[:S, :V - 1]
    q_bs = q_s[:S, :V - 1]
    vm = valid0[:S][None].expand(B, S)
    l_t0, _ = etm_kd_loss(z_full.detach(), q_b0[None].expand(B, S, -1), occ_ex, vm, mode="exact")
    l_t3, _ = etm_kd_loss(z_full.detach(), q_bs[None].expand(B, S, -1), occ_ex, vm, mode="exact")
    assert torch.equal(l_t0, l_t3)

    # 4) k2 pruned occupancy + pruned ETM-KD zero blank gradient
    ranges = k2.get_rnnt_prune_ranges(
        px_grad=torch.ones(B, S, T + 1), py_grad=torch.ones(B, S + 1, T), boundary=bd, s_range=3,
    )
    am_p, lm_p = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
    z_pr = m.joiner.logits(am_p, lm_p).detach().requires_grad_(True)
    occ_k2 = label_occupancy_pruned(z_pr, tgt, ranges, bd)
    lp_e, npe = etm_kd_pruned_loss(z_pr, ranges, q_batch, occ_k2, vmask, mode="exact")
    lp_b, npb = etm_kd_pruned_loss(z_pr, ranges, q_batch, occ_k2, vmask, mode="bound")
    assert npe == npb == 6 and 0.0 <= float(lp_e) <= float(lp_b) + 1e-6
    lp_e.backward()
    assert float(z_pr.grad[..., 0].abs().max()) == 0.0 and float(z_pr.grad[..., 1:].abs().sum()) > 0.0


def test_loss_trunc_and_augmentations_time_warp_and_bpe_dropout():
    # WER-7a loss truncation: drop, cauchy, tukey, and warmup no-op
    nll = torch.tensor([10.0, 12.0, 11.0, 13.0, 10.5, 500.0], requires_grad=True)
    lens = torch.tensor([5, 5, 5, 5, 5, 5])
    adj0, tok0, st0 = truncate_utterance_losses(nll, lens, q=0.0)
    assert torch.equal(adj0, nll) and tok0 == 30 and st0["trunc_dropped"] == 0.0
    adj_d, tok_d, st_d = truncate_utterance_losses(nll, lens, q=0.80, mode="drop")
    assert st_d["trunc_dropped"] == 1.0 and tok_d == 25 and float(adj_d[-1]) == 0.0
    (g_d,) = torch.autograd.grad(adj_d.sum(), nll)
    assert g_d[:-1].tolist() == [1.0] * 5 and float(g_d[-1]) == 0.0

    for mode in ("cauchy", "tukey"):
        adj_c, _, _ = truncate_utterance_losses(nll, lens, q=0.80, mode=mode)
        (g_c,) = torch.autograd.grad(adj_c.sum(), nll)
        assert all(math.isclose(float(v), 1.0, rel_tol=1e-5) for v in g_c[:-1])
        assert float(g_c[-1]) < (0.05 if mode == "cauchy" else 1e-6)

    lt = LossTruncator(q=0.85, warmup_steps=3, mode="drop", min_history=4)
    a_warm, _, st_w = lt(nll, lens, step=1)
    assert torch.equal(a_warm, nll) and st_w["trunc_dropped"] == 0.0
    a_post, _, st_p = lt(nll, lens, step=3)
    assert st_p["trunc_dropped"] == 1.0 and float(a_post[-1]) == 0.0

    # Time-warp (0 = bit-exact no-op with 0 RNG draws; 80 = modifies valid frames only)
    x = torch.randn(2, 200, 80)
    fl = torch.tensor([180, 120])
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(time_warp(x, fl, 0, g1), x)
    assert torch.equal(g1.get_state(), g2.get_state())
    xw = time_warp(x, fl, 80, torch.Generator().manual_seed(7))
    assert not torch.allclose(xw[0, :180], x[0, :180]) and torch.equal(xw[0, 180:], x[0, 180:])

    # BPE-dropout + auto-disable on accepted frame-KD rows
    class _FakeSP:
        def encode(self, s, enable_sampling=False, nbest_size=None, alpha=None):
            return [1, 2, 3, 4] if enable_sampling else [10, 20]

    class _FakeTok:
        upper = False
        sp = _FakeSP()

    from train.tokenizer import Tokenizer
    tok = object.__new__(Tokenizer)
    tok.sp, tok.upper = _FakeSP(), False
    assert tok.encode("hello world") == [11, 21]
    assert tok.encode("hello world", dropout_p=1.0, alpha=0.1) == [2, 3, 4, 5]
    pb = object.__new__(ParquetBatches)
    pb.bpe_dropout_p, pb.bpe_dropout_alpha, pb.seed = 1.0, 0.1, 42
    pb.feat = type("F", (), {"tok_rnnt": tok, "tok_ctc": tok})()
    ex_plain = Example("u1", "hello", torch.zeros(10, 80), [11, 21], [11, 21], kd=None)
    ex_kd = Example("u2", "hello", torch.zeros(10, 80), [11, 21], [11, 21], kd=(None, None, None, True))
    pb._apply_bpe_dropout(ex_plain, epoch=0)
    pb._apply_bpe_dropout(ex_kd, epoch=0)
    assert ex_plain.rnnt == [2, 3, 4, 5] and ex_kd.rnnt == [11, 21]  # auto-disabled on frame-KD!


def test_four_step_default_off_bit_exactness_and_cli_parsing(tmp_path):
    ap = argparse.ArgumentParser()
    add_wave2_args(ap)
    ns = ap.parse_args(["--mcr-weight", "0.1", "--etm-kd-weight", "0.2", "--loss-trunc-q", "0.995",
                        "--specaug-time-warp", "80", "--bpe-dropout-p", "0.5"])
    cfg_cli = TrainConfig()
    apply_wave2_args(cfg_cli, ns)
    assert (cfg_cli.mcr_weight, cfg_cli.etm_kd_weight, cfg_cli.loss_trunc_q,
            cfg_cli.specaug_time_warp, cfg_cli.bpe_dropout_p) == (0.1, 0.2, 0.995, 80, 0.5)

    # 4-step bit-exactness witness with all Wave 2 flags at default-off (0 / 0.0)
    torch.manual_seed(99)
    exs = [Example(f"u{i}", "a b", torch.randn(48 + 4 * i, 80), [1, 2, 3], [1, 2]) for i in range(4)]

    def run_four(run_name: str, cfg: TrainConfig):
        src = FixedBatches(exs, batch_size=2)
        tr = Trainer(str(tmp_path / run_name), cfg, src, model_cfg=PRESETS["TINY"], device="cpu", log=False)
        losses = [tr.train_step(b)["loss"] for _, (_, b) in zip(range(4), src.iterate(tr.pos))]
        return losses, {k: v.clone() for k, v in tr.model.state_dict().items()}

    base_cfg = TrainConfig(preset="TINY", seed=123, specaug=True, specaug_start=0, chunk=2,
                           mcr_weight=0.0, etm_kd_weight=0.0, loss_trunc_q=0.0,
                           specaug_time_warp=0, bpe_dropout_p=0.0)
    l1, s1 = run_four("r1", base_cfg)
    l2, s2 = run_four("r2", TrainConfig(preset="TINY", seed=123, specaug=True, specaug_start=0, chunk=2))
    assert l1 == l2
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    # SpecAug with time_warp=0 is bit-identical to SpecAug without time_warp arg
    x = torch.randn(2, 60, 80)
    fl = torch.tensor([60, 50])
    a1 = spec_augment(x, fl, SpecAugConfig(enabled=True), torch.Generator().manual_seed(5))
    a2 = spec_augment(x, fl, SpecAugConfig(enabled=True, time_warp=0), torch.Generator().manual_seed(5))
    assert torch.equal(a1, a2)
