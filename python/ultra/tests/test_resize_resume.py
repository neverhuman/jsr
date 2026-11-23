"""H53 Amendment 2 follow-up (for S2+): a resume that changes the learning rate or the batch shape.

1. The LR: the checkpoint's LambdaLR state carries base_lrs, so `--set lr=` used to be ignored on resume.
2. The cursor: with a new budget / max batch, the saved batch index points into the old packing of the current
   window, so a mid-window restart repeated or skipped part of it; the rest of that window now keeps the old
   packing and the new shape starts at the next window."""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_train_checkpoint import _trainer, workdir  # noqa: E402,F401
from train.data import Position  # noqa: E402
from train.resize import apply_resize, checkpoint_carry_shape  # noqa: E402


def test_resume_takes_the_configured_lr(workdir):  # noqa: F811
    a = _trainer(workdir)
    a.run(max_steps=4)
    b = _trainer(workdir)
    b.cfg.lr = a.cfg.lr * 2
    assert b.resume()
    lam = b.sched.lr_lambdas[0](b.sched.last_epoch)
    assert abs(b.sched.get_last_lr()[0] - b.cfg.lr * lam) < 1e-12
    assert all(abs(g["initial_lr"] - b.cfg.lr) < 1e-12 for g in b.opt.param_groups)
    b.run(max_steps=1)  # the next step steps the scheduler from the new base
    assert abs(b.sched.get_last_lr()[0] - b.cfg.lr * b.sched.lr_lambdas[0](b.sched.last_epoch)) < 1e-12


def test_same_lr_and_shape_change_nothing(workdir):  # noqa: F811
    a = _trainer(workdir)
    a.run(max_steps=3)
    b = _trainer(workdir)
    before = None
    assert b.resume()
    before = (list(b.sched.base_lrs), b.pos)
    assert apply_resize(b, {"lr": b.cfg.lr, "budget_s": b.cfg.budget_s, "max_batch": b.cfg.max_batch}) == {}
    assert (list(b.sched.base_lrs), b.pos) == before


def test_new_batch_shape_finishes_the_window_under_the_old_shape(tmp_path):
    """v3-reviewer / codex-full-fleet: a mid-window shape change must neither skip nor repeat. The resumed source
    keeps the checkpoint's shape for the rest of that window, then packs from the next window at the new shape."""
    from test_h52_extra_splits import _tree

    from train.data import ParquetBatches
    from train.features import LogMel

    root, _ = _tree(tmp_path, n_base=48)
    feat = SimpleNamespace(logmel=LogMel())

    def source(budget, mb):
        return ParquetBatches(["train-clean-100"], feat, seed=5, budget_s=budget, max_batch=mb, parquet_root=root,
                              window_groups=4)

    def groups(src, ep, w):
        rows = [r for fi, rg in src._window_units(ep, w) for r in src._read_rows(fi, rg)]
        return [[rows[i]["id"] for i in g] for g in src._header_groups(rows, ep, w)]

    old, new = source(0.35, 3), source(0.7, 6)
    g_old, g_new_next = groups(old, 0, 1), groups(new, 0, 2)
    assert len(g_old) >= 3 and groups(new, 0, 1) != g_old  # the shapes really pack differently
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    t = SimpleNamespace(cfg=SimpleNamespace(lr=1e-3, budget_s=0.7, max_batch=6), opt=opt,
                        sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), source=new, pos=Position(0, 1, 2))
    out = apply_resize(t, {"lr": 1e-3, "budget_s": 0.35, "max_batch": 3})
    assert t.pos == Position(0, 1, 2) and out["carry_shape"]["window"] == [0, 1]
    assert out["carry_shape"]["remaining_batches"] == len(g_old) - 2
    assert groups(new, 0, 1) == g_old  # the rest of window 1 is the old packing: nothing skipped or repeated
    assert groups(new, 0, 2) == g_new_next  # window 2 onward packs at the new shape


    t.pos = Position(0, 1, 1)
    t.source = source(0.7, 6)
    apply_resize(t, {"lr": 1e-3, "budget_s": 0.35, "max_batch": 3})
    t.pos = Position(0, 1, 2)
    checkpoint = {
        "train_config": vars(t.cfg),
        "position": t.pos.to_dict(),
        "carry_shape": checkpoint_carry_shape(t),
    }
    assert checkpoint["carry_shape"] == (0.35, 3)
    t.source = source(0.7, 6)
    t.pos = Position(**checkpoint["position"])
    assert g_old[2:] and groups(t.source, 0, 1)[2:] != g_old[2:]
    apply_resize(t, checkpoint["train_config"], checkpoint["carry_shape"])
    assert groups(t.source, 0, 1)[t.pos.batch:] == g_old[2:]
    assert groups(t.source, 0, 2) == g_new_next
    t.pos = Position(0, 2, 1)
    assert checkpoint_carry_shape(t) is None



def test_boundary_resume_needs_no_carry(tmp_path):
    from test_h52_extra_splits import _tree

    from train.data import ParquetBatches
    from train.features import LogMel

    root, _ = _tree(tmp_path, n_base=24)
    src = ParquetBatches(["train-clean-100"], SimpleNamespace(logmel=LogMel()), seed=5, budget_s=0.7, max_batch=6,
                         parquet_root=root, window_groups=4)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    t = SimpleNamespace(cfg=SimpleNamespace(lr=1e-3, budget_s=0.7, max_batch=6), opt=opt,
                        sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), source=src, pos=Position(0, 1, 0))
    assert apply_resize(t, {"lr": 1e-3, "budget_s": 0.35, "max_batch": 3}) == {} and src.carry_shape is None


def test_skipped_remainder_is_counted_from_the_old_packing(tmp_path):
    """codex-full-fleet: a move to the next window must count what it skips. The count replays the loader's own
    header-only packing of that window at the checkpoint's batch shape."""
    from test_h52_extra_splits import _tree

    from train.data import ParquetBatches
    from train.features import LogMel
    from train.resize import window_remainder

    root, _ = _tree(tmp_path, n_base=40)
    feat = SimpleNamespace(logmel=LogMel())
    src = ParquetBatches(["train-clean-100"], feat, seed=3, budget_s=0.35, max_batch=3, parquet_root=root,
                         window_groups=4)
    rows = [r for fi, rg in src._window_units(0, 1) for r in src._read_rows(fi, rg)]
    n_old = len(src._header_groups(rows, 0, 1))
    assert n_old > 2
    batches, utts = window_remainder(src, 0, 1, 2)
    assert batches == n_old - 2 and 0 < utts <= len(rows)
    assert window_remainder(src, 0, 1, n_old) == (0, 0)  # a consumed window skips nothing


def test_zero_checkpoint_lr_takes_the_configured_lr():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    t = SimpleNamespace(cfg=SimpleNamespace(lr=2e-4, budget_s=1.0, max_batch=1), opt=opt, sched=sched,
                        source=SimpleNamespace(), pos=Position(0, 0, 0))
    assert apply_resize(t, {"lr": 0.0, "budget_s": 1.0, "max_batch": 1}) == {"lr": [0.0, 2e-4]}
    assert sched.base_lrs == [2e-4] and opt.param_groups[0]["lr"] == 2e-4
