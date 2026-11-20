"""Checkpoint save/resume is bit-exact on CPU; incomplete or corrupt checkpoints are refused;
a non-finite loss pauses the run with diagnostics."""
import os
import shutil
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import checkpoint  # noqa: E402
from train.asr import PRESETS, ModelConfig  # noqa: E402
from train.data import Example, FixedBatches  # noqa: E402
from train.errors import CheckpointError, NonFiniteLossError  # noqa: E402
from train.model import EncoderConfig  # noqa: E402
from train.trainer import TrainConfig, Trainer  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture
def workdir(request):
    # Repository target/ instead of /tmp (host rule: nothing in /tmp).
    d = os.path.join(REPO, "target", "test-train", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _examples(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        T = 40 + 7 * i
        out.append(Example(f"u{i}", "x", torch.randn(T, 80, generator=g),
                           torch.randint(1, 11, (3 + i % 3,), generator=g).tolist(),
                           torch.randint(1, 7, (2 + i % 2,), generator=g).tolist()))
    return out


def _cfg():
    enc = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5,
                        chunk=2, left_context=4, dropout=0.1)
    return ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16)


def _trainer(d):
    torch.set_num_threads(1)
    cfg = TrainConfig(preset="TINY", warmup=2, total_steps=100, ckpt_every_steps=1000,
                      ckpt_every_minutes=1e9, eval_every=0, log_every=1000, specaug=True,
                      specaug_start=2)
    src = FixedBatches(_examples(), batch_size=2, shuffle=True)
    return Trainer(d, cfg, src, model_cfg=_cfg(), device="cpu", log=False)


def test_resume_is_bit_exact(workdir):
    a = _trainer(os.path.join(workdir, "a"))
    a.run(max_steps=7)
    b = _trainer(os.path.join(workdir, "b"))
    b.run(max_steps=4)
    c = _trainer(os.path.join(workdir, "b"))
    assert c.resume() and c.step == 4 and c.pos == b.pos
    c.run(max_steps=3)
    assert c.step == a.step == 7 and c.pos == a.pos
    for (k, va), (_, vc) in zip(a.model.state_dict().items(), c.model.state_dict().items(),
                                strict=True):
        assert torch.equal(va, vc), k
    sa, sc = a.opt.state_dict()["state"], c.opt.state_dict()["state"]
    for i in sa:
        assert torch.equal(sa[i]["exp_avg_sq"], sc[i]["exp_avg_sq"])
    assert a.sched.state_dict() == c.sched.state_dict()


def test_incomplete_or_corrupt_checkpoint_refused(workdir):
    t = _trainer(workdir)
    t.run(max_steps=1)
    path = checkpoint.latest(workdir)
    with open(path, "r+b") as fh:
        fh.seek(100)
        fh.write(b"\x00\x01\x02")
    with pytest.raises(CheckpointError):
        checkpoint.load(path)
    os.remove(path[:-3] + ".sha256")
    with pytest.raises(CheckpointError):
        checkpoint.load(path)


def test_manifest_change_refused(workdir):
    t = _trainer(workdir)
    t.run(max_steps=1)
    u = _trainer(workdir)
    u.source.manifest = ["other"]
    with pytest.raises(CheckpointError):
        u.resume()


def test_non_finite_loss_pauses(workdir):
    t = _trainer(workdir)
    with torch.no_grad():
        t.model.joiner.out.bias[0] = float("nan")
    with pytest.raises(NonFiniteLossError) as e:
        t.run(max_steps=1)
    assert os.path.exists(e.value.diag_path)
    u = _trainer(workdir)
    with pytest.raises(CheckpointError):
        u.resume()


def test_presets_build():
    assert set(PRESETS) >= {"S", "D0", "TINY"}
