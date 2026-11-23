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


def test_weights_snapshots_are_kept_and_loadable(workdir):
    t = _trainer(workdir)
    t.cfg.weights_every = 2
    t.run(max_steps=6)
    names = sorted(f for f in os.listdir(workdir) if f.startswith("model-") and f.endswith(".pt"))
    assert names == ["model-00000002.pt", "model-00000004.pt", "model-00000006.pt"]
    p = checkpoint.load(os.path.join(workdir, names[-1]), restore_rng=False)
    for k, v in t.model.state_dict().items():
        assert torch.equal(p["model"][k], v), k


def test_divergence_guard_stops_on_loss_trend_and_wer(workdir):
    from train.errors import DivergenceError

    t = _trainer(workdir)
    t.cfg.diverge_after, t.cfg.diverge_ratio, t.cfg.diverge_patience = 0, 0.3, 20
    for _ in range(200):
        t._watch_loss({"loss": 1.0, "grad_norm": 1.0})
    with pytest.raises(DivergenceError):
        for _ in range(2000):
            t.step += 1
            t._watch_loss({"loss": 3.0, "grad_norm": 50.0})
    assert os.path.exists(os.path.join(workdir, "DIVERGED.json"))
    with pytest.raises(CheckpointError):
        _trainer(workdir).resume()
    u = _trainer(os.path.join(workdir, "w"))
    u._watch_wer(21.0)
    u._watch_wer(24.0)  # within 5 points
    with pytest.raises(DivergenceError):
        u._watch_wer(27.0)


def test_average_of_identical_is_identity_and_mean_is_mean(workdir):
    from train.average import average, save_average

    t = _trainer(workdir)
    t.run(max_steps=1)
    a = checkpoint.latest(workdir)
    avg = average([a, a, a])
    src = checkpoint.load(a, restore_rng=False)["model"]
    assert list(avg["model"]) == list(src)
    for k, v in src.items():
        assert avg["model"][k].shape == v.shape and avg["model"][k].dtype == v.dtype
        assert torch.equal(avg["model"][k], v), k
    t.run(max_steps=1)
    b = checkpoint.latest(workdir)
    out = save_average([a, b], os.path.join(workdir, "avg.pt"))
    m = checkpoint.load(out, restore_rng=False)
    sa, sb = src, checkpoint.load(b, restore_rng=False)["model"]
    k = next(k for k in sa if torch.is_floating_point(sa[k]) and not torch.equal(sa[k], sb[k]))
    assert torch.allclose(m["model"][k], (sa[k].double() + sb[k].double()).div(2).float())
    assert [s["step"] for s in m["averaged_from"]] == [1, 2]


def test_average_refuses_mismatched_configs(workdir):
    from train.average import average

    t = _trainer(os.path.join(workdir, "a"))
    t.run(max_steps=1)
    cfg = TrainConfig(preset="TINY", warmup=2, total_steps=100, ckpt_every_steps=1000,
                      ckpt_every_minutes=1e9, eval_every=0, log_every=1000)
    u = Trainer(os.path.join(workdir, "b"), cfg, FixedBatches(_examples(), batch_size=2),
                model_cfg=PRESETS["TINY"], device="cpu", log=False)
    u.run(max_steps=1)
    with pytest.raises(CheckpointError):
        average([checkpoint.latest(os.path.join(workdir, "a")),
                 checkpoint.latest(os.path.join(workdir, "b"))])


def test_qkv_constraint_caps_rows_and_gain_and_persists(workdir):
    t = _trainer(workdir)
    t.cfg.qkv_cap_from, t.cfg.qkv_cap_ratio = 1, 1.2
    t.run(max_steps=1)  # records the reference
    att = [m for m in t.model.modules() if hasattr(m, "qkv")]
    ref = [m.qkv.weight.norm(dim=1).clone() for m in att]
    with torch.no_grad():
        for m in att:
            m.qkv.weight.mul_(5.0)
            m.norm.weight.fill_(9.0)
    t._qkv_constraint()
    for m, r in zip(att, ref):
        assert torch.all(m.qkv.weight.norm(dim=1) <= 1.2 * r * (1 + 1e-5))
    assert all(float(m.norm.weight.abs().max()) <= 9.0 for m in att)
    t.save(wait=True)
    u = _trainer(workdir)
    assert u.resume() and u.qkv_caps is not None and u.qkv_ref_norm == t.qkv_ref_norm


def test_weight_ema_tracks_and_resumes_bit_exact(workdir):
    a = _trainer(os.path.join(workdir, "a"))
    a.cfg.ema_decay = 0.9
    a.run(max_steps=6)
    assert a.ema is not None
    w = a.model.state_dict()
    k = next(k for k, v in w.items() if v.is_floating_point() and v.dim() == 2)
    assert not torch.equal(a.ema[k], w[k])  # lags the weights
    b = _trainer(os.path.join(workdir, "b"))
    b.cfg.ema_decay = 0.9
    b.run(max_steps=3)
    c = _trainer(os.path.join(workdir, "b"))
    c.cfg.ema_decay = 0.9
    assert c.resume() and c.ema is not None
    c.run(max_steps=3)
    for name in a.ema:
        assert torch.equal(a.ema[name], c.ema[name]), name
    off = _trainer(os.path.join(workdir, "off"))
    off.run(max_steps=2)
    assert off.ema is None


def test_model_heads_follow_configured_tokenizers():
    from train.trainer import model_config_for

    d = model_config_for(TrainConfig(preset="S"))
    assert (d.vocab_rnnt, d.vocab_ctc) == (4097, 1025) == (PRESETS["S"].vocab_rnnt,
                                                          PRESETS["S"].vocab_ctc)
    h13 = model_config_for(TrainConfig(preset="S", rnnt_pieces=512, ctc_pieces=512))
    assert (h13.vocab_rnnt, h13.vocab_ctc) == (513, 513)
    assert h13.encoder == PRESETS["S"].encoder
    from train.asr import ASRModel

    m = ASRModel(h13)
    assert m.joiner.out.out_features == 513 and m.ctc_head.out_features == 513
    with pytest.raises(Exception):
        model_config_for(TrainConfig(preset="S", rnnt_pieces=1))


class _Tok:
    sha256 = None

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_logs_data_wait_pad_pct_and_ema_eval_restores_weights(workdir):
    import json

    t = _trainer(workdir)
    t.cfg.ema_decay, t.cfg.eval_every, t.cfg.log_every, t.log_enabled = 0.9, 4, 2, True
    t.dev, t.tok = _examples(3, seed=5), _Tok()
    t.run(max_steps=4)
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    ema = t._evaluate_ema()
    assert set(ema) == {"ema_wer", "ema_sub", "ema_del", "ema_ins"}
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, before[k]), k  # live weights restored after the EMA eval
    recs = [json.loads(line) for line in open(os.path.join(workdir, "train.jsonl"))]
    train = [r for r in recs if r["kind"] == "train"]
    assert train and all(0 <= r["data_wait_pct"] <= 100 and 0 <= r["pad_pct"] < 100 for r in train)
    assert any(r["kind"] == "eval" and "ema_wer" in r for r in recs)


def test_swap_weights_keeps_optimizer_and_schedule(workdir):
    from train.trainer import swap_weights

    t = _trainer(workdir)
    t.run(max_steps=3)
    t.save(wait=True)
    other = _trainer(os.path.join(workdir, "other"))
    other.run(max_steps=1)
    checkpoint.save_weights(os.path.join(workdir, "soup"), 1, {"model": other.model.state_dict(),
                                                             "model_config": other.model_cfg.to_dict()})
    u = _trainer(workdir)
    assert u.resume()
    opt_before = {k: v.clone() for k, v in u.opt.state_dict()["state"][0].items() if torch.is_tensor(v)}
    sched_before = u.sched.state_dict()
    info = swap_weights(u.model, os.path.join(workdir, "soup", "model-00000001.pt"))
    for k, v in u.model.state_dict().items():
        assert torch.equal(v, other.model.state_dict()[k]), k
    for k, v in opt_before.items():
        assert torch.equal(u.opt.state_dict()["state"][0][k], v)
    assert u.sched.state_dict() == sched_before and u.step == 3 and info["kept"] == []


def test_divergence_guard_keeps_a_margin_when_the_best_loss_is_negative(workdir):
    """A delay penalty can push the train loss below zero. The rule best + ratio x |best| equals
    (1 + ratio) x best for a positive best; for a negative best it must not trip on a small rise
    (H41 arm 13b stopped on -0.783 against a best of -0.846), and must still trip on a real one."""
    import pytest

    from train.errors import DivergenceError

    t = _trainer(workdir)
    t.cfg.diverge_after, t.cfg.diverge_patience, t.cfg.diverge_ratio = 0, 0, 0.3
    t.best_ema, t.step = -0.846, 1300
    for _ in range(50):
        t._watch_loss({"loss": -0.783, "grad_norm": 1.0})
    assert t.over_since is None and t.best_ema == -0.846
    with pytest.raises(DivergenceError):
        for _ in range(500):
            t._watch_loss({"loss": 0.5, "grad_norm": 1.0})
    u = _trainer(workdir + "_pos")
    u.cfg.diverge_after, u.cfg.diverge_patience, u.cfg.diverge_ratio = 0, 0, 0.3
    u.best_ema = 1.0
    u.loss_ema = 1.29
    u._watch_loss({"loss": 1.29, "grad_norm": 1.0})  # under (1 + 0.3) x 1.0: unchanged behaviour
    assert u.over_since is None
