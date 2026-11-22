"""H23 optimiser options (train/optim): the default path is bit-for-bit the original AdamW trainer;
every alternative trains a tiny model on CPU, resumes bit-exactly, and has its defining property
checked (ScaledAdam's scale-relative step, Eden's shape, Schedule-Free's x/y switch, Muon's
orthogonalised update and parameter split, adaptive/AGC clipping, Lookahead, joiner z-loss)."""
import importlib.util
import math
import os
import shutil
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.data import Example, FixedBatches  # noqa: E402
from train.errors import OptimConfigError  # noqa: E402
from train.model import EncoderConfig  # noqa: E402
from train.optim import (Lookahead, ScaledAdam, ScheduleFreeAdamW, build_optimizer,  # noqa: E402
                         eval_weights)
from train.optim.clip import GradClipper  # noqa: E402
from train.optim.muon import Muon, is_muon_param, newton_schulz  # noqa: E402
from train.optim.scaled_adam import eden_lambda  # noqa: E402
from train.optim.zloss import band_zloss, diagonal_zloss  # noqa: E402
from train.trainer import TrainConfig, Trainer, lr_lambda  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HAVE_K2 = importlib.util.find_spec("k2") is not None


@pytest.fixture
def workdir(request):
    d = os.path.join(REPO, "target", "test-train", request.node.name)  # not /tmp (host rule)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _examples(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [Example(f"u{i}", "x", torch.randn(40 + 7 * i, 80, generator=g),
                    torch.randint(1, 11, (3 + i % 3,), generator=g).tolist(),
                    torch.randint(1, 7, (2 + i % 2,), generator=g).tolist()) for i in range(n)]


def _mcfg(rnnt_loss="exact"):
    enc = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5,
                        chunk=2, left_context=4, dropout=0.1)
    return ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16,
                       rnnt_loss=rnnt_loss)


def _trainer(d, rnnt_loss="exact", **kw):
    torch.set_num_threads(1)
    cfg = TrainConfig(preset="TINY", warmup=2, total_steps=100, ckpt_every_steps=1000,
                      ckpt_every_minutes=1e9, eval_every=0, log_every=1000, specaug=True,
                      specaug_start=2, **kw)
    src = FixedBatches(_examples(), batch_size=2, shuffle=True)
    return Trainer(d, cfg, src, model_cfg=_mcfg(rnnt_loss), device="cpu", log=False)


def _params_equal(a, b):
    for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(va, vb), k


# ---- default path ----------------------------------------------------------------------------
def test_default_path_is_bit_for_bit_the_original_adamw(workdir):
    """The default config builds the same AdamW/LambdaLR/clip as the pre-H23 trainer and produces
    identical weights, optimiser state and gradient norms after several steps."""
    new = _trainer(os.path.join(workdir, "new"))
    ref = _trainer(os.path.join(workdir, "ref"))
    # Re-create the pre-H23 construction verbatim on the reference trainer.
    m, cfg = ref.model, ref.cfg
    decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in m.named_parameters() if p.ndim < 2]
    ref.opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=cfg.lr, betas=cfg.betas)
    ref.sched = torch.optim.lr_scheduler.LambdaLR(ref.opt, lr_lambda(cfg))
    ref.clipper = lambda model: float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                      cfg.clip))
    ref.clipper.mode = "fixed"
    assert type(new.opt) is torch.optim.AdamW and new.clipper.mode == "fixed"
    assert new.opt.defaults == ref.opt.defaults
    norms_new, norms_ref = [], []
    it = FixedBatches(_examples(), batch_size=2, shuffle=False).iterate(new.pos)
    batches = [next(it) for _ in range(6)]  # the source repeats epochs forever
    for _, b in batches:  # same dropout draws for both: restore the global RNG before each
        rng = torch.get_rng_state()
        norms_new.append(new.train_step(b)["grad_norm"])
        torch.set_rng_state(rng)
        norms_ref.append(ref.train_step(b)["grad_norm"])
    assert norms_new == norms_ref
    _params_equal(new.model, ref.model)
    sn, sr = new.opt.state_dict()["state"], ref.opt.state_dict()["state"]
    for i in sr:
        assert torch.equal(sn[i]["exp_avg"], sr[i]["exp_avg"])
        assert torch.equal(sn[i]["exp_avg_sq"], sr[i]["exp_avg_sq"])
    assert new.sched.get_last_lr() == ref.sched.get_last_lr()
    assert "clipper" not in new.payload()


def test_default_losses_unchanged_without_zloss():
    torch.manual_seed(0)
    m = ASRModel(_mcfg())
    b = FixedBatches(_examples(), batch_size=3, shuffle=False)
    _, batch = next(iter(b.iterate(_pos())))
    args = (batch.feats, batch.feat_lens, batch.rnnt, batch.rnnt_lens, batch.ctc, batch.ctc_lens)
    m.eval()
    a = m.losses(*args)
    z = m.losses(*args, zloss_weight=0.0)
    assert torch.equal(a["loss"], z["loss"]) and "zloss" not in a and "zloss" not in z


def _pos():
    from train.data import Position

    return Position()


# ---- every optimiser trains and resumes bit-exactly ------------------------------------------
ARMS = [
    {"optimizer": "adamw"},
    {"optimizer": "scaled_adam", "lr": 0.045, "eden_steps_per_epoch": 3.0},
    {"optimizer": "schedulefree_adamw", "lr": 2e-3},
    {"optimizer": "muon"},
    {"optimizer": "adamw", "grad_clip_mode": "adaptive", "clip_window": 4},
    {"optimizer": "adamw", "grad_clip_mode": "agc"},
    {"optimizer": "adamw", "lookahead_k": 3},
    {"optimizer": "adamw", "zloss": 1e-3},
]


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: "-".join(f"{k}={v}" for k, v in a.items()))
def test_arm_trains_and_resumes_bit_exact(workdir, arm):
    a = _trainer(os.path.join(workdir, "a"), **arm)
    before = {k: v.clone() for k, v in a.model.state_dict().items()}
    a.run(max_steps=7)
    assert any(not torch.equal(before[k], v) for k, v in a.model.state_dict().items())
    b = _trainer(os.path.join(workdir, "b"), **arm)
    b.run(max_steps=4)
    c = _trainer(os.path.join(workdir, "b"), **arm)
    assert c.resume() and c.step == 4
    c.run(max_steps=3)
    _params_equal(a.model, c.model)
    assert a.sched.state_dict() == c.sched.state_dict()
    assert math.isfinite(a.loss_ema)


def test_unknown_options_rejected(workdir):
    with pytest.raises(OptimConfigError, match="unknown optimizer"):
        _trainer(workdir, optimizer="sgdx")
    with pytest.raises(OptimConfigError, match="grad_clip_mode"):
        _trainer(workdir, grad_clip_mode="soft")
    with pytest.raises(OptimConfigError, match="lookahead"):
        _trainer(workdir, optimizer="schedulefree_adamw", lookahead_k=2)


# ---- ScaledAdam + Eden ----------------------------------------------------------------------
def test_scaled_adam_step_is_relative_to_parameter_rms():
    torch.manual_seed(0)
    small = torch.nn.Parameter(torch.randn(8, 8) * 0.01)
    big = torch.nn.Parameter(small.detach() * 100)
    opt = ScaledAdam([small, big], lr=0.01, scalar_lr_scale=0.0)  # isolate the direction term
    g = torch.randn(8, 8)
    small.grad, big.grad = g.clone(), g.clone()
    s0, b0 = small.detach().clone(), big.detach().clone()
    opt.step()
    ds, db = (small.detach() - s0).norm(), (big.detach() - b0).norm()
    assert torch.allclose(db / ds, torch.tensor(100.0), rtol=1e-3)
    # First Adam step has |m / sqrt(v)| = (1 - b1) / sqrt(1 - b2) * sqrt(1 - b2)... per element ~0.1
    rel = ds / s0.norm()
    assert 1e-4 < float(rel) < 1e-2


def test_scaled_adam_learns_scale_and_respects_ceiling():
    p = torch.nn.Parameter(torch.full((4, 4), 4.0))  # RMS 4 > param_max_rms 3
    opt = ScaledAdam([p], lr=0.01, param_max_rms=3.0)
    for _ in range(5):
        p.grad = -p.detach().clone()  # gradient pushes the scale up
        opt.step()
    assert float(p.detach().pow(2).mean().sqrt()) <= 4.0 * (1 + 0.01 * 3 * 5) + 1e-6
    q = torch.nn.Parameter(torch.full((4, 4), 0.5))
    opt = ScaledAdam([q], lr=0.01)
    for _ in range(20):
        q.grad = q.detach().clone()  # loss grows with the scale -> scale should shrink
        opt.step()
    assert float(q.detach().abs().mean()) < 0.5


def test_eden_shape():
    f = eden_lambda(base_warmup=500, lr_batches=7500, lr_epochs=3.5, steps_per_epoch=0)
    assert f(0) == pytest.approx(0.5)
    assert f(500) == pytest.approx((1 + (500 / 7500) ** 2) ** -0.25)
    assert f(7500) == pytest.approx(2 ** -0.25)
    assert f(1000) > f(5000) > f(50000) > 0
    g = eden_lambda(base_warmup=0, lr_batches=7500, lr_epochs=3.5, steps_per_epoch=1000)
    assert g(3500) == pytest.approx(f(3500) * 2 ** -0.25)


# ---- Schedule-Free ---------------------------------------------------------------------------
def test_schedulefree_eval_train_switch_and_convergence():
    torch.manual_seed(0)
    target = torch.randn(16)
    p = torch.nn.Parameter(torch.zeros(16))
    opt = ScheduleFreeAdamW([p], lr=0.05, betas=(0.9, 0.999))
    for _ in range(400):
        p.grad = 2 * (p.detach() - target)
        opt.step()
    y = p.detach().clone()
    opt.eval()
    x = p.detach().clone()
    assert torch.equal(x, opt.state[p]["x"])
    assert float((x - target).norm()) < 0.05 * float(target.norm())
    opt.eval()  # idempotent
    assert torch.equal(p.detach(), x)
    with pytest.raises(OptimConfigError, match="eval mode"):
        opt.step()
    opt.train()
    assert torch.allclose(p.detach(), y, atol=1e-6)


def test_eval_weights_swaps_only_for_schedulefree(workdir):
    t = _trainer(workdir, optimizer="schedulefree_adamw", lr=2e-3)
    t.run(max_steps=3)
    y = {k: v.clone() for k, v in t.model.state_dict().items()}
    with eval_weights(t.opt):
        x = {k: v.clone() for k, v in t.model.state_dict().items()}
    assert any(not torch.equal(y[k], x[k]) for k in y)
    _ = [torch.testing.assert_close(t.model.state_dict()[k], y[k]) for k in y]
    a = _trainer(os.path.join(workdir, "adamw"))
    s = {k: v.clone() for k, v in a.model.state_dict().items()}
    with eval_weights(a.opt):
        _params_equal_dict(a.model.state_dict(), s)


def _params_equal_dict(a, b):
    for k in a:
        assert torch.equal(a[k], b[k]), k


# ---- Muon ------------------------------------------------------------------------------------
def test_newton_schulz_orthogonalises():
    torch.manual_seed(0)
    for shape in ((24, 8), (8, 24), (16, 16)):
        o = newton_schulz(torch.randn(*shape))
        s = torch.linalg.svdvals(o)
        # NS5 with Jordan's coefficients is deliberately inexact: most singular values land in
        # ~[0.7, 1.2]; the very smallest of a random square matrix may stay lower.
        assert float(s.max()) < 1.3 and float(s.min()) > 0.3, (shape, s)
        assert float((s > 0.6).float().mean()) >= 0.9, (shape, s)


def test_muon_parameter_split():
    m = ASRModel(_mcfg())
    muon = {n for n, p in m.named_parameters() if is_muon_param(n, p)}
    assert any(".att.qkv.weight" in n for n in muon) and any(".ff1.w1.weight" in n for n in muon)
    for n in muon:
        assert n.startswith("encoder.blocks.") or n.startswith("encoder.") and "stem" not in n
        assert not n.endswith(("bias",)) and "norm" not in n
    for bad in ("joiner.out.weight", "joiner.enc_proj.weight", "ctc_head.weight",
                "predictor.emb.weight", "encoder.stem.out.weight"):
        assert bad not in muon
    cfg = TrainConfig(optimizer="muon")
    opt, _ = build_optimizer(m, cfg, lr_lambda)
    assert isinstance(opt, Muon)
    in_groups = sum(len(g["params"]) for g in opt.param_groups)
    assert in_groups == len(list(m.parameters()))
    assert [g["use_muon"] for g in opt.param_groups] == [True, False, False]


def test_muon_update_rms_matches_adamw_scale():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(64, 32))
    opt = Muon([{"params": [p], "use_muon": True}], lr=1.0, momentum=0.0)
    p.grad = torch.randn(64, 32)
    opt.step()
    rms = float(p.detach().pow(2).mean().sqrt())
    assert 0.12 < rms < 0.3  # 0.2 x sqrt(64) x RMS(orthogonal 64x32 ~ 1/sqrt(64))


# ---- clipping --------------------------------------------------------------------------------
def _toy(grads):
    m = torch.nn.Linear(4, 3)
    m.weight.grad, m.bias.grad = grads
    return m


def test_fixed_clip_equals_clip_grad_norm():
    g = (torch.randn(3, 4) * 10, torch.randn(3) * 10)
    a, b = _toy((g[0].clone(), g[1].clone())), _toy((g[0].clone(), g[1].clone()))
    b.load_state_dict(a.state_dict())
    n1 = GradClipper("fixed", 5.0)(a)
    n2 = float(torch.nn.utils.clip_grad_norm_(b.parameters(), 5.0))
    assert n1 == n2 and torch.equal(a.weight.grad, b.weight.grad)


def test_adaptive_clip_uses_running_median():
    c = GradClipper("adaptive", clip=100.0, k=2.0, window=8)
    for v in (1.0, 1.0, 2.0):  # fewer than window // 4 = 2 ... third makes 3
        m = _toy((torch.full((3, 4), v), torch.zeros(3)))
        c(m)
    assert c.threshold() == pytest.approx(2.0 * 1.0 * math.sqrt(12))
    spike = _toy((torch.full((3, 4), 50.0), torch.zeros(3)))
    pre = c(spike)
    post = float(torch.linalg.vector_norm(spike.weight.grad))
    assert pre == pytest.approx(50 * math.sqrt(12))
    assert post == pytest.approx(c.last_threshold, rel=1e-4) and c.clipped == 1
    s = c.state_dict()
    d = GradClipper("adaptive", clip=100.0, k=2.0, window=8)
    d.load_state_dict(s)
    assert d.threshold() == c.threshold()


def test_agc_clips_unitwise_and_exempts_output_layers():
    torch.manual_seed(0)
    m = ASRModel(_mcfg())
    for p in m.parameters():
        p.grad = torch.ones_like(p) * 1e3
    GradClipper("agc", clip=1e9, agc_lambda=0.01)(m)
    w = m.encoder.blocks[0].ff1.w1.weight
    ratio = w.grad.norm(dim=1) / w.detach().norm(dim=1).clamp(min=1e-3)
    assert float(ratio.max()) <= 0.01 * (1 + 1e-4)
    assert torch.equal(m.joiner.out.weight.grad, torch.ones_like(m.joiner.out.weight) * 1e3)


# ---- Lookahead -------------------------------------------------------------------------------
def test_lookahead_interpolates_every_k_steps():
    p = torch.nn.Parameter(torch.zeros(3))
    la = Lookahead(torch.optim.SGD([p], lr=1.0), k=2, alpha=0.5)
    for _ in range(2):
        p.grad = -torch.ones(3)
        la.step()
    assert torch.allclose(p.detach(), torch.ones(3))  # fast reached 2, slow -> 0 + 0.5 * 2
    sd = la.state_dict()
    q = torch.nn.Parameter(torch.zeros(3))
    lb = Lookahead(torch.optim.SGD([q], lr=1.0), k=2, alpha=0.5)
    lb.load_state_dict(sd)
    assert lb.counter == 2 and torch.equal(lb.slow[0], la.slow[0])


# ---- z-loss ----------------------------------------------------------------------------------
def test_diagonal_zloss_matches_manual_and_trains_joiner():
    torch.manual_seed(0)
    m = ASRModel(_mcfg())
    enc_p = torch.randn(2, 5, 16)
    pred_p = torch.randn(2, 4, 16)
    enc_lens, tgt_lens = torch.tensor([5, 3]), torch.tensor([3, 2])
    z = diagonal_zloss(m.joiner, enc_p, pred_p, enc_lens, tgt_lens)
    terms = []
    for b in range(2):
        for t in range(int(enc_lens[b])):
            u = min(t * (int(tgt_lens[b]) + 1) // int(enc_lens[b]), int(tgt_lens[b]))
            terms.append(torch.logsumexp(m.joiner.logits(enc_p[b, t], pred_p[b, u]), -1) ** 2)
    torch.testing.assert_close(z, torch.stack(terms).mean())
    z.backward()
    assert m.joiner.out.bias.grad is not None and float(m.joiner.out.bias.grad.abs().sum()) > 0


def test_band_zloss_masks_padded_frames():
    logits = torch.randn(2, 4, 3, 5)
    lens = torch.tensor([4, 2])
    logits[1, 2:] = 1e3  # padding garbage must not count
    z = band_zloss(logits, lens)
    lse = torch.logsumexp(logits, -1)
    ref = torch.cat([lse[0].flatten(), lse[1, :2].flatten()]).pow(2).mean()
    torch.testing.assert_close(z, ref)


def test_trainer_logs_zloss(workdir):
    t = _trainer(workdir, zloss=1e-3)
    _, batch = next(iter(t.source.iterate(t.pos)))
    r = t.train_step(batch)
    assert "zloss" in r and r["zloss"] > 0


def test_pruned_band_zloss_or_requires_k2(workdir):
    """With k2 the pruned path adds the band z-loss; without it the option refuses clearly."""
    t = _trainer(workdir, rnnt_loss="pruned", zloss=1e-3)
    _, batch = next(iter(t.source.iterate(t.pos)))
    if not HAVE_K2:
        with pytest.raises(ImportError, match="needs k2"):
            t.train_step(batch)
        return
    r = t.train_step(batch)
    assert r["zloss"] > 0 and math.isfinite(r["loss"])


def test_ema_eval_scores_ema_weights_not_schedulefree_x(workdir):
    """The EMA dev eval must see the EMA weights; the main eval sees Schedule-Free's x; the live
    y weights are back afterwards."""
    t = _trainer(workdir, optimizer="schedulefree_adamw", lr=2e-3, ema_decay=0.5)
    t.run(max_steps=3)
    t.dev, t.tok = _examples()[:1], object()
    seen = []
    t._evaluate = lambda max_batch_s: seen.append(
        t.model.joiner.out.weight.detach().clone()) or {"wer": 1.0}
    y = t.model.joiner.out.weight.detach().clone()
    t.evaluate()
    t._evaluate_ema()
    sf = t.opt
    x = sf.state[t.model.joiner.out.weight]["x"]
    assert torch.equal(seen[0], x)
    assert torch.equal(seen[1], t.ema["joiner.out.weight"])
    torch.testing.assert_close(t.model.joiner.out.weight.detach(), y)


def test_zloss_survives_the_ddp_loss_rebuild(workdir):
    """Under DDP the trainer rebuilds the loss from its parts (_global_loss); the z-loss term
    and its gradient must be part of it (single-process gloo group)."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        t = _trainer(workdir, zloss=0.5)
        _, batch = next(iter(t.source.iterate(t.pos)))
        out = t.model.losses(batch.feats, batch.feat_lens, batch.rnnt, batch.rnnt_lens, batch.ctc,
                             batch.ctc_lens, zloss_weight=0.5)
        loss, _ = t._global_loss(out)
        torch.testing.assert_close(loss, out["loss"])
        assert out["zloss"].requires_grad
    finally:
        dist.destroy_process_group()
