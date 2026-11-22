"""Two-view CR-CTC training step (H41): one optimiser step over two SpecAugment views, full form
(CTC + transducer on both views) or half form (transducer on view A only).

Kept out of trainer.py; `self` is the Trainer.
"""
from __future__ import annotations

import math

import torch


def ctc_view(self, x, b, chunk, cr_target) -> dict:
    """Encoder and CTC head only, on one already augmented view (the half form's second view):
    its CTC loss and, given the other view's detached posteriors, the consistency KL."""
    from train.ctc import ctc_loss

    m = self.model
    enc, enc_lens = m.encoder(x, b.feat_lens, chunk)
    with torch.autocast(device_type=enc.device.type, enabled=False):
        lp = torch.log_softmax(m.ctc_head(enc.float()).float(), dim=-1)
        total, tokens, _ = ctc_loss(lp, enc_lens, b.ctc, b.ctc_lens, None)
        ctc = total / max(1, tokens)
        loss, cr = m.cfg.ctc_weight * ctc, lp.new_zeros(())
        if cr_target is not None:
            kl = (cr_target.exp() * (cr_target - lp)).sum(-1)
            valid = torch.arange(lp.shape[1], device=lp.device)[None] < enc_lens[:, None]
            cr = (kl * valid).sum() / valid.sum().clamp(min=1)
            loss = loss + self.cfg.cr_ctc_weight * cr
    return {"loss": loss, "ctc": ctc, "cr_ctc": cr, "ctc_lp": lp.detach()}


def two_view_step(self, batch, b, gen: torch.Generator) -> dict:
    """One optimiser step of full-form CR-CTC (H41): two SpecAugment views of the batch, CTC and
    transducer loss on both, and the symmetric consistency KL between their CTC posteriors.
    The KL's targets are detached, so the loss is one term per view: the views run one after the
    other and their gradients accumulate, which keeps memory at one view's graph (a 650 s batch
    with both graphs alive does not fit 24 GB). View A's masks are those of the single-view
    step. View B's dropout draws are replayed, so its target pass and its training pass see
    the same network."""
    from train.asr import spec_augment

    c, m = self.cfg, self.model
    if self.world > 1 or c.kd_weight > 0 or c.zloss:
        raise ValueError("cr_ctc_full runs single-process, without KD or z-loss")
    aug = self._step_augment()
    with torch.no_grad():
        x_norm = m.cmvn(b.feats)
        xa = spec_augment(x_norm, b.feat_lens, aug, gen) if aug is not None else x_norm
        xb = spec_augment(x_norm, b.feat_lens, aug, gen) if aug is not None else x_norm
    args = (b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.ctc, b.ctc_lens)
    kw = {"chunk": self._step_chunk(), "pruned_scale": self._pruned_scale(),
          "simple_detach": self.step < c.prune_hold, "cr_ctc_weight": c.cr_ctc_weight,
          **({"delay_penalty": c.delay_penalty} if c.delay_penalty else {})}
    cuda = self.device.type == "cuda"
    ac = {"device_type": self.device.type, "dtype": torch.bfloat16, "enabled": cuda}

    def rng(state=None):
        if state is None:
            return torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()
        return torch.cuda.set_rng_state(state, self.device) if cuda else torch.set_rng_state(state)

    half = c.cr_ctc_half and not c.cr_ctc_full  # transducer loss on view A only
    s0 = rng()
    with torch.no_grad(), torch.autocast(**ac):  # view B's posteriors: view A's target
        lp_b = (ctc_view(self, xb, b, kw["chunk"], None) if half else m.losses(*args, view=xb, **kw))["ctc_lp"]
    self.opt.zero_grad(set_to_none=True)
    with torch.autocast(**ac):
        out_a = m.losses(*args, view=xa, cr_target=lp_b, stats=self.stats, **kw)
    wa = m.cfg.ctc_weight * out_a["ctc"] + c.cr_ctc_weight * out_a["cr_ctc"]  # view A's CTC + KL part
    ((out_a["loss"] - 0.5 * wa) if half else 0.5 * out_a["loss"]).backward()
    s2 = rng()
    rng(s0)  # replay view B's dropout draws
    with torch.autocast(**ac):
        out_b = (ctc_view(self, xb, b, kw["chunk"], out_a["ctc_lp"]) if half
                 else m.losses(*args, view=xb, cr_target=out_a["ctc_lp"], **kw))
    (0.5 * out_b["loss"]).backward()
    rng(s2)  # continue the stream after everything drawn in this step

    def mean(k):
        return 0.5 * (out_a[k].detach() + out_b[k].detach())

    def one(k):  # half form: the transducer terms come from view A alone
        return out_a[k].detach() if half else mean(k)

    loss = (out_a["loss"] - 0.5 * wa + 0.5 * out_b["loss"]).detach() if half else mean("loss")
    out = {"rnnt": one("rnnt"), "ctc": mean("ctc"), "cr_ctc": mean("cr_ctc"), "nll": out_a["nll"],
           "ctc_masked": out_a["ctc_masked"],
           **({"rnnt_simple": one("rnnt_simple")} if "rnnt_simple" in out_a else {})}
    if not math.isfinite(float(loss)):
        self._pause(batch, out, float("nan"), "loss")
    gnorm = self.clipper(self.model)
    if not math.isfinite(gnorm):
        self._pause(batch, out, gnorm, "gradient norm")
    self.opt.step()
    self._qkv_constraint()
    self._ema_update()
    self.sched.step()
    self.step += 1
    return {"loss": float(loss), "rnnt": float(out["rnnt"]), "ctc": float(out["ctc"]),
            "grad_norm": gnorm, "ctc_masked": out["ctc_masked"], "lr": self.sched.get_last_lr()[0],
            "audio_s": batch.audio_seconds, "utts": len(batch.uids), "cr_ctc": float(out["cr_ctc"]),
            **({"rnnt_simple": float(out["rnnt_simple"])} if "rnnt_simple" in out else {})}
