"""Train the DENSE reference student (spec v1.0-B §17, §26) on CPU or GPU.

  python -m train.trainer --run-dir /mnt/xbabe3-external/jsr/runs/train/<name> --preset S \
      --splits train-clean-100 --threads 32

Resumes automatically from `<run-dir>/LATEST`. Logs JSON lines to `<run-dir>/train.jsonl`.
A non-finite loss or gradient pauses the run: diagnostics go to `<run-dir>/PAUSED.json` and the
process exits with status 3; a paused run refuses to resume until that file is removed.
Optimiser: AdamW with linear warm-up and cosine decay; bf16 autocast on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field

import torch

from train import checkpoint
from train.asr import PRESETS, ASRModel, ModelConfig, SpecAugConfig
from train.ctc import FeasibilityStats
from train.data import Batch, Example, FixedBatches, Position, collate
from train.errors import CheckpointError, DivergenceError, NonFiniteLossError, ShapeError

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")


@dataclass
class TrainConfig:
    preset: str = "S"
    lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    warmup: int = 1500
    total_steps: int = 60000
    min_lr_ratio: float = 0.02
    clip: float = 5.0
    seed: int = 1234
    budget_s: float = 200.0
    max_batch: int = 64
    max_seconds: float = 20.0
    splits: tuple[str, ...] = ("train-clean-100",)
    specaug: bool = True
    # SpecAugment off for the first N optimiser steps. From step 0 it held the S model on the
    # unigram plateau (D1c, 2026-11-20: CTC 5.9/token flat to step 2250; without it the same
    # model left the plateau by step ~900).
    specaug_start: int = 4000
    log_every: int = 10
    ckpt_every_steps: int = 500
    ckpt_every_minutes: float = 30.0
    eval_every: int = 2000
    dev_split: str = "dev-clean"
    dev_limit: int = 200
    chunk: int = 0  # 0 = model default
    workers: int = 4  # data-decoding threads
    eval_each_epoch: bool = True
    # Divergence guard (D3MR 2026-11-21 diverged at lr 8e-4: attention qkv weights of one block
    # grew 10x, loss rose for ~3k steps, then the model collapsed). The run stops, writes
    # DIVERGED.json and refuses to resume when, after `diverge_after` steps, the train-loss EMA
    # stays above (1 + diverge_ratio) x its best value for `diverge_patience` steps, or a dev
    # eval is worse than the best one by more than `diverge_wer_points`. 0 disables a rule.
    diverge_after: int = 3000
    diverge_ratio: float = 0.3
    diverge_patience: int = 300
    diverge_wer_points: float = 5.0
    # Attention-weight constraint (D3MR/D3MRb: one block's qkv norm ran away, 70 -> 426, until
    # the model collapsed). From step qkv_cap_from, every row of each attention qkv weight and
    # every element of its pre-attention LayerNorm gain is held at <= qkv_cap_ratio x its value
    # at that step (projection after each optimiser step; the inference graph is unchanged).
    # qkv_guard_ratio: stop as diverged if max_qkv_norm exceeds that multiple of its value at
    # qkv_cap_from. 0 disables each.
    # Weight EMA (training-only; 0 disables): ema = d*ema + (1-d)*weights after every optimiser
    # step. Saved in checkpoints and in the eval-only model-<step>.pt snapshots under "ema";
    # score it with `train.evaluate --ema`.
    ema_decay: float = 0.0
    # SentencePiece sizes (pieces, excluding blank) of the RNN-T and CTC heads; the model's output
    # sizes follow (pieces + 1). H13 shares one 512-piece model between both heads.
    rnnt_pieces: int = 4096
    ctc_pieces: int = 1024
    # RNN-T loss (ModelConfig.rnnt_loss): "exact" or "pruned" (k2; H13b), band width, and a linear
    # ramp of the pruned-loss weight from 0.1 to 1 over the first prune_warmup steps while the
    # simple joiner that picks the band is still untrained (0 = weight 1 from the start).
    # H22 distillation from R0: cache dir of train/teacher_cache.py outputs and the KL weight;
    # tokenizer overrides both heads' SentencePiece model path (R0's bpe.model for H22).
    # Batch loader: "thread" (decode in a thread pool; GIL-bound) or "process" (spawned worker
    # processes running the same decode/perturb/log-mel code; bit-identical batches, test).
    loader: str = "thread"
    # Row groups sorted together per window before packing (more = less padding per batch).
    window_groups: int = 4  # a change takes effect at the next epoch boundary on resume
    kd_dir: str = ""
    kd_weight: float = 0.0
    tokenizer: str = ""
    rnnt_loss: str = "exact"
    prune_range: int = 5
    prune_warmup: int = 0
    qkv_cap_from: int = 0
    qkv_cap_ratio: float = 1.5
    qkv_guard_ratio: float = 0.0
    speed_perturb: str = ""
    pad_frames: int = 0  # round batch time axis up to this multiple (MPS: fewer shapes)
    pad_tokens: int = 0  # round target axes up to this multiple  # e.g. "0.9,1.0,1.1" (train batches only)
    weights_every: int = 0  # >0: also keep an eval-only model-<step>.pt every N steps (not rotated)
    extra: dict = field(default_factory=dict)


def lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        p = min(1.0, (step - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


class Trainer:
    def __init__(self, run_dir: str, cfg: TrainConfig, source, model_cfg: ModelConfig | None = None,
                 dev: list[Example] | None = None, tokenizer=None, device: str | None = None,
                 log: bool = True):
        self.run_dir, self.cfg, self.source, self.dev, self.tok = run_dir, cfg, source, dev, tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_cfg = model_cfg or PRESETS[cfg.preset]
        torch.manual_seed(cfg.seed)
        self.model = ASRModel(self.model_cfg).to(self.device)
        decay = [p for n, p in self.model.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in self.model.named_parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                                      {"params": no_decay, "weight_decay": 0.0}],
                                     lr=cfg.lr, betas=cfg.betas)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg))
        self.step = 0
        self.pos = Position()
        self.stats = FeasibilityStats()
        self.augment = SpecAugConfig(enabled=cfg.specaug)
        self.log_enabled = log
        self.stop_requested = False
        self.saver = checkpoint.AsyncSaver()
        self.loss_ema: float | None = None
        self.best_ema: float | None = None
        self.over_since: int | None = None
        self.best_wer: float | None = None
        self.qkv_caps: dict | None = None
        self.qkv_ref_norm: float | None = None
        self.ema: dict | None = None
        self.precision = "bf16-autocast" if self.device.type == "cuda" else "fp32"
        os.makedirs(run_dir, exist_ok=True)

    # ---- state -------------------------------------------------------------------------------
    def payload(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(), "position": self.pos.to_dict(),
                "precision": self.precision, "train_config": asdict(self.cfg),
                "model_config": self.model_cfg.to_dict(),
                "manifest": list(getattr(self.source, "manifest", [])),
                "tokenizer_sha256": getattr(self.tok, "sha256", None),
                "ctc_stats": self.stats.summary(),
                "qkv_caps": self.qkv_caps, "qkv_ref_norm": self.qkv_ref_norm, "ema": self.ema}

    def save(self, wait: bool = False) -> None:
        """Snapshot now; the write finishes in the background unless `wait`."""
        self.saver.save(self.run_dir, self.step, self.payload())
        if wait:
            self.saver.wait()

    def resume(self) -> bool:
        for stop in ("PAUSED.json", "DIVERGED.json"):
            if os.path.exists(os.path.join(self.run_dir, stop)):
                raise CheckpointError(f"run is stopped; inspect and remove {self.run_dir}/{stop}")
        path = checkpoint.latest(self.run_dir)
        if path is None:
            return False
        p = checkpoint.load(path)
        if ModelConfig.from_dict(p["model_config"]) != self.model_cfg:
            raise CheckpointError("checkpoint model config differs from the requested one")
        manifest = list(getattr(self.source, "manifest", []))
        if p["manifest"] != manifest:
            raise CheckpointError("data manifest changed since the checkpoint; refusing to resume")
        self.model.load_state_dict(p["model"])
        self.opt.load_state_dict(p["optimizer"])
        self.sched.load_state_dict(p["scheduler"])
        self.pos = Position(**p["position"])
        old_wg = int((p.get("train_config") or {}).get("window_groups", 4))
        if old_wg != self.cfg.window_groups and hasattr(self.source, "carry_width"):
            # the new width starts at the next epoch; this epoch finishes with the old one
            self.source.carry_width = (old_wg, self.pos.epoch)
        self.step = p["step"]
        self.qkv_caps = p.get("qkv_caps")
        self.qkv_ref_norm = p.get("qkv_ref_norm")
        self.ema = ({k: v.to(self.device) for k, v in p["ema"].items()}
                    if p.get("ema") is not None else None)
        return True

    def init_cmvn(self, n_batches: int = 8) -> None:
        feats = []
        for i, (_, b) in enumerate(self.source.iterate(Position())):
            feats.extend(b.feats[j, :b.feat_lens[j]] for j in range(len(b.uids)))
            if i + 1 >= n_batches:
                break
        self.model.cmvn.set_stats(feats)

    # ---- one optimiser step ------------------------------------------------------------------
    def train_step(self, batch: Batch) -> dict:
        self.model.train()
        b = batch.to(self.device)
        gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + self.step)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            out = self.model.losses(b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.ctc, b.ctc_lens,
                                    stats=self.stats,
                                    augment=self.augment if self.step >= self.cfg.specaug_start
                                    else None, gen=gen,
                                    chunk=self.cfg.chunk or None,
                                    pruned_scale=self._pruned_scale(),
                                    kd=b.kd, kd_weight=self.cfg.kd_weight)
        loss = out["loss"]
        if not torch.isfinite(loss):
            self._pause(batch, out, float("nan"), "loss")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip))
        if not math.isfinite(gnorm):
            self._pause(batch, out, gnorm, "gradient norm")
        self.opt.step()
        self._qkv_constraint()
        self._ema_update()
        self.sched.step()
        self.step += 1
        return {"loss": float(loss), "rnnt": float(out["rnnt"]), "ctc": float(out["ctc"]),
                "grad_norm": gnorm, "ctc_masked": out["ctc_masked"],
                "lr": self.sched.get_last_lr()[0], "audio_s": batch.audio_seconds,
                "utts": len(batch.uids),
                **({"rnnt_simple": float(out["rnnt_simple"])} if "rnnt_simple" in out else {}),
                **({"kd": float(out["kd"]), "kd_frames": out["kd_frames"]} if "kd" in out else {})}

    def _diverged(self, detail: str, extra: dict) -> None:
        diag = os.path.join(self.run_dir, "DIVERGED.json")
        with open(diag, "w") as fh:
            json.dump({"step": self.step, "detail": detail, "loss_ema": self.loss_ema,
                       "best_ema": self.best_ema, "best_wer": self.best_wer,
                       "position": self.pos.to_dict(), **extra}, fh, indent=1)
        self.save(wait=True)  # keep the state at the stop for inspection
        raise DivergenceError(self.step, detail, diag)

    def _pruned_scale(self) -> float:
        w = self.cfg.prune_warmup
        return 1.0 if not w else min(1.0, 0.1 + 0.9 * self.step / w)

    @torch.no_grad()
    def _ema_update(self) -> None:
        d = self.cfg.ema_decay
        if not d:
            return
        state = self.model.state_dict()
        if self.ema is None:
            self.ema = {k: v.detach().clone() for k, v in state.items()}
            return
        for k, v in state.items():
            if v.is_floating_point():
                self.ema[k].lerp_(v, 1.0 - d)
            else:
                self.ema[k].copy_(v)

    def _attention_modules(self):
        return [(n, m) for n, m in self.model.named_modules() if hasattr(m, "qkv")]

    @torch.no_grad()
    def _qkv_constraint(self) -> None:
        c = self.cfg
        if not c.qkv_cap_from or self.step + 1 < c.qkv_cap_from:
            return
        if self.qkv_caps is None:  # record the reference at the first eligible step
            self.qkv_caps = {n: {"rows": m.qkv.weight.norm(dim=1).cpu() * c.qkv_cap_ratio,
                                 "gain": m.norm.weight.abs().cpu() * c.qkv_cap_ratio}
                             for n, m in self._attention_modules()}
            self.qkv_ref_norm = self._max_qkv_norm()
            return
        for n, m in self._attention_modules():
            cap = self.qkv_caps[n]
            w = m.qkv.weight
            rows = w.norm(dim=1)
            scale = torch.clamp(cap["rows"].to(w.device) / rows.clamp(min=1e-12), max=1.0)
            w.mul_(scale[:, None])
            g = m.norm.weight
            lim = cap["gain"].to(g.device)
            g.copy_(torch.maximum(torch.minimum(g, lim), -lim))
        if c.qkv_guard_ratio and self.qkv_ref_norm:
            cur = self._max_qkv_norm()
            if cur > c.qkv_guard_ratio * self.qkv_ref_norm:
                self._diverged(f"max_qkv_norm {cur} > {c.qkv_guard_ratio} x {self.qkv_ref_norm} "
                               f"(its value at step {c.qkv_cap_from})", {"max_qkv_norm": cur})

    def _max_qkv_norm(self) -> float:
        """Largest attention qkv weight norm (D3MR's divergence showed up here first)."""
        norms = [float(m.qkv.weight.detach().norm()) for m in self.model.modules()
                 if hasattr(m, "qkv")]
        return round(max(norms), 2) if norms else 0.0

    def _watch_loss(self, r: dict) -> None:
        c = self.cfg
        x = r["loss"]
        self.loss_ema = x if self.loss_ema is None else 0.99 * self.loss_ema + 0.01 * x
        if not c.diverge_ratio or self.step < c.diverge_after:
            return
        if self.best_ema is None or self.loss_ema < self.best_ema:
            self.best_ema = self.loss_ema
        if self.loss_ema > (1 + c.diverge_ratio) * self.best_ema:
            self.over_since = self.over_since or self.step
            if self.step - self.over_since >= c.diverge_patience:
                self._diverged(f"train-loss EMA {self.loss_ema:.3f} > (1 + {c.diverge_ratio}) x "
                               f"best {self.best_ema:.3f} for {c.diverge_patience} steps",
                               {"grad_norm": r["grad_norm"]})
        else:
            self.over_since = None

    def _watch_wer(self, wer: float | None) -> None:
        if wer is None or not self.cfg.diverge_wer_points:
            return
        if self.best_wer is not None and wer > self.best_wer + self.cfg.diverge_wer_points:
            self._diverged(f"dev WER {wer} > best {self.best_wer} + {self.cfg.diverge_wer_points}",
                           {"wer": wer})
        self.best_wer = wer if self.best_wer is None else min(self.best_wer, wer)

    def _pause(self, batch: Batch, out: dict, gnorm: float, what: str):
        diag = os.path.join(self.run_dir, "PAUSED.json")
        info = {"step": self.step, "what": what, "uids": batch.uids, "grad_norm": gnorm,
                "rnnt": float(out["rnnt"]), "ctc": float(out["ctc"]),
                "nll": out["nll"].tolist(), "position": self.pos.to_dict()}
        with open(diag, "w") as fh:
            json.dump(info, fh, indent=1)
        raise NonFiniteLossError(self.step, what, diag)

    # ---- evaluation ----------------------------------------------------------------------------
    def evaluate(self, max_batch_s: float = 300.0) -> dict:
        from ultra.wer import corpus_wer

        if not self.dev or self.tok is None:
            return {}
        self.model.eval()
        exs = sorted(self.dev, key=lambda e: e.seconds)
        refs, hyps, forced = [], [], 0
        i = 0
        while i < len(exs):
            j = i + 1
            while j < len(exs) and exs[j].seconds * (j - i + 1) <= max_batch_s:
                j += 1
            b = collate(exs[i:j]).to(self.device)
            toks, f = self.model.greedy(b.feats, b.feat_lens, self.cfg.chunk or None)
            forced += f
            refs.extend(b.texts)
            hyps.extend(self.tok.decode(t) for t in toks)
            i = j
        w = corpus_wer(refs, hyps)
        w["forced_advances"] = forced
        w["examples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps, strict=True))[:3]]
        return w

    @torch.no_grad()
    def _evaluate_ema(self) -> dict:
        """Dev WER of the EMA weights (swapped in, then the live weights restored)."""
        if self.ema is None or not self.dev or self.tok is None:
            return {}
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema)
        try:
            ev = self.evaluate()
        finally:
            self.model.load_state_dict(live)
        return {"ema_wer": ev.get("wer"), "ema_sub": ev.get("sub"), "ema_del": ev.get("del"),
                "ema_ins": ev.get("ins")}

    # ---- loop --------------------------------------------------------------------------------
    def log(self, rec: dict) -> None:
        rec = dict(rec, time=time.time())
        with open(os.path.join(self.run_dir, "train.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self.log_enabled:
            print(json.dumps(rec), flush=True)

    def run(self, max_steps: int | None = None) -> None:
        last_ckpt = time.time()
        acc = {"audio_s": 0.0, "loss": 0.0, "rnnt": 0.0, "ctc": 0.0, "n": 0, "masked": 0}
        t0 = time.time()
        steps_here = 0
        t_wait = time.time()
        for pos, batch in self.source.iterate(self.pos):
            acc["data_wait_s"] = acc.get("data_wait_s", 0.0) + (time.time() - t_wait)
            acc["frames"] = acc.get("frames", 0) + int(batch.feat_lens.sum())
            acc["padded"] = acc.get("padded", 0) + batch.feats.shape[0] * batch.feats.shape[1]
            r = self.train_step(batch)
            new_epoch = pos.epoch > self.pos.epoch
            self.pos = pos
            steps_here += 1
            acc["audio_s"] += r["audio_s"]
            acc["n"] += 1
            acc["masked"] += r["ctc_masked"]
            for k in ("loss", "rnnt", "ctc", "rnnt_simple", "kd"):
                if k in r:
                    acc[k] = acc.get(k, 0.0) + r[k]
            self._watch_loss(r)
            if self.step % self.cfg.log_every == 0:
                dt = time.time() - t0
                self.log({"kind": "train", "step": self.step, "pos": pos.to_dict(),
                          "loss": acc["loss"] / acc["n"], "rnnt": acc["rnnt"] / acc["n"],
                          "ctc": acc["ctc"] / acc["n"], "lr": r["lr"], "grad_norm": r["grad_norm"],
                          "ctc_masked": acc["masked"], "audio_s": acc["audio_s"], "wall_s": dt,
                          "dropped_long": getattr(self.source, "dropped_long", 0),
                          **({"rnnt_simple": acc["rnnt_simple"] / acc["n"]} if "rnnt_simple" in acc
                             else {}),
                          **({"kd": acc["kd"] / acc["n"]} if "kd" in acc else {}),
                          "max_qkv_norm": self._max_qkv_norm(), "loss_ema": self.loss_ema,
                          "audio_h_per_h": acc["audio_s"] / max(dt, 1e-9),
                          "data_wait_pct": round(100.0 * acc.get("data_wait_s", 0.0) / max(dt, 1e-9), 2),
                          "pad_pct": round(100.0 * (1 - acc.get("frames", 0) / max(1, acc.get("padded", 0))), 2)})
                acc = {k: 0.0 if isinstance(v, float) else 0 for k, v in acc.items()}
                t0 = time.time()
            epoch_end = new_epoch and self.cfg.eval_each_epoch and self.dev
            if epoch_end or (self.cfg.eval_every and self.step % self.cfg.eval_every == 0):
                ev = self.evaluate()
                ema = self._evaluate_ema()
                self.log({"kind": "eval", "step": self.step, "split": self.cfg.dev_split,
                          "epoch": pos.epoch, "epoch_end": bool(epoch_end),
                          **ev, **ema, **self.stats.summary()})
                self._watch_wer(ev.get("wer"))
                t0 = time.time()
            if self.cfg.weights_every and self.step % self.cfg.weights_every == 0:
                self.saver.save(self.run_dir, self.step,
                                {"model": self.model.state_dict(),
                                 "model_config": self.model_cfg.to_dict(),
                                 "train_config": asdict(self.cfg),
                                 "tokenizer_sha256": getattr(self.tok, "sha256", None),
                                 "ema": self.ema},
                                weights_only=True)
            due = (self.step % self.cfg.ckpt_every_steps == 0
                   or time.time() - last_ckpt > 60 * self.cfg.ckpt_every_minutes)
            if due or self.stop_requested:
                self.save()
                last_ckpt = time.time()
            if self.stop_requested or (max_steps and steps_here >= max_steps) \
                    or self.step >= self.cfg.total_steps:
                if not due and not self.stop_requested:
                    self.save()
                self.saver.wait()
                return
            t_wait = time.time()


def load_dev(split: str, limit: int, feat) -> list[Example]:
    from ultra.libri import load_split

    return [feat.example(u.uid, u.text, u.pcm) for u in load_split(split, limit)]


def _teacher(cfg: TrainConfig):
    if not cfg.kd_dir:
        return None
    from train.kd import TeacherStore

    store = TeacherStore(cfg.kd_dir)
    print(json.dumps({"kind": "teacher", "entries": len(store), "topk": store.topk}), flush=True)
    return store


def model_config_for(cfg: TrainConfig) -> ModelConfig:
    """The preset's model config with output heads sized to the configured tokenizers."""
    import dataclasses

    if cfg.rnnt_pieces < 2 or cfg.ctc_pieces < 2:
        raise ShapeError("rnnt_pieces and ctc_pieces must be >= 2")
    return dataclasses.replace(PRESETS[cfg.preset], vocab_rnnt=cfg.rnnt_pieces + 1,
                               vocab_ctc=cfg.ctc_pieces + 1, rnnt_loss=cfg.rnnt_loss,
                               prune_range=cfg.prune_range)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--preset", default="S", choices=sorted(PRESETS))
    ap.add_argument("--splits", nargs="+", default=["train-clean-100"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps this run")
    ap.add_argument("--overfit", type=int, default=0, help="D0: train on the first N utterances")
    ap.add_argument("--init-from", default=None,
                    help="fresh run only: start from this checkpoint's model weights (and CMVN)")
    ap.add_argument("--set", nargs="*", default=[], help="TrainConfig overrides key=value")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
    cfg = TrainConfig(preset=a.preset, splits=tuple(a.splits))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(json.loads(v)) if not isinstance(cur, str) else v)

    from train.data import Featurizer, ParquetBatches
    from train.tokenizer import Tokenizer, tokenizer_path

    tok_r = Tokenizer(cfg.tokenizer or tokenizer_path(cfg.rnnt_pieces))
    tok_c = Tokenizer(cfg.tokenizer or tokenizer_path(cfg.ctc_pieces))
    if cfg.kd_weight > 0 and not cfg.kd_dir:
        raise ValueError("kd_weight > 0 needs kd_dir (the R0 teacher cache)")
    model_cfg = model_config_for(cfg)
    if tok_r.num_outputs != model_cfg.vocab_rnnt or tok_c.num_outputs != model_cfg.vocab_ctc:
        raise ShapeError(f"tokenizer sizes {tok_r.num_outputs}/{tok_c.num_outputs} != model heads "
                         f"{model_cfg.vocab_rnnt}/{model_cfg.vocab_ctc}")
    feat = Featurizer(tok_r, tok_c)
    if a.overfit:
        from ultra.libri import load_split

        exs = [feat.example(u.uid, u.text, u.pcm) for u in load_split(a.splits[0], a.overfit)]
        source, dev = FixedBatches(exs, batch_size=len(exs)), exs
        cfg.eval_each_epoch = False  # every step is an epoch here
    else:
        source = ParquetBatches(list(a.splits), feat, cfg.seed, cfg.budget_s, cfg.max_batch,
                                cfg.max_seconds, workers=cfg.workers,
                                speed_perturb=tuple(float(v) for v in cfg.speed_perturb.split(",")
                                                    if v.strip()),
                                pad_frames=cfg.pad_frames, pad_tokens=cfg.pad_tokens,
                                teacher=_teacher(cfg), loader=cfg.loader,
                                window_groups=cfg.window_groups)
        dev = load_dev(cfg.dev_split, cfg.dev_limit, feat) if cfg.eval_every else None
    tr = Trainer(a.run_dir, cfg, source, dev=dev, tokenizer=tok_r, device=a.device,
                 model_cfg=model_cfg)
    with open(os.path.join(a.run_dir, "config.json"), "w") as fh:
        json.dump({"train": asdict(cfg), "model": tr.model_cfg.to_dict(), "argv": sys.argv,
                   "threads": torch.get_num_threads(), "params": tr.model.num_params(),
                   "tokenizers": {"rnnt": tok_r.sha256, "ctc": tok_c.sha256}}, fh, indent=1)

    def on_signal(signum, frame):
        tr.stop_requested = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if tr.resume():
        tr.log({"kind": "resume", "step": tr.step, "pos": tr.pos.to_dict()})
    elif a.init_from:
        p = checkpoint.load(a.init_from, restore_rng=False)
        import dataclasses

        src = ModelConfig.from_dict(p["model_config"])
        if dataclasses.replace(src, qat="none", qat_stem="none") != \
                dataclasses.replace(tr.model_cfg, qat="none", qat_stem="none"):
            raise CheckpointError("--init-from checkpoint has a different model config")
        tr.model.load_state_dict(p["model"])
        tr.log({"kind": "start", "init_from": a.init_from, "init_step": p["step"],
                "params": tr.model.num_params(), "device": str(tr.device),
                "threads": torch.get_num_threads(), "precision": tr.precision})
    else:
        tr.init_cmvn()
        tr.log({"kind": "start", "params": tr.model.num_params(), "device": str(tr.device),
                "threads": torch.get_num_threads(), "precision": tr.precision})
    try:
        tr.run(a.max_steps)
    except DivergenceError as e:
        tr.log({"kind": "diverged", "error": str(e)})
        return 4
    except NonFiniteLossError as e:
        tr.log({"kind": "paused", "error": str(e)})
        return 3
    tr.log({"kind": "eval", "step": tr.step, "split": cfg.dev_split, **tr.evaluate(),
            **tr.stats.summary()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
