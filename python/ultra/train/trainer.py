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
from train.errors import CheckpointError, NonFiniteLossError

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
                "ctc_stats": self.stats.summary()}

    def save(self, wait: bool = False) -> None:
        """Snapshot now; the write finishes in the background unless `wait`."""
        self.saver.save(self.run_dir, self.step, self.payload())
        if wait:
            self.saver.wait()

    def resume(self) -> bool:
        if os.path.exists(os.path.join(self.run_dir, "PAUSED.json")):
            raise CheckpointError(f"run is paused; inspect and remove {self.run_dir}/PAUSED.json")
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
        self.step = p["step"]
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
                                    chunk=self.cfg.chunk or None)
        loss = out["loss"]
        if not torch.isfinite(loss):
            self._pause(batch, out, float("nan"), "loss")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip))
        if not math.isfinite(gnorm):
            self._pause(batch, out, gnorm, "gradient norm")
        self.opt.step()
        self.sched.step()
        self.step += 1
        return {"loss": float(loss), "rnnt": float(out["rnnt"]), "ctc": float(out["ctc"]),
                "grad_norm": gnorm, "ctc_masked": out["ctc_masked"],
                "lr": self.sched.get_last_lr()[0], "audio_s": batch.audio_seconds,
                "utts": len(batch.uids)}

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
        for pos, batch in self.source.iterate(self.pos):
            r = self.train_step(batch)
            new_epoch = pos.epoch > self.pos.epoch
            self.pos = pos
            steps_here += 1
            acc["audio_s"] += r["audio_s"]
            acc["n"] += 1
            acc["masked"] += r["ctc_masked"]
            for k in ("loss", "rnnt", "ctc"):
                acc[k] += r[k]
            if self.step % self.cfg.log_every == 0:
                dt = time.time() - t0
                self.log({"kind": "train", "step": self.step, "pos": pos.to_dict(),
                          "loss": acc["loss"] / acc["n"], "rnnt": acc["rnnt"] / acc["n"],
                          "ctc": acc["ctc"] / acc["n"], "lr": r["lr"], "grad_norm": r["grad_norm"],
                          "ctc_masked": acc["masked"], "audio_s": acc["audio_s"], "wall_s": dt,
                          "audio_h_per_h": acc["audio_s"] / max(dt, 1e-9)})
                acc = {k: 0.0 if isinstance(v, float) else 0 for k, v in acc.items()}
                t0 = time.time()
            epoch_end = new_epoch and self.cfg.eval_each_epoch and self.dev
            if epoch_end or (self.cfg.eval_every and self.step % self.cfg.eval_every == 0):
                self.log({"kind": "eval", "step": self.step, "split": self.cfg.dev_split,
                          "epoch": pos.epoch, "epoch_end": bool(epoch_end),
                          **self.evaluate(), **self.stats.summary()})
                t0 = time.time()
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


def load_dev(split: str, limit: int, feat) -> list[Example]:
    from ultra.libri import load_split

    return [feat.example(u.uid, u.text, u.pcm) for u in load_split(split, limit)]


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

    tok_r, tok_c = Tokenizer(tokenizer_path(4096)), Tokenizer(tokenizer_path(1024))
    feat = Featurizer(tok_r, tok_c)
    if a.overfit:
        from ultra.libri import load_split

        exs = [feat.example(u.uid, u.text, u.pcm) for u in load_split(a.splits[0], a.overfit)]
        source, dev = FixedBatches(exs, batch_size=len(exs)), exs
        cfg.eval_each_epoch = False  # every step is an epoch here
    else:
        source = ParquetBatches(list(a.splits), feat, cfg.seed, cfg.budget_s, cfg.max_batch,
                                cfg.max_seconds, workers=cfg.workers)
        dev = load_dev(cfg.dev_split, cfg.dev_limit, feat) if cfg.eval_every else None
    tr = Trainer(a.run_dir, cfg, source, dev=dev, tokenizer=tok_r, device=a.device)
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
        if ModelConfig.from_dict(p["model_config"]) != tr.model_cfg:
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
    except NonFiniteLossError as e:
        tr.log({"kind": "paused", "error": str(e)})
        return 3
    tr.log({"kind": "eval", "step": tr.step, "split": cfg.dev_split, **tr.evaluate(),
            **tr.stats.summary()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
