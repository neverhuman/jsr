"""Train the DENSE reference student (spec v1.0-B §17, §26) on CPU or GPU.

Resumes from `<run-dir>/LATEST`; logs JSON lines to `<run-dir>/train.jsonl`; non-finite loss/grad
writes `<run-dir>/PAUSED.json` and exits 3. AdamW + cosine by default (H23 options in train/optim).
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
from train.anchored_rnnt import anchor_kwargs, load_anchors
from train.checkpoint import swap_weights  # noqa: F401  (re-exported: tests and tools import it from here)
from train.asr import PRESETS, ASRModel, ModelConfig, SpecAugConfig
from train.ctc import FeasibilityStats
from train.data import Batch, Example, FixedBatches, Position, collate
from train.errors import CheckpointError, DivergenceError, NonFiniteLossError, ShapeError
from train.resize import apply_resize, checkpoint_carry_shape
from train.optim import build_optimizer, eval_weights, make_clipper

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
    # SpecAugment off for the first N optimiser steps (avoids unigram plateau on from-scratch runs).
    specaug_start: int = 4000
    log_every: int = 10
    ckpt_every_steps: int = 500
    ckpt_every_minutes: float = 30.0
    eval_every: int = 2000
    dev_split: str = "dev-clean"
    dev_limit: int = 200
    chunk: int = 0  # 0 = model default
    cmvn_batches: int = 8  # initial CMVN batches on from-scratch runs
    pred_ctx: int = 0  # H41 arms 16-18 (train/recipe_opts.py; 0 = preset / built-in value)
    left_context: int = 0
    specaug_time_masks: int = 0
    specaug_time_width: int = 0
    chunk_modes: str = ""  # comma list of per-step chunks (e.g. "4,8,16"); "" = off
    eval_chunks: str = ""  # extra eval chunks logged as wer_c<k> / ema_wer_c<k>
    workers: int = 4  # data-decoding threads
    eval_each_epoch: bool = True
    diverge_after: int = 3000  # divergence guard (0 disables)
    diverge_ratio: float = 0.3
    diverge_patience: int = 300
    diverge_wer_points: float = 5.0
    ema_decay: float = 0.0  # weight EMA (0 disables); ema_half_life_h > 0 indexes by audio-hours
    ema_half_life_h: float = 0.0
    data_rank: int = 0  # H50 DiLoCo island shard (data_world 0 = use DDP rank/world)
    data_world: int = 0
    stratify_chunks: bool = False  # rank r: modes[(draw + data_rank) % len] when data_world > 1
    ddp_stratify_lengths: bool = False  # needs ddp_global_batches=false
    diloco_hub: str = ""  # live DiLoCoHub URL; diloco_every > 0 enables hot-swap sync
    diloco_worker: str = ""
    diloco_every: int = 0
    diloco_dtype: str = "float32"
    rnnt_pieces: int = 4096  # SentencePiece sizes excluding blank (H13: 512 shared)
    ctc_pieces: int = 1024
    loader: str = "thread"  # "thread" | "process"
    window_groups: int = 4  # row groups sorted per window (next epoch on resume)
    extra_splits: str = ""  # H52: "/abs/dir[@weight],..." + conf floor
    extra_min_conf: float = 0.0
    extra_base_weight: float = 0.0
    dur_tables: str = ""  # H55 per-row FLAC frame-count sidecar JSONLs ("" = off)
    label_map: str = ""  # train/label_map.py: sealed uid<TAB>text TSV for extra splits ("" = off)
    label_map_sha256: str = ""
    label_map_only: int = 0  # 1: extra-split rows absent from the map are dropped before packing
    node_shards: str = ""  # H55 option (iv) node-local disjoint shard spec ("" = off)
    collate_ahead: int = 0  # >0: collate next N batches off main thread (0 = inline)
    ddp_util_target: float = 0.0  # DDP options (0/"" = off)
    ddp_global_batches: bool = False
    ddp_rank_weights: str = ""
    ddp_native: bool = False
    ddp_bf16_comm: bool = False
    kd_dir: str = ""  # H22 R0 teacher cache & KL weight
    kd_weight: float = 0.0
    tokenizer: str = ""
    rnnt_loss: str = "exact"  # exact | pruned (k2)
    prune_range: int = 5
    prune_warmup: int = 0
    prune_hold: int = 0
    qkv_cap_from: int = 0
    qkv_cap_ratio: float = 1.5
    qkv_guard_ratio: float = 0.0
    speed_perturb: str = ""
    pad_frames: int = 0
    pad_tokens: int = 0
    weights_every: int = 0
    optimizer: str = "adamw"  # H23 optimizer / clipping / zloss (defaults = AdamW bit-exact)
    grad_clip_mode: str = "fixed"
    clip_k: float = 2.0
    clip_window: int = 200
    agc_lambda: float = 0.1
    lookahead_k: int = 0
    lookahead_alpha: float = 0.5
    zloss: float = 0.0
    cr_ctc_weight: float = 0.0  # H26 CR-CTC (0 = off)
    stream_ctx_s: float = 0.0  # H41 arm 11 stream-context aug
    stream_ctx_cap: float = 1.5
    cr_ctc_full: bool = False
    cr_ctc_half: bool = False
    delay_penalty: float = 0.0  # H34 k2 delay_penalty
    mfa_anchors: str = ""  # H48c MFA-anchored latency loss ("" / 0 = off)
    anchor_kappa: float = 0.0
    anchor_margin: int = 1
    anchor_cap: float = 12.0
    anchor_simple: bool = True
    inter_ctc_weight: float = 0.0  # H28 V2 options (all default-off)
    balancer_weight: float = 0.0
    synth_aug: bool = False
    specaug_ramp_start: int = 0
    eden_lr_batches: float = 7500.0
    eden_lr_epochs: float = 3.5
    eden_steps_per_epoch: float = 0.0
    sf_beta2: float = 0.999
    muon_momentum: float = 0.95
    # Wave 2 training objectives & regularization (default-off):
    mcr_weight: float = 0.0
    etm_kd_weight: float = 0.0
    etm_kd_mode: str = "exact"
    loss_trunc_q: float = 0.0
    loss_trunc_warmup: int = 0
    loss_trunc_mode: str = "drop"
    specaug_time_warp: int = 0
    bpe_dropout_p: float = 0.0
    bpe_dropout_alpha: float = 0.1
    # H57 FACT Factors B/D/E (default-OFF, bit-identical when 0 / ""):
    layer_drop: float = 0.0
    text_tables_dim: int = 0
    text_loss_weight: float = 0.0
    text_stream_path: str = ""
    text_batch_size: int = 2048
    text_enull_batches: int = 4
    gain_aug_p: float = 0.0
    gain_min_db: float = -35.0
    gain_max_db: float = 6.0
    noise_aug_p: float = 0.0
    noise_dir: str = ""
    noise_snr_min_db: float = 5.0
    noise_snr_max_db: float = 25.0
    rir_aug_p: float = 0.0
    rir_dir: str = ""
    extra: dict = field(default_factory=dict)


def lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        p = min(1.0, (step - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


class _ForwardLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.losses(*args, **kwargs)


class Trainer:
    def __init__(self, run_dir: str, cfg: TrainConfig, source, model_cfg: ModelConfig | None = None,
                 dev: list[Example] | None = None, tokenizer=None, device: str | None = None,
                 log: bool = True, rank: int = 0, world: int = 1, local_rank: int = 0):
        self.run_dir, self.cfg, self.source, self.dev, self.tok = run_dir, cfg, source, dev, tokenizer
        self.rank, self.world, self.local_rank = rank, world, local_rank
        self.data_rank, self.data_world = (cfg.data_rank, cfg.data_world) if cfg.data_world > 0 else (rank, world)
        self._diloco_state: dict | None = None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if world > 1 and self.device.type == "cuda":
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        log = log and rank == 0  # one log per run
        self.model_cfg = model_cfg or PRESETS[cfg.preset]
        torch.manual_seed(cfg.seed)
        self.model = ASRModel(self.model_cfg).to(self.device)
        self._ddp_loss = None
        self.opt, self.sched = build_optimizer(self.model, cfg, lr_lambda)
        self.clipper = make_clipper(cfg)
        self.step = 0
        self.pos = Position()
        self.stats = FeasibilityStats()
        from train.fact_hooks import FactTextHelper
        from train.recipe_opts import make_loss_truncator, specaug

        self._fact_text = FactTextHelper.build(self.model, cfg, self.data_rank)
        self.augment = specaug(cfg)
        self.loss_trunc = make_loss_truncator(cfg)
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
                "scheduler": self.sched.state_dict(), "position": self.pos.to_dict(), "precision": self.precision,
                "train_config": asdict(self.cfg), "model_config": self.model_cfg.to_dict(), "carry_shape": checkpoint_carry_shape(self),
                "manifest": list(getattr(self.source, "manifest", [])), "tokenizer_sha256": getattr(self.tok, "sha256", None),
                "ctc_stats": self.stats.summary(), "qkv_caps": self.qkv_caps, "qkv_ref_norm": self.qkv_ref_norm,
                "ema": self.ema, "loss_ema": self.loss_ema, "best_ema": self.best_ema, "best_wer": self.best_wer,
                "watch_sig": self._watch_sig(), **({"diloco": self._diloco_state} if self._diloco_state else {}),
                **({"clipper": self.clipper.state_dict()} if self.clipper.mode != "fixed" else {})}

    def _step_chunk(self) -> int | None:
        c = self.cfg
        if not c.chunk_modes:
            return c.chunk or None
        modes = [int(v) for v in c.chunk_modes.split(",") if v.strip()]
        pick = torch.Generator().manual_seed(c.seed * 1_000_003 + self.step + 500_009)
        draw = int(torch.randint(0, len(modes), (1,), generator=pick))
        return modes[(draw + self.data_rank) % len(modes) if c.stratify_chunks and self.data_world > 1 else draw]

    def _watch_sig(self) -> dict:
        c = self.cfg
        sig = [c.kd_weight, c.cr_ctc_weight, c.zloss, c.delay_penalty, c.chunk_modes, c.rnnt_loss,
               c.prune_range, c.budget_s, c.max_batch, self.model_cfg.ctc_weight]
        # options added later are appended only when set, so a checkpoint written before they
        # existed still matches and a resumed run keeps its divergence baselines
        sig += [[k, v] for k, v in (("inter_ctc", c.inter_ctc_weight), ("balancer", c.balancer_weight),
                                    ("cr_ctc_full", c.cr_ctc_full), ("cr_ctc_half", c.cr_ctc_half),
                                    ("stream_ctx", c.stream_ctx_s), ("mcr", c.mcr_weight),
                                    ("etm_kd", c.etm_kd_weight and [c.etm_kd_weight, c.etm_kd_mode]),
                                    ("loss_trunc", c.loss_trunc_q and [c.loss_trunc_q, c.loss_trunc_mode]),
                                    ("text_loss", c.text_loss_weight),
                                    ("anchor", c.anchor_kappa and [c.anchor_kappa, c.anchor_margin, c.anchor_cap, c.anchor_simple])) if v]
        return {"loss": sig, "dev": [c.dev_split, c.dev_limit, c.chunk]}

    def reset_watch(self) -> None:
        """Re-arm the divergence guard (e.g. after --swap-weights replaced the model)."""
        self.loss_ema = self.best_ema = self.best_wer = None
        self.over_since = None

    def save(self, wait: bool = False) -> None:
        """Snapshot now; the write finishes in the background unless `wait`. Under data
        parallelism one rank per node writes (every node keeps its own copy for resume)."""
        if self.local_rank != 0:
            return
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
        from train.recipe_opts import check_resume_compatible

        check_resume_compatible(ModelConfig.from_dict(p["model_config"]), self.model_cfg)
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
        self._diloco_state = p.get("diloco")
        if self.cfg.diloco_hub and self.cfg.diloco_every > 0:
            from train.diloco_d1 import validate_diloco_resume
            validate_diloco_resume(self, p)
        self.qkv_caps = p.get("qkv_caps")
        self.qkv_ref_norm = p.get("qkv_ref_norm")
        # divergence baselines carry over only while they measure the same thing: a resume that
        # changes the loss definition (KD / CR-CTC / z-loss weights, loss type, batch) or the
        # dev panel re-arms the guard from scratch instead of comparing against an old scale
        sig, old = self._watch_sig(), p.get("watch_sig") or {}
        if old.get("loss") == sig["loss"]:
            self.loss_ema, self.best_ema = p.get("loss_ema"), p.get("best_ema")
        if old.get("dev") == sig["dev"]:
            self.best_wer = p.get("best_wer")
        self.clipper.load_state_dict(p.get("clipper"))
        self.ema = ({k: v.to(self.device) for k, v in p["ema"].items()}
                    if p.get("ema") is not None else None)
        if resized := apply_resize(self, p.get("train_config") or {}, p.get("carry_shape")):  # new lr / batch shape (train/resize.py)
            self.log({"kind": "resize", "step": self.step, **resized})
        return True

    def init_cmvn(self, n_batches: int = 8) -> None:
        """Initial CMVN statistics of a from-scratch run, over up to `n_batches` batches
        (TrainConfig.cmvn_batches; 8 is what every run before 2026-11-22 used)."""
        feats = []
        for i, (_, b) in enumerate(self.source.iterate(Position())):
            feats.extend(b.feats[j, :b.feat_lens[j]] for j in range(len(b.uids)))
            if i + 1 >= n_batches:
                break
        self.model.cmvn.set_stats(feats)

    def _step_augment(self) -> SpecAugConfig | None:
        """Return active SpecAugConfig for self.step (supports smooth ramp from specaug_ramp_start)."""
        if not self.augment.enabled:
            return None
        if self.step >= self.cfg.specaug_start:
            return self.augment
        rs = self.cfg.specaug_ramp_start
        if rs > 0 and self.step >= rs and self.cfg.specaug_start > rs:
            frac = (self.step - rs) / float(self.cfg.specaug_start - rs)
            return SpecAugConfig(
                enabled=True,
                freq_masks=self.augment.freq_masks,
                freq_width=max(1, int(round(self.augment.freq_width * frac))),
                time_masks_per_10s=self.augment.time_masks_per_10s,
                time_width=max(1, int(round(self.augment.time_width * frac))),
                time_warp=self.augment.time_warp,
            )
        return None

    # ---- one optimiser step ------------------------------------------------------------------
    def train_step(self, batch: Batch) -> dict:
        self.model.train()
        b = batch.to(self.device)
        gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + self.step)
        if (self.cfg.cr_ctc_full or self.cfg.cr_ctc_half) and self.cfg.cr_ctc_weight > 0:
            from train.two_view import two_view_step

            return two_view_step(self, batch, b, gen)
        if self.world > 1 and self.cfg.ddp_native and self._ddp_loss is None:
            from torch.nn.parallel import DistributedDataParallel

            devices = [self.device.index] if self.device.type == "cuda" else None
            self._ddp_loss = DistributedDataParallel(_ForwardLoss(self.model), device_ids=devices,
                                                     broadcast_buffers=False,
                                                     gradient_as_bucket_view=True)
            if self.cfg.ddp_bf16_comm:
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

                self._ddp_loss.register_comm_hook(None, default_hooks.bf16_compress_hook)
        forward = self._ddp_loss if self._ddp_loss is not None else self.model.losses
        t_fwd = time.time()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            out = forward(b.feats, b.feat_lens, b.rnnt, b.rnnt_lens, b.ctc, b.ctc_lens,
                                    stats=self.stats if b.active else None,
                                    augment=self._step_augment(), gen=gen,
                                    chunk=self._step_chunk(),
                                    pruned_scale=self._pruned_scale(),
                                    kd=b.kd, kd_weight=self.cfg.kd_weight,
                                    simple_detach=self.step < self.cfg.prune_hold,
                                    **({"zloss_weight": self.cfg.zloss} if self.cfg.zloss else {}),
                                    **({"cr_ctc_weight": self.cfg.cr_ctc_weight}
                                       if self.cfg.cr_ctc_weight else {}),
                                    **({"delay_penalty": self.cfg.delay_penalty}
                                       if self.cfg.delay_penalty else {}),
                                    **({"inter_ctc_weight": self.cfg.inter_ctc_weight}
                                       if self.cfg.inter_ctc_weight else {}),
                                    **({"balancer_weight": self.cfg.balancer_weight}
                                       if self.cfg.balancer_weight else {}),
                                    **({"ctx": b.ctx} if b.ctx is not None else {}),
                                    **anchor_kwargs(self.cfg, b),
                                    **self._wave2_kwargs())
        if self.world > 1:  # this rank's own forward time: its speed for throughput-weighted shards
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._fwd_s = getattr(self, "_fwd_s", 0.0) + (time.time() - t_fwd)
        loss = out["loss"] if self.world > 1 or b.active else out["loss"] * 0.0  # island zero-support step
        if self.world > 1:
            hb = self._hb_acc()
            hb["loss"] += float(loss.detach()) if b.active else 0.0
            hb["n"] += int(b.active)
            hb["audio_s"] += batch.audio_seconds if b.active else 0.0
            t_c = time.time()
            loss, out = self._global_loss(out, active=b.active)
            hb["comm_s"] += time.time() - t_c
        if self._fact_text is not None:
            loss = self._fact_text.step(self.model, b, self.device, loss, out, self.world)
        # finite-loss agreement: under DDP the all-reduced loss decides for every rank at once
        if not math.isfinite(float(out.get("loss_global", loss))):
            self._pause(batch, out, float("nan"), "loss")
        self.opt.zero_grad(set_to_none=True)
        if self.world > 1 and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_bwd = time.time()
        loss.backward()
        if self.world > 1 and self._ddp_loss is None:
            from train.dist import all_reduce_grads

            all_reduce_grads([p for p in self.model.parameters() if p.requires_grad])
        if self.world > 1 and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # backward + gradient collective (overlapped when native)
        self._bwd_s = getattr(self, "_bwd_s", 0.0) + (time.time() - t_bwd)
        gnorm = self.clipper(self.model)  # after the gradient sync (DDP: same norm on every rank)
        if not math.isfinite(gnorm):
            self._pause(batch, out, gnorm, "gradient norm")
        self.opt.step()
        self._qkv_constraint()
        step_audio_s = self._timed_global_audio(batch.audio_seconds)
        self._ema_update(step_audio_s)
        self.sched.step()
        self.step += 1
        return {"loss": float(out.get("loss_global", loss)), "rnnt": float(out["rnnt"]),
                "ctc": float(out["ctc"]),
                "grad_norm": gnorm, "ctc_masked": out["ctc_masked"] if b.active else 0,
                "lr": self.sched.get_last_lr()[0], "audio_s": step_audio_s,
                "utts": len(batch.uids) if b.active else 0,
                **({"rnnt_simple": float(out["rnnt_simple"])} if "rnnt_simple" in out else {}),
                **({"kd": float(out["kd"]), "kd_frames": out["kd_frames"]} if "kd" in out else {}),
                **({"zloss": float(out["zloss"].detach())} if "zloss" in out else {}),
                **({"cr_ctc": float(out["cr_ctc"].detach())} if "cr_ctc" in out else {}),
                **({"inter_ctc": float(out["inter_ctc"].detach())} if "inter_ctc" in out else {}),
                **({"balancer": float(out["balancer"].detach())} if "balancer" in out else {}),
                **({"anchor_risk": float(out["anchor_risk"]),
                    "anchor_cov": out["anchor_tokens"] / max(1, out["rnnt_tokens"])}
                   if "anchor_risk" in out else {}),
                **({"mcr_kl": float(out["mcr_kl"].detach())} if "mcr_kl" in out else {}),
                **({"etm_kd": float(out["etm_kd"].detach())} if "etm_kd" in out else {}),
                **({"text_loss": float(out["text_loss"])} if "text_loss" in out else {})}

    def _wave2_kwargs(self) -> dict:
        from train.recipe_opts import wave2_loss_kwargs
        return wave2_loss_kwargs(self.cfg, self.loss_trunc, self.step)

    def _ddp_interval_stats(self, acc: dict) -> dict:
        """Per-rank sample counts, padding and backward(+collective) seconds for the interval."""
        if self.world <= 1:
            return {}
        from train.dist import all_gather_floats

        pad = 100.0 * (1 - acc.get("frames", 0) / max(1, acc.get("padded", 0)))
        th = getattr(self, "_throttle", None)
        ts = th.stats() if th is not None else {"duty": 1.0, "util_5min": -1.0, "target": 0.0, "slept_s": 0.0}
        slept = ts["slept_s"] - getattr(self, "_slept_mark", 0.0)
        self._slept_mark = ts["slept_s"]
        rows = all_gather_floats([float(acc.get("frames", 0)), pad, getattr(self, "_bwd_s", 0.0),
                                  float(acc.get("step_s", 0.0)), float(acc.get("data_wait_s", 0.0)),
                                  slept, ts["duty"], ts["util_5min"],
                                  getattr(self, "_fwd_s", 0.0)], self.device)
        self._bwd_s = self._fwd_s = 0.0
        col = lambda k, nd=2: [round(r[k], nd) for r in rows]  # noqa: E731
        return {"rank_frames": [int(r[0]) for r in rows], "rank_pad_pct": col(1),
                "rank_bwd_comm_s": col(2), "rank_step_s": col(3), "rank_data_wait_s": col(4),
                "rank_throttle_sleep_s": col(5), "rank_duty": col(6, 3), "rank_gpu_util_5min": col(7, 3),
                "rank_fwd_s": col(8), "util_target": ts["target"]}

    def _hb_acc(self) -> dict:
        if getattr(self, "_hb", None) is None:
            self._hb = {"audio_s": 0.0, "loss": 0.0, "n": 0, "comm_s": 0.0, "step_s": 0.0,
                        "wait_s": 0.0, "sleep_s": 0.0, "frames": 0, "padded": 0, "t": time.time()}
        return self._hb

    def _timed_global_audio(self, seconds: float) -> float:
        t = time.time()
        out = self._global_audio(seconds)
        if self.world > 1:
            self._hb_acc()["comm_s"] += time.time() - t
        return out

    def _rank_heartbeat(self) -> None:
        """DDP: every rank appends one line of LOCAL values per log interval to
        <run_dir>/rank<R>.jsonl (no collective; best-effort, never raises)."""
        if self.world <= 1:
            return
        import socket

        from train.dist import physical_gpu

        hb, now = self._hb_acc(), time.time()
        gpu = physical_gpu(self.device.index or 0) if self.device.type == "cuda" else "cpu"
        rec = {"kind": "rank", "rank": self.rank, "world": self.world, "host": socket.gethostname(),
               "gpu": int(gpu) if gpu.isdigit() else gpu, "step": self.step, "time": now,
               "interval_s": round(now - hb["t"], 3), "audio_s": round(hb["audio_s"], 2),
               "loss": hb["loss"] / hb["n"] if hb["n"] else None,
               "compute_s": round(max(0.0, hb["step_s"] - hb["comm_s"]), 3),
               "comm_s": round(hb["comm_s"], 3), "wait_s": round(hb["wait_s"], 3),
               "sleep_s": round(hb["sleep_s"], 3),
               "pad_pct": round(100.0 * (1 - hb["frames"] / max(1, hb["padded"])), 2)}
        self._hb = None
        try:
            with open(os.path.join(self.run_dir, f"rank{self.rank}.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except (OSError, TypeError, ValueError):
            pass

    def _global_audio(self, seconds: float) -> float:
        """Audio seconds in this step over all ranks (logged audio-h/h is then the run's total)."""
        if self.world <= 1:
            return seconds
        from train.dist import all_reduce_sum

        return all_reduce_sum([seconds], self.device)[0]

    def _global_loss(self, out: dict, active: bool = True):
        """Each objective over its own global denominator (train/ddp_loss.py)."""
        from train.ddp_loss import global_loss

        return global_loss(out, self.cfg, self.model.cfg, self.world, self.device,
                           self._pruned_scale(), active)

    def _diverged(self, detail: str, extra: dict) -> None:
        diag = os.path.join(self.run_dir, "DIVERGED.json")
        with open(diag, "w") as fh:
            json.dump({"step": self.step, "detail": detail, "loss_ema": self.loss_ema,
                       "best_ema": self.best_ema, "best_wer": self.best_wer,
                       "position": self.pos.to_dict(), **extra}, fh, indent=1)
        self.save(wait=True)  # keep the state at the stop for inspection
        raise DivergenceError(self.step, detail, diag)

    def _pruned_scale(self) -> float:
        if self.step < self.cfg.prune_hold:
            return 0.0
        w, s = self.cfg.prune_warmup, self.step - self.cfg.prune_hold
        return 1.0 if not w else min(1.0, 0.1 + 0.9 * s / w)

    @torch.no_grad()
    def _ema_update(self, audio_s: float = 0.0) -> None:
        if self.cfg.ema_half_life_h > 0 and audio_s > 0:
            d = math.exp(-math.log(2.0) * (audio_s / 3600.0) / self.cfg.ema_half_life_h)
        else:
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
                                 **({"gain": m.norm.weight.abs().cpu() * c.qkv_cap_ratio}
                                    if m.norm.weight is not None else {})}
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
            if g is not None and "gain" in cap:
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
        if self.loss_ema > self.best_ema + c.diverge_ratio * abs(self.best_ema):  # |best|: loss can be < 0
            self.over_since = self.over_since or self.step
            if self.step - self.over_since >= c.diverge_patience:
                self._diverged(f"train-loss EMA {self.loss_ema:.3f} > best {self.best_ema:.3f} + "
                               f"{c.diverge_ratio} x |best| for {c.diverge_patience} steps",
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
        if not self.dev or self.tok is None:
            return {}
        with eval_weights(self.opt):
            return self._evaluate(max_batch_s)

    def _evaluate(self, max_batch_s: float, chunk: int | str | None = None) -> dict:
        from train.chunks import resolve as resolve_chunk
        from ultra.wer import corpus_wer

        self.model.eval()
        exs = sorted(self.dev, key=lambda e: e.seconds)
        refs, hyps, forced = [], [], 0
        i = 0
        while i < len(exs):
            j = i + 1
            while j < len(exs) and exs[j].seconds * (j - i + 1) <= max_batch_s:
                j += 1
            b = collate(exs[i:j]).to(self.device)
            toks, f = self.model.greedy(b.feats, b.feat_lens, resolve_chunk(chunk, self.cfg.chunk))
            forced += f
            refs.extend(b.texts)
            hyps.extend(self.tok.decode(t) for t in toks)
            i = j
        w = corpus_wer(refs, hyps)
        w["forced_advances"] = forced
        w["examples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps, strict=True))[:3]]
        return w

    @torch.no_grad()
    def _evaluate_extra_chunks(self) -> dict:
        """Dev WER of the live and EMA weights at each extra attention chunk in cfg.eval_chunks."""
        out: dict = {}
        if not self.cfg.eval_chunks or not self.dev or self.tok is None:
            return out
        from train.chunks import key, parse_list

        for c in parse_list(self.cfg.eval_chunks):  # positive chunks and the explicit 'offline'
            with eval_weights(self.opt):
                out[f"wer_{key(c)}"] = self._evaluate(300.0, c).get("wer")
            ema = self._evaluate_ema(c).get("ema_wer")
            if ema is not None:
                out[f"ema_wer_{key(c)}"] = ema
        return out

    @torch.no_grad()
    def _evaluate_ema(self, chunk: int | str | None = None) -> dict:
        """Dev WER of the EMA weights (swapped in, then the live weights restored)."""
        if self.ema is None or not self.dev or self.tok is None:
            return {}
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema)
        try:
            # as loaded: no Schedule-Free x swap over the EMA weights
            ev = self._evaluate(300.0, chunk) if chunk is not None else self._evaluate(300.0)
        finally:
            self.model.load_state_dict(live)
        return {"ema_wer": ev.get("wer"), "ema_sub": ev.get("sub"), "ema_del": ev.get("del"),
                "ema_ins": ev.get("ins")}

    # ---- loop --------------------------------------------------------------------------------
    def log(self, rec: dict) -> None:
        if self.rank != 0:  # one train.jsonl per data-parallel run
            return
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
        throttle = None
        if self.world > 1 and self.cfg.ddp_util_target > 0 and self.device.type == "cuda":
            from train.dist import StepThrottle, physical_gpu
            throttle = StepThrottle(self.cfg.ddp_util_target, physical_gpu(self.device.index or 0), live_dir=self.run_dir)
        self._throttle = throttle
        diloco = None
        if self.cfg.diloco_hub and self.cfg.diloco_every > 0:
            from train.diloco_client import TrainerDiLoCoHook
            diloco = TrainerDiLoCoHook.from_trainer(self)
        for pos, batch in self._batches():
            acc["data_wait_s"] = acc.get("data_wait_s", 0.0) + (time.time() - t_wait)
            t_step = time.time()
            acc["frames"] = acc.get("frames", 0) + int(batch.feat_lens.sum())
            acc["padded"] = acc.get("padded", 0) + batch.feats.shape[0] * batch.feats.shape[1]
            r = self.train_step(batch)
            acc["step_s"] = acc.get("step_s", 0.0) + (time.time() - t_step)
            if self.world > 1:
                from train.dist import all_reduce_max

                hb = self._hb_acc()
                hb["step_s"] += time.time() - t_step
                hb["wait_s"] += t_step - t_wait
                hb["frames"] += int(batch.feat_lens.sum())
                hb["padded"] += batch.feats.shape[0] * batch.feats.shape[1]
                t_c = time.time()
                nap = all_reduce_max(throttle.sleep_for(self.step, time.time() - t_step), self.device) \
                    if throttle is not None else 0.0
                self.stop_requested = all_reduce_max(float(self.stop_requested), self.device) > 0
                hb["comm_s"] += time.time() - t_c
                if throttle is not None:  # synchronised step-boundary sleep, never SIGSTOP
                    throttle.pause(nap)
                    hb["sleep_s"] += max(0.0, nap)
            new_epoch = pos.epoch > self.pos.epoch
            self.pos = pos
            steps_here += 1
            acc["audio_s"] += r["audio_s"]
            acc["n"] += 1
            acc["masked"] += r["ctc_masked"]
            for k in ("loss", "rnnt", "ctc", "rnnt_simple", "kd", "zloss", "cr_ctc", "inter_ctc", "balancer", "anchor_risk", "anchor_cov", "text_loss"):
                if k in r:
                    acc[k] = acc.get(k, 0.0) + r[k]
            self._watch_loss(r)
            if diloco is not None: diloco.on_step(self, r)
            if self.step % self.cfg.log_every == 0:
                self._rank_heartbeat()
                dt = time.time() - t0
                opt_avgs = {k: acc[k] / acc["n"] for k in ("rnnt_simple", "kd", "zloss", "cr_ctc", "inter_ctc", "balancer", "anchor_risk", "anchor_cov", "text_loss") if k in acc}
                self.log({"kind": "train", "step": self.step, "pos": pos.to_dict(),
                          "loss": acc["loss"] / acc["n"], "rnnt": acc["rnnt"] / acc["n"],
                          "ctc": acc["ctc"] / acc["n"], "lr": r["lr"], "grad_norm": r["grad_norm"],
                          "ctc_masked": acc["masked"], "audio_s": acc["audio_s"], "wall_s": dt,
                          "dropped_long": getattr(self.source, "dropped_long", 0), **opt_avgs,
                          **({"dropped_conf": self.source.dropped_conf} if getattr(self.source, "min_conf", 0) else {}),
                          **({"dropped_label": self.source.dropped_label} if getattr(self.source, "label_map", None) else {}),
                          **({"anchor_counts": dict(s.counts)} if (s := getattr(self.source, "anchors", None)) is not None else {}),
                          **({"kd_counts": dict(self.source.teacher.counts)} if getattr(self.source, "teacher", None) else {}),
                          **({"clip_thr": self.clipper.last_threshold} if self.clipper.mode == "adaptive" else {}),
                          **({"gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)} if self.device.type == "cuda" else {}),
                          "max_qkv_norm": self._max_qkv_norm(), "loss_ema": self.loss_ema, **(diloco.log_fields() if diloco is not None else {}),
                          "audio_h_per_h": acc["audio_s"] / max(dt, 1e-9),
                          "data_wait_pct": round(100.0 * acc.get("data_wait_s", 0.0) / max(dt, 1e-9), 2),
                          "pad_pct": round(100.0 * (1 - acc.get("frames", 0) / max(1, acc.get("padded", 0))), 2),
                          **self._ddp_interval_stats(acc)})
                acc = {k: 0.0 if isinstance(v, float) else 0 for k, v in acc.items()}
                t0 = time.time()
            epoch_end = new_epoch and self.cfg.eval_each_epoch and self.dev
            do_eval, ep_flag = (diloco.gate.allow_eval(self.step, self.cfg.eval_every, bool(epoch_end), diloco.just_synced)
                                if diloco is not None else (bool(epoch_end or (self.cfg.eval_every and self.step % self.cfg.eval_every == 0)), bool(epoch_end)))
            if do_eval:
                ev = self.evaluate() if self.rank == 0 else {}
                ema = {**(self._evaluate_ema() if self.rank == 0 else {}), **(self._evaluate_extra_chunks() if self.rank == 0 else {})}
                self.log({"kind": "eval", "step": self.step, "split": self.cfg.dev_split, "epoch": pos.epoch, "epoch_end": ep_flag, **ev, **ema, **self.stats.summary()})
                wer = ev.get("wer")
                if self.world > 1:
                    from train.dist import broadcast_value
                    wer = broadcast_value(wer, self.device)
                self._watch_wer(wer)
                t0 = time.time()
            do_w = (diloco.gate.allow_weights(self.step, self.cfg.weights_every, diloco.just_synced)
                    if diloco is not None else bool(self.cfg.weights_every and self.step % self.cfg.weights_every == 0))
            if do_w and self.local_rank == 0:
                with eval_weights(self.opt):
                    self.saver.save(self.run_dir, self.step, {"model": self.model.state_dict(), "model_config": self.model_cfg.to_dict(), "train_config": asdict(self.cfg), "tokenizer_sha256": getattr(self.tok, "sha256", None), "ema": self.ema}, weights_only=True)
            raw_due = self.step % self.cfg.ckpt_every_steps == 0 or time.time() - last_ckpt > 60 * self.cfg.ckpt_every_minutes
            due = diloco.gate.allow_ckpt(raw_due, diloco.just_synced) if diloco is not None else raw_due
            if due or (self.stop_requested and diloco is None):
                self.save(); last_ckpt = time.time()
            if self.stop_requested or (max_steps and steps_here >= max_steps) or self.step >= self.cfg.total_steps:
                if not due and not self.stop_requested and (diloco is None or diloco.just_synced):
                    self.save()
                self.saver.wait()
                return
            t_wait = time.time()


def _shard_batches(self):
    """Batches for this rank: sources that shard natively (ParquetBatches.world) yield only this
    rank's batches; any other source is sharded by batch index (data_rank, data_rank + data_world, ...)."""
    it = self.source.iterate(self.pos)
    dw = getattr(self, "data_world", self.world)
    dr = getattr(self, "data_rank", self.rank)
    if dw <= 1 or getattr(self.source, "world", 1) == dw:
        yield from it
        return
    for i, item in enumerate(it):
        if i % dw == dr:
            yield item


Trainer._batches = _shard_batches


def load_dev(split: str, limit: int, feat) -> list[Example]:
    from ultra.libri import load_split

    return [feat.example(u.uid, u.text, u.pcm) for u in load_split(split, limit)]


def _teacher(cfg: TrainConfig, tok=None):
    if not cfg.kd_dir:
        return None
    from train.kd import TeacherStore

    store = TeacherStore(cfg.kd_dir, tokenizer_sha256=getattr(tok, "sha256", None))
    print(json.dumps({"kind": "teacher", "entries": len(store), "topk": store.topk, **store.identity}),
          flush=True)
    return store


def model_config_for(cfg: TrainConfig) -> ModelConfig:
    """The preset's model config with output heads sized to the configured tokenizers."""
    import dataclasses

    if cfg.rnnt_pieces < 2 or cfg.ctc_pieces < 2:
        raise ShapeError("rnnt_pieces and ctc_pieces must be >= 2")
    from train.recipe_opts import model_overrides

    return dataclasses.replace(model_overrides(PRESETS[cfg.preset], cfg), vocab_rnnt=cfg.rnnt_pieces + 1,
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
    ap.add_argument("--swap-weights", default=None,
                    help="resume only: replace the model weights by this checkpoint's (e.g. a weight "
                         "soup), keeping optimizer state, schedule, position and any training-only "
                         "heads the file lacks")
    from train.recipe_opts import add_wave2_args, apply_wave2_args
    add_wave2_args(ap)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
    cfg = TrainConfig(preset=a.preset, splits=tuple(a.splits))
    apply_wave2_args(cfg, a)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(json.loads(v)) if not isinstance(cur, str) else v)

    from train.data import Featurizer, ParquetBatches, apply_data_shard, parse_extra_splits
    from train.tokenizer import Tokenizer, tokenizer_path

    tok_r = Tokenizer(cfg.tokenizer or tokenizer_path(cfg.rnnt_pieces))
    tok_c = Tokenizer(cfg.tokenizer or tokenizer_path(cfg.ctc_pieces))
    if cfg.kd_weight > 0 and not cfg.kd_dir:
        raise ValueError("kd_weight > 0 needs kd_dir (the R0 teacher cache)")
    model_cfg = model_config_for(cfg)
    from train.chunks import check_plan

    check_plan(cfg, model_cfg)  # illegal chunks fail before step 0, not at their first draw
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
        from train.fact_hooks import fact_aug_from_cfg
        from train.label_map import label_map_from_cfg

        source = ParquetBatches(list(a.splits), feat, cfg.seed, cfg.budget_s, cfg.max_batch,
                                cfg.max_seconds, workers=cfg.workers,
                                speed_perturb=tuple(float(v) for v in cfg.speed_perturb.split(",")
                                                    if v.strip()),
                                pad_frames=cfg.pad_frames, pad_tokens=cfg.pad_tokens,
                                teacher=_teacher(cfg, tok_r), loader=cfg.loader,
                                window_groups=cfg.window_groups, synth_aug=cfg.synth_aug,
                                extra=parse_extra_splits(cfg.extra_splits), min_conf=cfg.extra_min_conf,
                                base_weight=cfg.extra_base_weight, anchors=load_anchors(cfg, tok_r),
                                dur_tables=cfg.dur_tables, node_shards=cfg.node_shards,
                                bpe_dropout_p=cfg.bpe_dropout_p, bpe_dropout_alpha=cfg.bpe_dropout_alpha,
                                collate_ahead=cfg.collate_ahead, fact_aug=fact_aug_from_cfg(cfg),
                                label_map=label_map_from_cfg(cfg))
        dev = load_dev(cfg.dev_split, cfg.dev_limit, feat) if cfg.eval_every else None
    from train.dist import broadcast_state, init_from_env

    dev_type = (a.device or ("cuda" if torch.cuda.is_available() else "cpu")).split(":")[0]
    rank, world, local_rank = init_from_env(dev_type)
    apply_data_shard(source, cfg, world)
    if world > 1 and hasattr(source, "world"):
        source.rank, source.world = rank, world
        source.ddp_global_batches, source.stratify_lengths = cfg.ddp_global_batches, cfg.ddp_stratify_lengths
        if cfg.ddp_rank_weights:
            w = tuple(float(v) for v in cfg.ddp_rank_weights.split(","))
            if len(w) != world or min(w) <= 0 or not cfg.ddp_global_batches:
                raise ValueError("ddp_rank_weights needs one positive weight per rank and ddp_global_batches")
            source.rank_weights = w
    if cfg.stream_ctx_s:  # H41 arm 11, single-GPU and DDP alike (parquet loader only)
        source.stream_ctx_s, source.stream_ctx_cap = cfg.stream_ctx_s, cfg.stream_ctx_cap
        print(json.dumps({"stream_ctx_s": cfg.stream_ctx_s, "stream_ctx_cap": cfg.stream_ctx_cap}), flush=True)
    tr = Trainer(a.run_dir, cfg, source, dev=dev, tokenizer=tok_r, device=a.device,
                 model_cfg=model_cfg, rank=rank, world=world, local_rank=local_rank)
    cfg_name = "config.json" if rank == 0 else f"config.rank{rank}.json"
    with open(os.path.join(a.run_dir, cfg_name), "w") as fh:
        json.dump({"train": asdict(cfg), "model": tr.model_cfg.to_dict(), "argv": sys.argv,
                   "threads": torch.get_num_threads(), "params": tr.model.num_params(),
                   "tokenizers": {"rnnt": tok_r.sha256, "ctc": tok_c.sha256},
                   "world": world}, fh, indent=1)

    def on_signal(signum, frame):
        tr.stop_requested = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if tr.resume():
        tr.log({"kind": "resume", "step": tr.step, "pos": tr.pos.to_dict()})
        if a.swap_weights:
            tr.log({"kind": "swap_weights", "step": tr.step, **swap_weights(tr.model, a.swap_weights)})
            tr.ema = None  # the EMA restarts from the swapped weights (no carry-over of the old line)
            tr.reset_watch()  # and the divergence guard re-arms on the new weights
    elif a.init_from:
        p = checkpoint.load(a.init_from, restore_rng=False)
        from train.fact_hooks import load_init_state

        load_init_state(tr.model, p["model"], ModelConfig.from_dict(p["model_config"]), tr.model_cfg)
        tr.log({"kind": "start", "init_from": a.init_from, "init_step": p["step"],
                "params": tr.model.num_params(), "device": str(tr.device),
                "threads": torch.get_num_threads(), "precision": tr.precision})
    else:
        tr.init_cmvn(cfg.cmvn_batches)
        tr.log({"kind": "start", "params": tr.model.num_params(), "device": str(tr.device),
                "threads": torch.get_num_threads(), "precision": tr.precision})
    if world > 1:
        from train.dist import all_reduce_max

        # every rank must start from the same checkpoint state: same step, same position
        key = float(tr.step * 1_000_003 + tr.pos.epoch * 10_007 + tr.pos.window * 101 + tr.pos.batch)
        if all_reduce_max(key, tr.device) != -all_reduce_max(-key, tr.device):
            raise CheckpointError("ranks resumed different checkpoints (step/position differ)")
        broadcast_state(tr.model)  # identical weights (and CMVN) on every rank
        from train.dist import all_gather_floats, state_checksum

        sums = [v[0] for v in all_gather_floats([state_checksum(tr.model)], tr.device)]
        tr.log({"kind": "ddp", "world": world, "rank0_device": str(tr.device),
                "checksums": sums, "ranks_agree": len(set(sums)) == 1,
                "native": cfg.ddp_native, "bf16_comm": cfg.ddp_bf16_comm,
                "global_batches": cfg.ddp_global_batches, "util_target": cfg.ddp_util_target})
    try:
        tr.run(a.max_steps)
    except DivergenceError as e:
        tr.log({"kind": "diverged", "error": str(e)})
        return 4
    except NonFiniteLossError as e:
        tr.log({"kind": "paused", "error": str(e)})
        return 3
    if not (cfg.diloco_hub and cfg.diloco_every > 0) or (tr._diloco_state and tr._diloco_state.get("step") == tr.step):
        tr.log({"kind": "eval", "step": tr.step, "split": cfg.dev_split, **tr.evaluate(), **tr.stats.summary()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
