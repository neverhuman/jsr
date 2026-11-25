"""Trainer configuration (moved verbatim from train/trainer.py; trainer.py re-exports TrainConfig)."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    seal_data_identity: bool = False  # strict preflight and fixed-layout checkpoint restart
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
    stream_ctx_mode: str = "h41"  # "h41" | turn_switch | turn_switch_mixed (default = unchanged H41 behaviour)
    stream_ctx_p: float = 1.0
    stream_ctx_gap_s: float = 0.0
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
