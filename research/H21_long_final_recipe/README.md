# H21: Long final-recipe runs (~30 epochs of 960 h)

**Status:** `PRE-REGISTERED`, launching 2026-11-22.

## Hypothesis
Most of the remaining gap to strong references (D2 dev-clean ~7.0 with averaging + beam vs R0 2.85) comes from recipe and exposure rather than architecture. The recipe is:
- SentencePiece-512 shared by both heads;
- weight EMA;
- speed perturbation;
- a stable, cap-guarded optimisation;
- the pruned RNN-T loss, for more epochs per GPU-day;
- ~30 epochs.

A dense S and a multirate S_MR trained this way should both improve markedly on D2/D2MR at equal decode settings (last-3 average, beam 4, no blank penalty: H12 §5).

## Runs (one per A100 40 GB, each under its own fleet_guard at GPU target 0.88)

| Run | Host | Preset | Notes |
|---|---|---|---|
| L1 | xbabe8 (A100 40 GB) | S (dense) | |
| L2 | xbabe9 (A100 40 GB) | S_MR (multirate, fixed pad-before-pair, H12) | the throughput-headline model |

**Shared pinned settings:**
- **Data and tokenizer:** 960 h (train-clean-100/360, train-other-500), downloaded on each node from openslr/librispeech_asr and sha256-checked against our manifest. `rnnt_pieces=512 ctc_pieces=512` (SentencePiece-512, sha256 `4e213d44…`).
- **Loss:** `rnnt_loss=pruned prune_range=5 prune_warmup=2000`. Validated exact-equivalent at full band (tests/test_train_pruned.py); the simple-joiner scale is 0.5.
- **Optimisation:**
  - `seed=1234`;
  - `budget_s=1200 max_batch=256`: 2× H13a's 600 s per step, the A100 batch. The per-step audio budget is recorded in config.json.
  - `lr=0.0014` (H13a's 1e-3 × √2 for the 2× batch), `warmup=2000`, `clip=5`, cosine-style decay to the trainer's floor.
  - `total_steps=86400`, which is 30 × 960 h / 1200 s per step. Exposure is measured in audio-hours, logged per step.
- **Augmentation and averaging:** `specaug_start=4000` (delayed, as in H13a/H3), `speed_perturb=0.9,1.0,1.1`, `ema_decay=0.9995`.
- **Guards:** `qkv_cap_from=30000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0` (from scratch, as D4M/H13a), `diverge_after=5500`.
- **Evaluation:** dev-clean400 plus an eval-only snapshot every 5000 steps, auto-posted, with logs mirrored to H0 for research/curves. The full-dev / panel backfill (greedy, beam 4, last-3 average, EMA) runs on CPU as snapshots land.

## Decision points (dev-clean400 greedy at the snapshot; full dev with beam 4 at the decision)
- **5 epochs (~14.4k steps):** must beat H13a and D2 at equal audio-hours, or the recipe element at fault is reconsidered.
- **10 epochs (~28.8k steps):** compared against D2-final (102k steps at 600 s ≈ 17 epochs). If L1 is not ahead at equal exposure, stop L1 and give its GPU to the winning H13 variant.
- **20 epochs (~57.6k steps):** decide the paper's final checkpoints. If H13b–d contradicts a recipe element (for example, the pruned loss is worse than exact at equal exposure), warm-restart from the latest snapshot with that element changed.

## Links
- H13 (vocab and pruned-loss controls), H12 (decode point, MR padding fix), H9 (qkv cap).
