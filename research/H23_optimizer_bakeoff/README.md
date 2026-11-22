# H23: Optimiser bake-off (faster convergence per GPU-hour)

**Status:** `PRE-REGISTERED` on 2026-11-22. Nothing had been launched when this file was committed. Owner: optim-lane. Code: `python/ultra/train/optim/`, tests: `python/ultra/tests/test_train_optim.py`, runner: `python/ultra/train/ops/h23_bakeoff.sh`.

## Question
At a fixed audio exposure, does any of these beat our AdamW + warm-up/cosine recipe on dev WER, or reach the same WER in clearly less wall time, without hurting stability?
- ScaledAdam + Eden (Zipformer/icefall);
- Schedule-Free AdamW (Defazio et al. 2024);
- Muon + AdamW (Jordan 2024; Moonshot 2025 RMS matching);
- adaptive (median-relative) gradient clipping?

## Matched protocol (identical for every arm; pinned to the H13 winner, H13d)
- **Model and data:**
  - preset **S** (dense), from scratch;
  - **SentencePiece-512** for both heads (`rnnt_pieces=512 ctc_pieces=512`, sha256 `4e213d44…2dba`, as in H13);
  - **pruned RNN-T loss** (`prune_range=5 prune_warmup=2000`, simple-loss scale 0.5, lm-only 0.25);
  - train-clean-100.
- **Schedule and exposure:**
  - `seed=1234 budget_s=300 max_batch=64 total_steps=24000` (about 20 epochs, ~2,000 audio-h);
  - `clip=5 specaug_start=4000`, no speed perturbation, `ema_decay=0.9995`, `diverge_after=5000`;
  - `loader=process`: bit-identical batches to the thread loader (trainer-lane test), so every arm sees the same batches in the same order, and the same as H13d.
- **Evaluation:**
  - dev-clean400 greedy (chunk-masked) plus an eval-only snapshot every 2000 steps;
  - offline on the 4k/8k/12k/16k/20k/24k snapshots: the full dev-clean and dev-other, greedy and beam 4 (frozen H12 §5 setting), raw and EMA weights.
  - Schedule-Free snapshots and evals use its averaged weights `x` (trainer `eval_weights`).
- **Hardware:** each arm runs on one GPU under its own `fleet_guard` at 0.88, with guard_ceiling coverage. Wall-time comparisons are made only between arms on the **same GPU model**. The AdamW baseline is re-run (H23a) rather than reusing H13d, so it shares hardware and code with the other arms. H13d (A10, same protocol) is a secondary reference for WER at equal steps.

## Arms (only these settings differ)
| Arm | Optimiser | LR / schedule | Clipping | Other |
|---|---|---|---|---|
| **H23a** | AdamW (wd 0.01 on ≥ 2-D, betas 0.9/0.98) | lr 1e-3, warm-up 2000, cosine to 2% | fixed 5 | baseline (= H13d recipe) |
| **H23b** | ScaledAdam (scalar_lr_scale 0.1, RMS clamp [1e-5, 3]) | base lr 0.045, **Eden** (lr_batches 7500, lr_epochs 3.5, 1200 steps/epoch, warm-up 500 from 0.5) | adaptive, 2 × median of the last 200 norms (icefall's clipping_scale 2) | icefall defaults |
| **H23c** | Schedule-Free AdamW (wd 0.01 on ≥ 2-D, beta1 0.9, beta2 0.98) | lr 2e-3, linear warm-up 2000, then flat (no decay) | fixed 5 | eval/snapshots with `x` |
| **H23d** | Muon (momentum 0.95, Nesterov, NS5) on encoder-block 2-D matrices, AdamW on the rest | lr 1e-3, warm-up 2000, cosine (Muon update RMS-matched: 0.2·√max(m,n)) | fixed 5 | stem, heads, predictor, joiner, norms, biases on AdamW |
| **H23e** (side) | AdamW (as H23a) | as H23a | **adaptive**, 2 × median of the last 200 norms | isolates the clipping effect in H23b |

The LR choices are each method's published default or the obvious transfer from it, fixed before any result. There is no per-arm LR sweep. A method that diverges at its default gets **one** pre-declared retry at half its lr, reported as such.

## Metrics
1. **Primary:** dev-clean and dev-other WER (full sets, greedy and beam 4; EMA and raw) at matched audio-hours: 8k, 16k and 24k steps (the batches are identical, so equal steps means equal audio-hours).
2. **Wall time:** guarded wall-clock to 24k steps and mean s/step, only between arms on the same GPU model. Also step-time overhead measured unguarded over 200 steps on one GPU.
3. **Peak GPU memory** (`torch.cuda.max_memory_allocated`, plus nvidia-smi peak).
4. **Stability:**
   - grad-norm spikes (steps with grad_norm > 5 × running median);
   - `max_qkv_norm` trajectory and its value at 24k;
   - loss-EMA excursions;
   - any PAUSED or DIVERGED stop.

## Decision rule (fixed in advance)
- An arm **wins** if, against H23a, at 24k steps on the full dev-clean **and** dev-other (beam 4, best of raw/EMA, chosen the same way for every arm), it has either:
  - **≥ 5% relative lower WER** at equal audio-hours, or
  - **WER no worse than +2% relative** reached in **≥ 25% less wall time** on the same GPU model. For a 24k-step run, this means the arm's WER at step ≤ 18k matches H23a's 24k WER.
- It must also not be less stable: no DIVERGED/PAUSED, no more grad-norm spikes than H23a × 2, and a final `max_qkv_norm` ≤ 1.5 × H23a's.
- **Multiple winners:** the one with the lowest dev-other WER goes forward to an H21-style long run.
- **H23e:** adaptive clipping becomes the new default if it meets the same rule against H23a, or is WER-neutral (±2% rel.) with fewer spikes.
- **Single seed:** differences under 2% relative are treated as noise. A winner by < 8% relative is confirmed with a second seed before adoption.

## Not in this wave
SOAP and Lion: only if GPU budget remains after a–e. Lookahead and joiner z-loss are implemented (`lookahead_k`, `zloss`) but are not arms. They are candidates for a follow-up if an arm shows instability at high lr.

## Runs
Filled in as runs launch: host, GPU, start time, commit.

| Arm | Host / GPU | Start (UTC) | Commit | Status |
|---|---|---|---|---|
