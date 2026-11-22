# H37: production trunks A1 (accuracy) and F1 (fast)

**Status:** launched on 2026-11-22 (owner-approved plan, 19:42Z; layout update 19:50Z). This is a run record, written at launch. The 4,096-token D2 line ended with D2f.

## A1: accuracy trunk
- **Model and seed:** dense S, R0 BPE-500, warm from H22ctl `ckpt-00079800` EMA weights (`seeds/A1_seed_ema.pt`, sha256 prefix `c44e3e9d26617913`).
- **Schedule:** AdamW, 1,000 warmup steps to a peak of 1e-3, one cosine sized for about 8 hours. `total_steps` starts at 36,000 and is re-derived from the measured rate at the first checkpoint if it is far off.
- **Batch:** 1,300 s / 288 per rank (H22ctl's batch, about 85% of 40 GB), `window_groups` 96.
- **Chunk sampling:** each step draws the attention chunk from {4, 8, 16} (160/320/640 ms), the same on every rank. Dev eval at chunk 4 and chunk 8.
- **Ranks:** 4-rank synchronous DDP on xbabe5 GPU0, xbabe5 GPU1, xbabe7 GPU0 and xbabe9; hottest-rank throttle at 0.83.
- **Guards:** qkv cap at 1.5× the step-100 norms. The divergence guard is re-armed for the restart: the loss rule starts at step 8,000 and the WER rule only trips at +15 pp, because WER is expected to rise first under the high LR.
- **Expectation (lead):** WER rises early and falls below H22ctl's 4.95% (dev-clean400, EMA) late in the cycle.
- **Launcher:** `python/ultra/train/ops/a1_ddp.sh`.

## F1: fast trunk
- **Model:** S_MR, R0 BPE-500, from scratch, H22ctl's recipe: AdamW, lr 1e-3, 2,000 warmup steps, pruned loss from step 2,000, SpecAugment from step 4,000, speed perturbation, EMA.
- **Batch and length:** single A100 (xbabe8), 2,000 s / 400 (L2's S_MR batch on a 40 GB A100), 54,000 steps, about 30k audio-h.
- **qkv cap from step 100, ratio 6.0.** The cap is relative to the step-100 row norms, which are near their initial value (about 20). Ratio 6.0 puts the cap at about 120–135, where healthy runs plateau (H22ctl 118, L3 114–117) and below L2's runaway (189). The usual ratio 1.5 would pin the norms at about 33.
- **Launcher:** `python/ultra/train/ops/f1_launch.sh`.
