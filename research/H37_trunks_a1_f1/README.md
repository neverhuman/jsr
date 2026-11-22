# H37: production trunks A1 (accuracy) and F1 (fast)

**Status:** launched on 2026-11-22 (owner-approved plan, 19:42Z; layout update 19:50Z). This is a run record, written at launch. The 4,096-token D2 line ended with D2f.

## A1: accuracy trunk
- **Model and seed:** dense S, R0 BPE-500, warm from H22ctl `ckpt-00079800` EMA weights (`seeds/A1_seed_ema.pt`, sha256 prefix `c44e3e9d26617913`).
- **Schedule:** AdamW, 1,000 warmup steps to a peak of 1e-3, one cosine sized for about 8 hours: **49,000 steps**, about 87.6k audio-hours. It started at 36,000 steps and was re-sized at step 2,509 from the measured rate.
- **Batch:** 1,300 s / 288 per rank (H22ctl's batch, 75–84% of 40 GB as measured), `window_groups` 96. Global batch 5,200 s on 4 ranks, 6,500 s on 5.
- **Chunk sampling:** each step draws the attention chunk from {4, 8, 16} (160/320/640 ms), the same on every rank. Dev eval at chunk 4 and chunk 8.
- **Ranks:** synchronous DDP with a hottest-rank throttle at 0.83. Started at 19:53Z on 4 ranks (xbabe5 GPU0, xbabe5 GPU1, xbabe7 GPU0, xbabe9) at 9.3k audio-h/h. xbabe7 GPU1 joined as the fifth rank at step 2,509 (20:17Z) when L3 ended: 11.7k audio-h/h.
- **Restart at step 2,509:** one restart re-sized the cosine and added the fifth rank. The first relaunch failed the start-state check because the new rank's directory had no `LATEST` file; the second came up with equal checksums on all 5 ranks. The larger global batch re-packs the window the restart fell in, so a small part of that window is skipped in that epoch.
- **Guards:** qkv cap at 1.5× the step-100 norms. The divergence guard is re-armed for the restart: the loss rule starts at step 8,000 and the WER rule only trips at +15 pp, because WER is expected to rise first under the high LR.
- **Expectation (lead):** WER rises early and falls below H22ctl's 4.95% (dev-clean400, EMA) late in the cycle.
- **Launcher:** `python/ultra/train/ops/a1_ddp.sh`.

## F1: fast trunk
- **Model:** S_MR, R0 BPE-500, from scratch, H22ctl's recipe: AdamW, lr 1e-3, 2,000 warmup steps, pruned loss from step 2,000, SpecAugment from step 4,000, speed perturbation, EMA.
- **Batch and length:** single A100 (xbabe8), 1,800 s / 360, 60,000 steps, about 30k audio-h. A first start at 2,000 s / 400 (L2's batch) reserved 88% of GPU memory after 50 steps, so it was restarted smaller within two minutes.
- **qkv cap from step 100, ratio 6.0.** The cap is relative to the step-100 row norms, which are near their initial value (about 20). Ratio 6.0 puts the cap at about 120–135, where healthy runs plateau (H22ctl 118, L3 114–117) and below L2's runaway (189). The usual ratio 1.5 would pin the norms at about 33.
- **Launcher:** `python/ultra/train/ops/f1_launch.sh`.
