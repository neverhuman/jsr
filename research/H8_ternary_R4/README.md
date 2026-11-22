# H8: Ternary (R4) quantisation-aware training pilot

**Status:** EXPLORATORY; the pilot is concluded.

## Question
Can S_MR be trained with ternary weights {−1, 0, +1} (per-channel absmean scale) at a small quality cost? Does a 13–14 MiB weight set that fits a Zen 2 CCX's 16 MiB L3 make the engine faster?

## Setup
- **Trainer:** preset `S_MR_T`: 132 Linear layers ternary with a straight-through estimator. Activations are fake-quantised per row and centred, exactly as the engine does it (dfabd0a, 875a368).
- **Run R4T:** warm start from D1MR ckpt-40000, 12k steps at lr 3e-4, xbabe1.
- **Export:** `ultra.r4-ternary/v0` in T1 layout. The engine has a ternary layer type: T-LUT2 on x86 and sdot/smmla on NEON (c7ae75b, c4ec711).

## Results
- **Quality:** the R4T curve went 22.8% @1k → 18.6% @9k → **18.04% final** (torch), against the FP32 baseline of 17.41% on dev-clean400. The engine scored 17.98% (it passed the gate: engine WER ≤ torch + 0.10).
- **Size:** 14.26 MiB of trits, with 31.8% zeros.
- **Speed:**
  - The ternary kernel streams from DRAM without slowing down, and it wins at a single stream.
  - At batch (16 streams and up), the engine is **compute-bound** and the int8 GEMM is faster. Ternary density was 112/CCX vs 136 for int8 S_MR on x86, and 432 vs 504 on 6 Mac workers.
- **First stage:** post-training int8 of the stem saved 4–8% but failed the +0.10 margin.

## Consequence
- Ternary is valuable for memory and single-stream latency, **not** for batch density on these CPUs.
- The density claim stays on INT8 S_MR.
- Future recipes should train the stem as int8 or ternary.

## Pointers
- Commits: dfabd0a, 875a368, c7ae75b, c4ec711, 7f89da8.
- Checkpoint copy: `/mnt/xbabe3-external/jsr/runs/train/R4T_S_MR_T_tc100_copy/`.

## 960 h follow-up (R4T-960): FAILED
- **Run:** S_MR_T ternary QAT from D2MR-final (ckpt-00050000), 960 h, lr 3e-4, warmup 500, 15k steps, SpecAugment from step 0, no qkv cap (the run predates it). xbabe2, 2026-11-21.
- **Result:** dev-clean400 was 17.13% at 2.5k. At step 4000 the gradient norm spiked to 94.5 and the loss jumped from 1.07 to 7.45. The divergence guard stopped the run at 4305 (loss EMA 2.76 vs best 1.15) while the loss was recovering. The max qkv norm rose steadily, 138 → 159.
- **Decision:** not relaunched; ternary is v1.1 at best. If revisited: the qkv cap and guard, a lower learning rate or a longer warmup, and the MR padding fix (H12).
- **Diagnostics:** `/mnt/xbabe3-external/jsr/runs/train/R4T960_S_MR_T_960h/diagnostics/` (DIVERGED.json, train.jsonl, config, stdout, guard log).
