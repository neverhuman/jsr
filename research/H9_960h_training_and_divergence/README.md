# H9: 960 h training, divergence, and the qkv-norm cap

**Status:** ongoing (D2, D2MR, D3MRc and D4M are training).

## Question
How far does accuracy improve with all 960 h? Which recipe is stable?

## Setup
- **Hosts:** xbabe3 3090 (D2), xbabe2 3090 (D2MR), xbabe1 4090 (D3*), xbabe4 M3 Ultra via MPS (D4M).
- **Fleet guard:** every run is under `python/ultra/ultra/fleet_guard.py` (macOS: `train/mac_guard.py`). The owner rule is ≤ 90% mean CPU/GPU per host, and our jobs yield to other users.

## Results (dev-clean400 unless noted)

| Run | Recipe | Curve | State |
|---|---|---|---|
| D2 | dense S, 960 h, warm from D1d | 21.9 @5k → 17.3 @10k → 13.9 @20k → 11.4 @30k → **9.99 @40k** (dev-other 21.9) | training to 102k |
| D2MR | S_MR, warm from D1MR@2182 | 34.2 @5k → 20.4 @15k → 12.3 @35k → **11.3 @45k** | finishing at 50k |
| D3MR | S_MR_I8S (int8-QAT stem) + speed perturbation, warm from D1MR-40k, lr 8e-4 | 21.2 @5k → **100% @10k** | **DIVERGED** |
| D3MRb | same, lr 3e-4, plus a divergence guard | 16.6 @5k → **15.0 @20k** | **stopped by the guard @28.5k** |
| D3MRc | same plus the qkv cap (from 2k at 1.5x), the qkv guard (2x), the MR padding fix and EMA 0.9995 | relaunched fresh 2026-11-21 22:15Z | training (xbabe1) |
| R4T-960 | S_MR_T ternary QAT from D2MR-final, lr 3e-4, warmup 500, 15k steps, no qkv cap | 17.13 @2.5k | **FAILED** @4305 (grad-norm spike 94.5 at 4k; guard stop) |
| H13a | dense S from scratch, SentencePiece-512 for RNN-T and CTC, exact loss, EMA, speed perturbation, cap from 30k | — | training (xbabe2), 544 audio-h/h |
| D4M | M_MR (d 512) from scratch on MPS | early | training (v1.1 candidate) |

## What went wrong
- **D3MR and D3MRb diverged.** One attention block's qkv weight norm ran away (70 → 684 in D3MR; 70 → 426 in D3MRb), together with its pre-attention LayerNorm gain.
- **Likely cause:** that same block (`mid_blocks.5`) was already the largest in the warm-start checkpoint (D1MR-40k). AdamW weight decay applies, but it is far too weak at these learning rates. Dense from scratch (D2) grows evenly and saturates.
- **The ablations were inconclusive** (D3 recipe minus the int8 stem: qkv 94.5 @5k; minus speed perturbation: 98.7 @5k; D3MR itself: 190 @5k). Both ablations used `total_steps=5000`, so the learning rate decayed to 0.13e-3 by 4k, while D3MR ran on a 150k schedule. The comparison therefore cannot attribute the runaway. A rerun on the 150k schedule with `--max-steps 5000` is deferred unless D3MRc's qkv runs away.
- **R4T-960 failed** (ternary QAT at 960 h). At step 4000 the gradient norm spiked to 94.5 (typically 1–4) and the loss jumped from 1.07 to 7.45. The divergence guard stopped the run at 4305 while it was recovering (loss 1.26 at 4300). qkv rose steadily, 138 → 159, with no cap (the run predates it). It will not be relaunched now: ternary is v1.1 at best. Diagnostics: `/mnt/xbabe3-external/jsr/runs/train/R4T960_S_MR_T_960h/diagnostics/`.

## Consequence
- **Divergence guard:** stop when the loss EMA is 1.3× its best for 300 steps, or dev WER is 5 points worse (69a9bf2).
- **qkv-row and LN-gain cap** at 1.5× a reference step, plus a qkv-norm guard at 2×. The inference graph is unchanged (4d4fc03).
- An external watcher covers runs started before the guard existed.

## Pointers
- Commits: 69a9bf2, 4d4fc03, f7425dd, de84b39, c2ac1f7, ade28c2.
- Run dirs: `/mnt/xbabe3-external/jsr/runs/train/D2_S_960h`, and the xbabe1/xbabe2/xbabe4 run dirs named in `docs/ultra/STATUS.md`.
