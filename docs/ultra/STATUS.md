# STATUS (updated 2026-11-21 ~22:00Z)

- **Qualified rung:** none yet. **Every number is exploratory.**
- **Controlling docs:** `docs/ultra/PLAN_TO_V1.md` (gates G1–G9), `docs/ultra/FREEZE.md` (rev 3, not yet tagged).
- **Research history:** `research/H1…H11` (next: H12).

## Accuracy (dev-clean400 unless noted; greedy streaming)

| Run | Host | Best so far | State |
|---|---|---|---|
| **D2** dense S, 960 h | xbabe3 3090 | **8.52%** @50k, epoch 7 (dev-other 19.60%) | training to 102k, ETA about 11-23 morning |
| D2MR S_MR, 960 h | xbabe2 3090 | **11.07%** @49.5k, epoch 5 | finishing at 50k; then the final eval and selection (B2) |
| D3MRb S_MR + int8 stem + speed perturbation | xbabe1 | 15.0% @20k | stopped by the divergence guard (qkv runaway); cap fix in 4d4fc03; ablations A/B running; then D3MRc |
| D4M M_MR (d 512), from scratch | xbabe4 M3 (MPS) | early | v1.1 candidate, about 3.5 days under the guard |
| R4T ternary, 100 h | — | 17.98% (engine) | done (H8) |

Reference: R0 Zipformer 2.85%, Nemotron 0.6B 2.16% on DEV400. **The R1 accuracy gap remains.**

## Speed (whole service, same build and session, H0 Zen 2; CPU-s per audio-s)
- **Engine INT8 R1: 0.041**, against R0 0.078 and ORT FP32 0.175. Nemotron is 0.472 (different scope).
- **Single stream INT8:** Zen 2 0.040; M3 Ultra 0.0132 (S) and 0.0105 (S_MR).
- **Density:** not claimed. It must come from the single-harness qualifying run (FREEZE §4.4).

## On main (5174e6c / 1b1b615)
- **ultra-engine:**
  - FP32 and per-row INT8, S/S_MR/ternary/int8-stem;
  - AVX2 and NEON, bit-identical across ISAs, including PCM→tokens, via G4 portable log-mel (7300/7300);
  - panel, recorded identities, `--qualifying`, bootstrap INT8 gate (exit codes), sealed test reads.
- **bench-suite:** single harness (unified A1, 320 ms / 2-chunk grid, ladder, panel, `--qualified` preflight, placement, sustained series, LOOKS ledger, ABAB runner).
- **Trainer:**
  - S/S_MR/S_MR_T/S_MR_I8S/M_MR;
  - MPS;
  - divergence guard and qkv cap;
  - panel eval with test splits refused;
  - ops scripts under `train/ops/`.
- **Independent reviews:**
  - pass 1 (7 blocking);
  - pass 2;
  - pass 3: 13 resolved / 6 partial / 1 deferred, plus N9–N12.

## Open before `bench-freeze-v1`
1. **runtime-lane, pass-3 items:**
   - N9 same-L3 check;
   - `--qualified` forces the frozen ladder;
   - N10 ABAB (8 blocks, paired t-test);
   - B4 loaded-artifact pins;
   - S6 suite guards;
   - N11.
2. **G1:** finalists selected on the 730-utterance panel: D2 (dense, about 11-23) and S_MR (D2MR-final vs D3MRb@20k vs D3MRc).
3. **G2:** engine prepared-INT8 gates for the finalists.
4. **Lead:** CANDIDATES.lock pins for the finalists, full hashes, the final FREEZE text, then the tag.
5. **G6, a quiet x86 host:** **owner decision needed.** H0, xbabe0 and xbabe2 all carry other projects' unpinned jobs.

## Risks
- Shared hosts: A/A replays vary by up to +25%.
- H0's root disk ran out once and now has about 11 GB free. Other projects write there.
- The ultra-bridge fused_native tests time out under heavy host load.
