# H25: Engine Runtime Hygiene, `M=1/2` Tile Fixes, INT8 K/V Cache & H17 Loose Ends

Status: **PRE-REGISTERED** (JSR v2 Phase 1 — zero-WER-risk exact speedups & state compression).

## 1. Question
How much whole-service CPU, tail latency (`p99`), and per-stream memory state (`6.70 MiB` Dense S, `4.83 MiB` `S_MR`) can we cut on **both H0 Zen 2 AVX2 and xbabe4 M3 Ultra NEON** without changing model architecture or retraining?

## 2. Scope & Pre-Registered Changes
1. **Prepared execution plan & zero-alloc runtime (`crates/ultra-engine`):**
   - Assign stable matrix IDs in `model/layers.rs` (eliminate per-call linear scans & `Vec` rebuilds).
   - Replace per-GEMM activation copies/allocs/locks in `pool.rs` with fixed-capacity reusable job slots.
   - Replace hypothesis cloning in `beam.rs` with a bounded prefix arena, context-keyed stateless predictor cache, and tropical max-plus `logsumexp` LUT (`0-ULP` or verified parity).
   - Cache 1 overlapping stem `conv1` row (`19.9 KB/stream`) in `stem.rs` and switch long-session detokenization to incremental `O(1)` appending (fixes quadratic 1-hour replay cost).
   - Add specialized small-`M` (`M = 1, 2`) micro-tiles so `S_MR` (12.5 Hz, `M=2` per 160 ms chunk) and future `V2` 6.25 Hz U-Net blocks (`M=1` per chunk) never waste half/three-quarters of `MR8 = 4` register tiles.
2. **K/V Cache Compression (`6.70 → 1.92 MiB/stream`, 3.5× state cut):**
   - Deterministic per-token/per-head INT8 K/V cache (quantized on write, never re-quantized from FP32; tile-dequantized in registers).
   - Directly increases L3 CCX residency at high multi-stream density (`32–144+` streams/CCX).
3. **H17 Spec-Correction Loose Ends:**
   - **DONE (verified on `main`):** `ternary + act7` silent arithmetic switch at `M = 8` is already fixed in `crates/ultra-engine/src/encoder/mod.rs:247` (`!q.act7` bypasses T-LUT2) and locked by `crates/ultra-engine/tests/ternary_act7.rs`.
   - **DONE (PR #24):** Persist and restore `loss_ema`, `best_ema`, and `best_wer` in `Trainer.payload()` / `Trainer.resume()` (`python/ultra/train/trainer.py`), covered by `python/ultra/tests/test_train_cr_ctc.py`.
   - Audit/fix `data.py` speed-perturbation post-duration-drop bias (`0.9×` slow clips dropped near `max_seconds`).
   - Unify `joiner_logprobs` QAT effective-weight interface before v2 codebook QAT.

## 3. Decision / Promotion Rule
- **Exact runtime cleanups:** Promote on `≥ 10%` whole-service CPU reduction at **100% token-identical** output on the 730-utterance panel (`x86 AVX2 == M3 NEON`).
- **INT8 K/V cache:** Promote if `≤ +0.05 pp` point WER delta (`≤ +0.25 pp` 95% bootstrap UCB) vs FP32 K/V on the 730-utt panel with measurable CCX density or RSS win.

## 4. Pointers
- Sources: `tips/v2/JSR_CPU_ASR_BREAKTHROUGH_RESEARCH_SPEC.md` §13, `tips/v2/JSR_CPU_Accuracy_Frontier_Engineering_Spec_2026-11-21.md` §6–7, `tips/v2/JSR_CPU_ASR_Engineering_Spec_2026-11-21.md` §4/§6, `research/H17_spec_corrections/README.md`.
