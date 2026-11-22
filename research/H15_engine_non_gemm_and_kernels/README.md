# H15: engine speed — bit-identical non-GEMM cuts, an unsigned-offset int8 tile, and two correctness checks

**Status:** `EXPLORATORY`. B1 is merged-ready and bit-identical. B2 failed its bar and stays experimental. Both checks pass. All timing comes from shared hosts with stated load, so read it as indicative.

Plan: `docs/ultra/NEXT_WAVES_PROPOSAL.md` §B (B1, B2), assigned 2026-11-21 23:00Z. Code: `lane/engine`
3e1882f (B2 + staging script), 39dcc4a (B1 + ternary/act7 fix), 103c983 (staging provenance).

## Questions

1. **B1.** Do bit-identical cuts outside the GEMM give the hoped-for 10–15%? The cuts: build the RoPE table once per step, not once per attention call, and fuse each residual add with the norm after it.
2. **B2.** Does an unsigned-offset 7-bit int8 tile reach ≥ 1.2× on the GEMM against the signed tile (K0) and `act7` on Zen 2? It drops `vpabsb` and both `vpsignb`, and its int32 sums are identical to `act7`.
3. **S_MR pair padding.** Can the engine's pair path mix padding into the last real frame? That was the trainer-side bug fixed in ac30d53.
4. **grok-lane bug 3.** Does ternary + `act7` change activation width at the 8-row T-LUT2 cap?

## Results

### B1: bit-identical, but small

- **Hosts and load.** Run on xbabe0 (3995WX), core 60, against the panel (730 utterances, prepared INT8, qualifying mode). Host load was 18–20 from another project's evaluation jobs.
- **Method.** ABAB order: 3e1882f (before) vs 103c983 (after), CPU-s per audio-s.

| Run | Before | After | Attention phase before → after |
|---|---:|---:|---:|
| D2MR-final, 1 stream (repeat 1) | 0.0393 (outlier) | 0.0358 | 0.0033 → 0.0028 |
| D2MR-final, 1 stream (repeat 2) | 0.0357 | 0.0360 | 0.0029 → 0.0028 |
| D2 @45k dense, 1 stream | 0.0373 | 0.0372 | 0.0046 → 0.0044 |
| D2MR-final, 16 streams | 0.0322 | 0.0304 | 0.0037 → 0.0034 |

- **Identity.** Tokens are identical to before, 730/730, at 1 and 16 streams. Mac NEON at 103c983 matches Mac at 4818a94, 730/730. The output and encoder continuity fixtures, batch invariance, layer parity and the prepared-INT8 tests pass unchanged.
- **Size of the gain.** Attention drops 3–10%. The fused add+norm saves about 0.0001: `activation_residual` goes 0.0007 → 0.0006 and `layer_norm` is unchanged. Whole-engine change is 0–5%, inside run-to-run noise on a loaded host. **Not the 10–15% projected.**
- **Why so small.** The non-GEMM work is not mostly RoPE or norm passes. In the after profile it is:
  - attention kernels (scores and value mixing), 0.0028–0.0044;
  - the stem, 0.0007;
  - quantisation, 0.0004;
  - depthwise conv, 0.0006.

  The int8 GEMM is still 75–80%.

### B2: the unsigned-offset tile is slower on Zen 2

- **Setup.** `examples/u7_bench`, real dense-S layer shapes. Hot means one layer's weights are resident; streaming means all 16 blocks, about 54 MiB.
- **Exactness.** Bit-exact against `act7` and scalar on every shape before timing (`kernels::tests::gemm_i8_u7_equals_act7_and_scalar_including_extremes`).
- **Numbers.** Best of 3, ps/MAC:

| Host (core, load) | Rows | K0 | act7 | U7 | U7 vs K0 | U7 vs act7 |
|---|---:|---:|---:|---:|---:|---:|
| H0 (62, ~22) | 4 hot | 16.66 | 15.75 | 17.63 | 0.94× | 0.89× |
| H0 (62, ~22) | 16 hot | 16.34 | 15.42 | 17.29 | 0.95× | 0.89× |
| H0 (62, ~22) | 64 stream | 16.82 | 15.95 | 17.36 | 0.97× | 0.92× |
| xbabe0 (52, ~20) | 16 hot | 18.83 | 17.25 | 18.72 | 1.01× | 0.92× |
| xbabe0 (56, ~20) | 64 stream | 18.68 | 17.64 | 19.88 | 0.94× | 0.89× |

- **Verdict: FAIL** (0.82–1.01×, bar 1.2×). U7 is not wired into the engine. The kernel is kept as an exactness-tested experiment. No NEON twin, per the plan.
- **Reading.** U7 removes three non-multiply ops but keeps four vector multiplies per 64 MACs. `act7` removes one multiply and wins by about 6%. On Zen 2, ps/MAC follows the multiply count more than the total instruction count. That partly rehabilitates the "one multiply pipe" model that `act7` alone had seemed to refute.
- **What it implies.** A real gain on this host needs fewer multiplies: 7-bit weights with paired accumulation (K2, a weight-numerics candidate), or a multiply-free lookup (B4).

### S_MR pair padding: the engine already pairs with zeros

- **Code.** `encode_chunk` zero-fills `pair_in` before copying the `n` real rows, so an odd final row pairs with zeros. Padding never reaches the pair projection. The middle rows are `ceil(n/2)`, and restoration drops the padded 25 Hz position.
- **Evidence.** Full-frame layer parity of D1MR s18200 against the trainer's streaming `ChunkEncoderMR`, 50 DEV utterances, **26 of which have an odd encoder length**. Every layer ≤ 3.8e-6 normwise, every frame including the last. ChunkEncoderMR has the behaviour that ac30d53 brought to the batched training path.

### grok-lane bug 3: confirmed and fixed

- **Bug.** T-LUT2, used for ≤ 8 rows, quantises activations to 8 bits. With `act7` the int8 tile uses 7 bits, so with act7 on a ternary layer's output depended on how many rows shared the call.
- **Fix (39dcc4a).** `act7` never routes to T-LUT2.
- **Test.** `tests/ternary_act7.rs` checks rows 1/8/9/16 with and without act7. It fails without the fix ("row 0 depends on the batch size (act7 = true)") and passes with it. The default path has act7 off, so it is unchanged.

### Also measured

- **D2MR-final, Zen 2 single stream (xbabe0, core 60, load ~20, 3e1882f):** 0.0446 cpu-s/s, then 0.0357–0.0393 in the ABAB runs. Both are contaminated by the co-running evaluation jobs. The qualifying-host measurement (xbabe2 window) supersedes them.
- **B6 sizing, from the same profiles.**
  - The greedy decoder costs about 0.0016 cpu-s/s, roughly 4–5% of single stream: joiner int8 0.0012, predictor fp32 0.0001, bookkeeping 0.0003.
  - A bounded beam of 4 multiplies joiner and predictor work per frame by about 4. With batched hypotheses at the same ps/MAC, that adds about 0.005–0.006 cpu-s/s (+13–17% single stream) before any blank-skip.
  - The exact blank-first certificate (V2 §2.5) would claw part of that back.
  - This is an estimate, not a measurement.

## Decision

- Merge B1: bit-identical, with a small and real attention win.
- Drop B2 for Zen 2.
- Next speed levers on this hardware:
  1. a multiply-free ternary lookup (B4);
  2. 7-bit-weight paired accumulation (a new numerics candidate);
  3. INT8 K/V attention (B3), since attention is the largest non-GEMM phase.

## Reproduce

```text
# B2 (any AVX2 host): taskset -c <core> target/release/examples/u7_bench
# B1 profiles: ultra-engine-eval --weights <prepared> --panel docs/ultra/data/dev_selection_panel.json \
#   --streams {1,16} --precision int8 --qualifying --profile --out <dir>
```

Raw outputs:
- H0: `/mnt/xbabe3-external/jsr/runs/engine/h15-u7bench-*`;
- xbabe0: `~/jsr-engine-data/runs/x0-3e1882f-232341`, `x0-b1-232831`;
- Mac: `~/jsr-engine-data/runs/engine/mac-b1-103c983`.

---

## Addendum (runtime-lane, 2026-11-22): B1 Frontend — 0-ULP Sparse Triangular Mel + Precomputed Bit-Reversal + Unfused SIMD FFT

**Status:** `EXPLORATORY` (merged onto `main` @ `4efa7f2` from `lane/runtime-h15-mel @ 809a72c`).

- **Question:** Can we cut portable log-mel frontend CPU cost while keeping **0-ULP bit-identity** across x86_64 AVX2, ARM64 NEON, and scalar reference (preserving G4's 7,300/7,300 cross-ISA identity from H11)?
- **Implementation:** `crates/ultra-frontend/src/{portable_math,trainer_mel,trainer_mel_avx2}.rs`, `crates/ultra-frontend/tests/sparse_mel_parity.rs`:
  1. Precomputed sparse triangular mel filterbank (`477` nonzeros vs `20,560` dense = **43× fewer MACs**, ~`5 KiB` L1d-resident vs `82 KiB`, automatic dense fallback on non-finite inputs).
  2. Precomputed 512-point FFT bit-reversal table.
  3. Unfused 4-wide AVX2 (`vmulpd`/`vaddpd`/`vsubpd` on `x86_64`) and 2-wide NEON (`vmulq_f64`/`vaddq_f64`/`vsubq_f64` on `aarch64`) FFT butterflies with zero FMA contraction.
- **Verified results (`cargo test --locked --release -p ultra-frontend --test sparse_mel_parity -- --nocapture`, 10,000 frames = 100 audio-s):**

| Host / ISA | Dense portable (CPU-ms/s) | Sparse + SIMD FFT (CPU-ms/s) | Speedup | ULP diff vs dense |
|---|---:|---:|---:|---:|
| H0 AMD Zen 2 (`x86_64` AVX2) | 1.126 | **0.317** | **3.55×** | **0 ULP** |
| `xbabe4` Apple M3 Ultra (`aarch64` NEON, verified by lead) | 0.546 | **0.193** | **2.83×** | **0 ULP** |

- **Continuity:** `ultra-engine --test output_continuity`, `batch_invariance`, and `prepared_int8` pass unchanged on both x86_64 and ARM64 (0 fixture updates).
