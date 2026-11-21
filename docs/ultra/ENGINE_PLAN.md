# ENGINE_PLAN: a pure-Rust R1 CPU engine (`ultra-engine`)

Owner request (2026-11-21): port R1 inference aggressively to the fastest possible CPU
path. Rust is preferred.

## Why Rust, and why not C++20, OpenMP or Fortran
- **Same machine code:** Rust `std::arch` intrinsics compile through LLVM, like clang C++. We
  already have AVX2 int8, T1 and T-LUT2 kernels in `ultra-kernels` that are bit-exact
  against scalar references.
- **No OpenMP:** threading is our own CCX-pinned workers. SP-A showed density is L3-bound
  per Zen 2 CCX, so the win is one worker per CCX with weights reused across that worker's
  streams, not a generic thread pool.
- **No Fortran:** it would buy nothing here and break the single-language audit, build and
  supply chain.
- **safetensors:** used as the weight format: mmap'd, zero-copy, hashed, no Python pickle.

## Where the time goes today (measured, exploratory)

| Path | CPU-s/s | Scope |
|---|---:|---|
| R1-D1 int8, ORT dynamic | 0.149 | engine |
| R1 FP32, native ORT | 0.194 | whole service |
| R0 Zipformer int8 | 0.065 | engine |
| R0 Zipformer int8 | 0.137 | whole service |

- **Model cost:** R1 S dense is d=384 with 16 Conformer blocks, about 1.36 GMAC per audio-s
  and about 52 MiB of int8 weights. S_MR does about 31% fewer MACs.
- **Hardware (H0 and xbabe0-2):** Threadripper 3995WX (Zen 2). AVX2 + FMA, no AVX-512 or
  VNNI. 16 CCX of 4 cores, 16 MiB L3 per CCX.
- **Roofline:** about 100+ int8 GMAC/s per core peak (`vpmaddubsw`/`vpmaddwd`). One stream
  needs 1.36 GMAC/s, so a fused int8 engine at 30–50% of peak is about 20–40 streams per
  core in compute. Today's 0.149 is about 7.
- **ORT costs:** dispatch, per-op memory traffic, unfused LayerNorm/activation passes and
  per-tensor dynamic quantisation (also batch-dependent: the INT8 finding).

## Levers, in order
1. **Fused int8 GEMM per layer, batched across all streams in the worker.** One weight pass
   serves `B streams × chunk frames` rows, so weights stream from DRAM/L3 once per chunk step.
   - Packed weights: per-output-channel scales, a BLIS-style register-blocked micro-kernel
     on AVX2 (`vpmaddubsw` u8×s8 into i16 pairs into i32, or the `vpmaddwd` path for exact
     i32).
   - Fused into the epilogue: bias, SiLU/GLU, residual add and the next LayerNorm.
2. **Per-row (per-frame) dynamic activation quantisation.** It is batch-invariant by
   construction: a stream's numbers never depend on its neighbours.
3. **Attention in fp32 with a KV ring cache:** 4 queries × ≤132 keys × 6 heads, tiny.
   RoPE is precomputed. The causal depthwise conv uses a state ring. No reallocation per
   chunk; everything lives in a per-worker arena.
4. **Threading and placement:**
   - One worker per CCX, holding all of that CCX's streams.
   - Option to split GEMM output rows over the CCX's 4 cores, so each core's 512 KiB L2
     holds its weight slice.
   - Measure both, with the observer on another CCX (freeze rule).
5. **Multirate S_MR:** fewer MACs in the middle 10 blocks at 12.5 Hz.
6. **Ternary (R4), T-LUT2:** 13 MiB of weights fit in one CCX's 16 MiB L3, so no DRAM
   traffic. The kernel cost law is already measured (fixed term 0.18× int8). This needs a
   ternary-trained model (QAT, GPU), so it is a later rung, not v1.0.0.
7. **ARM NEON (`sdot`/`i8mm`) backend** for the M3 Ultra hosts, after x86.

## Stages and acceptance
- **E0, weights:** a converter from the trainer checkpoint to safetensors plus an
  architecture JSON (`ultra.r1-weights/v1`), hashed. Read-only on checkpoints.
- **E1, fp32 reference engine (pure Rust, scalar plus AVX2 fp32):**
  - per-layer outputs within 1e-4 relative of ORT fp32 on 50 DEV utterances;
  - 400/400 identical transcripts versus the FP32 bundle.
- **E2, int8 per-row engine:**
  - WER within +0.10 of fp32 on 400 DEV;
  - batch invariance: identical output for B = 1, 3, 5, 16 and 32 with mixed positions and
    partial chunks.
- **E3, speed:**
  - single-stream engine CPU-s/s at or below **0.05** for S dense int8 (target 3× vs ORT
    0.149), and at or below 0.035 for S_MR;
  - per-CCX density at least 2.5× the ORT path at the A1 latency rule.
  - All exploratory until a qualifying window.
- **E4, integration:** an `ultra-stream` backend (runtime-lane owns the trait; the engine
  supplies a library), plus a whole-service row through `ultra-bridge` M2 (codex).
- **E5, later:** a T-LUT2 path for a ternary model and a NEON backend.

## Rules
- **Ownership:** engine-lane owns the new crate `crates/ultra-engine/**` and the additive
  kernel extensions in `crates/ultra-kernels/**` (no API breaks), plus
  `python/ultra/tools/r1_to_safetensors.py`.
- **Off-limits:** no edits to `ultra-stream`, `ultra-cli`, `ultra-gp` (runtime-lane) or
  `train/**` (trainer). Ask through AGENT_CHAT.
- **`unsafe`:** only in arch modules, each block with a `SAFETY:` comment. Every SIMD kernel
  has a scalar reference and a bit-exactness or tolerance test.
- **Cores:** CCX 56-59 and 60-63 for engine timing (never the SMT siblings 120-127). Builds
  use `CARGO_BUILD_JOBS=6` at nice 10 on 48-51. The owner's 90% host cap applies.
- **Gate:** `just required` (`--locked`) and jankurai ≥ 85 with 0 caps before every DONE.
  Codex reviews before any claim.
