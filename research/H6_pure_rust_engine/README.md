# H6: Pure-Rust inference engine (`ultra-engine`), x86 AVX2 and ARM NEON

**Status:** EXPLORATORY speed numbers. Correctness is gated (parity, continuity, quality).

## Question
How fast can R1 run on CPU if ONNX Runtime is replaced by a fused, hand-written Rust engine, with no loss of quality?

## Setup
- **Plan:** `docs/ultra/ENGINE_PLAN.md`.
- **Engine:** crate `crates/ultra-engine`:
  - safetensors weights;
  - FP32 path and per-row **centred** INT8 activations with per-channel int8 weights;
  - AVX2 (`vpmaddubsw`), NEON (`sdot`/`smmla` via stable asm);
  - prepared INT8 artifact;
  - one worker per CCX.
- **Gates:**
  - per-layer parity vs torch (≤ 1e-4);
  - 400/400 FP32 transcripts identical vs the reference bundle;
  - INT8 ≤ +0.10 pp (later a bootstrap bound on the 730-utterance panel);
  - output-continuity fixtures (bit-identical across kernel changes).

## Results
- **FP32:** 400/400 identical transcripts; worst per-layer error 2.4e-6.
- **INT8 centred per-row:** WER **+0.000**, batch-invariant (B = 1/3/5/16/32 bit-identical).

**Whole service, same build, D2 @5k (bridge, exploratory):**

| Path | WER | CPU-s/s | RSS |
|---|---:|---:|---:|
| ORT FP32 | 21.88% | 0.175 | 776 MB |
| engine FP32 | 21.88% | 0.104 | 478 MB |
| **engine INT8** | 21.80% | **0.041** | 647 MB → 259 MB (prepared) |
| R0 Zipformer (same session) | 2.85% | 0.078 | 239 MB |

- **ARM (M3 Ultra):**
  - the engine is bit-identical to x86 given the same features;
  - single-stream INT8 is 0.0131 (dense) and 0.0104 (S_MR);
  - per-core density about 2.5× Zen 2 (dev tool).
- **Panel INT8 gates** (730 utterances, 73 speakers): all pass on x86 and ARM (`ultra-engine-gate`).

## What went wrong
- **The first INT8 attempt (uncentred) gave +0.207.** Centring fixed it.
- **A faster attention kernel changed the summation order,** flipping 33/400 INT8 transcripts. Fixed by restoring the order; the rule since then is that kernel changes must be bit-identical.
- **The density harness first under-counted lateness and credited unprocessed audio.** The "24/core" figure was retracted.
- **The loader claimed checks it didn't perform** (an edit had silently failed). This was caught by review and fixed.

## Consequence
- The engine is the R1 deployment path.
- **The density comparison with R0/ORT must come from one harness** (see H10).

## Pointers
- Commits: 2b0b2d7, 9aa3223, d596935, c3e073d, 2a966c0, 114aaa6, 929bdd3, 87a4a51, b9acfb0, cc9d0fb.
- Reviews: `docs/ultra/reviews/2026-11-21-engine-e1-review.md`.
