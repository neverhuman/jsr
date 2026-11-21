# H4: INT8 export, and why a stream's output must not depend on its neighbours

**Status:** resolved (static export v1.1). The underlying principle later shaped the engine (H6).

## Question
Can R1 be exported to INT8 ONNX with WER within +0.10 pp of FP32, and with output that is identical whether a stream runs alone or batched?

## Setup
- **Export contract:** `ultra.r1-export/v1` and v1.1 (`python/ultra/train/export.py`).
- **Reference consumer:** `bundle_infer`.
- **Data:** 400 dev-clean utterances. Calibration used only TRAIN audio.

## Results
- **Dynamic INT8** (ORT DynamicQuantizeLinear, one scale per tensor) is **batch-composition dependent**. The same row alone vs in a batch of 8 differs by up to 0.27 in INT8 and by 0 in FP32 (codex probe `docs/ultra/reviews/probe-r1-batching.py`). Rust-vs-Python INT8 agreement was 88.5%.
- **Static QDQ, calibrated:**

| Variant | WER Δ vs FP32 | Batch-invariant |
|---|---:|---|
| encoder + joiner int8 | +0.56 to +0.67 | yes |
| **encoder int8, joiner FP32** | **+0.066** | yes |
| dynamic per-row (max_batch 1) | +0.04 | yes |

## What went wrong
The calibration capture initially:
- built joiner inputs that never actually occur;
- truncated instead of reservoir-sampling;
- used one hash name for two different things.

Codex found all three, and they were fixed.

## Consequence
- **Rule: batch invariance is a hard requirement.**
- The engine (H6) uses **per-row, centred activation quantisation**, which is batch-invariant by construction and gives +0.000 WER.

## Pointers
- Commits: 461909c, 1f27ada, 7b1774c, 5c370ec, d37893a.
- Bundle `/mnt/xbabe3-external/jsr/bundles/r1-d1-v1.1b`.
