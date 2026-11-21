# H2: Baselines (Nemotron 0.6B, Moonshine v2, Whisper small.en) and the kernel cost law

**Status:** EXPLORATORY. These systems have *unknown overlap* with the LibriSpeech test sets.

## Question
1. Where do strong open recognizers sit on quality vs CPU cost?
2. Does the ternary kernel cost model (W01) hold on our hardware?

## Setup
- **Nemotron 0.6B Q8_0:** run through NeMo-Speech.cpp with a process adapter, at 160 ms and 560 ms chunks (`docs/ultra/candidates/nemotron-0.6b-q8.yaml`, `nemotron_results.md`).
- **Contrast set (P05):** Moonshine v2 base/tiny int8 and Whisper small.en int8 through the sherpa offline runtime (`docs/ultra/waves/P05`).
- **Kernels:** `crates/ultra-kernels` AVX2 int8/T1/T-LUT2, bit-exact against scalar references; cost fit in `ultra-bench`.

## Results (LibriSpeech; PUBLIC-TEST reads were logged as historical reconstructions in `docs/ultra/LOOKS.jsonl`)

| System | test-clean WER | test-other WER | Single-stream CPU-s/s |
|---|---:|---:|---:|
| Nemotron 160 ms | 2.65% | 5.60% | 0.709 (quiet) |
| Nemotron 560 ms | 2.58% | 5.18% | 0.451 (quiet) |
| Moonshine base / tiny | 2.55% / 4.51% (100 utts) | — | 0.122 / 0.058 |
| Whisper small.en | 3.77% (50 utts) | — | 1.94 |

- **Cost law W01 PASS** on H0: the T-LUT2 fixed term is 0.18× int8 (4da6062).

## What went wrong
- **Adapter bugs** (codex N1): stale WAV reuse and shard-merge errors. Fixed in 1084550.
- **Test-set reads:** LibriSpeech test was read in exploratory runs before any freeze. This is disclosed in FREEZE §4.

## Consequence
- Nemotron is the accuracy reference, about 6–8× costlier per stream than R0.
- Later (H10), Nemotron was **excluded from the density and cost claims**: it runs as a child process with a non-comparable cost scope.

## Pointers
- Commits: d2f9f6c, 718f107, 2701d2d, 5232d61, 2fa2846, 4da6062.
