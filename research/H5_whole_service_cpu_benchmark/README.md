# H5: Whole-service CPU benchmark (M1/M2) and measurement hygiene

**Status:** EXPLORATORY.

## Question
What does each system cost end to end (the whole service process, warm), on identical audio, scored by one Rust scorer?

## Setup
- **Harness:** `crates/ultra-bridge`, an audio-only job with a reference-blind worker. Outputs are retained and Rust-scored afterwards.
- **Workload:** DEV400 panel, workload sha fb8241b9….
- **Measurement:** `ultra-measure`, process-tree CPU (`crates/ultra-eval`).

## Results (DEV400 dev-clean)

| System | WER | CPU-s/s | RSS |
|---|---:|---:|---:|
| R0 Zipformer int8 | 2.85% | 0.137 → **0.078** (see below) | 239 MB |
| R1-D1 (100 h) via bridge | 29.83% | 0.141 | 593 MB |
| Nemotron 560 ms (child process) | 2.16% | 0.472 | 1.28 GB |
| R1 D2 @5k, ORT FP32 | 21.88% | 0.175 | 776 MB |

## What went wrong / surprised us
- **Observer co-location:** the measuring process shared the service's core. Moving it to a separate CCX dropped R0 from 0.137 to 0.102 CPU-s/s.
- **Output buffering:** writing each output record in one write cut another 14.8%, to 0.087 and later 0.078.
- **Host noise:** a same-day A/A replay of one row varied by **+25%**. Single runs on shared hosts are noisy, so FREEZE requires a quiet host plus ABAB blocks.
- **Panel too narrow:** the first-400 DEV panel covers only **6 speakers**, so its intervals are fragile. A speaker-stratified 730-utterance panel was built (`docs/ultra/data/dev_selection_panel.json`).

## Consequence
- Freeze rules: the observer sits on a disjoint CCX; cost scope is labelled on every row; numbers are only quoted from the same build and session.

## Pointers
- Commits: ed7f1df, f200b1d, 24c1b4b, 64c1b4d, 217962a, 4ae588e.
- Reports: `docs/ultra/reviews/2026-11-20-trained-cpu-benchmark.md`, `2026-11-21-service-cpu-control.md`.
