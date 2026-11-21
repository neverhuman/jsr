# H1: Stream density of the pretrained recognizer (R0) and the SP-A batching question

**Status:** EXPLORATORY (shared hosts; not from a qualifying window).

## Question
How many live streams can one CPU host sustain with a good pretrained streaming recognizer? Does batching streams together (SP-A) lift throughput by ≥3×?

## Setup
- **R0:** icefall/sherpa Zipformer en 2023-06-26, int8 (`docs/ultra/candidates/zipformer-en-2023-06-26.yaml`).
- **Harness:** Python SP-A harness (`python/ultra/ultra/sp_a.py`, `sp_a_fleet.py`):
  - fixed capture deadlines, grid decode, oldest-unprocessed lateness, drain at cutoff;
  - pre-registration and amendment A1 (p99 ≤ 800 ms, no growth): `python/ultra/waves/SP-A/PREREG.md`.
- **Hosts:** H0/xbabe0, AMD TR PRO 3995WX (Zen 2, 16 CCX × 4 cores, 16 MiB L3 per CCX).

## Results
- **Batching lift:** 1.49× for R0 (1.2–1.46× for Nemotron), against the pre-registered 3×. Expected verdict **NO_LIFT**.
- **Mechanism:** density is bound by L3 per CCX. Density halves from 1 → 8 workers at a constant clock when workers share a CCX. One single-threaded worker per CCX is best.
- **Machine-wide:** **192 live streams sustained** (16 CCX workers × 12) at p99 ≤ 636 ms, dev-clean WER 2.80%, ~0.06 CPU-s/s (xbabe0, commit 6401b37).

## What went wrong
- **Early harness bugs** (found by codex review): deadlines rebased to "now", per-packet decode collapsing batches, cutoff spinning. Fixed in 1505371, 1a09202, 404c58c.
- **Invalid latency:** the Rust-runner latency figures from before the runtime clock fixes are INVALID_INSTRUMENT (467c48a).

## Consequence
- Density, not batching, became the headline lever.
- One worker per CCX became the deployment shape.
- **Caveat discovered later (H10):** this 192 figure uses 1 worker per CCX in the Python harness. It **must not** be compared with engine figures measured at 4 workers per CCX in a different harness.

## Pointers
- Commits: 8895600, 97612db, e537c3a, 6401b37.
- Docs: `python/ultra/waves/SP-A/README.md`, `PREREG.md`; ledger `docs/ultra/EXPERIMENTS.jsonl`.
