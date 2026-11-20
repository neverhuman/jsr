# SP-A: real-time stream density at unchanged WER

- **Owner:** lead.
- **Registered:** 2026-11-20, before any paced density run.
- **Status:** OPEN.

## Hypothesis
Batching every ready stream into one weight pass per tick lets one H0 core sustain at least 3× as many paced real-time streams as decoding each stream alone. It does this with the same transcripts (ΔWER within ±0.05 points, paired) and with p99 lateness under one model chunk (320 ms).

## Fixed identities
| Item | Value |
|---|---|
| Model | icefall streaming Zipformer (LibriSpeech 960 h only), sherpa-onnx export, HF `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` @ `672fbf1b30579d6585301139bb363f42a0ad4a24`, chunk 16, left 128 |
| Precision | int8 (primary), fp32 (secondary) |
| Runtime | sherpa-onnx (pip version recorded in each result line) on onnxruntime CPU EP |
| Decoder | greedy |
| Label | **in-corpus**: the model was trained on LibriSpeech train |
| Machine | H0, shared host (h464 GPU jobs present); one physical core pinned per run (`taskset`), core recorded |

## Arms
1. `single`: each ready stream decoded alone (one encoder call per stream per chunk), 1 intra-op thread.
2. `batch`: all ready streams decoded in one call (one encoder call per tick), 1 intra-op thread.

## Procedure
1. **Throughput screen (unpaced, DEV = dev-clean + dev-other):** N ∈ {1, 8, 32, 64, 128} concurrent streams. This gives audio-seconds per CPU-second, an upper bound on streams per core.
2. **Paced verification (DEV):** for each arm, N = 0.8 × the screen's bound, then step up. N is sustained iff p99 lateness < 320 ms, final lateness < 320 ms, and lateness does not grow over the run.
3. **Confirm once (PUBLIC-TEST = test-clean + test-other):** at the largest sustained N per arm, chosen on DEV and frozen first. Run each arm in 3 alternating blocks.

## Metrics
- **Primary:** sustained real-time streams per physical core (`batch` / `single` ratio).
- **Secondary:**
  - WER per set (provisional Python scorer, re-scored by `ultra-eval`);
  - CPU-s per audio-s;
  - lateness p50/p95/p99;
  - max ready backlog.

## Pass and fail
- **Pass:** ratio ≥ 3.0 in every block, ΔWER (batch − single) within ±0.05 points on each test set, and the p99 lateness rule met.
- **If not:** report the measured ratio with verdict NO_LIFT or INCONCLUSIVE.

## Numeric choices (at most 4)
- packet = 20 ms;
- max_batch = 1024;
- lateness bound = 320 ms;
- N grid as above.

## Not allowed
- Choosing N on test sets.
- Counting wall × threads as CPU.
- Mixing fp32 and int8 numbers in one claim.

## Ceilings
- 0 GPU-hours.
- ≤ 20 core-hours.
- ≤ 5 GB disk.
