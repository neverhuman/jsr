# STATUS (2026-11-20)

- **Qualified rung:** none yet.
- **R0 candidates measured so far:**
  - sherpa-onnx streaming Zipformer (LibriSpeech-only): int8 test-clean smoke WER 2.51%, 0.079 CPU-s/s single and 0.053 batched.
  - Nemotron 0.6B Q8_0: smoke WER 2.04%, 0.84 CPU-s/s single. The full run is in the nemotron lane.
- **Last done:**
  - Merged ultra-eval, ultra-kernels/bench (W01 PASS on H0) and the jankurai hygiene pass. Audit is 85 with 0 caps; `just required` is green.
  - SP-A throughput screen (`python/ultra/waves/SP-A/README.md`).
- **In progress:**
  - SP-A paced fleet (lead, cores 32–47).
  - Nemotron full test sets (cores 16–31).
  - Trainer: D0 and CPU throughput (cores 0–15).
- **Exact next task:** rerun `python -m ultra.sp_a_fleet --cores 8 --streams-per-core 14 --split dev-clean` once block 32–47 is quiet, then step N up to the largest sustained density, then confirm on the test sets.
- **Blockers:** the GPU is used by h464 jobs with no end in sight (owner: wait). The host is heavily shared, so timing needs the core partition and lease in AGENT_CHAT.md.
- **Spend:** 0 GPU-hours; about 6 core-hours of benchmarks.
