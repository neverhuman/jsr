# FREEZE (DRAFT, not frozen): ULTRA-Bench v1

**Status: DRAFT.** It becomes binding only at the commit tagged `bench-freeze-v1` (PLAN_TO_V1 C3). That needs:
- the selected finalists (G1);
- their engine gates (G2);
- an independent review of this file (G8).

Until then, any row produced is exploratory.

## 1. Identities

| Item | Value |
|---|---|
| Scorer | `ultra-eval` / `ultra-score` at the freeze commit |
| Normaliser | `ultra-en-wer-norm/v1`, rule-table sha256 `988c59acad873bf0a0c50bdcf434dde0b5125cf51c2cef887bfc63f4f89113e8` (pinned test) |
| Data | `DATA.lock`: LibriSpeech parquet manifest sha256 `3e21c0e4…bff5`, decoded caches, MFA timings, sealed OOD sets |
| DEV selection panel | `docs/ultra/data/dev_selection_panel.json`, manifest sha256 `76fead746d1f8b8ae57a30aa341e81364538eb69759af813d6f13d6d1f57fc36` (dev-clean 400 / 40 speakers, dev-other 330 / 33) |
| DEV continuity panel | first-400 dev-clean, workload `fb8241b9…`. Historical comparability only, not selection. |
| Hardware | `HARDWARE.json` H0 = xbabe3, TR PRO 3995WX (Zen 2, AVX2 + FMA, 16 CCX × 4 cores, 16 MiB L3 per CCX) |
| Secondary hardware (ARM context row) | xbabe4, Apple M3 Ultra (NEON sdot/i8mm). macOS cannot pin. Labelled exploratory-ARM. |

## 2. Systems (candidate list; final pins go in CANDIDATES.lock at the freeze)

| Id | System | Runtime | Precision | Honesty label |
|---|---|---|---|---|
| R0 | Zipformer en 2023-06-26 | sherpa (Rust) | int8 | in-corpus |
| N560 / N160 | Nemotron 0.6B Q8_0 | NeMo-Speech.cpp adapter | q8 | unknown overlap |
| R1-dense | D2 final (selected on the DEV selection panel) | ultra-engine | prepared INT8; FP32 control | in-corpus |
| R1-MR | best of D2MR-final / D3MRb-final (DEV selection panel) | ultra-engine | prepared INT8; FP32 control | in-corpus |
| R1-T (research row) | R4T ternary, 100 h | ultra-engine | ternary | in-corpus, not a finalist |
| context | Moonshine v2 base, Whisper small.en | sherpa offline | int8 | unknown overlap |

## 3. Settings (fixed at the freeze)
- **Streaming:**
  - 20 ms packets, chunk 160 ms (4 frames at 25 Hz).
  - 320 ms grid with 2-chunk steps for density.
  - One inference thread per worker, one worker per CCX. GEMM split is off.
- **Decoding:** greedy RNN-T with symbol cap 8.
- **INT8:** per-row centred activations, prepared artifact. No runtime weight quantisation.
- **Density rule A1:**
  - p99 lateness ≤ 800 ms;
  - no backlog growth;
  - drained at cutoff;
  - accepted, processed and unfinished audio accounted.
- **Measurement:**
  - The observer is pinned to a CCX disjoint from the service, with effective affinity recorded.
  - Topology fails closed.
  - Sampling period is fixed in the run manifest.
  - Observer CPU is reported separately.
  - Whole-service CPU-s per audio-s is the cost metric.
  - `cost_scope` is labelled on every row.

## 4. Protocol (qualifying window, G6)
1. **Quiet host:** H0 with all GPU training finished, other lanes' CPU work paused, and load recorded before and after. Announce in AGENT_CHAT at least 1 h ahead, take `BENCH_LEASE.host`, and get acks from active lanes.
2. **Quality, one read each, logged in LOOKS.jsonl before reading:**
   - PUBLIC-TEST test-clean and test-other;
   - sealed OOD sets (VoxPopuli EN, AMI IHM).
   - Run all systems. Rust re-scoring uses the pinned normaliser.
   - Bootstrap over speaker or session clusters (≥ 10,000 replicates), with Holm correction over the declared family.
3. **Timing:** at least 5 ABAB blocks for each finalist pair, single stream, whole service.
4. **Density:**
   - per-CCX: 4 workers, increasing N until A1 fails;
   - full machine: 16 CCX workers;
   - systems: R0, R1-dense, R1-MR, and ORT R1 as the runtime baseline.
5. **Sustained:** a 1 h run at the claimed density, with an RSS and backlog series.
6. **Latency:** CQ(250/750/2000) and FC(800/1500) against MFA timings, finalisation labelled `controlled_eof_finish_input`.
7. **Silence probe:** 10 min, inserted words per hour.

## 5. Pre-registered claims (thresholds set here before the PUBLIC-TEST read)
- **C-density:** R1-MR engine sustained streams on H0 (full machine), reported as a ratio against R0 and against Nemotron, with the WER of each.
- **C-cost:** R1 engine INT8 whole-service CPU-s/s against R0, at their measured WERs.
- **C-quality:** the R1 test-clean/other WER gap to R0, stated with intervals. No parity claim is made unless the interval supports it.
- **C-OOD:** the degradation row on VoxPopuli and AMI for all systems.
- **SP-A:** the batching-lift verdict (expected NO_LIFT) recorded.

## 6. Open before the freeze
- G1/G2 finalists and their engine gates.
- G3: an ORT density row in the same harness.
- G4: deterministic log-mel. If it isn't done, ARM rows are labelled with frontend-dependent agreement.
- G8: an independent review of this file.
