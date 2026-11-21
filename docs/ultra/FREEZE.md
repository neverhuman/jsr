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

## 3. Settings (fixed at the freeze); revised after the independent review (2026-11-21-independent-freeze-review.md)
- **One harness for every density, cost and latency row:** `ultra bench-suite` (runtime-lane, G3).
  - `engine_density`, the Python SP-A harness and the bridge are development tools only. Their rows are never compared with each other or with bench-suite rows.
  - bench-suite runs R0 (sherpa), ORT-R1 and engine-R1 through the same scheduler, the same A1 code, the same grid and step settings, the same cost scope and the same placement.
- **Equal hardware for every system.**
  - The density unit is **streams per CCX with 4 single-threaded workers on that CCX's 4 physical cores**, observer on a different CCX. Streams per physical core = that ÷ 4.
  - The **full machine is 15 CCX** (60 cores). The 16th CCX holds the observer and governance, because the placement checks fail closed.
  - No result is extrapolated from one CCX to the whole machine.
- **Streaming:**
  - 20 ms packets, model chunk 160 ms.
  - Density at a 320 ms grid with 2-chunk steps **for every system**.
  - GEMM split is off.
- **Decoding:** greedy RNN-T with symbol cap 8. Any beam operating point is a separate, declared row.
- **INT8:** per-row centred activations from the prepared artifact.
  - **Qualifying mode refuses runtime weight quantisation.**
  - The engine configuration (stem kinds, ISA, binary sha, commit, affinity) is recorded on every row. No configuration may come from an unrecorded environment variable.
- **A1 density rule (one implementation, in bench-suite):**
  - Capture deadline per chunk = arrival of its last required sample + processing budget.
  - Lateness is measured per chunk.
  - p99 lateness ≤ 800 ms, every worker in the same run.
  - No growth: the mean lateness of the last third is no more than 200 ms above the first third.
  - Drained at cutoff, with accepted, processed and unfinished audio accounted.
  - CPU divided by processed audio only. Run length 300 s per step.
- **Cost:** whole-service CPU-s per audio-s, warm (model load excluded and reported separately).
  - Whole process tree, the same scope for R0, ORT-R1, engine-R1 and Nemotron. Nemotron also gets a cold row.
  - Observer CPU is reported separately.

## 4. Protocol (qualifying window, G6)
1. **Quiet host:** H0, all GPU training finished, other lanes paused, load recorded before and after. Announce in AGENT_CHAT at least 1 h ahead, take `BENCH_LEASE.host`, and get acks.
   - `bench-suite --qualified` must pass its pre-flight: lease, freeze-tag ancestry, clean build, pinned identities.
2. **Quality, one read each, logged in LOOKS.jsonl first:**
   - PUBLIC-TEST test-clean/other and the sealed OOD sets (VoxPopuli EN, AMI IHM), for all systems.
   - Rust re-scoring, bootstrap over speaker or session clusters (≥ 10,000 replicates).
   - **Disclosure:** LibriSpeech test-clean/other were read several times before this freeze (LOOKS.jsonl, historical reconstructions). The test result is reported with that disclosure. Selection never used test data.
3. **Timing:** 5 ABAB blocks for each finalist pair, single stream. The statistic is the median of the block CPU-s/s, with a bootstrap CI over blocks.
   - **The 5% median CPU rerun tolerance (BENCH_PLAN §0a) applies.** A second-day rerun of one finalist must agree within 5%, or the cost row is labelled unstable.
4. **Density:**
   - **Ladder:** N per worker in steps of 2.
   - **Repeats:** 3 at each N, alternating systems.
   - **Reported N:** the largest N at which all 3 repeats pass A1 for every worker.
   - **Systems:** R0, ORT-R1, engine R1-dense and engine R1-MR, per CCX and on the full machine (15 CCX).
   - **Nemotron has no density path in bench-suite, so it is not in the density claim.** Its single-stream cost row gives an implied ceiling, labelled as an estimate.
5. **Sustained:** two 1 h runs (BENCH_PLAN §0a) **at the claimed density N**, with RSS, backlog and lateness series.
6. **Latency:** CQ(250/750/2000) and FC(800/1500) against MFA timings, `controlled_eof_finish_input` labelled.
7. **Silence probe:** 10 min, inserted words per hour.

## 5. Pre-registered claims (fixed here; the Holm family is the 4 claims below)
- **C-density (headline):** engine **R1-MR** streams per CCX and on the full machine (15 CCX), against **R0** and against **ORT-R1**, each measured with the §4.4 ladder.
  - Reported as a ratio, **always printed next to each system's test-clean and test-other WER**.
  - The headline sentence must state the WER gap. No density claim is made without the accuracy beside it.
- **C-cost:** engine R1 (dense and MR) INT8 whole-service warm CPU-s/s against R0. Median of 5 ABAB blocks with a bootstrap CI, with WER beside it.
- **C-quality:** the R1 test-clean/other WER gap to R0 with intervals. No parity claim unless the interval supports it.
- **C-OOD:** the degradation row on VoxPopuli and AMI.
- **SP-A:** the batching-lift verdict, reported as context outside the family.
- **INT8 gate (G2), revised:**
  - on the 730-utterance DEV selection panel;
  - point delta ≤ +0.10 pp **and** one-sided 95% bootstrap upper bound ≤ +0.50 pp against FP32 of the same checkpoint.
- **R1-MR inclusion:** a deviation from BENCH_PLAN §0a, which held R1-MR to an add-on block. R1-MR is the density design and has measured 960 h checkpoints, so it is in the main freeze, labelled. This is recorded in DECISIONS.md.

## 6. Open before the freeze
- G1/G2 finalists, with the gate on the 730-utterance panel.
- **G3 (runtime-lane):**
  - bench-suite A1 unified as §3;
  - R0, ORT and engine on the same grid, step, workers and placement;
  - `--qualified` implemented;
  - panel input;
  - the sustained run at density N;
  - fleet placement verifies shared L3 and fails closed for CPUs ≥ 64.
- **G4:** deterministic log-mel.
- **Panel input everywhere:** engine_eval and bench-suite take `--panel`; the trainer already has `--panel`.
- **G8:** re-review of this revision.
- **Full hashes:** at the freeze, the hashes above (data manifest, workload) are written out in full.
