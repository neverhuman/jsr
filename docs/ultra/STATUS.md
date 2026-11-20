# STATUS (updated 2026-11-20 ~22:30Z)

- **Qualified rung:** none yet. No qualifying (BENCH_LEASE.host) run has happened. Every number below is exploratory.
- **Plan:** `docs/ultra/BENCH_PLAN_v1.md` (ULTRA-Bench v1, owner-approved). Protocol deltas in §0a. Rulings O13–O15a.

## Built and on main (`f200b1d` and later)
- **Contracts and evaluation:**
  - `ultra-core` §33 event contract, hardened receipts, the `UnknownOverlap` label;
  - `ultra-eval`: WER, CQ(D), FC(D) with ownership-mapped episodes, bootstrap, process-tree CPU, paced replay.
- **Kernels:** `ultra-kernels` / `ultra-bench`. W01 PASS on H0.
- **Rust runtime and CLI** (`ultra-{model,frontend,stream,cli}`, runtime-lane):
  - the sherpa R0 backend;
  - trainer-parity LogMel+CMVN (B2);
  - a post-decode monotonic offline Final (WP5);
  - `try_text`.
- **M1 measured CPU bridge** (`ultra-bridge`, codex): whole-service, exploratory receipts only. Report: `docs/ultra/reviews/2026-11-20-trained-cpu-benchmark.md` (f200b1d).
- **R1 native backend** (`OrtTransducerBackend` + `ultra-gp`, abed965/dd5078c): EXPERIMENTAL. FP32 parity is 400/400 (codex re-scored); the dynamic INT8 is non-qualifying (355/400, +0.156 WER).
- **`ultra bench-suite`, Nemotron Rust adapter, `r1_costlaw`** (15e66a4, merged daf50bb): EXPERIMENTAL, with codex B3/B4 findings open.
- **Trainer** (`python/ultra/train`):
  - dense S and multirate S_MR;
  - delayed SpecAugment;
  - export contract `ultra.r1-export/v1`;
  - the numpy/ORT reference consumer.
- **Data:**
  - LibriSpeech 960 h and dev/test (`DATA.lock`);
  - MFA word timings;
  - sealed OOD VoxPopuli EN and AMI IHM (TED-LIUM 3 is BLOCKED_RIGHTS);
  - frozen slices;
  - the silence probe.
- **Fleet:** `fleet_guard` (owner rule: never more than 90% mean CPU/GPU on any host) and `fleet_util.sh`.

## Measured so far (exploratory; DEV unless stated)
- **R0 Zipformer int8, machine-wide density** (xbabe0, same CPU as H0, Python harness on the real clock): **192 live streams sustained**, with 16 single-threaded CCX workers × 12 streams. p99 ≤ 636 ms, dev-clean 2.80%, 0.06 CPU-s/s.
- **SP-A primary:** batched vs single is 1.49×, against the pre-registered 3×. The expected verdict is NO_LIFT. Density is L3-bound per Zen 2 CCX.
- **Nemotron 0.6B Q8_0 (PUBLIC-TEST, exploratory, unknown overlap):**

  | | test-clean WER | test-other WER | Single-stream CPU-s/s (quiet) |
  |---|---:|---:|---:|
  | 160 ms | 2.65% | 5.60% | 0.709 |
  | 560 ms | 2.58% | 5.18% | 0.451 |

- **M1 DEV400 board** (same audio, Rust-scored, whole service, guarded):

  | System | WER | CPU-s/s | RSS |
  |---|---:|---:|---:|
  | R0 Zipformer int8 | 2.85% | 0.137 | 239 MB |
  | R1-D1 | 29.83% | 0.141 | 593 MB |
  | Nemotron 560 ms | 2.16% | 0.472 | 1.28 GB |

  A/A is exact.
- **R1-D2 progress** (trainer-normalised, 400 DEV, read-only): 21.91% at step 5000 and 21.89% at step 6854, from R1-D1's 29.8%.
- **Rust-runner latency figures** from before the runtime clock fixes are INVALID_INSTRUMENT. Their CPU and WER figures stand.

## Training (GPU)

| Run | Host | State |
|---|---|---|
| D2 (S, 960 h) | xbabe3 3090 | Running under a guard (0.74, min-duty 0). ETA slips while other projects' jobs share the GPU. |
| D1MR (S_MR, 100 h) | xbabe1 4090 | At step ~2700, mostly paused by other users' jobs. About 4.5 h at full duty. Step 7800 kept as the matched control. |
| D2MR (S_MR, 960 h, cut to 5 epochs) | xbabe2 3090 | Shared with another user's job. About 1.6 days at full duty. |

- **GPU spend:** observed training spans about 4.0 h before D2 (D0 and the D1 attempts). The D2, D1MR and D2MR ledgers are kept in each run directory.

## Exact next tasks
1. **runtime-lane:** one hardening commit covering codex's B1/B3/B4 findings:
   - reference-blind latency;
   - `--precision` reaching every subjob;
   - a `cost_scope` label;
   - `--qualified` gating;
   - positive ggml-CPU evidence;
   - pipe draining;
   - non-finite caches;
   - terminal errors;
   - `retained_bytes`.
2. **trainer-lane:**
   - a batch-invariant INT8 export. v1.1 static MinMax is at +0.67, so the joiner moves to fp32 and an encoder calibration sweep is running.
   - A2 loader hardening and per-example proofs.
   - D2 to completion, with selection on DEV.
3. **codex:** the D1 TRAIN eligibility audit; E1/E2 review of the runtime hardening.
4. **lead:**
   - merges and gates;
   - FREEZE.md drafted once the INT8 path and B3 are accepted;
   - a speaker-stratified freeze panel;
   - scheduling the qualifying window after D2.

## Blockers
- GPU contention from other projects on all three GPUs. We yield under the owner's cap.
- The owner listening check of the MFA timings is open.
- TED-LIUM 3 has no rights-clear source.
