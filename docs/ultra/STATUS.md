# STATUS (updated 2026-11-20 ~21:20Z)

- **Qualified rung:** none yet. No qualifying (BENCH_LEASE.host) run has happened. Every number below is exploratory.
- **Plan:** `docs/ultra/BENCH_PLAN_v1.md` (ULTRA-Bench v1, owner-approved). Protocol deltas in §0a. Rulings O13–O15a.

## Built and on main (`ed7f1df` and later)
- **Contracts and evaluation:**
  - `ultra-core` §33 event contract, hardened receipts, the `UnknownOverlap` label;
  - `ultra-eval`: WER, CQ(D), FC(D) with ownership-mapped episodes, bootstrap, process-tree CPU, paced replay.
- **Kernels:** `ultra-kernels` / `ultra-bench`. W01 PASS on H0.
- **Rust runtime and CLI** (`ultra-{model,frontend,stream,cli}`, runtime-lane):
  - the sherpa R0 backend;
  - trainer-parity LogMel+CMVN (B2);
  - a post-decode monotonic offline Final (WP5);
  - `try_text`.
- **M1 measured CPU bridge** (`ultra-bridge`, codex): whole-service, exploratory receipts only.
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

- **R1-D1** (our S, 100 h, D1d step 7800): 29.8% dev-clean. The M1 bridge reproduces it (Rust-scored 29.825%, 400/400 transcripts identical) at 0.141 CPU-s/s whole-service through the Python worker.
- **Rust-runner latency figures** from before the runtime clock fixes are INVALID_INSTRUMENT. Their CPU and WER figures stand.

## Training (GPU)

| Run | Host | State |
|---|---|---|
| D2 (S, 960 h) | xbabe3 3090 | Running under a guard (0.74, min-duty 0). ETA slips while other projects' jobs share the GPU. |
| D1MR (S_MR, 100 h) | xbabe1 4090 | At step ~2700, mostly paused by other users' jobs. About 4.5 h at full duty. Step 7800 kept as the matched control. |
| D2MR (S_MR, 960 h, cut to 5 epochs) | xbabe2 3090 | Shared with another user's job. About 1.6 days at full duty. |

- **GPU spend:** observed training spans about 4.0 h before D2 (D0 and the D1 attempts). The D2, D1MR and D2MR ledgers are kept in each run directory.

## Exact next tasks
1. **runtime-lane:**
   - B1, the Rust ORT R1 backend. Same-precision parity is currently 88.5% for int8; codex's batch-composition hypothesis is under test.
   - WP4, the online clock fix.
   - B3 bench-suite.
   - B4 Nemotron.
2. **trainer-lane:** A2 loader hardening and per-example parity proofs; pin onnx.
3. **codex:** M1 400-row DEV baselines (R0, N560); review of the native deliveries.
4. **lead:** merges and gates; the freeze draft (FREEZE.md) once parity holds; scheduling the qualifying window after D2.

## Blockers
- GPU contention from other projects on all three GPUs. We yield under the owner's cap.
- The owner listening check of the MFA timings is open.
- TED-LIUM 3 has no rights-clear source.
