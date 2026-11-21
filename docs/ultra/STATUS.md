# STATUS (updated 2026-11-21 ~09:05Z)

- **Qualified rung:** none yet. No qualifying (BENCH_LEASE.host) window has run, so every number below is **exploratory**. The host is shared, and a same-day A/A CPU replay varied by +25%.
- **Plan:** `docs/ultra/BENCH_PLAN_v1.md` (ULTRA-Bench v1). The engine plan is in `docs/ultra/ENGINE_PLAN.md`.

## What is on main (7ae39f0)
- **Contracts, eval and kernels:**
  - ultra-core events and receipts;
  - ultra-eval: WER, CQ, FC, bootstrap, `ultra-measure` with a pinned disjoint observer;
  - ultra-kernels: AVX2 int8, T1, T-LUT2 including a signed path.
- **Runtime (runtime-lane):**
  - sherpa R0, and the ORT R1 backend (experimental; FP32 parity 400/400);
  - `ultra bench-suite`: C4 slices, silence probe, MFA latency joined after inference, cost_scope, job manifests;
  - a Nemotron adapter;
  - the ultra-engine backend integration (E4).
- **ultra-engine (engine-lane), pure Rust:**
  - **Precision:** FP32, and int8 with centred per-row activations (batch-invariant, +0.000 WER) from a prepared-int8 artifact (RSS 259 MiB).
  - **Models:** dense S and multirate S_MR; ternary R4 layers (T-LUT2 on x86).
  - **Architectures:** AVX2 and aarch64 NEON (sdot/i8mm). The engine is bit-identical across ISAs.
  - **Guards:** output-continuity tests.
- **Bridge (codex):** M1/M2 whole-service measurement, with ORT and fused-engine adapters (FP32 and prepared INT8).
- **Trainer:**
  - dense, S_MR and S_MR_T (ternary QAT);
  - a batch-invariant static-INT8 ORT export (v1.1);
  - `export_ternary`.

## Measured (exploratory)
Same build 6df5c8f, DEV400 dev-clean, whole service, observer separated:

| System | WER | CPU-s/s | RSS |
|---|---:|---:|---:|
| R0 Zipformer int8 | 2.85% | 0.078 | 239 MB |
| R1 D2-s5000 ORT FP32 | 21.88% | 0.175 | 776 MB |
| R1 D2-s5000 engine FP32 | 21.88% | 0.104 | 478 MB |
| **R1 D2-s5000 engine INT8** | **21.80%** | **0.041** | 647 MB (now 259 MB prepared) |

**Engine density** (A1 rule, p99 ≤ 800 ms, int8):

| Host | Dense S | D1MR (S_MR) |
|---|---|---|
| Zen 2, per 4-core CCX | 96 streams | **136 streams** |
| M3 Ultra (xbabe4) | 64 per core | **84 per core**; 504 on 6 workers |

**Retracted pending the single-harness rerun (independent review 2026-11-21):** the engine density figures above come from `engine_density` with 4 workers per CCX, while R0's 192 came from the Python harness with 1 worker per CCX, and the ORT figure came from bench-suite. These are not comparable. Valid comparisons will come only from bench-suite on equal hardware (FREEZE §3).

**Single-stream int8:**

| Host | Dense | S_MR |
|---|---:|---:|
| Zen 2 | 0.039–0.043 | — |
| M3 Ultra | 0.0131 | 0.0104 |

## Training (dev-clean 400 unless noted)

| Run | Data | Latest result |
|---|---|---|
| D2 dense | 960 h, xbabe3 | 21.9% @5k → 17.3% @10k → **13.9% @20k** (dev-other 27.1%); training continues under guard |
| D2MR multirate | 960 h, xbabe2 | 34.2% @5k → 26.9% @10k → **20.4% @15k** |
| D1MR multirate | 100 h, DONE @40k | **18.0% full dev-clean** / 36.5% dev-other (dense 100 h was 29.8%) |
| R4T ternary QAT | from D1MR, xbabe1 | 22.8% @1k → **18.6% @9k** vs FP32 17.4% (gap 1.2); finishes at 12k, then export plus engine gate |

## Next
1. **R4T:** finish → export → engine ternary gate (engine WER within +0.10 of the torch ternary eval) → ternary speed on Zen 2 and M3.
2. **D2 / D2MR:** converge → DEV selection → INT8 prepared → engine rows.
3. **runtime-lane:**
   - a same-harness ORT density row;
   - a platform-deterministic log-mel (cross-ISA PCM → tokens);
   - E4 follow-ups.
4. **Freeze:** FREEZE.md, the speaker-stratified panel, and the qualifying window (quiet host, observer pinned, A/B blocks).

## Blockers / owner decisions
- GPUs are shared with other projects; we yield under the 90% cap.
- Open owner items:
  - the MFA 20-utterance listening check;
  - the TED-LIUM 3 rights;
  - data expansion beyond LibriSpeech (needed for the accuracy gap to R0).
