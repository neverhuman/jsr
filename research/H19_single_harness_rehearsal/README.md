# H19: Single-harness rehearsal of the qualifying protocol (R0 vs dense vs multirate)

**Status:** EXPLORATORY. Shared host xbabe0, 20 s ladder rungs, not the frozen 120 s / 3×300 s confirmation.

## Question
Does the frozen protocol run end to end before the qualifying window: one bench-suite harness, the A1 lateness rule, the per-CCX ladder that stops at the first failure, and the 8-block counterbalanced ABAB with Holm? And what does it say about R1 against R0 when every system runs under the same rules?

## Setup
- **Harness:** `ultra bench-suite` (runtime-lane).
- **Builds:** 4fafb45 (ABAB), 57fc3c5 (ladder A), then ce58952 (ladders D and E; includes the H15 sparse mel and the per-thread `sched_getaffinity` fix).
- **Host and placement:** xbabe0, a 3995WX (Zen 2). Four single-thread workers on one CCX (cores 96–99, L3 8). The observer sits on another CCX (100–103, L3 9). `nice 10`.
- **Grid:** 320 ms, max 2 chunks.
- **Systems:**
  - R0: Zipformer int8 (sherpa-onnx);
  - dense: `d2-45000` prepared INT8, and the finalist `d2-avg3-45-50-55` prepared INT8;
  - multirate: `d2mr-50000` prepared INT8.
- **Workloads:**
  - ABAB: dev-clean, 50 utterances, 232 audio-s per block;
  - panel frontier: the 400-utterance dev-clean half of the 730-utterance panel, manifest sha 76fead74…;
  - ladder: 4:48:4 streams per worker.

## Results (exploratory)
**Single-stream cost, 8 counterbalanced blocks** (Job B, n = 8, df = 7, exact t + Holm):

| Arm vs R0 | Geometric-mean CPU ratio | 95% CI | Speed-up | p (Holm) |
|---|---:|---|---:|---:|
| dense `d2-45000` | 0.6665 | [0.6553, 0.6779] | 1.50× | 1.4e-10 |
| MR `d2mr-50000` | 0.6289 | [0.5779, 0.6845] | 1.59× | 1.9e-6 |

**Per-CCX density** (four workers, p99 ≤ 800 ms; highest passing rung):

| Run (build) | R0 | dense | MR |
|---|---:|---:|---:|
| A (57fc3c5) | 32 | 96 (`d2-45000`) | 144 |
| D (ce58952, sparse mel) | 32 | 112 (`d2-45000`) | 144 |
| E (ce58952) | 32 | **96 (`d2-avg3`, the finalist)** | 144 |

- R0 = 32 and MR = 144 reproduced exactly in all three runs. Dense ranged 96–112, so the dense ratio to R0 is 3.0–3.5×; MR is 4.5×.
- **30 s sustained at 64 streams:** MR held a flat backlog (about 441–444 ms) and flat PSS (about 1.18 GB). R0 could not sustain 64 (p99 881 ms).

**Same-harness WER beside the cost** (Job C, panel dev-clean 400, Rust scorer):

| System | WER |
|---|---:|
| R0 | 2.952% |
| `d2-45000` | 9.561% |
| `d2mr-50000` | 11.561% |

## What went wrong or surprised us
- **The per-thread affinity check read the main thread's mask** (`/proc/self/status`) instead of each worker's. Fixed (ce58952) before any qualifying run.
- **Run D vs E:** dense moved one rung (112 vs 96) with a different checkpoint and on a shared host. The paper reports the finalist value plus the range.
- **The accuracy gap is large.** R0 is 2.95% against our 9.6–11.6% at these checkpoints. Every density ratio must be printed beside it.

## Decision / consequence
- The protocol runs end to end, and the qualifying window on xbabe2 will use the same commands with the frozen rung lengths.
- The density and cost numbers feed the paper (H18) as exploratory, warm-boundary results.

## Pointers
- Receipts on xbabe0, `~/jsr-data/runtime-dryrun/`:
  - `out-abab-4fafb45-20261122T004755Z/`
  - `out-ladder-57fc3c5-20261122T010703Z/`
  - `out-ladder-ce58952-h15mel-20261122T011741Z/`
  - `out-ladder-d2avg3-ce58952-20261122T024516Z/` (sha 70bfd3fc…)
  - `out-panel-ce58952-20261122T011028Z/` (sha 7c355dec…)
- Copies with sha256 values: `docs/paper/data/raw/MANIFEST.txt`.
- AGENT_CHAT runtime-lane posts, 2026-11-22T01:16Z–02:59Z; FREEZE §4–5.

---

## Addendum (runtime-lane, 2026-11-22): Negative Preflight, Full 5-Way 730-Utt Panel (Jobs F1/F2), and Receipt Ledger

### 1. Negative Preflight Verification (`4fafb45`, `xbabe0` core `108`)

All 6 fail-closed contract checks exited with code `2` before execution:
1. `--density-workers 2` (non-4-worker CCX count) -> exit `2`
2. `--grid-ms 160` (non-frozen pacer grid) -> exit `2`
3. Wrong ABAB arm order (`[d2mr, R0, d2]`) -> exit `2`
4. Split/cache basename mismatch (`--split dev-clean` with `test-clean.npz`) -> exit `2`
5. `nemotron-560 --panel` (non-qualifying 560 ms chunk candidate) -> exit `2`
6. Observer sharing L3 cache with worker cores -> exit `2`

### 2. Full 5-Way Frozen 730-Utt DEV Panel (`Jobs F1 + F2`, `ce58952`, Rust `ultra-en-wer-norm/v1` `988c59acad87`)

Run on `xbabe0` cores `104` (`dev-clean`, 400 utts, `3,052.2 s`) and `108` (`dev-other`, 330 utts, `2,065.3 s`), single executable (`sha256: 86a1277a…`), `sherpa-onnx 2.0.2` / `ONNX Runtime 1.28.2`:

| Candidate | Mode | `dev-clean` (400) | `dev-other` (330) | Pooled 730-utt WER | `s1` CPU-s/s (clean / other) | `b16` CPU-s/s (clean / other) |
|---|---|---:|---:|---:|---:|---:|
| `zipformer-int8` (`R0`) | streaming 320 ms | `2.95%` | `8.21%` | **`5.11%`** | `0.2180 / 0.2462` | `0.0642 / 0.0672` |
| `d2-avg3-45-50-55` (dense INT8) | streaming 160 ms | `7.76%` | `17.34%` | **`11.70%`** | `0.1563 / 0.1665` | `0.0445 / 0.0480` |
| `d2mr-50000` (MR INT8) | streaming 160 ms | `11.56%` | `22.19%` | **`15.93%`** | `0.1695 / 0.1755` | `0.0363 / 0.0410` |
| `moonshine-v2-tiny-en-int8` | offline `B=1` | `5.67%` | `12.14%` | **`8.33%`** | `0.0465 / 0.0545` | — (`p99 = 1.39–1.41 s`) |
| `moonshine-v2-base-en-int8` | offline `B=1` | `3.60%` | `8.83%` | **`5.75%`** | `0.0829 / 0.0996` | — (`p99 = 2.53–2.72 s`) |

### 3. Additional Live Catches & SHA-256 Receipt Ledger

- **Cross-route OOD `LOOKS.jsonl` key mismatch:** caught by `codex-review` and unified to `kind="ood-sealed-read"` keyed by canonical `set` (`ami`/`voxpopuli`) matching `python/ultra/ultra/ood.py` & `score-ood` (`faec803`).
- **Standalone binary `"source_commit": "unknown"`:** fixed in `crates/ultra-cli/src/lib.rs` to fall back to compile-time `ULTRA_BUILD_GIT_COMMIT` when executed outside a git checkout.
- **Full SHA-256 receipt ledger (`xbabe0:~/jsr-data/runtime-dryrun/`):**
  - Job A (`57fc3c5`): `out-ladder-57fc3c5-20261122T010703Z/density_ladder.json` (`08d38a5ed97ee60517473385a297b1e63514dea69930bface8f914306722e1b5`)
  - Job B (`4fafb45`): `out-abab-4fafb45-20261122T004755Z/abab_timing.json` (`add1100793bd2f3d613b6f1a1935b9f1c64ecb53eba4d7f63cdbd4d25d577687`)
  - Job C (`ce58952`): `out-panel-ce58952-20261122T011028Z/frontier.json` (`7c355dec2266f6ef5f445f8e0cd8dfe312e78acf1a3b640980f14607ffdfe227`)
  - Job D (`ce58952`): `out-ladder-ce58952-h15mel-20261122T011741Z/density_ladder.json` (`5b2054d1ed343762cb25826932137788b772dfb73ecbfad622c4cfc79ba96299`)
  - Job E (`ce58952`): `out-ladder-d2avg3-ce58952-20261122T024516Z/density_ladder.json` (`70bfd3fc16ef7c9dd315dc3325a404442fa0ca4ac5c3d64063bb3be9a7291cb4`)
  - Job F1 (`ce58952`, clean): `out-panel5-clean-ce58952-20261122T024522Z/frontier.json` (`59312f45c3c6…`)
  - Job F2 (`ce58952`, other): `out-panel5-other-ce58952-20261122T024528Z/frontier.json` (`6635bf8fc65d…`)
