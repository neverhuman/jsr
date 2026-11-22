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
