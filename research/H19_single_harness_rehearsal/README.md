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

### 4. Job G: Current Best Checkpoint `D2 avg(55/60/65k)` INT8 — Greedy & Beam-4 (`43f31a1`, `xbabe0` cores `96–111`)

Run from the single executable `ultra-43f31a1` (`sha256: a6ca5c4124f60aed1bb9c6964bd1cb89527763d047f6bb1d6df2db1720072520`, `--beam` wired through `NativeEngineBackend` / `ultra bench-suite`) against `~/jsr-engine-data/runs/d2-avg3-55-60-65/weights-int8prep` (`weights.safetensors sha256: e9ffe4632f6aa087…`):

| Arm / Decode | Per-CCX Ladder (`4:48:4`, 20 s) | Pass `p99` | Fail Rung (`p99`) | 64-Stream 30 s Sustained (`p99`) | Same-Harness Panel WER (`clean / other / pooled`) |
|---|---|---:|---:|---:|---:|
| **`d2-avg3-55-60-65` INT8 (greedy)** | **`96/CCX` (`24/core`)** | `600.85 ms` | `112` (`938.42 ms`) | PASS (`497.97 ms`) | **`7.444% / 16.720%`** (**`11.26%` pooled**) |
| **`d2-avg3-55-60-65` INT8 (`--beam 4`)** | **`96/CCX` (`24/core`)** | `600.29 ms` | `112` (`1989.86 ms`) | PASS (`509.52 ms`) | **`6.962% / 15.896%`** (**`10.64%` pooled**) |
| **`d2mr-50000` INT8 (greedy)** | **`144/CCX` (`36/core`)** | `610.87 ms` | `160` (`1840.44 ms`) | PASS (`447.13 ms`) | `11.561% / 22.187%` (`15.93%` pooled) |

- **Key finding:** Bounded beam-4 search drops pooled panel WER by **−0.62 pp (`11.26% -> 10.64%`)** with **ZERO serving-density loss (`96/CCX -> 96/CCX`)** and `+4%` batch CPU (`+0.5 MB` RSS), while `d2mr-50000` reproduced **`144/CCX` (`36/core`)** across **4/4 runs and 3 builds** (`57fc3c5`, `ce58952`, `43f31a1`).
- **Shared-host disclosure:** During G1's initial 40 s window, concurrent `xbabe0` backfill load temporarily reduced `R0` to `16/CCX` (`p99 585.82 ms`, `rtf 0.120`, failing `32` at `1575.56 ms`). G1's `R0` is therefore **not** used as a speedup denominator; ratios remain anchored to `R0 = 32/CCX` from clean-session Jobs A, D, and E.
- **Job G SHA-256 receipts (`xbabe0:~/jsr-data/runtime-dryrun/`):**
  - G1 ladder: `out-ladder-g1-d2avg556065-greedy-43f31a1-20261122T041016Z/density_ladder.json` (`031f5ed5766cbeb47201d9ca7dbf50c054e411dd34549c8fbd7e5d8a67d601e8`)
  - G2 beam-4 ladder: `out-ladder-g2-d2avg556065-beam4-43f31a1-20261122T041016Z/density_ladder.json` (`b8f9cf5068e9a980cc5340e5d6db2db74745d2d56414cd7d761a5044824390d9`)
  - G3a clean greedy: `out-panel-g3a-clean-greedy-43f31a1-20261122T041024Z/frontier.json` (`ae01af005f2361d9e294fdf6f39223e2d4ba5fae24b5675ab01ffe57dad13c6a`)
  - G3a clean beam-4: `out-panel-g3a-clean-beam4-43f31a1-20261122T041024Z/frontier.json` (`30e7048d45c5eadb5d670dc4525559656ffbc8dc6bec06577a34ba6c52ef5ea1`)
  - G3b other greedy: `out-panel-g3b-other-greedy-43f31a1-20261122T041032Z/frontier.json` (`367b59bfc131904fbb0087ad6ff7931837e0b83a4bfc4662dfdb8b8548e4fd70`)
  - G3b other beam-4: `out-panel-g3b-other-beam4-43f31a1-20261122T041032Z/frontier.json` (`6aac8285f9a6adf1de5b9de4601a2a1c9d4b0b28ccf0a16b3ca4084c4aa735fe`)
