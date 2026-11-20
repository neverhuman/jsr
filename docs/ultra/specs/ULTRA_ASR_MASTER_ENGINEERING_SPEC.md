# ULTRA — master engineering specification for an event-clocked, integer-exact streaming speech recogniser on CPUs

**Version:** 1.0, 2026-11-20 · **Status:** specification; nothing here has been trained
**Supersedes:** TACTUS, KAIROS v0.2, APEX-ASR/CADENCE v1 and CADENCE-CPU v2. It was written after scoring all of them (`ULTRA_SCORECARD.md`). Appendix F says what came from where.
**Audience:** an engineering agent taking charge, and the owner who signs its gates.

**Claim tags.** `[REPO: path]` read in the JopeDime code-only archive of 2026-11-20 (no history, no build manifests). `[SRC n]` fetched from the source's page (Appendix H). `[MEM n]` cited from memory. `[DOC]` stated in one of the reviewed documents and not checked by me. `[SANDBOX]` measured by me (Appendix A). `[D]` arithmetic on tagged numbers. `[EST]` my estimate. `[BET]` a hypothesis with a wave and a fallback. Untagged design statements are decisions of this specification.

---

## 0. Agent: start here

### 0.1 The plan in six lines

1. Build the measuring instrument and run the existing recognisers through it. The best one is packaged as **R0**: usable at once, and the control for everything after.
2. Train a conventional dense streaming model, **R1**. From here on the project always owns a working recogniser of its own.
3. Add one mechanism per rung, each tested by switching it on and off in the same weights or in a matched pair: right context and final text (**R2**), the event clock (**R3**), ternary integer inference (**R4**), scale (**R5**).
4. A rung that fails its gate is skipped; the previous rung ships.
5. Every number has a label, a receipt and a pre-registration.
6. Nothing irreversible is done to text a user has been told is committed.

### 0.2 Rules that do not bend

1. **Inference is CPU-only.** From rung R4 on it is also integer-only and bit-exact across targets (§5).
2. **"Commit" means irrevocable** (§3). No later pass changes committed or final text; a later opinion is a `CorrectionSuggested` event.
3. **Every run-time mode is a training mode.** A dial setting, grid, delay, precision or depth the runtime may use must have been trained and tested, including any switch between them inside an utterance.
4. **Inference never uses what only training knows.** No fallback keyed on the reference transcript or its length.
5. **A loss must be feasible or be skipped and counted.** A CTC term whose target cannot be aligned is masked for that example and logged; it is never silently zeroed.
6. **The anchor rule.** Text the model produced may adapt only preview text. Committed and final text adapt only from labels a user confirmed.
7. **Sealed data is read only through the look ledger.** No training, tuning, search, calibration or debugging on it.
8. **Pre-register before running.** Hypothesis, arms, primary metric, margin, stop rule, at most four numeric choices `[REPO: research/flywheel_selfplay/PREREG_TEMPLATE.md]`.
9. **A cached-screen win is a lead.** Confirm on fresh data: H92 looked positive on a cached screen and H94 measured −51.9 Elo prospectively `[REPO: research/flywheel_selfplay/h94_w6_prospective_hedge/README.md]`.
10. **CPU cost is process-tree user plus system time** per second of audio. Threads × wall time is "reserved", and is reported separately.
11. **An optimisation is admitted only if** it matches the scalar reference bit for bit and improves the full streaming benchmark. A micro-benchmark win alone admits nothing.
12. **No silent degradation.** Every shed step, mode change, skipped pass, refused adaptation and lost-audio gap is an event in the stream and a line in the receipt.
13. **Rust for what ships.** The repository forbids new Python outside emitted code and its harness `[REPO: docs/adr/ADR-0006-rust-only-stack.md]`; the trainer needs an owner ruling (O1).
14. **Downloaded models, data and archives are untrusted** until hashes and licences are recorded.

### 0.3 Roles

| Role | Holds | Must differ from |
|---|---|---|
| Lead | this specification, the task board, pre-registrations | Scorer |
| Scorer | frozen scorer, look ledger, every sealed read | Lead |
| Red-team | one script per wave that tries to explain the result away: tail-only gains, tie accepts, seed dependence, leakage | the wave's author |
| Fleet operator | claims, manifests, budgets, quarantine | may be the Lead |

Two agents are the minimum: a builder and a scorer. Separation is enforced by file permissions on the sealed vault, not by good intentions.

### 0.4 Decision rights

| Decide alone | Propose; the owner decides | Never |
|---|---|---|
| Implementation, module boundaries, test design | Anything in §16 | Read sealed labels outside a registered look |
| Kernel choice per target, by the admission rule | Change a gate, margin or rule of §0.2 | Train, pseudo-label, search or calibrate on sealed audio |
| Values inside a pre-registered range | Add a corpus or a teacher | Edit a recorded verdict |
| Take a listed fallback after a failed gate | Exceed a budget ceiling | Apply a self-label to committed or final text |
| Order of work off the critical path | Any public claim | Assign a registry identifier |
| Stop a broken run | Drop a wave on the critical path | Delete evidence, shards or ledgers |

### 0.5 Stop the affected work and write to the owner when

1. K-EVENT or K-TERN fails after its permitted revisions.
2. A budget ceiling would be passed by more than 20%.
3. A parity failure survives two working days.
4. Decontamination finds evaluation audio in training or pseudo-label data.
5. A licence is unclear.
6. Measured training throughput is below the planning floor (§8.8).
7. W01 shows the cost law does not hold on the reference machines.
8. Two consecutive critical-path waves return `NO_LIFT`.

Everything not affected continues. A failed experiment is a reason to take its fallback, not to halt.

### 0.6 Working records

| Record | Content |
|---|---|
| `STATUS.md` | current qualified rung, last task done, exact next task, blockers, spend |
| `DECISIONS.md` | decision, alternatives, evidence, consequence, when to revisit |
| `EXPERIMENTS.jsonl` | pre-registration hash, status, verdict, receipts |
| `CANDIDATES.lock` | exact runtime, model, tokenizer and options for every external recogniser |
| `DATA.lock` | source, split, deduplication and rights manifests |
| `HARDWARE.json` | every machine a number came from |
| `LOOKS.jsonl` | one line per pre-registered sealed read |
| `FACTS.md` | every item of §2.6 resolved against the full repository |

An agent that loses context reads `STATUS.md`, the last receipt and `CANDIDATES.lock`, and continues. It does not redo finished work or invent a missing result.

**Verdicts:** `PASS`, `NO_LIFT`, `INCONCLUSIVE`, `INVALID_INSTRUMENT`, `BLOCKED_RESOURCE`, `BLOCKED_RIGHTS`, `STOPPED_BUDGET`. A verdict is never edited; a later wave supersedes it.

**Done** means: the artifact exists, its acceptance test passes on the recorded commit, the result is written down, and the next task can consume it without guessing.

### 0.7 First ten working days

| Day | Action | End-of-day evidence |
|---|---|---|
| 1 | Obtain the full repository; read its `AGENTS.md`; file rulings O1–O10 (§16); inventory the deployment CPU as H0 | `FACTS.md` started; `HARDWARE.json`; `RULINGS.md` requested |
| 2 | Scorer, normaliser, known-answer fixtures; event and receipt schemas (§3, Appendix D) | scoring fixtures pass; schemas validate |
| 3 | Paced replay harness; process-tree CPU and memory accounting | arbitrary packet sizes give identical sample timelines |
| 4 | First external recogniser through an adapter (Nemotron via its CPU runtime, or whichever is available) | one receipt with text, CPU seconds, delays |
| 5 | Two more adapters; port the Appendix A benchmark to `ultra-bench` | three candidates or written blockers; cost-law file for H0 |
| 6 | Program parser, canonical printer and static cost estimator (§6) | round-trip test; estimator reproduces Appendix B |
| 7 | Audio identity graph on LibriSpeech; bank; quarantine test; boards | planted sealed shard refused; zero overlap |
| 8 | W00 and W01: incumbents' fixed and marginal cost with counters; cost law on H0 | verdicts recorded |
| 9 | Trainer skeleton on LibriSpeech train-clean-100; throughput per GPU | loss falls; measured audio-hours per GPU-hour; re-budget note |
| 10 | Package R0; write `STATUS.md`; propose sizes and the next three waves | a usable recogniser; owner reply |

---

## 1. What is being built and what winning means

### 1.1 The model in one paragraph

Audio becomes 7-bit features in an integer front end (**Cochlea**). A small always-on network (**Reflex**) reads them at 12.5 Hz and produces token posteriors on a dense grid. A short symbolic rule (**the gate**) keeps the frames that carry information as **events** and pools the rest into them. The large causal network (**Cortex**) runs only on events, all events of a chunk in one pass. A few more blocks (**the Tower**) see D events of right context; the same blocks with the whole utterance in view produce final text at a pause. One **transducer** decodes all three delays: preview, commit, final. Large matrices are ternary at two bits per weight; activations are 7-bit integers. The model is one program in the GP language.

### 1.2 Why this shape: the cost law

A CPU pays twice for a network: to bring the weights to the core, and to multiply. Both terms are measured `[SANDBOX]`:

> **time per wake-up = weights × (a + c × events in the wake-up)**
> packed ternary, out of cache: a = 19.4 ps per weight, c = 11.0 ps per multiply-add (R² = 0.999 over eight batch sizes)
> int8, out of cache: a = 96.4, c = 10.5 (R² = 0.967)

One frame through out-of-cache int8 weights spends 90% of its time fetching them. The incumbent's published timings have the same shape: 24.4 ms of a 27.0 ms chunk does not depend on the audio in it `[D]` from `[SRC 2]`. What causes that intercept in the incumbent is inferred, not observed; W00 measures it. The design minimises, in order: bytes fetched per wake-up (ternary), wake-ups (event gate; one pass per chunk), multiply-adds (fewer events, not narrower layers).

### 1.3 The ladder

Each rung is a bundle that passes the release tests of §11.8. Each adds one mechanism, has one wave, and falls back to the rung below.

| Rung | Adds | Control | Gate | If it fails |
|---|---|---|---|---|
| **R0** | Harness; best existing recogniser, pinned and packaged | — | runs on H0 with receipts | report the SLO miss |
| **R1** | Own dense model: Reflex + Cortex on every 12.5 Hz frame, transducer, float then int8 | a public streaming recipe reproduced on the same data | W10: within 5% relative of that reproduced recipe; the gap to R0's reference is recorded, not gated | fix data and recipe; nothing else proceeds |
| **R2** | Tower: right context D, and final mode | R1, same weights with D = 0 | W11: commit at D = 2 ≥ 5% relative better than D = 0; final ≥ 5% better than commit | single tap with fixed lookahead |
| **R3** | Event clock | R2, same weights with the dial at "keep all" | **K-EVENT** (§1.5) | frame clock; the gate stays as a low-power option |
| **R4** | 7-bit activations, ternary weights, integer runtime, bit-exact targets | R3 in int8 | **K-TERN** (§1.5) | ladder: two planes → int8 for sensitive blocks → int8 throughout |
| **R5** | Data and capacity scale; M and S by distillation | R4 at the smaller data scale | W20 per doubling; **K-ACC** | stop buying data; ship the best rung |
| Side | Text events; anchor adaptation; phrase biasing; policy search; specialists; refinement controller | the rung they attach to | own waves (§12) | off |

### 1.4 Targets

Two kinds, both needed.

**Relative, in our harness (these decide the win).** At matched p95 stable-word delay, on machine A (§11.5), against the best existing recogniser measured in the same harness:

- **ULTRA-L:** word error rate lower by at least 3% relative with a paired interval excluding zero, at no more than one third of its CPU cost.
- **ULTRA-M:** word error rate within +0.10 points (one-sided 95% bound) at no more than one fifth of its CPU cost.
- **Final text** within 0.3 points of the best full-context recogniser in the tournament.

**Absolute anchors (macro average of the eight public English test sets after the leaderboard's normalisation `[SRC 1]`).** For orientation: Nemotron Speech Streaming 0.6B reports 7.67 at 0.16 s and 7.07 at 0.56 s `[SRC 1]`; Parakeet Unified 0.6B reports 6.70 at 0.40 s, 6.52 at 0.56 s and 5.91 offline `[SRC 3]`. Vendor "latency" is algorithmic; ours is measured stable-word delay, so these are anchors, not the comparison.

| ULTRA-L | G1 viable | G2 parity | G3 lead |
|---|---:|---:|---:|
| Commit, D = 1 (about 0.2 s typical) | ≤ 8.2 | ≤ 7.3 | ≤ 6.9 |
| Commit, D = 2 (about 0.4 s typical) | ≤ 7.6 | ≤ 6.7 | ≤ 6.3 |
| Final (at a pause) | ≤ 7.0 | ≤ 6.3 | ≤ 5.9 |
| Preview | ≤ 8.8 | ≤ 7.8 | ≤ 7.3 |
| CPU-seconds per second, one thread, machine A, 160 ms grid | ≤ 0.30 | ≤ 0.20 | ≤ 0.14 |
| p99 compute per 160 ms chunk, one thread | ≤ 80 ms | ≤ 60 ms | ≤ 45 ms |
| Preview flicker, changed words per 100 | ≤ 12 | ≤ 8 | ≤ 5 |
| Noise panel: WER rise ÷ the float twin's rise | ≤ 1.30 | ≤ 1.15 | ≤ 1.05 |

ULTRA-M: the same WER rows plus 0.6 / 0.6 / 0.5 points; cost ≤ 0.12 / 0.08 / 0.055; p99 ≤ 30 / 20 / 15 ms. ULTRA-S: G1 and G2 only; commit at D = 2 ≤ 10.0 / 9.0; cost ≤ 0.03 / 0.02; p99 ≤ 8 / 5 ms.

At every gate: committed and final text are never revised (by construction, tested); no slice (names, numbers, technical terms, noise, long form, fast speech) worse than the reference by more than 10% relative at G2; from R4, identical hashes on every target.

### 1.5 Kill gates

| Gate | Wave | Test | Passes if | Otherwise |
|---|---|---|---|---|
| **K-EVENT** | W12 | Same R2 weights across a retention sweep (100, 75, 60, 45, 35% of frames kept) | at the chosen retention: WER rises ≤ 0.10 macro (one-sided 95% bound) and ≤ 0.30 on any set; deletions on the fast-speech slice not worse than +5% relative; full-pipeline CPU falls ≥ 15% | highest retention that passes; if none, the frame clock |
| **K-TERN** | W15 | R3 in float, int8, ternary one plane, ternary with plane 2 | WER rises ≤ 0.25 macro and ≤ 0.50 on the noise panel against float; full-pipeline CPU falls ≥ 15% against int8 | first passing rung of the ladder in §1.3 |
| **K-ACC** | W24 | Sealed comparison in our harness | within 0.5 points of the best streaming incumbent at matched delay | release as the fastest, not the most accurate; remaining budget goes to data |

### 1.6 Odds

My estimates; each is replaced by a measurement at the wave named.

| Outcome | Odds | Wave |
|---|---:|---|
| R1 within 5% of a reproduced public recipe on the same data | 85% | W10 |
| The tower pays at D = 2 and in final mode | 75% | W11 |
| ULTRA-M at ≤ one fifth of the incumbent's CPU cost | 70% | W01, W24 |
| K-EVENT passes at ≤ 60% retention | 60% | W12 |
| K-EVENT passes at ≤ 45% retention | 45% | W12 |
| K-TERN passes at the first or second rung | 55% | W15 |
| Bit-exact parity on four targets | 80% | W16 |
| ULTRA-L passes G2 | 35% | W24 |
| ULTRA-L passes G3 | 15% | W24 |

The G2 odds are low for one reason: about 300,000 gross hours here, of unknown unique size, against 530,000 for the incumbent `[SRC 1]`.

---

## 2. Evidence base

### 2.1 Measured `[SANDBOX]`

Cloud Xeon at 2.1 GHz, one thread, AVX-512 VNNI and VBMI, `gcc -O3 -march=native`. Two kernels with identical checksums: int8 by `vpdpbusd`; 2-bit packed ternary unpacked by `vpmovzxwq` and `vpmultishiftqb`, then the same instruction. One to sixteen input vectors per pass over 1M, 64M and 1,536M weights. Data, fit and source: Appendix A.

| Weights | Kernel | Fixed, ps per weight | Marginal, ps per multiply-add | R² |
|---|---|---:|---:|---:|
| 1,536M, out of cache | ternary | 19.4 | 11.0 | 0.999 |
| 1,536M, out of cache | int8 | 96.4 | 10.5 | 0.967 |
| 64M (16 MB packed) | ternary | 16.5 | 10.4 | 0.998 |
| 1M, in L2 | ternary | 14.1 | 10.9 | 0.995 |
| 1M, in L2 | int8 | 5.4 | 10.6 | 0.996 |

In cache, int8 is cheaper because ternary pays to unpack. Out of cache, ternary's fixed term is five times smaller. Hence: large matrices ternary, small hot ones int8.

**Limits.** One host, one thread, a synthetic matrix–vector kernel, one run of the fine sweep. No desktop, laptop or Arm core has been measured by anyone in this document set.

### 2.2 A second run, reported `[DOC]`

KAIROS v0.2 Appendix B reports the same source on a 2.3 GHz Xeon with all eighteen checksums matching. Reading its three out-of-cache ternary points gives roughly a = 30 ps and c = 6.5 ps `[D]`: a faster core and a slower fetch. With either set of constants ULTRA-L's estimated cost at the 160 ms grid is 0.10 to 0.11 CPU-seconds per second before the allowance (Appendix B). The total is stable; the split is not. W01 measures it where it matters.

### 2.3 Derived `[D]`

Incumbent, i7-11700K, eight threads, 8-bit `[SRC 2]`: marginal (42.7 − 27.0) ÷ 12 = 1.31 ms per 80 ms frame; fixed 27.0 − 2 × 1.31 = 24.4 ms per chunk, 90% of the 160 ms chunk. 27.0 ms × 8 threads ÷ 160 ms = 1.35 is **reserved** thread-seconds per second, an upper bound on its CPU cost, not a measurement of it.

### 2.4 Published `[SRC]`

| Model | Weights | WER, 8-set mean | CPU, as published |
|---|---:|---|---|
| Nemotron Speech Streaming EN | 600M | 8.43 / 7.67 / 7.07 / 6.93 at 0.08 / 0.16 / 0.56 / 1.12 s `[SRC 1]` | 27.0 ms mean, 28.9 ms p99 per 160 ms chunk; i7-11700K, 8 threads `[SRC 2]` |
| Parakeet Unified EN | 600M | 8.44 / 7.35 / 6.92 / 6.70 / 6.52 at 0.16 / 0.24 / 0.32 / 0.40 / 0.56 s; 5.91 offline `[SRC 3]` | — |
| Moonshine v2 Medium | 245M | 6.65 `[SRC 4]` | — |
| FUTO ASR4ALL-L | 96.8M | 8.90 raw; 6.66 with its corrector `[SRC 5]` | 14.1× real time, one i9-12900K thread `[SRC 5]` |
| Granite 5.0 TurboCTC | 470M | 5.72 offline `[SRC 6]` | — |
| Parakeet Redux (ternary) | 178 MB | 6.55 on seven sets; 9.04 under noise `[SRC 7]` | 113× real time, EPYC 9575F `[SRC 7]` |

The incumbent's card gives 530,000 training hours: a 250,000-hour in-house set plus YouTube-Commons 109,500, YODAS2 102,000, LibriLight 49,500 and MOSEL 14,000 `[SRC 1]`.

### 2.5 Assumed `[EST]`, measured early

| Quantity | Value used | Measured by |
|---|---|---|
| Events per second at the default dial | 5.6 (45% of frames) | W12 |
| Cortex wake-ups per second, 160 ms grid | 3.75 | W12 |
| Allowance on itemised kernel time | 1.5× | P13 |
| Plane-2 share of ternary rows | 15% | W15 |
| Training throughput | 300–600 audio-hours per GPU-hour | P11 |

### 2.6 Facts: confirmed this round, and still to confirm

| Fact | Status |
|---|---|
| `step(a, c)` is 1 when `a > c`, strictly; one argument is an error | Confirmed `[REPO: crates/feat-math/src/math_expr.rs, math_expr/parser.rs]`. TACTUS and TEMPO thresholds on grid values were therefore late by one step; APEX's endpoint rule does not parse. All rules here use half-step thresholds. |
| `(sym affine\|raw "expr")` genomes exist | Confirmed `[REPO: crates/feat-math/src/genome.rs]` |
| Registry v2 has 90 entries; "a future opcode needs a coordinated allocation; reservations 53+ are not appropriated" | Confirmed `[REPO: docs/contracts/operator-registry-v2.md]` |
| Typed-graph contract; `state_codec.rs` beside sample-time kernels; `wire_v2.rs`; `lexicase.rs`; `pareto.rs`; grammar-masking automaton; `docs/model-objects.md`; `docs/emitters.md` | Present `[REPO]` |
| Pre-registration template limits numeric threads to four | Confirmed `[REPO: research/flywheel_selfplay/PREREG_TEMPLATE.md]` |
| `registry/operators.toml`, `deploy/ci/coverage.json`, commit `df4f5250`, git history | Not in the archive. **Confirm** in the full repository. |
| "One open pull request at a time" | Not found by search in the archive's `AGENTS.md`, which describes claimed PR branches and worktree lanes. **Confirm** and follow whatever it says. |
| Expression caps: 64 nodes and depth 16 hand-written; 15 and 4 in search | From my first reading; **confirm** against the parser |
| Crate names, features, test commands | Unknown: no manifests. **Confirm** on day 1 |

---

## 3. Output contract

Adopted from CADENCE-CPU v2, which is the only reviewed plan whose text semantics are consistent. Every other plan, TACTUS included, called text "committed" and then let a later pass replace it.

### 3.1 Events

| Event | Meaning | Can earlier text change? |
|---|---|---|
| `Preview` | Replaces the current mutable suffix of the segment; carries a revision number | Yes, within the suffix |
| `Commit` | Appends text to the immutable prefix | No |
| `Final` | Closes the segment and commits what remains of its suffix | No model revises that segment afterwards |
| `CorrectionSuggested` | A later alternative for finalised or committed text | Does not mutate the transcript; the application decides |
| `Gap` | A range of input samples was lost or discontinuous | Nothing is invented for it |
| `Status` | Dial, grid or precision change; shed step; endpoint decision; skipped pass; refused adaptation | No |

Invariants, each a test in §11.7: the event sequence number strictly increases; `Preview` touches only the mutable suffix; the immutable prefix only grows; `Final` closes a segment exactly once; nothing after `Final` or a cancel claims new text for that segment; payload sizes are bounded.

### 3.2 Two modes

| Mode | For | Behaviour | Scored as |
|---|---|---|---|
| **DICTATE** | editable dictation, captions | `Preview` while the user speaks. At the endpoint the Tower runs in final mode over the whole segment and its transcript is emitted as `Final`. If the finalisation deadline passes first, the best eligible text is finalised and a `Status` says the pass was skipped. | preview snapshots; final text |
| **COMMIT** | applications that act on text before the utterance ends | `Commit` as the Tower clears each event at delay D. At the endpoint, final mode decodes **only the uncommitted suffix**, continuing from the committed decoder state. If the unconstrained final transcript differs inside the committed region, it is offered as `CorrectionSuggested`. | the commit ledger exactly as emitted |

Word error rate is always computed on what the interface emitted. A diagnostic full-context transcript is kept beside it to show what was left on the table; it never substitutes for the output.

### 3.3 Segments and boundaries

- A segment ends at an endpoint (§4.4), at a soft cap of 20 s where a low-energy gap or a word boundary exists, or at a hard cap of 30 s or 192 events.
- Every event belongs to exactly one segment and every token to exactly one event. A word that straddles a forced cut is owned by the segment holding its first event.
- Causal state (Reflex, Cortex and decoder history) carries across a cut. Final mode may read up to 2 s of the previous segment as context and may not re-emit its tokens.
- Transport packet size does not change results: 10, 20 and 37 ms packets give identical events (test in §11.7).

### 3.4 Audio, discontinuity, overload

- Internal timebase: mono 16 kHz int16, 64-bit sample counters. Other rates go through a stateful anti-aliased resampler whose delay is recorded. Capture callbacks never block and never run inference.
- **Gap.** On missing or duplicated samples: close the segment as incomplete, keep its immutable prefix, drop its mutable suffix, reset encoder and decoder state, emit `Gap` with the sample range, start a new segment. No delayed worker may emit into the new segment.
- **Overload.** Queues are bounded. Shed steps (§4.8) take effect at segment boundaries unless the mode switch was trained in-utterance (rule 3 of §0.2). When buffers would overflow, emit `Gap` for the lost range. A server refuses new streams that would break existing deadlines. The phrase "never behind" is not used.

### 3.5 Stream lifecycle

`ACCEPTING → DRAINING → FINISHED`, with `CANCELLED` and `FAILED` as terminal alternatives. `finish_input` is idempotent and starts one flush; `cancel` is idempotent; a failure is never reported as completion. After a worker crash the last valid immutable prefix is preserved and a restart begins a new state, not a pretended continuation.

---

## 4. Architecture

```
 16 kHz int16
     │
 COCHLEA   integer FFT → 80 mel → log code → table → 7-bit features         100 Hz → 25 Hz
     │
 REFLEX    causal memory blocks, ternary; dense CTC over 1,024 pieces       12.5 Hz, always on
     │        │ posterior, hidden state
     │     GATE   (sym raw "…") over integer features; dial θ; pools skipped frames
     │        │ events (about 45% of frames at the default dial)
 CORTEX    causal attention blocks, ternary; one pass per chunk of events   event clock
     │        │ causal top states c_j ──────────────► PREVIEW tap
 TOWER     m blocks; block 1 sees c_i up to j + D ──► COMMIT tap  (delay D events)
     │     same blocks over the closed segment ─────► FINAL tap   (at a pause)
     │
 TRANSDUCER  one stateless predictor and one output layer shared by the three taps
```

### 4.1 Cochlea

Integer from the first sample; no transcendental at run time (KAIROS).

| Step | Operation | Format |
|---|---|---|
| 1 | DC block (one pole, shift 10); pre-emphasis 0.97 | int16 in; Q15 |
| 2 | 25 ms frames every 10 ms; Hann | Q15 table |
| 3 | 512-point radix-2 FFT, one bit of scaling per stage | int32, block exponent |
| 4 | Power; 80 triangular mel filters, 60–7,600 Hz | u64; Q15 weights |
| 5 | Log code `floor(8·log2 E)`, 0..255 (0.38 dB steps; 96 dB of range) | leading-zero count and a compare ladder |
| 6 | Per-band energy smoother, shift 5 (about 320 ms), coded the same way | u8 |
| 7 | Normaliser: one 256 × 256 table indexed by (energy code, smoother code) | 7-bit out |
| 8 | Stack 4 frames | 320 values at 25 Hz |

The table is generated at build from a GP expression; default `(c0/(1e-6+c1)^0.98+2)^0.5-1.4142135623730951` (13 nodes, depth 7) `[MEM 1]`. All 65,536 inputs are enumerated at build, so no input is untested. Integer DSP has no subnormal slow path.

**This is a bet, isolated.** R1 trains with a standard floating-point log-mel front end as the control. W13 swaps in the Cochlea and requires WER within +0.10 points. A second arm computes the normaliser directly with an integer square root instead of the table (APEX's objection to 8-bit log codes). The loser is dropped.

### 4.2 Reflex

Purpose: a dense, cheap acoustic grid; token hints for the Cortex; the features the gate reads; a transcript of last resort.

- Linear 320 → d_s (int8); two blocks at d_s, 25 Hz; stack two frames; linear 2d_s → d_r (int8); n_r blocks at d_r, 12.5 Hz; CTC head over 1,024 lower-case pieces plus blank (int8).
- **Block:** per-channel causal FIR over 16 frames (int8 taps) `[MEM 4]` plus three leaky integrators `m ← m − (m >> s) + (x >> s)`, s ∈ {2, 4, 6} (TEMPO); integer RMS norm; feed-forward d → 4d → d, clipped ReLU, ternary. Every fourth block adds 4-head attention over 32 frames.
- Strictly causal. Ternary: the Cortex pass evicts it between chunks, so it is fetched every chunk, and an int8 Reflex-L would add 0.014 CPU-seconds per second `[D]`.
- **CTC feasibility is checked, not assumed.** A CTC target with U tokens and R adjacent repeats needs at least U + R positions `[MEM 17]`. For every training example the trainer tests this at 12.5 Hz; infeasible examples have the term masked and are counted by speaking-rate decile. If more than 0.5% of the fastest decile is infeasible, the head becomes a **two-position head** (one linear layer emitting two posteriors per frame, a 25 Hz grid at 12.5 Hz cost). That remedy is pre-declared in W02.
- The Reflex is never adapted. Its outputs steer the gate, so changing it would change committed text.

### 4.3 The gate

A frame becomes an **event** when a rule fires. The rule is a genome in the repository's form `[REPO: crates/feat-math/src/genome.rs]`.

| Feature | Meaning |
|---|---|
| `col(0)` | 1 − p(blank) |
| `col(1)` | posterior entropy ÷ ln 1025 |
| `col(2)` | frames since the last event (integer) |
| `col(3)` | spectral flux, 0..1 |
| `col(4)` | largest non-blank probability |
| `col(5)` | 1 if the top piece differs from the previous frame's |
| `col(6)` | speech activity (smoothed `col(0)`, shift 3) |
| `col(7)` | **dial** θ in −1..1 |

Default (15 nodes, depth 4; fits the search caps). `step(a, c)` is 1 when a > c, so thresholds sit between grid values:

```
(sym raw "max(step(col(0)+col(1), 0.1+col(7)), step(col(2), 2.5)*step(col(6), 0.5))")
```

First term: the frame is informative. Second: in speech, never more than 240 ms without an event.

- **Integer lowering.** Gate rules use only `step`, `max`, `min`, `+`, `−`, `*`, constants and `col(n)`. Features are Q7 integers, constants are rounded to Q7 at build, and the rule compiles to integer comparisons. Anything else is refused.
- **The dial.** θ = −1 keeps every frame: the same-weights control, and the highest-quality tier. Training samples θ per utterance and, for modes that may switch in-utterance, at random switch points (§8.4).
- **Pooling is lossy and says so.** Skipped frames enter the next event as a weighted mean (weights `1 + 127·(1 − p(blank))`). Every frame contributes; order and detail within the pool are lost. Whether that loss matters is what K-EVENT measures.
- **Guard rails outside the rule:** one event per second in silence; a flush 240 ms after the last event if tokens await commit; θ changes are `Status` events.
- **Careful mode from run-time observables only.** If the last 2 s show mean normalised entropy above 0.45, more than 8.5 events per second, or any decoder cap hit, the **next segment** runs at θ = −1:

```
(careful (sym raw "max(step(col(0), 0.45), max(step(col(1), 8.5), step(col(2), 0.5)))"))
```

False triggers and their cost are part of the measured policy. The reference transcript plays no part (rule 4).

Event input: `x = W_in·[h_event ; pooled] + E_gap[min(gap, 15)] + Σ_{k≤4} p_k·E_tok[piece_k] + E_style[style]`. The fourth term hands the Reflex's four best pieces to the Cortex (KAIROS). `style` selects cased-and-punctuated or plain output (§4.6).

### 4.4 Cortex

- n blocks at width d. Block: integer RMS norm → attention → causal depthwise convolution over the event index (kernel 5) → feed-forward d → 4d → d, clipped ReLU → residual in int16.
- **Attention:** h heads over the last 128 events plus the current chunk. Position is a learned bias per head indexed by a 32-bucket log scale of the time gap in frames: a table lookup, exact, and indifferent to irregular spacing.
- **One pass per chunk** (TEMPO). All events of a chunk go through the layers together, so the weights are fetched once per chunk. Within a chunk each event sees the whole chunk.
- **Grid:** 1, 2, 4, 7 or 14 frames (80 ms to 1.12 s), fixed by the mode and tied to audio time, so a transcript depends on the audio and the logged mode, never on machine load.
- **Every admitted event updates every block's state.** There is no per-event early exit. Precision and depth are fixed for a segment.
- The convolution over the event index has a receptive field that stretches in physical time as the gate thins; it is trained across the dial range and reported as such.
- **Not in the default:** self-conditioning by intermediate CTC on the event grid. It needs feasible CTC targets on events. It is wave W14b, with masking and a 1% ceiling on masked examples.

### 4.5 Tower: right context and final text (from KAIROS v0.2)

Let `c_j` be the Cortex's causal top state for event j.

1. **Block 1** produces `u¹_j` at step j + D. Query: `c_j`. Keys and values: `c_i` for i from j − 128 to j + D. Right context enters here and nowhere else.
2. **Blocks 2..m** are causal in the delayed stream, so the total delay stays D events.
3. **Final mode.** At an endpoint the same m blocks run once over the closed segment with every block attending to all of it.
4. **Training** draws D from {0, 1, 2, 3} per batch and runs final mode with probability 0.2, with a consistency term between modes (§8.2).
5. **Delay.** At the default dial, D = 1 is about 200 ms typical and at most 240 ms to the next event plus one grid step plus compute; D = 2 is about 400 ms typical and at most 480 ms plus one grid step. In a pause the flush bounds it at 240 ms.

One mechanism and one set of weights give three points on the delay axis. W11 includes a dedicated-weights arm (a separate final-pass stack, as in TEMPO and TACTUS) in case sharing costs accuracy.

### 4.6 Transducer and the three taps

- **Predictor:** stateless `[MEM 19]`; the sum of two embeddings of the last two non-blank pieces.
- **Joiner:** `u = clippedReLU(P_tap·h + pred)`; `logits = W_out·u` over 4,096 pieces plus blank. `P_tap` is one d × J projection per tap; predictor and `W_out` are shared. No `tanh`, so no table.
- **Greedy decoding:** at an event, loop: blank → advance to the next event; non-blank → emit and update the predictor. At most 8 non-blank pieces per event; a cap hit forces an advance and increments a counter that the careful-mode rule reads.
- **Commit tap** consumes Tower outputs in event order. Its state (last two pieces, event cursor) persists across chunks and packets.
- **Preview tap, with no splice.** After each chunk: copy the commit decoder's state, decode the Cortex's causal states for the events not yet committed, emit the result as the mutable suffix, discard the copy. Preview and commit can never duplicate or drop a word at a join, because the preview always starts from the committed state and covers exactly the uncommitted events.
- **Final tap** decodes Tower final-mode states: the whole segment in DICTATE mode, the uncommitted suffix in COMMIT mode.
- **Why a transducer on events, and CTC only on the dense grid.** A transducer may emit several pieces at one event, so no count of events can make a transcript unreachable. CTC cannot. All four reviews converge on this.
- **Vocabulary:** 4,096 cased pieces with punctuation. Decoding with the full output layer is 3% of the per-event cost at L and 22% at S `[D]`, so no shortlist at L and M; W14 tests 1,024 / 4,096 / 16,384 and, for S, a smaller head.
- **Case and punctuation without changing the loss.** Each training example carries a style flag. References without case or punctuation are trained as style *plain* with lower-case targets; cased references as style *cased*. No token is relabelled as blank. Formatting quality is scored separately, and a formatted transcript must normalise to the same words as the plain one.

### 4.7 Scheduler

Per chunk of the grid: Cochlea and Reflex on its frames; gate; if no event, stop; else one Cortex pass, then the Tower for every event whose right context is now complete; commit; preview.

**Itemised cost per wake-up** (Appendix B): ternary weights of Cortex and Tower; key and value state read per block; attention scores and mixing; tap projections and joiner calls (two for commit, six for preview at D = 2). At L the split is about 95% feed-forward and projection weights, 1% attention and 3% decoding `[D]`.

**Many streams on one core.** In a server, the events of all streams in the same tick go through one weight pass. The fixed term is paid once, so about 14 streams per core at L, 36 at M and 150 at S `[D]` with the allowance. Integer kernels make this safe: a stream's hashes are identical alone and in a batch (test in §5.5). Cross-stream batching waits at most 5 ms and is switched off if it hurts the latency profile.

### 4.8 Shedding

In order, each a `Status` event: (1) raise θ; (2) widen the grid; (3) drop plane 2; (4) skip final mode and finalise the commit-tap text; (5) Reflex transcript only; (6) `Gap`. Steps 1 to 3 apply at the next segment boundary unless W21 has shown the in-utterance switch is trained and harmless.

### 4.9 Adaptation

| Part | Changes | Learns from | Default |
|---|---|---|---|
| **Phrase list** | A bounded bonus on transducer paths that follow a user phrase (beam 4 only when the list is non-empty) | phrases the user supplies | on when a list is given |
| **Anchor prior (A1)** | A bias per piece on commit and final logits: shrunk log-ratio of confirmed counts to the model's prior | enrolment text, explicit corrections | off until enrolment |
| **Anchor readout (A2)** | A ridge residual on commit and final logits (below) | the same confirmed labels, force-aligned to events | off until the guard passes |
| **Preview teacher** `[BET]` | The preview tap's projection only | final text of earlier segments | off; wave W18 |

Nothing here touches the Reflex, the gate, the Cortex or the Tower. The preview tap is downstream of everything committed text depends on, so the preview teacher cannot change committed or final text by construction.

**A2, exactly.** Let φ ∈ ℝ¹²⁸ be a fixed int8 projection of the joiner hidden vector, RMS-normalised. For the N lattice points on the forced alignment of confirmed text, stack Φ ∈ ℝ^{N×128}, targets Y (one-hot, including blank) and the model's probabilities P, restricted to the pieces that occurred plus blank. With R = Y − P:

```
A   = ΦᵀΦ + λI                      (128 × 128)
ΔW  = A⁻¹ Φᵀ R                      applied as  logits + η · ΔW φ
```

**The guard, with dimensions.** Folds are whole utterances. For a fold B with rows Φ_B:

```
primal:  S_BB = Φ_B A⁻¹ Φ_Bᵀ        E = R − Φ ΔW        E_B(held out) = (I − S_BB)⁻¹ E_B
dual:    C = (ΦΦᵀ + λI)⁻¹ (N × N)   α = C R             E_B(held out) = (C_BB)⁻¹ α_B
```

These are the same quantity. APEX wrote the dual form while defining C in feature space; CADENCE v2 wrote the primal. Use the primal when N > 128. It needs one factorisation of A and one small solve per fold, not a refit `[REPO: crates/feat-math/src/kernel/ridge_fast.rs]`. Held-out predictions `P_B + η(R_B − E_B(held out))` give held-out log-loss and token error on confirmed labels for each η ∈ {0, 0.25, 0.5, 1}.

Adopt η > 0 only if at least four of five folds improve held-out log-loss by more than 2% relative, held-out token error is not worse, and the update is not already spanned by the existing readout:

```
(anchor (sym raw "step(col(0), 3.5)*step(col(1), 0.02)*step(0.5, col(2))"))
```

η = 0 skips the add and is bit-identical to the unadapted model. The solve runs off the audio path in floating point with a fixed operation order; its result is quantised to int8 rows with multipliers, and the hash of that table is in the receipt.

**What the guard does not prove.** It bounds a regression loss on confirmed labels. It is not a theorem about word error rate. The product claim is tested on later sessions of the same user (the ADAPT-TEST partition) with a per-user harm rate, and every such number is labelled *recall*.

**Privacy defaults.** Nothing leaves the device. Corrections and adapter state are stored only if the user turns that on; a profile can be erased; receipts carry hashes, not text.

### 4.10 Refinement controller `[BET]` (from CADENCE-CPU v2)

Final mode is cheap (88 ms for a 10 s segment at L on one thread `[D]`), so by default it always runs. A controller earns its place in two cases: under load, to choose which segments get final mode; and for a **cascade**, where a stronger full-context recogniser from the R0 tournament re-transcribes retained audio for selected segments. The controller is trained on measured benefit (errors of the fast text minus errors of the refined text) from features available before the pass: confidence and blank statistics, agreement of Reflex and transducer, rare pieces, speech rate, segment length, noise proxies, revisions already seen. Always-refine and never-refine are the controls. Gate (W19): at least 3% relative better final WER for at most 15% more CPU, or equal WER for 20% less CPU than always-refine.

### 4.11 Capacity, specialists

- **Capacity or data? (KAIROS v0.2)** Train the full-precision, non-causal twin of L on all data. If it misses the accuracy target, the limit is capacity or data: W23 compares a wider dense Cortex against a mixture-of-experts Cortex (APEX) at equal bytes per wake-up, and against more data at equal cost. If the twin reaches the target and the student does not, the gap is streaming or quantisation, and more weights are not the remedy.
- **Specialists** (after G1): per-cluster scaling of hidden units `[MEM 2]`, folded into the integer row multipliers so they cost nothing at run time. The router is part of the committed recognition path; its choice is fixed per segment and logged. A specialist is credited only on sessions routed to it unprompted `[REPO: docs/research/waves/h458-h465-black-hole-20261116/]`.

### 4.12 Presets

From `sizing3.py` (Appendix B). "Host A" uses my measured constants; "host B" the reported second run. Kernel time is itemised; "allowed" multiplies by 1.5.

| | S | M | L |
|---|---:|---:|---:|
| Reflex: d_s / d_r / blocks | 192 / 256 / 8 | 256 / 384 / 12 | 256 / 512 / 12 |
| Cortex: width / blocks / heads | 512 / 10 / 8 | 1024 / 13 / 16 | 1536 / 16 / 16 |
| Tower blocks; joiner width | 2; 384 | 3; 512 | 4; 640 |
| Reflex / Cortex / Tower / heads, weights | 5.8M / 32.3M / 6.3M / 5.3M | 17.7M / 165.6M / 37.7M / 7.9M | 30.3M / 456.4M / 113.2M / 10.8M |
| **Total weights** | **49.7M** | **228.9M** | **610.8M** |
| of which ternary / int8 | 43.1M / 6.6M | 218.3M / 10.6M | 595.6M / 15.2M |
| RAM for weights; download | 19.0 MB; 16.5 MB | 73.4 MB; 60.8 MB | 186.4 MB; 152.2 MB |
| State per stream | 1.8 MB | 4.6 MB | 8.5 MB |
| Bytes fetched per wake-up | 12.4 MB | 62.1 MB | 170.7 MB |
| Multiply-adds per event | 58M | 254M | 684M |
| Fixed cost per wake-up, host A / B | 0.9 / 1.0 ms | 4.9 / 7.3 ms | 13.4 / 20.2 ms |
| Cost per event, host A / B | 0.6 / 0.4 ms | 2.8 / 1.7 ms | 7.5 / 4.4 ms |
| CPU-seconds per second, 160 ms grid, host A / B | 0.008 / 0.007 | 0.039 / 0.041 | 0.100 / 0.109 |
| … with allowance | 0.013 / 0.011 | 0.058 / 0.062 | 0.150 / 0.163 |
| … 560 ms grid | 0.006 / 0.005 | 0.027 / 0.023 | 0.069 / 0.061 |
| … every frame kept | 0.015 / 0.013 | 0.070 / 0.071 | 0.186 / 0.190 |
| Chunk with three events, with allowance | 4.6 / 3.6 ms | 21 / 20 ms | 56 / 52 ms |
| Final mode, 10 s segment | 6.8 / 4.2 ms | 31 / 19 ms | 88 / 54 ms |
| Streams per core, batched | 150 / 250 | 36 / 62 | 14 / 24 |

Three readings. (1) The gate saves 1.8× in CPU, not the 2.2× the event count suggests, because wake-ups fall less than events; K-EVENT asks for 15% and reports the curve. (2) L is 611M weights, the incumbents' class, and 119M fewer than TACTUS-L, because the Tower replaces a separate final-pass network. (3) G3's cost row (0.14) is below the allowed estimate (0.15 to 0.16); it needs a smaller allowance or the 560 ms grid. That is why it is a stretch gate.

---

## 5. Numeric contract (rung R4)

R1 to R3 may run on any floating-point or int8 runtime. From R4 the following holds, and cross-target bit-exactness becomes a release test.

### 5.1 Formats

| Name | Values | Use |
|---|---|---|
| **A7** | u8 in 0..127, zero point 0 | outputs of clipped ReLU; features |
| **A7z** | u8 in 0..127, zero point 64 | signed, normalised inputs to linear layers; attention queries |
| **R16** | int16, saturating adds, one fixed exponent per stack | residual stream |
| **ACC** | int32 | accumulators |
| **I8** | int8 in −127..127 | embeddings, first and last layers, FIR taps, heads, attention keys and values |
| **T1** | ternary {−1, 0, +1}, 2 bits; code 3 is refused at load | large matrices |
| **T2** | two ternary planes, `w = m₁t₁ + m₂t₂` | the most sensitive rows; plane 2 is a trained, droppable mode |

### 5.2 Operations, to the rounding rule

1. **Linear.** `acc = Σ wᵢaᵢ` in int32. For A7z inputs the true product is `Σ wᵢ(aᵢ − 64) = acc − 64·Σwᵢ`; `Σwᵢ` is precomputed per row and folded into the row bias.
2. **Requantise.** `y = sat16((acc·M + b + 2^(s−1)) >> s)`, computed in int64, arithmetic shift, 1 ≤ s ≤ 31 (s = 0 is refused at build). This rounds halves toward +∞ for both signs; test vectors include −0.5 → 0 and −1.5 → −1.
3. **Clipped ReLU.** `clamp(y >> e, 0, 127)`.
4. **RMS norm.** `n = Σ rᵢ²` in int64; `rms = isqrt(n / d)` with floor division and floor square root; `inv = 2³⁰ / max(rms, 1)` (floor); `outᵢ = clamp(((rᵢ·gᵢ·inv) >> s_n) + 64, 0, 127)`, an A7z value. All products fit in int64 for d ≤ 4,096.
5. **Attention scores.** Query A7z, keys I8, same kernel as a linear row; each cached key stores its sum for the zero-point correction.
6. **Softmax.** `q = score >> s_a`, so one step is 1/16 nat. `m = max q`; `dist = min(m − q, 255)`, non-negative by construction; `e = EXP[dist]` with `EXP[k] = round(65535·exp(−k/16))` and `EXP[0] = 65535`; `Z = Σ e` in u32 (at most 200 keys, so below 2²⁴); `p = (e·2¹⁵ + Z/2) / Z` (floor). Equal scores give equal weights; a row always contains itself, so Z > 0. (KAIROS's wording, "subtract the maximum, clamp to 0..255", gives zero everywhere if read literally.)
7. **Mixing.** `Σ pᵢ·vᵢ` in int32 (bounded by 200 × 32,768 × 127 < 2³¹), then requantise.
8. **Tables** (Cochlea normaliser, `EXP`, position biases) are bundle data covered by its hash and regenerated bit for bit at build.
9. **Floating point** appears only at build time and in the anchor solve, which runs off the audio path and ships its result as an integer table.

### 5.3 Rules

| # | Rule | Why |
|---|---|---|
| N1 | **The 7-bit rule.** Activations never exceed 127; int8 weights never reach −128. | AVX2 `vpmaddubsw` saturates at 32,767; the worst pair is 2 × 127 × 127 = 32,258. The same bound makes Arm `sdot` and the WebAssembly relaxed dot product `[SRC 16]` exact. |
| N2 | The build proves `fan-in × 127 × max\|w\| < 2³¹` per row and refuses the bundle otherwise. | No run-time overflow checks. |
| N3 | Row partitions for threads are static; integer sums do not depend on order. | Thread count and batch size cannot change a result. |
| N4 | Plane 2 off is a mode with its own training term and its own published WER. | Rule 3 of §0.2. |
| N5 | One arena per stream; no allocation after start, verified by an allocator counter. | Bounded tails. |
| N6 | Weights are stored in the order a pass reads them, in one read-only mapping. | The fixed term is sequential memory traffic. |

**Open question (W17).** Ternary layers do not need the 7-bit rule on x86: with weights in {−1, 0, +1}, 8-bit inputs give a worst pair of 510. One more bit of input precision into ternary layers may recover part of the ternary loss; Arm and WebAssembly then need a bias trick. W17 decides whether that is worth it.

### 5.4 Kernels

| Target | Ternary | int8 | Status |
|---|---|---|---|
| x86 AVX-512 VNNI + VBMI | `vpmovzxwq` + `vpmultishiftqb` unpack; `vpdpbusd`; four accumulators per input | `vpdpbusd` | measured `[SANDBOX]` |
| x86 AVX2 | shift-and-mask unpack to signed bytes; `vpmaddubsw` + `vpmaddwd` | same | to build |
| Arm NEON with dot product | table unpack; `sdot` | `sdot` | to build |
| WebAssembly | shift-and-mask unpack; relaxed dot product | same | to build |
| Scalar | loop | loop | reference |

Bake-off candidates in W01: table-lookup kernels over groups of trits (`[SRC 21]`, `[SRC 23]`, and APEX's 3-trit form with separate low and high byte tables), masked add and subtract straight from packed bits `[SRC 20]`. Admission is by rule 11 of §0.2.

### 5.5 Three different parity questions

| Question | Test |
|---|---|
| Does each kernel implement the declared arithmetic? | Bit-exact against the scalar reference on a randomised property test (extreme operands, zero point, negative rounding, every valid packed code) and on 1,000 parity utterances |
| Does quantisation preserve the model? | Quantised against float WER: K-TERN |
| Does the system reproduce across backends, threads and batches? | Identical per-utterance hashes of events, logits and tokens on all four targets; with 1, 2, 4, 8 threads; alone and in a batch of 16; and between the trainer's integer simulation and the runtime |

---

## 6. The model as a GP program

The model is one S-expression. The trainer, the cost estimator, search and the emitters all read it. CADENCE-CPU v2 argued for plain typed configuration files until a fixed model works; the program text **is** that typed configuration, and the heavier machinery is staged:

| Stage | Built when | Contents |
|---|---|---|
| G0 | days 5–6 | parser, canonical printer, type and rate checker, static cost estimator |
| G1 | with R4 | lowering to a typed graph; Rust emitter; integer reference interpreter; parity corpus |
| G2 | after R3 | search over genes on the fleet; C17 and WebAssembly emitters; registry admission |

### 6.1 What is reused as it exists `[REPO]`

| Piece | Where | Use |
|---|---|---|
| `(pipe X (op k=v …))`, `(portfolio …)`, `(sym affine\|raw "expr")`; canonical print | `crates/feat-math/src/dsl.rs`, `genome.rs` | base syntax; every policy is a `sym` |
| Scalar expressions: 19 functions, `col(n)` and `cN`, two-argument `step` with strict `>` | `math_expr.rs`, `math_expr/parser.rs` | rules and table-generating expressions |
| Ordered-axis typing: `Axis::SampleTime`, `Causality::{Causal, Offline}`, `FitScope::{TrainFold, Stateless}` | `kernel/ordered_axis.rs` | every audio stage declares all three |
| Causal sample-time kernels with a `state_codec` | `kernel/timescale_nabla/`, siblings | the pattern for checkpointing and resuming a stream bit-exactly |
| Typed graph and edit contract | `kernel/ir/`, `docs/contracts/typed-graph-and-edit-v1.md` | lowering target, under a new payload identifier |
| Bit-exact ridge with leave-one-out and Schur gain | `kernel/ridge_fast.rs` | front-end probes; anchor solve; rounding |
| Goertzel bank; Savitzky–Golay | `kernel/goertzel.rs`, `savgol.rs` | optional modulation and delta features, once admitted on sample time |
| ε-lexicase; Pareto archive | `kernel/lexicase.rs`, `pareto.rs` | gene selection |
| Grammar-masking automaton; 401-token policy vocabulary | `crates/jope-policy/src/grammar.rs`; `deploy/appliance/SPECIFICATION.md` | proposal model for search |
| Emitters with bit-pattern parity tests | `emit_*.rs`, `docs/emitters.md` | one program, several targets |
| Model objects with a manifest of hashes | `docs/model-objects.md` | the bundle |
| Memory-mapped bank, exact deck, quarantine gate; atomic claims and tail stealing | `research/flywheel_selfplay/h456_ultra_entropy_recipe/` | data bank and fleet jobs |

### 6.2 Rules for extending it

1. Same lexer, atoms and canonical print. A program that does not re-print to itself is refused.
2. `(asr …)` is a new top-level form. No existing program, codec tag or receipt is reinterpreted.
3. **No identifier is assigned here.** The registry says a new opcode needs a coordinated allocation. Until the owner rules (O7), audio operators live in an unregistered dialect table.
4. Additions to `ordered_axis` are a versioned change: an event-time axis; fit scopes `Pretrained` (tensors bound by hash), `Session` and `Profile`.
5. Every stage is `Causal` except Tower block 1 with a declared `right` bound and final mode, which runs only over a closed segment.
6. Expression caps: 64 nodes and depth 16 hand-written; 15 and 4 from search. A gate rule stays in the comparison fragment.
7. Cost is static: from the program alone the estimator returns weights by format, bytes per wake-up, multiply-adds per frame and per event, and state bytes. The runtime counts the same quantities and a test asserts equality.

### 6.3 Grammar

```
asr      := "(asr" meta stream reflex gate careful? cortex tower decoder endpoint sched anchor? route? ")"
meta     := "(meta" kv+ ")"
stream   := "(stream" name stage+ ")"
reflex   := "(stack" "reflex" pipe ctc ")"
gate     := "(gate" sym kv* ")"
careful  := "(careful" sym ")"
cortex   := "(stack" "cortex" pipe ")"
tower    := "(stack" "tower" pipe ")"
decoder  := "(transducer" kv+ tap tap tap ")"
tap      := "(tap" kv+ ")"
endpoint := "(endpoint" sym kv* ")"
sched    := "(sched" kv+ ")"
anchor   := "(anchor" kv+ sym ")"
route    := "(route" kv+ ")"
pipe     := "(pipe" source stage+ ")"            ; existing
sym      := "(sym" ("affine" | "raw") string ")" ; existing
stage    := "(" op kv* ")"                       ; existing
```

### 6.4 ULTRA-L

```
(asr
  (meta lookahead_ms=0 name=ultra-l sample_rate=16000 version=1)
  (stream cochlea
    (dc_block shift=10)
    (preemph coef=0.97)
    (frame hop=160 win=400 window=hann)
    (fft_power n=512)
    (melbank bins=80 hi=7600.0 lo=60.0)
    (logcode steps=8)
    (ema name=m shift=5)
    (lut2 expr="(c0/(1e-6+c1)^0.98+2)^0.5-1.4142135623730951" out=u7 scale=24.0)
    (stack_frames n=4))
  (stack reflex
    (pipe cochlea
      (linear fmt=i8 out=256)
      (fsmn_ffn d=256 ffn=4 fmt=t1 layers=2 taps=16 timemix="2 4 6")
      (stack_frames n=2)
      (linear fmt=i8 out=512)
      (fsmn_ffn attn_every=4 d=512 ffn=4 fmt=t1 heads=4 layers=12 taps=16 timemix="2 4 6" window=32))
    (ctc delay_penalty=0.3 fmt=i8 name=reflex positions=1 vocab=reflex-bpe1024))
  (gate
    (sym raw "max(step(col(0)+col(1), 0.1+col(7)), step(col(2), 2.5)*step(col(6), 0.5))")
    flush_ms=240 pool=mass silence_ms=1000)
  (careful (sym raw "max(step(col(0), 0.45), max(step(col(1), 8.5), step(col(2), 0.5)))"))
  (stack cortex
    (pipe events
      (event_embed d=1536 gap=16 style=2 topk=4)
      (attn_conv_ffn cache=128 conv=5 d=1536 ffn=4 fmt=t1 heads=16 layers=16 plane2=0.15 pos=timebucket32)))
  (stack tower
    (pipe cortex
      (attn_ffn cache=128 d=1536 ffn=4 fmt=t1 heads=16 layers=4 plane2=0.15 right=2)))
  (transducer context=2 fmt=i8 joiner=640 max_symbols=8 vocab=ultra-bpe4k
    (tap name=preview source=cortex)
    (tap name=commit source=tower)
    (tap name=final mode=segment source=tower))
  (endpoint
    (sym raw "max(step(col(0), 7.5), max(step(col(0), 3.5)*step(col(2), 0.5), step(col(3), 191.5)))")
    hard_cap_s=30 soft_cap_s=20)
  (sched grid="1 2 4 7 14" mode=2 shed="theta grid plane2 final reflex")
  (anchor eta="0.0 0.25 0.5 1.0" folds=5 kind=ridge rank=128 source=confirmed
    (sym raw "step(col(0), 3.5)*step(col(1), 0.02)*step(0.5, col(2))")))
```

Endpoint features: `col(0)` consecutive non-speech frames, `col(2)` 1 if the last piece ends a sentence, `col(3)` events in the segment. So: 640 ms of silence; or 320 ms after a sentence end; or 192 events. Expression sizes on a replica of the parser: gate 15 nodes, depth 4; careful 11, 4; endpoint 15, 5 (hand-written cap); anchor 11, 4; normaliser 13, 7. P03 asserts them under the real parser.

### 6.5 What search may change

"Save the gene, not the model" `[REPO: research/flywheel_selfplay/PREREG_TEMPLATE.md]`.

| Gene | Instrument | Caveat |
|---|---|---|
| Cochlea expression, scale, filter edges | closed-form ridge from features to aligned phone classes, exact leave-one-out; survivors get a stem-only retrain | — |
| Endpoint rule; phrase bonus; anchor thresholds | replay on cached decoder outputs | truly downstream of the cache, so replay is valid |
| Gate rule, dial, careful-mode rule | **re-run the Cortex**; cached posteriors from another gate are invalid | a cached screen may rank candidates; it cannot decide |
| Grid; shed order | the runtime on the latency panel | — |
| Format per block (T1, T2, I8) | sensitivity as a proposal, then end-to-end measurement; layer errors do not add | a mask may enable only rows whose plane-2 values were trained and stored |
| Widths, depths, vocabulary, Tower depth | waves | — |

Selection: ε-lexicase over (condition, speaker group) cells, a Pareto archive over WER, CPU cost and commit delay, simple controls (hand-set and random search) in every search wave, and confirmation on a partition the search never saw. Thresholds are refitted per checkpoint, never carried across training runs (the H461 dose effect `[REPO: research/flywheel_selfplay/h461_pairnf_dose_fbal]`).

---

## 7. Data and labels

### 7.1 Count unique audio, not rows

Collections overlap: LibriSpeech, MLS, Libriheavy and LibriLight are all LibriVox; Libriheavy is LibriLight audio with transcripts `[DOC]`; Loquacious *large* contains 11,000 hours of Libriheavy `[SRC 26]`. Hours summed across them are gross.

Maintain an **audio identity graph**: source collection → original recording → time interval → decoded-PCM hash, with transcript variants, speaker and session, licence record and split membership hanging off the interval. Dataset rows are references to intervals. Report gross hours, exact duplicates removed, overlaps merged, uncertain near-duplicates quarantined, and **unique admitted hours** with speaker and session counts by domain. Every budget in §8.8 is computed on unique hours.

### 7.2 Stages

| Stage | Material | Purpose | Gate to expand |
|---|---|---|---|
| D0 | a few hours | ingestion, losses, export, stream state | a tiny subset can be overfit |
| D1 | LibriSpeech train-clean-100 | training mechanics; measured throughput | stable training; valid alignments |
| D2 | LibriSpeech 960 h | reproducible acoustic baseline; rung proofs | matches a public recipe within an explained tolerance |
| D3 | Loquacious *medium* (2,500 h) then *clean* (13,150) then *large* (25,150) `[SRC 26]` | breadth: talks, spontaneous speech, YouTube | unique hours, label audit and rights pass |
| D4 | MLS English (about 44,660 h gross) and other labelled sets | read-speech scale | gain per added cost beats another pass over D3 |
| D5 | Open pool with pseudo-labels: YouTube-Commons, YODAS2, LibriLight, MOSEL — 275,000 h gross, the incumbent's own open sources `[SRC 1]` | scale | pilot quality, diversity and budget shown (§7.4) |
| Text | 10⁹ words or more | text events (§8.5) | licence |
| Synthetic | 500–3,000 h | numbers, names, addresses, URLs | never the whole of any test |

Nothing proceeds to the next stage automatically. I am not a lawyer; every corpus and teacher needs an entry in the licence ledger first (ruling O2). SPGISpeech is evaluation-only unless licensed.

### 7.3 Partitions

| Partition | Use |
|---|---|
| TRAIN | gradients, tokenizer |
| CALIBRATION | quantisation ranges and fleet statistics; drawn from training, never from tests |
| DEV | iteration; reused and labelled as reused |
| CONFIRM | one decision round each, then retired to DEV; settings frozen before it is read |
| PUBLIC-TEST | the eight public test sets, for comparability |
| DOMAIN-SEALED | newly recorded target-domain speech; the product endpoint |
| ADAPT-TEST | earlier sessions for enrolment and corrections, later sessions for scoring |
| PARITY | 1,000 redistributable utterances for hashes; unrelated to quality selection |

Splits are by the strongest grouping available: recording, speaker, book or show. Decontamination is by canonical recording id, audio fingerprint and transcript similarity; 8-word text matches are alerts, not proof. LibriVox-derived sources are filtered at bank build against a book and speaker exclusion list from all LibriSpeech development and test material, and the bank refuses to build without it. A shard tagged sealed is refused at load `[REPO: h456_ultra_entropy_recipe/mmap_bank_h456.py]`.

### 7.4 Teachers and label quality

- **Pilot before factory.** 500 hours of candidate audio, stratified by source, accent, noise, speech rate and teacher agreement, with a human audit of a sample. Fit a label-quality estimator against audited errors, using teacher agreement, duration, alignment score, language check and acoustic quality as features.
- **Teachers.** Two of different lineage, plus the in-house full-context twin when it exists; a third only if the pilot shows its value exceeds its cost. Record each teacher's lineage: two derivatives of one model are not independent votes `[DOC]`. Candidates, licence permitting: Parakeet Unified offline `[SRC 3]`, Granite TurboCTC `[SRC 6]`, one sequence-to-sequence model.
- **Tiers** are thresholds on the fitted estimator, chosen for an explicit error-versus-coverage trade: *gold* (estimated error under 2%), *silver* (under 5%), *bronze* (distillation targets only), *reject*. Agreement is a feature of the estimator, not the definition of truth: a correct unusual name does not become wrong because models disagree.
- **Keep the hard speech.** A stratified reserve of disagreeing, accented and rare-term audio goes to human review instead of being filtered out.
- **Checks.** A label-poison canary (200 deliberately wrong transcripts must land in *reject* at 95% or better) is a unit test of the pipeline; it does not establish quality on natural errors. Corrupting reference labels on disk must change no model output.

### 7.5 Bank, deck, streams

- **Bank.** 16 kHz int16 in memory-mapped shards with an index (identity-graph ids, offsets, transcript hash, tier, licence), size and hash checked, a `READY` marker. Authoritative audio stays compressed; decoded PCM is a bounded cache (115.2 MB per hour: 300,000 hours would be 34.6 TB `[D]`).
- **Family** = (corpus, speaker). **Deck:** corpora with mass ∝ √hours; within a corpus an exactly uniform deck over speakers. H456 found exact uniformity necessary on its task `[REPO: h456_ultra_entropy_recipe/README.md]`; here natural and exactly-uniform mixes are arms of W20. The loader logs the **realised** hours and effective sample size per source; a table of intended shares is not evidence.
- **Streams**, as starting shares: human-labelled 0.38; teacher targets 0.26; residual (student disagrees with teacher consensus, sampled by percentile-ranked regret, at most 20% of any batch, with an effective-sample-size floor) 0.14; synthetic 0.12; text events 0.10 (returns to human if W18a fails). Raw regret collapsed to an effective size near 3 in H449; percentile ranks fixed it in H450 `[REPO: research/flywheel_selfplay/h450_scalefree_regret_gate/README.md]`.
- **Augmentation**, a small auditable set first: gain and clipping; additive noise; room responses; codecs; speed with alignments updated; spectral masking. Clean examples in every phase. Long concatenated sessions with carried state only after short streams are correct.

---

## 8. Training

### 8.1 Stages

Each stage keeps its output, has one gate, and has a budget from §8.8.

| Stage | What | Data | Gate |
|---|---|---|---|
| T0 | Bank, boards, vault, noise floor | D0–D2 | W03: the board resolves 2% relative at 95% |
| T1 | Reflex, float, dense CTC | D2 then D3 | W02: CTC infeasible on ≤ 0.5% of the fastest decile; dev WER reported |
| T2 | **R1**: Cortex on every frame, transducer from the Cortex top, float; int8 export. Control: one public streaming recipe reproduced on the same data | D2, then D3 | W10 |
| T3 | Full-context twin ("Oracle") for alignments and lattice targets; teacher pilot; pseudo-label factory | D3; D5 on the fleet | §7.4 checks |
| T4 | **R2**: Tower with D sampled, final mode | D3 | W11 |
| T5 | **R3**: event clock; dial, grid and switch-point sampling | D3 | **K-EVENT** (W12) |
| T6 | Text events; vocabulary and optional self-conditioning decisions | text + D3 | W18a, W14, W14b |
| T7 | **R4**: 7-bit activations, then ternary, block by block; fleet rounding; integer export | D3 | **K-TERN** (W15); parity (W16) |
| T8 | **R5**: all admitted data; capacity diagnosis | D4, D5 | W20, W23 |
| T9 | Distil L into M and S | all + L's targets | tier gates of §1.4 |
| T10 | Anchor adaptation; specialists; policy search | DEV, ADAPT-TEST | W22, W22c, W21b |

### 8.2 Objective

Starting weights; at most four are tuned in one wave. Each term is normalised per sequence and logged separately.

```
L =  1.0 · RNNT(commit tap; Tower at sampled D; emission restricted to events k..k+1)
   + 0.5 · RNNT(preview tap; Cortex top; restricted to events k..k+1)
   + 0.5 · RNNT(final tap; Tower in final mode)                 on the 20% of batches in final mode
   + 0.3 · CTC(Reflex; delay penalty)                           masked and counted when infeasible
   + 0.1 · consistency: KL between commit posteriors on the reference alignment path under two modes
   + λ_seq · sequence distillation                              teacher text as targets
   + λ_lat · lattice distillation                               §8.3
   + λ_feat · feature distillation                              §8.3
   + text-event terms                                           §8.5
```

- The transducer loss is a tested pruned implementation `[MEM 16]`, validated against an unpruned reference on a small fixture. The lattice has about 5.6 rows per second on events, which is what makes a 4,096-way joiner affordable.
- **Alignment restriction** `[MEM 5]`: forced alignments from the Oracle give each reference piece an event index k; the piece may be emitted at k or k + 1 of that tap's stream. The commit tap's stream is already delayed by D.
- **Consistency pairs**, one sampled per batch: dial default against keep-all; plane 2 on against off; D = 2 against final mode; grid 2 against grid 7. This is how rule 3 of §0.2 is paid for.
- λ_seq, λ_lat and λ_feat start at zero while the supervised path is validated, then take small pre-registered values.

### 8.3 Distillation, defined

| Kind | Teacher | Target | Valid because |
|---|---|---|---|
| **Sequence** | any teacher, any tokenizer | its transcript, normalised, re-tokenised for the student | text is tokenizer-independent |
| **Lattice** | the in-house Oracle only: same vocabulary, same events | at each point (t, u) on the Oracle's best alignment path, KL from the Oracle's joiner distribution to the student's, over the Oracle's ten most likely pieces, blank, and one **tail bucket** holding the rest; the student's tail is the sum of its probabilities outside that set | both distributions live on the same (event, prefix) lattice |
| **Feature** `[BET]` | an incumbent whose encoder frames are 80 ms, the same grid as ours `[SRC 1]`, licence permitting | its encoder states, through a learned linear projection, at matched frames (R1) or at event frames (R3) | no temporal mapping needs learning; masked to valid frames |

Posterior KL between unrelated systems is not used: their lattices and vocabularies do not share a support. Feature distillation (W10b) is the cheapest way I can see to borrow from a model trained on 530,000 hours.

### 8.4 Mode coverage

1. R3 starts from R2 weights at θ = −1; anneals toward 0 over 20% of the stage; then samples θ ∈ [−1, 0.3] per utterance, 10% of batches at −1.
2. Frames the forced alignment marks as piece onsets are force-kept with probability 0.9, annealed to 0 (TEMPO).
3. No gradient through the gate decision. 10% of gate decisions are flipped at random.
4. Grid sampled from {1, 2, 4, 7, 14}; D from {0, 1, 2, 3}; plane 2 dropped for 15% of utterances (from T7).
5. **Switch points.** In 20% of utterances one mode variable changes at a random event. Only modes covered this way may switch inside an utterance at run time (W21).
6. The Cortex always trains on the frozen Reflex's actual outputs, never on reference tokens standing in for them (the H446 mismatch) `[REPO: research/flywheel_selfplay]`.
7. Two controls are kept: the same checkpoint with the gate switched (isolates sensitivity), and separately trained dense and gated models at matched budget (isolates what training can recover).

### 8.5 Text events (from KAIROS v0.2) `[BET]`

The Cortex reads the Reflex's token guesses, so it can be trained as a corrector from text.

1. Tokenise a sentence twice: Reflex vocabulary for inputs, output vocabulary for targets.
2. One event per Reflex piece; gaps and extra keep-alive events sampled from real aligned statistics.
3. Fill the four-best field from a confusion table: for each true piece, the observed distribution of the Reflex's four-best lists.
4. Fill the acoustic field from a per-piece diagonal Gaussian of real Reflex states (integer class sums from the fleet), or a learned null vector with probability 0.3.
5. Train the transducer terms on the result.
6. **Twin consistency:** for 10% of audio batches, build the text twin of each utterance and add a KL between the two commit posteriors on the alignment path.

Gate (W18a): the rare-word slice improves by at least 5% relative and the macro WER is not worse beyond the margin.

### 8.6 Quantisation (T7)

1. Freeze the float checkpoint and its evaluation.
2. Activations to 7 bits with learned clipping `[MEM 14]`; normalisation and softmax switch to their integer forms in the forward pass.
3. int8 weights, per row. This is R3's int8 control for K-TERN.
4. Ternary, bottom up, block by block: codes and row multipliers initialised by second-order rounding on the fleet from integer statistics `[SRC 24]` `[SRC 25]`, then a short straight-through fine-tune. Plane 2 goes to the 15% of rows with the largest remaining error; its values are stored for every row a mask may enable.
5. **Precision is an optimisation, not a creed.** Per block choose among T1, T2 and I8 by measured end-to-end cost and WER, starting by restoring the most harmful blocks.
6. **Export check.** The integer reference interpreter and the Rust runtime give identical hashes on the parity corpus. The emitted program is the reference; the trainer's simulated forward is not.

**Exact statistics.** Layer inputs are integers in 0..127, so `Σ x xᵀ` is exact in int64 and independent of shard order. The asymmetric variant needs a cross-term with float teacher outputs; that is exact only after the teacher's values are rounded to a fixed int16 grid. Then each product is at most 127 × 32,767 and the sum over 2³⁹ frames stays below 2⁶³. (APEX and TACTUS called the cross-term exact without that step.)

### 8.7 Optimiser and soups

Use the reproduced recipe's optimiser for R1. For new stages: AdamW, β = (0.9, 0.98), weight decay 10⁻², clip 1.0; peak rate from a range test; warm-up then cosine. The last 15% of a stage uses valley cycles with checkpoints at each valley, and uniform soups of valley checkpoints scored on CPU hosts `[REPO: h456_ultra_entropy_recipe]`. H456 measured +1.1 to +5.7 Elo from soups on its task; W20b measures it here. No optimiser zoo: fifteen waves between H449 and H471 returned `NO_LIFT` on theirs `[DOC]`.

Checkpoints store model, optimiser, scheduler, averages, random states, sampler position and manifests, with atomic completion markers. Non-finite loss, repeated out-of-memory or a changed manifest pauses the run; skipped batches are counted and bounded.

### 8.8 Budget

```
GPU-hours = unique audio-hours × passes ÷ measured audio-hours per GPU-hour
```

A FLOP count is a floor on cost. Until P11 measures the rate, plan at 300 to 600 `[EST]`:

| Job | Audio-hours consumed | GPU-hours at 600 / 300 |
|---|---:|---:|
| Public-recipe control, 2,500 h × 20 passes | 50,000 | 85 / 170 |
| R1 dense, 25,000 h × 8 passes | 200,000 | 330 / 670 |
| Oracle, 25,000 h × 6 | 150,000 | 250 / 500 |
| R2 Tower, 25,000 h × 3 | 75,000 | 125 / 250 |
| R3 event clock, 25,000 h × 4 | 100,000 | 170 / 330 |
| R4 quantisation, 25,000 h × 3 | 75,000 | 125 / 250 |
| R5 scale, 300,000 h × 3 (if unique hours allow) | 900,000 | 1,500 / 3,000 |
| R5 re-quantise and heal, 300,000 h × 1 | 300,000 | 500 / 1,000 |
| M and S distillation, 300,000 h × 2 each | 1,200,000 | 2,000 / 4,000 |
| Waves, seeds, failed runs (planning share 35%) | — | 1,750 / 3,500 |
| **Total** | | **≈ 6,900 / 13,700** |

Fleet: pseudo-labels over 275,000 gross hours with two teachers at about 15× real time per core is about 37,000 core-hours `[EST]`; statistics, scoring, search and parity add about 15,000.

**Envelopes.** With a workstation only: R0, the harness, the cost law, the runtime, the bank, policy search. With one 24 GB GPU: T0–T5 on D2 and Loquacious *medium*; every rung proof at small scale. With several GPUs and the fleet: everything.

**Rules.** A stage that reaches 120% of its line without passing its gate stops and reports. Before any scale-up: a measured pilot, a full estimate with 30% contingency, the unique-hours manifest, and a learning curve that shows why more data is worth it.

---

## 9. CPU fleet

Every job follows the repository's contract: units claimed by atomic `mkdir`; largest-first buckets with tail stealing; memory capped by cgroup; pinned cores; at most four numeric threads per worker, recorded by the worker; create-only uploads with verified read-back; a manifest of commit, commands, inputs and hashes `[REPO: h456_ultra_entropy_recipe/steal_distributed_tails_cpu.py, PREREG_TEMPLATE.md]`.

| Job | Unit | Output |
|---|---|---|
| J1 Ingest and identity graph | one archive | verified shards; interval records |
| J2 Decontaminate | one shard × boards | exclusion flags; audit trail |
| J3 Pseudo-label | 10 min of audio × teacher | transcripts, confidences |
| J4 Label quality | one shard | estimator features, tier, canary result |
| J5 Align | one shard through the Oracle | piece-to-event alignments; lattice targets |
| J6 Stage | one microbatch | augmented, bucketed audio |
| J7 Integer statistics | one shard through the integer model | Gram and fixed-grid cross-Gram matrices; per-piece class sums; Reflex confusion tables |
| J8 Second-order rounding | one row block | ternary codes, plane-2 codes, multipliers |
| J9 Valley scoring and soups | (checkpoint, board shard) | per-speaker error counts |
| J10 Gene search | (gene set, board shard) | WER, cost, delay per cell |
| J11 Hard-example mining | one shard through the student | per-utterance regret |
| J12 Text-event preparation | one text shard | dual tokenisation, sampled events |
| J13 Kernel bake-off | (kernel, CPU model) | throughput, counters, parity |
| J14 Baseline census | (recogniser, board shard, grid) at real-time pacing | CPU cost, delays, WER |
| J15 Parity | parity corpus on one target | per-utterance hashes |

Scoring never touches a trainer. What search evaluates is the emitted integer runtime, not a simulation.

---

## 10. Harness, runtime and packaging

### 10.1 Workspace

Names are proposals; map them to the real tree on day 1.

```
crates/ultra-core/       events, config, bundle validation, formats, requantisation, tables, arena
crates/ultra-kernels/    scalar, avx512_vnni, avx2, neon_dot, wasm_simd; property tests
crates/ultra-frontend/   Cochlea; float log-mel control
crates/ultra-model/      Reflex, gate, Cortex, Tower, transducer, anchor; bundle loader
crates/ultra-stream/     clocks, scheduler, shedding, multi-stream batcher, lifecycle
crates/ultra-gp/         program parser, checker, canonical print, cost, lowering, emitters
crates/ultra-eval/       replay, CPU and memory accounting, normaliser, scoring, statistics, ledgers
crates/ultra-bench/      cost law, kernel bake-off, timing protocol
crates/ultra-adapters/   one process-isolated adapter per external recogniser
crates/ultra-cli/        inventory | transcribe | stream | bench | score | compare | bundle verify | doctor
research/ultra/          trainer (the Python exception), bank, one folder per wave
configs/ultra/           programs; candidate manifests; profiles
docs/ultra/              STATUS, DECISIONS, FACTS, RULINGS, boards, ledgers, model cards
```

### 10.2 External recognisers (rung R0)

Each candidate is pinned completely: model repository and immutable revision, artifact and tokenizer hashes, runtime commit, compiler and flags, quantisation, chunk and context settings as the runtime reports them, thread counts, affinity, licence ids. A run refuses unresolved fields. First wave: Nemotron streaming through its CPU runtime at 160 and 560 ms; Parakeet Unified at 320 and 560 ms; Moonshine v2; one exact streaming Zipformer checkpoint; FUTO ASR4ALL; one low-bit model (Parakeet Redux or Phonon); Granite TurboCTC for final text. The catalogue is frozen per comparison round.

Two views are reported: a controlled view (common audio, common normaliser, comparable context, no personalisation) and a product view (each runtime's best supported end-to-end path under the same deadlines). The harness owns pacing, clocks, references and scoring. Adapters receive audio and any allowed phrase list, never references or filenames that contain text.

### 10.3 Bundle

`models/sha256-<manifest-digest>/` `[REPO: docs/model-objects.md]`: `manifest.json` (program hash, training commit, data manifest hash, formats, required instruction set, both WERs for plane 2 on and off, tested hardware, limits); `program.dsl`; `weights.t2` in pass order; `weights.i8`; `tables.bin`; two vocabularies; `parity.json`; `licences/`; `provenance.json`; `card.md`; `SHA256SUMS`. The loader refuses: a hash mismatch, a truncated payload, an unsupported schema or instruction set, a table that does not regenerate from the program, a missing overflow proof, an unknown special token, or a path that escapes the bundle. Hashes are verified at load, not per chunk.

### 10.4 Interface

```rust
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> BackendCaps;
    fn create_stream(&self, config: &StreamConfig) -> Result<Box<dyn AsrStream>, AsrError>;
}
pub trait AsrStream: Send {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, AsrError>;
    fn poll_event(&mut self) -> Result<Option<TranscriptEvent>, AsrError>;
    fn finish_input(&mut self) -> Result<(), AsrError>;
    fn cancel(&mut self) -> Result<(), AsrError>;
    fn set_phrases(&mut self, phrases: &[&str]) -> Result<(), AsrError>;
    fn confirm_text(&mut self, segment: u64, text: &str) -> Result<(), AsrError>;   // anchor labels
    fn receipt(&self) -> Receipt;
}
```

This is a contract sketch (from CADENCE-CPU v2), not compiled code. One owner per stream; weights are shared read-only. `push_pcm` reports how many samples it accepted. Across the C boundary: opaque handles, versioned structs, no unwinding, stable error codes. `ULTRA_CPU_THREADS` defaults to 1 and is capped at 4, like `JOPE_CPU_THREADS` `[REPO: crates/feat-math/src/emit_rs.rs]`.

### 10.5 The loop

```
per chunk of the grid:
    frames   = cochlea.push(pcm)
    post, h  = reflex.step(frames)
    events  += gate(post, h, θ)                       # pooled frames ride with the next event
    if no event in this chunk: continue               # no Cortex wake-up
    c        = cortex.pass(events of this chunk)      # one fetch of the weights
    u        = tower.step(c, D)                       # outputs for events whose right context is complete
    emit Commit( transducer.greedy(commit_tap, u) )           # COMMIT mode only
    emit Preview( decode(preview_tap, c[uncommitted], from = commit state) )
    if endpoint or cap:
        f = tower.final(segment)                              # unless shed or the controller says no
        emit Final( transducer.greedy(final_tap, f, from = segment start or commit cursor) )
        careful_next = careful_rule(last 2 s)
```

---

## 11. Evaluation and governance

### 11.1 Boards

Partitions as in §7.3. The public board is the eight English test sets (AMI, Earnings-22, GigaSpeech, LibriSpeech clean and other, SPGISpeech, TED-LIUM, VoxPopuli) with dataset versions, channels, segmentation and normaliser recorded; if one cannot be lawfully used, the smaller suite is named as such and never reported as eight. The domain board is newly recorded speech: a 2–5 hour pilot for development, then 10–20 hours across sessions, microphones and speakers for the sealed set (ruling O5), covering quiet dictation, conversation, technical terms and names, numbers and dates, low volume and far field, noise, fast speech and repeated words, long uninterrupted speech, and silence and music.

Slices at every gate: names, numbers, technical terms, noise, long form, fast speech, repeated words.

### 11.2 Labels on every number

*Fair*: no data from the benchmark's corpus in training. *In-corpus*: its training split was used. *Recall*: the evaluated speaker's audio or profile was used. Plus the partition (DEV, CONFIRM, PUBLIC-TEST, DOMAIN-SEALED) and the output mode. For pretrained external models, upstream overlap with public tests is unknown and is stated as unknown.

### 11.3 Metrics

| Group | Metrics |
|---|---|
| Accuracy | WER of preview snapshots, the commit ledger and final text; substitutions, deletions, insertions; per set; slices; per-speaker win, tie, loss; harm rate |
| Silence | inserted words per hour and false-transcript episodes per hour (WER is undefined with zero reference words) |
| Delay | first correct preview word; stable-word delay p50 and p95; commit delay; finalisation delay; coverage (how many reference words were eligible for timing) |
| Stability | preview flicker per 100 words; commit revisions (must be 0); share of committed words a `CorrectionSuggested` would change |
| Cost | process-tree CPU-seconds per second; reserved threads; p50, p95, p99 compute per chunk; queue wait; maximum backlog; events and wake-ups per second |
| Memory | peak RSS and PSS; allocator high-water mark; bundle bytes; state per stream |
| Formatting | cased and punctuated accuracy; numeric and entity correctness under a conservative normaliser |
| Events | retention; share of aligned pieces with an event within one frame; careful-mode triggers; cap hits |

"Stable time" for editable text is retrospective: the first appearance of a word after which it never changed. For COMMIT mode it is the actual `Commit` event. Reference timings come from human-checked alignments on the latency panel.

### 11.4 Workloads

Unpaced single file (throughput); paced single stream (delay and backlog, 10 or 20 ms packets); paced concurrent streams (raise N until the profile fails); cold load; a warm stream of at least one hour with speech, silence and speaker changes; coexistence with a fixed background load. Paced elapsed time is the audio duration by construction and is never inverted into a throughput.

### 11.5 Machines and timing

| Machine | Why |
|---|---|
| H0 | the owner's actual deployment CPU; the primary selection machine |
| A: Intel i9-12900K, one performance core | FUTO's published reference `[SRC 5]`; the cost rows of §1.4 |
| B: Intel i7-11700K | the incumbent's published reference `[SRC 2]` |
| C: an Arm laptop or phone-class core; D: a small Arm board | NEON path; sustained thermal behaviour |
| E: a Zen 5 server | Redux's reference `[SRC 7]`; multi-stream batching |
| A in a browser | WebAssembly path |

A machine that is not available is marked `UNTESTED`; nothing is extrapolated to it. Protocol: pinned physical cores (SMT tested separately); governor, power source and thermal state recorded; fixed warm-up; at least five repetitions for finalists; candidate and control in alternating blocks; a qualification lock, with the host qualified **before** looking at who won; no two timing runs on one machine at once; hardware counters for cache misses and memory bandwidth in W00 and W01. A portable build is not compiled with host-native flags.

### 11.6 Promotion rules and statistics

Fixed before confirmation:

- **Speed without loss:** CPU cost down by at least 10%; one-sided 95% upper bound on WER change ≤ +0.10 points; slices within their margins; deadlines met.
- **Accuracy:** at least 3% relative lower WER with a paired interval excluding zero; CPU cost up by no more than 10% unless the accurate profile is being selected.
- **Compression research result:** at least 15% lower full-pipeline CPU at the same margin. A smaller file alone passes nothing.

Statistics: paired bootstrap over speaker or recording clusters, at least 5,000 replicates, within dataset strata with frozen weights for the macro average; sign-flip across corpora for the macro claim; Holm correction (or the bound at 1 − 0.05/m for m frozen comparisons) across a wave's confirmatory claims; three seeds for training comparisons where affordable, reported separately from evaluation intervals. Both absolute points and relative change are reported. When the data cannot resolve the margin the verdict is `INCONCLUSIVE`.

Two different A/A questions: two identical inference runs test the instrument; two trainings with independent seeds test training noise (W03). A fresh-seed null arm rides in every training wave and is **reported**; if it "wins", uncertainty is reassessed, and the wave is not selectively voided. Best-of-N winners are re-run fresh before they count (H451 → H452 `[REPO]`).

### 11.7 Correctness tests

| Family | Cases | Catches |
|---|---|---|
| Transport | 10, 20, 37 ms packets; random boundaries; a short last packet | packet-dependent text; accidental cache reset |
| Causality | change audio after the declared horizon; earlier eligible output must not change | hidden lookahead; a chunk mask that leaks |
| Repetition | "go go"; repeated letters and pieces; blanks at boundaries | CTC collapse; decoder state errors |
| Decoder | long bursts; cap hits; flush with pending pieces | loops; dropped tokens |
| Ledger | preview replacement; commit append; final once; late suggestion | mutation or double ownership of words |
| Discontinuity | missing packets; restart; cancel during final mode | stale output in a new segment |
| Long sessions | one hour; no pauses; long silence | growing buffers; counter overflow; backlog |
| Modes | every dial, grid, D, plane setting; every trained switch point | an untrained mode reachable at run time |
| Kernels | extreme operands; zero point; negative rounding; padding; every packed code | saturation; overflow; packing errors |
| Loader | wrong hash, shape, schema, instruction set, tokenizer | corrupt or misidentified bundles |
| Scoring | known edits; zero-word references; macro weights; poison canary; shuffled ids | wrong numbers; label leakage |
| Accounting | child workers; final pass; cancellation; background load | hidden CPU or memory |
| Phrases | supplied names that are spoken; plausible distractors that are not | bias-induced insertions |

### 11.8 Release tests for any rung

Identities frozen; CPU-only verified; the profile's delay, memory and backlog targets pass; the emitted text meets the promotion rule; slices and silence behaviour reported and within limits; §11.7 passes; the bundle loads on H0 and each claimed target; rights resolved for the intended distribution; a second run reproduces the report; a rollback bundle and a model card with known limits exist. No "best", "leader", "10×" or "no-harm" appears in any report unless it names hardware, workload, quality and delay.

---

## 12. Waves

One folder per wave with a committed `PREREG.md` (Appendix C), a run script, an aggregate script, a red-team script and a `README.md` whose first line is the verdict. Run only the next wave needed.

| Wave | Question | Arms | Passes if | Otherwise |
|---|---|---|---|---|
| W00 | Is the incumbents' fixed cost memory traffic? | two incumbents; chunk 2..14 frames; 1..8 threads; counters | fixed share ≥ 70% at 160 ms; bandwidth within 20% of the machine's limit | add a term to the cost model before building on it |
| W01 | Does the cost law hold on H0 and A–E? Which kernel per target? | Appendix A; interleaved Reflex and Cortex passes; bake-off candidates | R² ≥ 0.98; ternary fixed term ≤ 0.4× int8 out of cache; estimator within ±30% of a real pass | choose kernels by measurement; revise §4.12 |
| W02 | Is dense CTC feasible; how few events keep the pieces? | T ≥ U + R by speaking-rate decile; gate replay as a screen | infeasible ≤ 0.5% in the fastest decile; replay reaches ≤ 60% retention with ≥ 99.5% of aligned pieces within one frame | two-position head; richer gate features |
| W03 | What can the board resolve? | control trained twice, independent seeds | 2% relative at 95% | enlarge the board |
| W10 | Is R1 sound? | R1 against a public streaming recipe reproduced on the same data; both against R0's reference in the harness | within 5% relative of the reproduced recipe; gap to R0 recorded | fix basics |
| W10b | Does feature distillation from an 80 ms-frame incumbent help? | with and without, matched steps | accuracy rule | off |
| W11 | Does the Tower pay? Shared or dedicated final weights? | m ∈ {0, 2, 4} × D ∈ {0, 1, 2, 3}; final mode; dedicated final stack | commit at D = 2 ≥ 5% relative better than D = 0; final ≥ 5% better than commit; smallest m that passes | single tap with fixed lookahead |
| W12 | **K-EVENT** | same weights, retention sweep; separately trained dense and gated pair | §1.5 | §1.5 |
| W13 | Integer Cochlea | float log-mel; table form; integer square-root form | WER within +0.10; hashes equal on all targets | keep float log-mel until R4, then the better integer form |
| W14 | Vocabulary | 1,024 / 4,096 / 16,384 | lowest macro; ties by rare-word slice, then size | — |
| W14b | Self-conditioning on events | with (masked when infeasible) and without | accuracy rule; masked ≤ 1% | off |
| W15 | **K-TERN** | float; int8; T1; T1 + plane 2 | §1.5 | §1.5 |
| W16 | Parity | four targets; threads; batches; simulation against runtime | all hashes equal | release blocked |
| W17 | 8-bit inputs to ternary layers | 7-bit against 8-bit | ≥ 0.05 points better | 7-bit everywhere |
| W17b | Fleet rounding as initialisation | second-order rounding from 5,000 h of statistics against none | same WER in ≤ 70% of the steps, or ≥ 0.05 better | plain fine-tune |
| W18a | Text events | share 0 / 0.10 / 0.25, with twin consistency | rare-word slice ≥ 5% relative better; macro within margin | share returns to human audio |
| W18 | Preview teacher | on and off, long sessions | flicker down ≥ 20% relative after 60 s; preview WER not worse; committed hashes **identical** | off |
| W19 | Refinement controller; cascade | never, always, random, learned | §4.10 | fixed policy |
| W20 | Does more data keep paying? Which mix? | doublings of unique hours; √hours, uniform, natural | each doubling ≥ 3% relative; mix by accuracy rule with no set worse by > 3% | stop buying data |
| W20b | Soups | soup against best single valley checkpoint | ≥ 0.05 better, fresh rerun | skip |
| W21 | Shedding and in-utterance switches | each step and each trained switch on the latency panel | cost of each step monotone and published; switches within +0.05 of a segment-boundary switch | switch at boundaries only |
| W22 | Anchor adaptation | off; phrases; A1; A1 + A2; A2 without the guard | on later sessions: slice WER ≥ 5% relative better; no user worse by > 1% relative; η = 0 path bit-identical | phrases and A1 only |
| W22b | Multi-stream batching | 1, 4, 16 streams per core | identical hashes; ≥ 70% of predicted streams | single-stream |
| W22c | Specialists | routed against none | ≥ 2% relative under the coverage rule; no cluster worse than identity | omit |
| W23 | Capacity or data? | Oracle with all data; wider dense; mixture of experts at equal bytes | accuracy rule at ≤ 0.30 CPU-s/s | stay at L; spend on data |
| W21b | Policy search | searched rules against hand-set and random search; fresh confirmation | cost −10% at equal WER, or WER −1% relative at equal cost, per unit of search compute | hand-set rules |
| W24 | Sealed looks; **K-ACC** | frozen finalists against frozen references | §1.4 | §1.5 |

---

## 13. Work packages

**Depends → deliver → accept when.** Effort in focused agent-days is a guess `[EST]`.

| ID | Package | Depends | Accept when | Days |
|---|---|---|---|---:|
| P00 | Repository, hardware, rulings | — | build and tests green on a clean clone; `FACTS.md` closes §2.6; `HARDWARE.json`; rulings requested | 1 |
| P01 | Scorer and fixtures | — | known edit counts, macro weights, zero-word handling, normaliser pinned; identical systems score identically | 3 |
| P02 | Events, lifecycle, receipts | P00 | every invariant of §3.1 and §3.5 has a passing test | 3 |
| P03 | Program parser, printer, checker, cost estimator | P00 | three programs re-print byte-identically; all expressions parse under the real parser with asserted sizes; one-argument `step` refused; estimator reproduces Appendix B to the weight | 5 |
| P04 | Paced replay; CPU and memory accounting | P02 | packet sizes do not change the sample timeline; process-tree CPU counted; idle wait excluded | 3 |
| P05 | Adapters for external recognisers | P02 | three candidates pinned and running CPU-only with receipts | 6 |
| P06 | Cost law and bake-off (W00, W01) | P00 | verdicts recorded; cost-law file per machine | 5 |
| P07 | Identity graph, bank, deck, partitions, vault | P00 | planted sealed shard refused; unique hours reported; zero overlap with boards; realised mixture logged | 6 |
| P08 | Baseline tournament; package **R0** | P01, P04, P05, P07 | frontier report on H0; §11.8 passes for R0 | 4 |
| P09 | Scalar integer runtime and loader | P03 | random weights: reference interpreter and Rust agree on 1,000 utterances; tampered bundle refused; checkpoint and resume bit-exact | 12 |
| P10 | AVX-512 kernels | P09, P06 | bit-exact with scalar; streaming benchmark improves | 5 |
| P11 | Trainer; throughput pilot; noise floor (W03) | O1, P07 | loss falls on train-clean-100; audio-hours per GPU-hour logged; budget re-filed | 6 |
| P12 | Reflex (T1, W02); **R1** (T2, W10) | P11 | gates of §8.1 | — |
| P13 | Streaming runtime: scheduler, taps, shedding, batching | P09, P02 | §11.7 passes on R1 weights; measured overhead over itemised kernel time replaces the 1.5× allowance | 12 |
| P14 | Oracle; alignments; teacher pilot (T3) | P12 | alignments cover ≥ 99% of pieces; pilot report with estimator and audit | — |
| P15 | **R2** Tower (T4, W11) | P12, P14 | verdict; chosen m | — |
| P16 | **R3** event clock (T5, W12) | P15, P13 | K-EVENT verdict with receipts; chosen retention | — |
| P17 | Cochlea (W13) | P03 | parity on four targets; table closure; verdict | 5 |
| P18 | **R4** quantisation (T7; W15, W16, W17, W17b) | P16, P10 | K-TERN verdict; all parity hashes equal | — |
| P19 | AVX2, NEON, WebAssembly kernels | P10 | bit-exact; admitted by rule 11 | 10 |
| P20 | Text events; vocabulary (T6) | P16 | verdicts | — |
| P21 | Pseudo-label factory; **R5** (T8; W20, W23) | P14, P18 | tiered pool with unique hours; verdict per doubling | — |
| P22 | Anchor adaptation, phrases, preview teacher (W22, W18) | P18, P13 | verdicts; η = 0 bit-identical | 8 |
| P23 | Shedding, batching, policy search (W21, W22b, W21b) | P18 | verdicts | 8 |
| P24 | M and S (T9); release (W24) | P21 | gates evaluated once each; cards; owner sign-off; a second agent reproduces from receipts | — |

Packages without a day count are dominated by training time, set by §8.8.

---

## 14. Milestones

Weeks assume the compute of §8.8 without queueing and are ordering, not promises.

| Milestone | Week | Contains | Decision |
|---|---:|---|---|
| M0 | 2 | P00–P08 | **R0 ships.** Does the cost law hold on real desktops? |
| M1 | 6 | P09–P11, T1 | Is dense CTC feasible? How low can retention go on replay? What does training cost? |
| M2 | 12 | T2, P13 | **R1**: an own conventional recogniser |
| M3 | 17 | T3, T4 | **R2**: does right context pay? |
| M4 | 22 | T5 | **K-EVENT** |
| M5 | 30 | T6, T7, P19 | **K-TERN**; parity on four targets |
| M6 | 38 | T8 | scale; capacity or data |
| M7 | 44 | T9, T10, W24 | M and S; adaptation; sealed looks |

Critical path: P00 → P07 → P11 → T1 → T2 → T4 → T5 → T7 → T8 → W24. The runtime track (P03, P09, P10, P13, P19) and the data track (P07, P14, P21) run beside it. KAIROS v0.2 estimated six to eight months with eight GPUs and a thousand cores for a similar scope; this is ten, because R0 and R1 come first and every rung is measured.

---

## 15. Risks

| Risk | Early sign | Response |
|---|---|---|
| The gate loses short words | deletions rise on fast speech in W12 | higher retention; shorter maximum gap; frame clock |
| Ternary will not hold at 600M weights | W15 misses by > 0.5 | the ladder; two planes still move half the bytes of int8 |
| The cost law is different on desktops | W01 fixed term far from §2.1 | re-derive §4.12; smaller Cortex; 560 ms grid; two threads |
| Unique hours are far below gross | identity graph report | spend on targeted labelled data; feature distillation |
| Pseudo-labels teach teacher errors | estimator fails the audit | tighten tiers; more human labels |
| Tower with restricted transducer on events is unstable | loss spikes at mode changes | train with the dial fixed first; dedicated final weights |
| Text events do not match real events | W18a hurts | raise twin consistency; else remove |
| Training costs several times the plan | P11 below 300 audio-hours per GPU-hour | fewer passes; M as the main model; owner re-budgets |
| The allowance is wrong | P13 | publish the measured factor; retune targets |
| Anchor adaptation harms | W22 harm cap exceeded | phrases and A1 only |
| WebAssembly relaxed SIMD missing | feature detection | exact scalar fallback, slower, same bits |
| A licence blocks a corpus or teacher | ledger review | drop it; report the hours lost |
| Scope | P09 or P13 slip | scalar and AVX-512 first; side waves dropped first |
| Plan drift | acceptance tests weakened | tests change only with the owner |

---

## 16. Owner rulings

| ID | Question | Blocks | Default if unanswered |
|---|---|---|---|
| O1 | Where may the trainer live under ADR-0006? Proposed: `research/ultra/` with an exception entry | all training | wait; runtime and harness continue |
| O2 | Which corpora and teachers are acceptable for the intended use? (The training-set list in the original request was empty.) | D3 onward | LibriSpeech only |
| O3 | GPU-hour and core-hour ceilings | every wave budget | D2 scale only |
| O4 | Which machine is H0? Access to A–E? | W01, cost gates | sandbox numbers, labelled |
| O5 | Who records the domain sets, and how many hours? | W24 | public sets only |
| O6 | Are the gates of §1.4 and margins of §11.6 accepted? | W24 | as written |
| O7 | Identifier allocation for audio operators | registry merge | unregistered dialect table |
| O8 | Which mode is the product: DICTATE, COMMIT or both? Is cased, punctuated text required at first release? | priority of W11, formatting work | both researched |
| O9 | May confirmed corrections and adapter state be stored? Where, how long? | W22 | on device, opt-in, erasable |
| O10 | Publication policy; licences of external runtimes used in the tournament | release | internal only |

---

## Appendix A. The cost law: benchmark, data and fit

**What was run.** `wstream.c` multiplies a matrix of N weights (rows of 1,024) by F input vectors per pass, with int8 weights (`vpdpbusd`) and with 2-bit packed ternary weights (unpack, then the same instruction). Inputs are 7-bit. Both kernels must return the same checksum. Build: `gcc -O3 -march=native -o wstream wstream.c`. Host: cloud Xeon at 2.1 GHz, 2 virtual CPUs, 4 MiB L2 per core, AVX-512 VNNI and VBMI. One thread.

**History.** Three runs with F ∈ {1, 2, 16} for KAIROS v0.1; one run with F ∈ {1, 2, 3, 4, 6, 8, 12, 16} for TACTUS (0.4 s minimum per point, best of three repetitions), reproduced below. KAIROS v0.2 reports an independent re-run of the three-point version on a similar host with matching checksums `[DOC]`.

**Fit.** `ps per weight per pass = a + c·F`, least squares over the eight values of F:

```
kernel,size,intercept_ps_per_weight,slope_ps_per_MAC,asymptote_GMAC_s,R2,max_rel_err
int8_vnni,1M,5.36,10.62,94.2,0.9955,0.071
int8_vnni,64M,30.43,16.40,61.0,0.9353,0.318
int8_vnni,1536M,96.38,10.50,95.3,0.9667,0.169
tern2_vnni,1M,14.14,10.90,91.7,0.9951,0.094
tern2_vnni,64M,16.46,10.43,95.8,0.9979,0.055
tern2_vnni,1536M,19.36,10.98,91.1,0.9993,0.050
incumbent: marginal 1.308 ms/frame, fixed 24.38 ms, fixed share 0.903, stream 25.3 GB/s, fixed 39.4 ps/weight (8 thr), marginal 2.12 ps/MAC (8 thr)
```

The 64M int8 row is noisy (67 MB against a shared last-level cache) and is not used.

**Raw output.**

```
kernel,weights,frames,weight_bytes,ms_per_pass,GMAC_per_s,weight_GB_per_s,checksum
int8_vnni,1M,1,1048576,0.016,65.2,65.22,-2382
tern2_vnni,1M,1,262144,0.024,43.1,10.78,-2382
int8_vnni,1M,2,1048576,0.027,76.7,38.36,-17142
tern2_vnni,1M,2,262144,0.037,56.6,7.07,-17142
int8_vnni,1M,3,1048576,0.040,78.9,26.29,36833
tern2_vnni,1M,3,262144,0.053,59.1,4.92,36833
int8_vnni,1M,4,1048576,0.054,78.3,19.58,21659
tern2_vnni,1M,4,262144,0.064,65.1,4.07,21659
int8_vnni,1M,6,1048576,0.074,85.1,14.18,-8181
tern2_vnni,1M,6,262144,0.086,72.9,3.04,-8181
int8_vnni,1M,8,1048576,0.092,91.4,11.42,-76071
tern2_vnni,1M,8,262144,0.098,85.6,2.67,-76071
int8_vnni,1M,12,1048576,0.132,95.6,7.96,-169674
tern2_vnni,1M,12,262144,0.150,83.7,1.74,-169674
int8_vnni,1M,16,1048576,0.189,88.6,5.54,-242266
tern2_vnni,1M,16,262144,0.201,83.4,1.30,-242266
int8_vnni,64M,1,67108864,2.884,23.3,23.27,-591026
tern2_vnni,64M,1,16777216,1.711,39.2,9.80,-591026
int8_vnni,64M,2,67108864,3.639,36.9,18.44,-1222025
tern2_vnni,64M,2,16777216,2.469,54.4,6.80,-1222025
int8_vnni,64M,3,67108864,4.055,49.6,16.55,-1656515
tern2_vnni,64M,3,16777216,3.242,62.1,5.17,-1656515
int8_vnni,64M,4,67108864,5.344,50.2,12.56,-2293234
tern2_vnni,64M,4,16777216,3.996,67.2,4.20,-2293234
int8_vnni,64M,6,67108864,11.437,35.2,5.87,-3868138
tern2_vnni,64M,6,16777216,5.463,73.7,3.07,-3868138
int8_vnni,64M,8,67108864,12.598,42.6,5.33,-4661303
tern2_vnni,64M,8,16777216,6.728,79.8,2.49,-4661303
int8_vnni,64M,12,67108864,15.324,52.6,4.38,-6520161
tern2_vnni,64M,12,16777216,9.157,87.9,1.83,-6520161
int8_vnni,64M,16,67108864,18.273,58.8,3.67,-9251333
tern2_vnni,64M,16,16777216,12.482,86.0,1.34,-9251333
int8_vnni,1536M,1,1610612736,147.313,10.9,10.93,-1644903
tern2_vnni,1536M,1,402653184,46.527,34.6,8.65,-1644903
int8_vnni,1536M,2,1610612736,177.547,18.1,9.07,-2181117
tern2_vnni,1536M,2,402653184,65.645,49.1,6.13,-2181117
int8_vnni,1536M,3,1610612736,215.602,22.4,7.47,-6017931
tern2_vnni,1536M,3,402653184,87.957,54.9,4.58,-6017931
int8_vnni,1536M,4,1610612736,228.424,28.2,7.05,-5822672
tern2_vnni,1536M,4,402653184,104.478,61.7,3.85,-5822672
int8_vnni,1536M,6,1610612736,281.672,34.3,5.72,-12027474
tern2_vnni,1536M,6,402653184,134.738,71.7,2.99,-12027474
int8_vnni,1536M,8,1610612736,304.830,42.3,5.28,-19606366
tern2_vnni,1536M,8,402653184,170.887,75.4,2.36,-19606366
int8_vnni,1536M,12,1610612736,352.124,54.9,4.57,-31876366
tern2_vnni,1536M,12,402653184,245.636,78.7,1.64,-31876366
int8_vnni,1536M,16,1610612736,413.402,62.3,3.90,-43717451
tern2_vnni,1536M,16,402653184,313.304,82.3,1.29,-43717451
```

**Source.**

```c
// Weight-stream micro-benchmark: how fast can one core push weights through a
// GEMV when the weight image is int8 (1 byte/weight) versus packed ternary
// (2 bits/weight), as a function of the weight-image footprint and of how many
// frames share one pass over the weights.
//
// Build: gcc -O3 -march=native -o wstream wstream.c
// (needs AVX-512F/BW/VBMI/VNNI)
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIN 1024 // activation width (bytes per frame)

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static inline uint64_t xorshift(void) {
  uint64_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return rng_state = x;
}

// int8 weights (1 byte/weight) and packed ternary (2 bits/weight); u8 activations; F frames per pass.
#define U 4 // independent accumulators per frame
static int64_t gemv_i8(const int8_t *w, size_t rows, const uint8_t *act, int F) {
  int64_t checksum = 0;
  for (size_t r = 0; r < rows; ++r) {
    const int8_t *wr = w + r * DIN;
    __m512i acc[16][U];
    for (int f = 0; f < F; ++f)
      for (int u = 0; u < U; ++u) acc[f][u] = _mm512_setzero_si512();
    for (int j = 0; j < DIN; j += 64 * U) {
      __m512i wv[U];
      for (int u = 0; u < U; ++u) wv[u] = _mm512_loadu_si512((const void *)(wr + j + 64 * u));
      for (int f = 0; f < F; ++f) {
        const uint8_t *a = act + (size_t)f * DIN + j;
        for (int u = 0; u < U; ++u)
          acc[f][u] = _mm512_dpbusd_epi32(acc[f][u], _mm512_loadu_si512((const void *)(a + 64 * u)), wv[u]);
      }
    }
    for (int f = 0; f < F; ++f) {
      __m512i s = _mm512_add_epi32(_mm512_add_epi32(acc[f][0], acc[f][1]), _mm512_add_epi32(acc[f][2], acc[f][3]));
      checksum += _mm512_reduce_add_epi32(s);
    }
  }
  return checksum;
}

static int64_t gemv_t2(const uint8_t *p, size_t rows, const uint8_t *act, int F) {
  const __m512i ctrl = _mm512_set1_epi64(0x0E0C0A0806040200ull);
  const __m512i three = _mm512_set1_epi8(3);
  const __m512i one = _mm512_set1_epi8(1);
  int64_t checksum = 0;
  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *pr = p + r * (DIN / 4);
    __m512i acc[16][U];
    for (int f = 0; f < F; ++f)
      for (int u = 0; u < U; ++u) acc[f][u] = _mm512_setzero_si512();
    for (int j = 0; j < DIN; j += 64 * U) {
      __m512i wv[U];
      for (int u = 0; u < U; ++u) {
        __m128i pk = _mm_loadu_si128((const void *)(pr + (j + 64 * u) / 4));
        __m512i q = _mm512_cvtepu16_epi64(pk);
        __m512i b = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, q), three);
        wv[u] = _mm512_sub_epi8(b, one);
      }
      for (int f = 0; f < F; ++f) {
        const uint8_t *a = act + (size_t)f * DIN + j;
        for (int u = 0; u < U; ++u)
          acc[f][u] = _mm512_dpbusd_epi32(acc[f][u], _mm512_loadu_si512((const void *)(a + 64 * u)), wv[u]);
      }
    }
    for (int f = 0; f < F; ++f) {
      __m512i s = _mm512_add_epi32(_mm512_add_epi32(acc[f][0], acc[f][1]), _mm512_add_epi32(acc[f][2], acc[f][3]));
      checksum += _mm512_reduce_add_epi32(s);
    }
  }
  return checksum;
}

int main(int argc, char **argv) {
  // weight counts: small (L2-resident), medium, large (DRAM-resident on this host)
  size_t counts[3] = {(size_t)1 << 20, (size_t)64 << 20, (size_t)1536 << 20};
  const char *names[3] = {"1M", "64M", "1536M"};
  int frames[8] = {1, 2, 3, 4, 6, 8, 12, 16};
  double min_seconds = argc > 1 ? atof(argv[1]) : 1.0;

  size_t maxw = counts[2];
  int8_t *w8 = aligned_alloc(64, maxw);
  uint8_t *wp = aligned_alloc(64, maxw / 4);
  uint8_t *act = aligned_alloc(64, 16 * DIN);
  if (!w8 || !wp || !act) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  for (size_t i = 0; i < maxw / 4; ++i) {
    uint64_t x = xorshift();
    uint8_t byte = 0;
    for (int k = 0; k < 4; ++k) {
      uint8_t code = (uint8_t)((x >> (8 * k)) % 3);
      byte |= (uint8_t)(code << (2 * k));
      w8[4 * i + k] = (int8_t)code - 1;
    }
    wp[i] = byte;
  }
  for (size_t i = 0; i < 16 * DIN; ++i) act[i] = (uint8_t)(xorshift() & 0x7F);

  printf("kernel,weights,frames,weight_bytes,ms_per_pass,GMAC_per_s,weight_GB_per_s,checksum\n");
  for (int c = 0; c < 3; ++c) {
    size_t rows = counts[c] / DIN;
    for (int fi = 0; fi < 8; ++fi) {
      int F = frames[fi];
      for (int kernel = 0; kernel < 2; ++kernel) {
        double best = 1e30;
        int64_t cs = 0;
        for (int rep = 0; rep < 3; ++rep) {
          double t0 = now();
          int passes = 0;
          double elapsed;
          do {
            cs = kernel == 0 ? gemv_i8(w8, rows, act, F) : gemv_t2(wp, rows, act, F);
            ++passes;
            elapsed = now() - t0;
          } while (elapsed < min_seconds);
          double per = elapsed / passes;
          if (per < best) best = per;
        }
        double bytes = kernel == 0 ? (double)counts[c] : (double)counts[c] / 4.0;
        printf("%s,%s,%d,%.0f,%.3f,%.1f,%.2f,%lld\n", kernel == 0 ? "int8_vnni" : "tern2_vnni",
               names[c], F, bytes, best * 1e3, (double)counts[c] * F / best / 1e9,
               bytes / best / 1e9, (long long)cs);
        fflush(stdout);
      }
    }
  }
  return 0;
}
```

**W01 must add:** H0 and machines A–E; 1 to 8 threads; AVX2, NEON and WebAssembly kernels; interleaved Reflex and Cortex passes; a real layer mix; hardware counters.

---

## Appendix B. Sizing and cost model

`sizing3.py` produces every number in §4.12. It itemises ternary weights, key and value state read per wake-up, attention, tap projections and joiner calls. `sandbox` constants are mine; `host2` constants are read off KAIROS v0.2's table and are document-reported.

```python
# ULTRA sizing and cost model. SANDBOX constants: fit_law.py on bench/results_fsweep.csv (my host).
# HOST2 constants: read off the re-run table in KAIROS v0.2 Appendix B (document-reported, 3 points).
import sys
HOSTS = {"sandbox": dict(aT=19.4e-12, aTc=16.5e-12, c=11.0e-12, aI=96.4e-12),
         "host2":   dict(aT=30.0e-12, aTc=20.0e-12, c=6.5e-12,  aI=83.0e-12)}
P2, DERATE = 1.15, 1.5
EV, RHO160, RHO560, D = 5.6, 3.75, 1.6, 2          # events/s; Cortex wake-ups/s; commit delay in events [EST]
VR, VC, KV = 1025, 4097, 128
cfg = {"S": dict(ds=192, dr=256, nr=8,  d=512,  n=10, m=2, J=384),
       "M": dict(ds=256, dr=384, nr=12, d=1024, n=13, m=3, J=512),
       "L": dict(ds=256, dr=512, nr=12, d=1536, n=16, m=4, J=640)}
def model(c, h):
    ds, dr, nr, d, n, m, J = (c[x] for x in ("ds", "dr", "nr", "d", "n", "m", "J"))
    r_t = 8*dr*dr*nr + 4*dr*dr*(nr//4) + 16*ds*ds
    r_i = 16*nr*dr + 320*ds + 2*ds*dr + VR*dr
    c_t, t_t = 12*d*d*n, 12*d*d*m
    c_i = 2*dr*d + VR*d + 64*d + 5*d*(n+m) + 32*16*(n+m)          # event embed, hint emb, gap/style, conv, time-bias
    h_i = 3*d*J + 3*VC*J                                           # 3 tap projections; 2 predictor tables + output
    tern, i8 = r_t + c_t + t_t, r_i + c_i + h_i
    aT = h["aTc"] if (c_t + t_t)*0.25*P2 < 16e6 else h["aT"]
    state = (n+m)*KV*d*2 + 2*nr*16*dr + 256*d
    fixed = (c_t + t_t)*P2*aT + (n+m)*KV*d*2*h["aI"]                # weights + key/value state read per wake-up
    attn = 2*KV*d*(n+m)                                            # QK scores + AV mix per event
    calls = 2 + 2*(D+1)                                            # joiner calls: commit + preview re-decode
    dec = 2*d*J + calls*J*VC
    per_ev = ((c_t + t_t)*P2 + 2*dr*d + attn + dec)*h["c"]
    r_mac = 16*ds*ds*25 + (r_t - 16*ds*ds)*12.5 + r_i*12.5
    refl = lambda chunks: chunks*r_t*h["aT"] + r_mac*h["c"]
    rtf160 = refl(6.25) + RHO160*fixed + EV*per_ev
    rtf560 = refl(1.786) + RHO560*fixed + EV*per_ev
    dense = refl(6.25) + 6.25*fixed + 12.5*per_ev
    rchunk = r_t*h["aT"] + r_mac*0.16*h["c"]
    final = t_t*P2*(h["aT"] + 56*h["c"]) + 56*(2*KV*d*m + 2*J*VC + d*J)*h["c"]
    i8x = 6.25*r_t*(h["aI"] - h["aT"])
    streams = 1.0/(DERATE*(EV*per_ev + r_mac*h["c"]))
    return dict(reflex_M=(r_t+r_i)/1e6, cortex_M=(c_t+c_i)/1e6, tower_M=t_t/1e6, heads_M=h_i/1e6, total_M=(tern+i8)/1e6,
                ternary_M=tern/1e6, int8_M=i8/1e6, ram_MB=(tern*0.25*P2+i8)/1e6, download_MB=(tern*0.2*P2+i8)/1e6,
                state_MB=state/1e6, bytes_per_wake_MB=((c_t+t_t)*0.25*P2 + (n+m)*KV*d*2)/1e6, mac_per_event_M=per_ev/h["c"]/1e6,
                attn_share=attn*h["c"]/per_ev, decode_share=dec*h["c"]/per_ev,
                fixed_ms=fixed*1e3, per_event_ms=per_ev*1e3, reflex_rtf=refl(6.25), rtf160=rtf160, rtf560=rtf560,
                rtf_dense=dense, gate_saving=dense/rtf160, rtf160_allow=rtf160*DERATE,
                p50_ms=(rchunk+fixed+per_ev)*1e3, p99_ms=(rchunk+fixed+3*per_ev)*1e3, p99_allow_ms=(rchunk+fixed+3*per_ev)*DERATE*1e3,
                final_ms_10s=final*1e3, i8_reflex_extra=i8x, streams_per_core=streams)
for host, h in HOSTS.items():
    res = {k: model(c, h) for k, c in cfg.items()}
    print(f"# host={host}")
    for key in next(iter(res.values())):
        vals = [res[k][key] for k in cfg]
        fmt = "{:.4f}" if max(vals) < 1 else ("{:.2f}" if max(vals) < 100 else "{:.1f}")
        print(f"{key:20s}", *[fmt.format(v).rjust(10) for v in vals])
```

Output:

```
# host=sandbox
reflex_M                   5.76      17.72      30.33
cortex_M                   32.3      165.6      456.4
tower_M                     6.3       37.7      113.2
heads_M                    5.31       7.87      10.82
total_M                    49.7      228.9      610.8
ternary_M                  43.1      218.3      595.6
int8_M                     6.62      10.60      15.19
ram_MB                     19.0       73.4      186.4
download_MB                16.5       60.8      152.2
state_MB                   1.77       4.60       8.45
bytes_per_wake_MB          12.4       62.1      170.7
mac_per_event_M            58.2      254.3      683.5
attn_share               0.0270     0.0165     0.0115
decode_share             0.2229     0.0701     0.0336
fixed_ms                   0.87       4.90      13.39
per_event_ms               0.64       2.80       7.52
reflex_rtf               0.0015     0.0046     0.0079
rtf160                   0.0084     0.0387     0.1002
rtf560                   0.0060     0.0267     0.0689
rtf_dense                0.0149     0.0702     0.1856
gate_saving                1.79       1.82       1.85
rtf160_allow             0.0125     0.0580     0.1503
p50_ms                     1.75       8.44      22.17
p99_ms                     3.03      14.03      37.21
p99_allow_ms               4.55      21.05      55.81
final_ms_10s               6.82      30.98      87.55
i8_reflex_extra          0.0026     0.0082     0.0141
streams_per_core          149.5       36.5       14.4
# host=host2
reflex_M                   5.76      17.72      30.33
cortex_M                   32.3      165.6      456.4
tower_M                     6.3       37.7      113.2
heads_M                    5.31       7.87      10.82
total_M                    49.7      228.9      610.8
ternary_M                  43.1      218.3      595.6
int8_M                     6.62      10.60      15.19
ram_MB                     19.0       73.4      186.4
download_MB                16.5       60.8      152.2
state_MB                   1.77       4.60       8.45
bytes_per_wake_MB          12.4       62.1      170.7
mac_per_event_M            58.2      254.3      683.5
attn_share               0.0270     0.0165     0.0115
decode_share             0.2229     0.0701     0.0336
fixed_ms                   1.00       7.29      20.19
per_event_ms               0.38       1.65       4.44
reflex_rtf               0.0015     0.0047     0.0081
rtf160                   0.0074     0.0413     0.1086
rtf560                   0.0045     0.0234     0.0613
rtf_dense                0.0125     0.0710     0.1898
gate_saving                1.69       1.72       1.75
rtf160_allow             0.0111     0.0620     0.1630
p50_ms                     1.62       9.70      25.92
p99_ms                     2.38      13.01      34.81
p99_allow_ms               3.56      19.51      52.21
final_ms_10s               4.16      19.11      54.15
i8_reflex_extra          0.0018     0.0056     0.0097
streams_per_core          252.9       61.8       24.3
```

---

## Appendix C. Pre-registration, filled in for W12 (K-EVENT)

```
# W12 — K-EVENT: what does the event clock cost at equal weights, and what does it save?

Hypothesis      There is a retention ≤ 60% at which committed WER rises by at most 0.10 points
                macro and full-pipeline CPU falls by at least 15%.
Unit            speaker (recording where speakers are unknown)
Partitions      choose retention on DEV; confirm once on a CONFIRM partition, then retire it
Model           one R3 checkpoint, float, selected by DEV loss before any arm is scored
Arms            dial settings giving 100, 75, 60, 45, 35% retention, same weights
                matched pair: separately trained dense and gated models, same data and steps
                null: the 100% arm retrained with a fresh seed (reported, not used to void)
Primary         Δ commit-ledger WER at D = 2, 160 ms grid, arm − 100% arm, macro over sets
Secondary       per set; fast-speech and repeated-word slices; deletions; events/s; wake-ups/s;
                careful-mode triggers; process-tree CPU on machine A, alternating blocks, five repetitions
Pass            at the chosen retention: one-sided 95% bound on macro Δ ≤ +0.10; every set ≤ +0.30;
                fast-speech deletions ≤ +5% relative; CPU ratio ≥ 1.15
Numeric choices (≤ 4) maximum gap 240 ms; pooling weights; D = 2; grid = 2 frames
Revisions       two rule revisions, each a new experiment id
If pass         adopt the lowest passing retention as the default dial; proceed to T6
If fail         highest passing retention; if none, frame clock (R2 ships); the gate stays optional
Not allowed     PUBLIC-TEST or DOMAIN-SEALED; choosing the checkpoint by a gated arm's WER;
                reusing cached posteriors across gate settings
Ceilings        GPU-hours, core-hours and disk written here before the first job
```

---

## Appendix D. Schemas

```json
{ "schema": "ultra.transcript-event/v1", "stream_id": "…", "segment_id": 12, "event_sequence": 87,
  "revision_id": 5, "event_type": "Preview", "immutable_prefix_utf8_bytes": 143,
  "replacement_suffix": "example pending text",
  "audio_start_sample": 1840000, "audio_end_sample": 1936000, "observed_monotonic_ns": 123456789000,
  "mode": "dictate", "degraded": false }
```

```json
{ "schema": "ultra.receipt/v1", "bundle": "sha256-…", "program_hash": "…", "source_commit": "…",
  "hardware": "sha256-…", "target": "x86_64-avx512vnni", "configured_threads": 1, "observed_threads": 1,
  "cpu_only_verified": true, "audio_seconds": 600.0, "process_tree_cpu_seconds": null,
  "mode_schedule": [ { "t_ms": 0, "grid_frames": 2, "theta_q7": 0, "delay_events": 2, "plane2": true } ],
  "status_events": [], "gaps": [], "events": null, "wakeups": null, "cap_hits": null,
  "chunk_ms": { "p50": null, "p95": null, "p99": null },
  "anchor": { "table": "sha256-…", "eta": 0.0 },
  "hashes": { "events": "…", "preview": "…", "commit": "…", "final": "…" },
  "label": "fair", "partition": "DEV", "completion_status": "schema_example" }
```

A receipt whose `completion_status` is `schema_example`, or with a null where a measurement belongs, is refused as evidence.

```yaml
schema: ultra.candidate/v1            # one external recogniser, pinned
candidate_id: REQUIRED
model: { repository: REQUIRED, revision: REQUIRED, artifact_sha256: REQUIRED, tokenizer_sha256: REQUIRED }
runtime: { repository: REQUIRED, commit: REQUIRED, compiler_and_flags: REQUIRED, backend: cpu, quantization: REQUIRED }
stream: { requested_chunk_ms: REQUIRED, actual_right_context_ms: REQUIRED_FROM_RUNTIME, left_context_policy: REQUIRED }
resources: { intra_op_threads: REQUIRED, affinity: REQUIRED }
licences: { code_id: REQUIRED, weights_id: REQUIRED, distribution_status: internal_evaluation_only_until_resolved }
```

```json
{ "schema": "ultra.audio-record/v1", "record_id": "…", "canonical_recording_id": "…",
  "start_sample": 0, "num_samples": 160000, "pcm_sha256": "…", "speaker_group": "known-or-unknown",
  "session_group": "…", "domain": "read|spontaneous|other", "transcript_id": "…",
  "supervision_origin": "aligned_book_text", "human_review_status": "not_individually_reviewed",
  "label_tier": "gold|silver|bronze|reject", "licence_record_id": "…", "partition": "TRAIN",
  "dedup_status": "admitted" }
```

```json
{ "schema": "ultra.costlaw/v1", "machine": "A", "cpu": "…", "kernel": "tern2_avx512vnni", "threads": 1,
  "residency": "out_of_cache", "a_ps_per_weight": null, "c_ps_per_mac": null, "r2": null, "protocol": "alternating-5x" }
```

```json
{ "schema": "ultra.look/v1", "board": "DOMAIN-SEALED", "who": "scorer", "when": "…", "why": "G1, tier L",
  "commit": "…", "bundle": "sha256-…", "read": ["wer_macro", "wer_per_set"], "decision_changed": false }
```

Unknown metadata stays unknown; speaker ids and review status are never filled in by inference.

---

## Appendix E. Lessons taken from the research waves

| Wave | What happened | Rule here |
|---|---|---|
| H443–H472, adapter lane | most waves ended `NO_LIFT` | every bet has a margin and a fallback (§1.3, §1.5) |
| H444 → H445 | a one-seed gain did not survive other seeds | three seeds where affordable (§11.6) |
| H451 → H452 | a best-of-N winner shrank on confirmation | fresh rerun; CONFIRM partitions (§7.3) |
| H92 → H94 | positive on a cached screen; −51.9 Elo prospectively | rule 9 of §0.2; gate search re-runs the Cortex (§6.5) |
| H447 | gains came from tails | per-speaker win, tie, loss and harm rate (§11.3) |
| H449 → H450 | raw regret weighting collapsed to an effective sample size near 3 | percentile ranks and a floor (§7.5) |
| H457, H470, H471 | stabilisers "helped" by shrinking updates | compare at equal update size; no optimiser zoo (§8.7) |
| H461 | larger doses inflated margins; family balancing gave no lift | thresholds refitted per checkpoint (§6.5); mixes are tested (W20) |
| H446 | training and serving channels differed | train on the real Reflex outputs and the real mode distribution (§8.4) |
| H455, H456 | exactly uniform deck; stream mix; valley soups; CPU-only scoring | §7.5, §8.7, §9 |
| H458 | per-family specialists; fair, in-corpus and recall labels; coverage rule; look ledger | §4.11, §11.2 |
| H460 | closed-form held-out guard; 4-of-5 folds; η = 0 is identity | the anchor guard (§4.9) |
| H110 | a learned proposal scorer gave +269 Elo | search uses a scorer over proposals (§6.5) |

Paths: `research/flywheel_selfplay/<wave folder>/`; H458–H465 summary in `docs/research/waves/h458-h465-black-hole-20261116/`.

---

## Appendix F. What came from where

| Element | From | Changed here |
|---|---|---|
| Measured two-term cost law; one pass per chunk on an audio-time grid; many streams in one pass | TACTUS | state traffic, attention and decoding itemised; second host shown |
| Fixed-cost derivation from the incumbent's timings; dial; TimeMix; kill gates; odds table; label canary | TEMPO | cause of the intercept left to W00 |
| Ladder with a pretrained R0 and a conventional R1 first; tournament with pinned candidates | CADENCE-CPU v2 | every later bet is a rung on the same ladder |
| Output contract, lifecycle, segment ownership, `Gap` | CADENCE-CPU v2 | mapped onto the three taps |
| Integer Cochlea by table lowering; 7-bit rule; ternary planes; Reflex FIR memory; token hints | KAIROS v0.1 | Cochlea isolated behind W13; integer square-root arm from APEX |
| CTC only on the dense grid; feasibility T ≥ U + R; mask and count | CADENCE-CPU v2 | two-position head as the pre-declared remedy (new) |
| Gate as a `sym` rule | KAIROS, TEMPO | strict-`>` thresholds; retention sweep; careful mode from observables |
| Tower with right context D; final mode in the same weights; capacity-or-data diagnosis | KAIROS v0.2 | transducer taps instead of CTC heads; dedicated-weights arm |
| Transducer as the main decoder | TEMPO, APEX | one joiner, three taps; cap 8 with a counter |
| Preview re-decoded from the committed decoder state | new | removes the two-head splice found by the APEX audit |
| Anchor rule; text events with confusion tables and twin consistency; roles, stop conditions, owner rulings | KAIROS v0.2 | preview teacher confined to the preview tap, so isolation is structural |
| Ridge readout with a fold guard | TEMPO, after H460 | leave-block-out stated in primal and dual form with dimensions |
| Refinement by predicted benefit | CADENCE-CPU v2 | applied to final mode and to a cascade with R0 |
| Mixture of experts; grouped-trit table kernel; cost-estimator parity test | APEX v1 | the first two demoted to arms of W23 and W01 |
| Zero-point correction; exact softmax; three parity questions; mixed precision as an optimisation | CADENCE-CPU v2 on KAIROS's formats | full operation list in §5.2 |
| Mode coverage and trained switch points | new | makes "every run-time mode is a training mode" enforceable |
| Audio identity graph; unique hours; partitions; pilot before factory; quality estimator | CADENCE-CPU v2 | open pool is the incumbent's own sources (KAIROS v0.2) |
| Distillation defined by support; tail bucket | CADENCE-CPU v2 | feature distillation at matched 80 ms frames (new) |
| Exact fleet statistics | KAIROS v0.1 | cross-term exact only on a fixed int16 grid (new) |
| Budget by measured throughput; 300–600× planning range | KAIROS v0.2, CADENCE-CPU v2 | table in §8.8; envelopes |
| Promotion rules; process-tree CPU; workloads; correctness test families | CADENCE-CPU v2 | modes row added |
| Program staging G0–G2 | new | answers "no new dialect before a model works" without giving up the GP language |

---

## Appendix G. Fair score of this plan

**87 / 100** on the rubric of `ULTRA_SCORECARD.md`, where TACTUS scores 74, KAIROS v0.2 76, CADENCE-CPU v2 78 and APEX v1 57.

| Criterion | Score | What holds it down |
|---|---:|---|
| A Correctness | 14 / 16 | Tower with a restricted transducer on gated events is untried; text events and integer attention are unproven at this scale |
| B Output contract | 7 / 8 | no adapter exists yet to exercise the tail cases |
| C CPU cost | 12 / 14 | one measured host; the split between fixed and marginal cost differs between the two runs |
| D Accuracy and data | 11 / 14 | unique hours unknown; teacher licences open; G2 at 35% |
| E Evidence | 9 / 10 | repository facts checked against an archive without history or manifests |
| F Innovation | 7 / 8 | new in combination and separately testable; none proven |
| G JopeDime | 7 / 8 | nothing compiled against the tree |
| H Evaluation | 7 / 8 | the domain sealed set does not exist yet |
| I Agent-readiness | 9 / 10 | commands and crate names unverified |
| J Risk and rights | 4 / 4 | — |

**What would raise it:** the cost law confirmed on a desktop core (W01); R0 and R1 shipped with receipts; the retention curve (W12); the ternary gap (W15). **What would lower it:** a measured allowance well above 1.5; unique hours far below gross; a licence that removes the open pool.

**Where this plan is most likely wrong.** (1) The event gate may cost more than 0.10 points at any useful retention; the plan then ships a frame-clock model that is still ternary. (2) Ternary at 600M weights may not hold under noise; the ladder ends at int8, where the fixed cost is five times higher and L's CPU cost is about two and a half times the ternary estimate `[D]`. (3) An incumbent trained on 530,000 hours may simply be out of reach on accuracy with the data available; K-ACC then repositions the release.

---

## Appendix H. Sources

`[SRC n]` and `[MEM n]` numbering is carried from the KAIROS v0.1 spec so tags stay stable. `[SRC 1]` and `[SRC 7]` were fetched again on 2026-11-20.

### H.1 Retrieved from the source

| # | Source |
|---|---|
| 1 | Nemotron Speech Streaming EN 0.6B model card — https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |
| 2 | NeMo-Speech.cpp benchmark — https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/BENCHMARK.md |
| 3 | Parakeet Unified EN 0.6B model card — https://huggingface.co/nvidia/parakeet-unified-en-0.6b |
| 4 | Moonshine v2 — https://arxiv.org/abs/2602.12241 |
| 5 | FUTO ASR4ALL-L model card — https://huggingface.co/futo-org/asr4all-l |
| 6 | Design of the IBM Granite 5.0 TurboCTC ASR model — https://arxiv.org/abs/2609.20104 |
| 7 | Parakeet Redux model card — https://huggingface.co/moondream/parakeet-redux |
| 8 | Phonon-2 research page — https://www.fermionresearch.com/research/phonon-2/ |
| 9 | Phonon-2.5 English model card — https://huggingface.co/FermionResearch/Phonon-2.5-English |
| 10 | E-Gaborformer — https://arxiv.org/abs/2611.08092 |
| 11 | Pushing the Limits of On-Device Streaming ASR — https://arxiv.org/abs/2604.14493 |
| 12 | Accelerating RNN-T Training and Inference Using CTC Guidance — https://arxiv.org/abs/2210.16481 |
| 13 | Blank-regularized CTC for Frame Skipping in Neural Transducer — https://arxiv.org/abs/2305.11558 |
| 14 | CTC Blank Triggered Dynamic Layer-Skipping — https://arxiv.org/abs/2401.02046 |
| 15 | Single-utterance test-time adaptation (SUTA) — https://arxiv.org/abs/2203.14222; continual test-time adaptation for ASR — https://arxiv.org/abs/2406.11064 |
| 16 | Rust `core::arch::wasm32` relaxed dot product documentation — https://doc.rust-lang.org/core/arch/wasm32/fn.i16x8_relaxed_dot_i8x16_i7x16.html |
| 17 | kNN-CTC — https://arxiv.org/abs/2312.13560 |
| 18 | Spiralformer — https://arxiv.org/abs/2510.00982 |
| 19 | Gated DeltaNet-2 — https://arxiv.org/abs/2605.22791 |
| 20 | FairyFuse: multiplication-free ternary kernels — https://arxiv.org/abs/2604.20913 |
| 21 | Bitnet.cpp — https://arxiv.org/abs/2502.11880 |
| 22 | Reducing the Offline-Streaming Gap for Unified ASR — https://arxiv.org/abs/2604.19079 |
| 23 | Ternary matrix multiply with AVX-512 lookup tables (write-up) — https://silverriver.bearblog.dev/ternary-mm-avx512/ |
| 24 | GPTQ — https://arxiv.org/abs/2210.17323 |
| 25 | GPTAQ (asymmetric calibration) — https://arxiv.org/abs/2504.02692 |
| 26 | Loquacious Set — https://arxiv.org/abs/2505.21578 |
| 27 | bitnet-cpu kernel notes (instruction-set selection) — https://huggingface.co/kernels/phanerozoic/bitnet-cpu |

Items 12–15, 17–21, 23–25 and 27 were read as search results, not as full pages.

### H.2 Cited from memory

| # | Reference |
|---|---|
| 1 | Wang et al., "Trainable frontend for robust and far-field keyword spotting" (PCEN), ICASSP 2017 |
| 2 | Swietojanski, Li and Renals, "Learning hidden unit contributions for unsupervised acoustic model adaptation" (LHUC), 2016 |
| 3 | Kim et al., "I-BERT: Integer-only BERT quantization", ICML 2021 |
| 4 | Zhang et al., "Deep-FSMN for large vocabulary continuous speech recognition", ICASSP 2018 |
| 5 | Mahadeokar et al., "Alignment restricted streaming recurrent neural network transducer", SLT 2021 |
| 6 | Yao et al., "Delay-penalized CTC implemented based on finite state transducer", Interspeech 2023 |
| 7 | Dong and Xu, "CIF: Continuous integrate-and-fire for end-to-end speech recognition", ICASSP 2020 |
| 8 | Nozaki and Komatsu, "Relaxing the conditional independence assumption of CTC-based ASR by conditioning on intermediate predictions", Interspeech 2021 |
| 9 | Fiscus, "A post-processing system to yield reduced word error rates: ROVER", 1997 |
| 10 | Ma et al., "The era of 1-bit LLMs: all large language models are in 1.58 bits", 2024 |
| 11 | You et al., "SpeechMoE: scaling to large acoustic models with dynamic routing mixture of experts", 2021 |
| 12 | Narayanan et al., "Cascaded encoders for unifying streaming and non-streaming ASR", ICASSP 2021 |
| 13 | Gales, "Maximum likelihood linear transformations for HMM-based speech recognition", 1998 |
| 14 | Esser et al., "Learned step size quantization", ICLR 2020; Choi et al., "PACT", 2018 |
| 15 | Yao et al., "Zipformer: a faster and better encoder for automatic speech recognition", ICLR 2024 |

### H.3 Added from memory for this plan

| # | Reference |
|---|---|
| 16 | Kuang et al., "Pruned RNN-T for fast, memory-efficient ASR training", Interspeech 2022 |
| 17 | Graves et al., "Connectionist temporal classification", ICML 2006 |
| 18 | Graves, "Sequence transduction with recurrent neural networks", 2012 |
| 19 | Ghodsi et al., "RNN-transducer with stateless prediction network", ICASSP 2020 |

### H.4 Cited by a reviewed document; not re-read by me `[DOC]`

CTC paper and the adjacent-repeat constraint; Stateful Conformer with cache-based inference (arXiv 2312.17279); Fast Conformer (2305.05084); CTC-guided frame reduction (2210.16481); Skipformer (2403.08258); FastEmit (2010.11148); Token-and-Duration Transducer (ICML 2023); T-MAC (2407.00088); Zipformer (2310.11230); the Libriheavy repository (shared audio with LibriLight); the Loquacious dataset card; the fifteen `NO_LIFT` optimiser waves and the re-run benchmark table reported in KAIROS v0.2.

### H.5 Repository files relied on

`crates/feat-math/src/{dsl.rs, genome.rs, math_expr.rs, math_expr/parser.rs, tokens.rs, emit_rs.rs}`; `crates/feat-math/src/kernel/{ordered_axis.rs, ridge_fast.rs, goertzel.rs, savgol.rs, lexicase.rs, pareto.rs, wire_v2.rs, ir/, timescale_nabla/state_codec.rs}`; `crates/jope-policy/src/grammar.rs`; `docs/contracts/{operator-registry-v2.md, typed-graph-and-edit-v1.md}`; `docs/{model-objects.md, emitters.md}`; `docs/adr/ADR-0006-rust-only-stack.md`; `AGENTS.md`; `research/flywheel_selfplay/PREREG_TEMPLATE.md`; wave folders `h92_w6_downside_controlled`, `h94_w6_prospective_hedge`, `h110_w9_scorer_replay_scored`, `h444`–`h472`, `h450_scalefree_regret_gate`, `h456_ultra_entropy_recipe`, `h460_apex_eq`, `h461_pairnf_dose_fbal`; `docs/research/waves/h458-h465-black-hole-20261116/`; `deploy/appliance/SPECIFICATION.md`.

---

## Appendix I. Glossary

| Term | Meaning |
|---|---|
| Rung | A shippable bundle that adds one mechanism to the one below |
| Event | A 12.5 Hz frame the gate keeps; the unit the Cortex runs on |
| Retention | Share of frames kept as events |
| Wake-up | One pass of the Cortex and Tower over a chunk's events; costs one fetch of their weights |
| Grid | Chunk length in frames (1, 2, 4, 7, 14) |
| Dial (θ) | Gate threshold offset; −1 keeps every frame |
| Tower | The blocks above the Cortex that carry right context; final mode is the same blocks over a closed segment |
| Tap | A projection from a point in the stack into the shared transducer: preview, commit, final |
| Preview / Commit / Final | Mutable suffix / irrevocable append / segment closed |
| Segment | The span between endpoints or caps; the unit of final mode and of mode changes |
| A7, A7z, I8, T1, T2 | 7-bit activations without and with zero point 64; int8 weights; one and two ternary planes |
| Fixed / marginal cost | Time to fetch weights per wake-up / time per multiply-add |
| CPU cost | Process-tree user plus system seconds per second of audio |
| Oracle | The full-precision, non-causal twin used for alignments and lattice targets |
| Anchor rule | Self-made text adapts only preview; committed and final text adapt only from confirmed labels |
| Careful mode | A segment run with every frame kept, triggered by run-time observables |
| Fair / in-corpus / recall | Labels on reported numbers |
| Look | One pre-registered read of sealed data |
