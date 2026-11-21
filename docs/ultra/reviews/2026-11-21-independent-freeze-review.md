# Independent review of the FREEZE draft and its measurement code (2026-11-21)

**Reviewer:** fresh-context Claude reviewer (PLAN_TO_V1 B4 / G8). I started with no shared conclusions and read the code myself.
**Scope:**
- documents: `docs/ultra/FREEZE.md` (DRAFT), `PLAN_TO_V1.md`, `BENCH_PLAN_v1.md`, the DEV selection panel and `python/ultra/tools/dev_panel.py`;
- engine code: `ultra-engine` `engine_density` / `engine_eval` / engine / quant;
- runtime code: `ultra-cli` bench-suite (`suite_cmd`, `suite_engine`, `suite_r1_z`, `suite_runners`, `suite_latency`);
- harness code: `ultra-stream` `grid_sim` / `native_engine_fleet`, and `ultra-eval` `ultra-measure` / `affinity` / `procfs` / `score` / `bootstrap` / `normalise`;
- evidence: AGENT_CHAT entries behind the "136/CCX" and "192 machine-wide" numbers.

**What I ran:** read-only checks only, no benchmarks:
- the panel manifest hash recomputes;
- both panel cache hashes match;
- panel speakers and uids are disjoint from test-clean and test-other (0 overlap);
- first-400 dev-clean covers 6 speakers;
- H0 topology is 16 L3 domains, e.g. cpu0 shares L3 with `0-3,64-67`.

**Verdict:** the draft cannot be frozen as written. In its current form the headline density ratio would compare different amounts of hardware, different harnesses and different lateness rules. One of its two named baselines, Nemotron, has no density measurement path at all. Rankings: **B** = blocking, **S** = should-fix before the tag, **N** = note.

---

## Blocking

### B1. The "×N density vs R0" compares 4 cores per CCX against 1 core per CCX
- **R0's number:** "192 machine-wide" is 16 single-threaded workers, one per CCX, so 16 cores in use. It is 12 streams per worker, measured with the Python harness on xbabe0 (AGENT_CHAT 2026-11-20T20:19Z: "16 single-threaded workers, one per CCX"; BENCH_PLAN §1.3: "12 streams/core sustained with one worker per CCX").
- **The engine's number:** "136/CCX" is **4 separate `engine_density` processes on cores 56–59, 34 streams each** (AGENT_CHAT, "x86 per-CCX, prepared int8, harness v2": "4 x 34 = 136/CCX").
- **Effect:** a per-CCX or machine ratio built from these two numbers (136/12 ≈ 11×, or 16×136 vs 192) gives the engine 4× the cores. On cores alone the observation is ~34 vs ~12 per core, about 2.8×.
- **FREEZE contradicts itself:**
  - §3 says "One inference thread per worker, one worker per CCX";
  - §4.4 says "per-CCX: 4 workers";
  - `engine_density.rs:1-3` says "pin the process to one core … one worker per CCX".
- **The L3-bound argument is untested:** "R0 is L3-bound, so more cores per CCX would not help it" (BENCH_PLAN §1.3 finding 1) is a hypothesis. Nobody has measured R0 or ORT with 4 workers per CCX.
- **Fix:**
  - Fix one worker topology in §3 and apply it to every system: either 1 worker = 1 core, with W workers per CCX, or 1 worker per CCX.
  - Measure R0, ORT R1 and the engine under it, and report streams per **core** next to streams per CCX and per machine.
  - If the engine uses 4 workers per CCX, R0 and ORT must also be run at 4 workers per CCX before any ratio is published.

### B2. The engine density number and the baseline numbers come from different harnesses with different A1 implementations
There are three A1 implementations:
- Python `sp_a.py` / `sp_a_fleet.py`, which produced R0's 192;
- Rust `grid_sim::simulate_grid` / `sustains_lateness`, which bench-suite uses for R0, ORT and the engine;
- the standalone `engine_density.rs`, which produced 136/CCX.

They differ on every axis that decides where the knee falls:

| Aspect | Python SP-A (R0 192) | Rust `grid_sim` (bench-suite) | `engine_density` (engine 136) |
|---|---|---|---|
| Lateness reference | oldest unprocessed packet deadline | oldest unprocessed packet deadline (`grid_sim.rs:272`) | per chunk, decode end minus arrival of the chunk's last needed sample (`engine_density.rs:107-131`) |
| Sample weighting | one sample per decode tick | one per tick (`grid_sim.rs:280-283`) | one per chunk (dozens per tick) |
| Drain lateness | excluded | excluded (`!cutoff`, `grid_sim.rs:282`) | **included** (`engine_density.rs:222-227`) |
| No-growth rule | last third − first third ≤ 0.25 × 800 = 200 ms (`sp_a_fleet.py:51,62`) | same 200 ms (`grid_sim.rs:59`) | second half − first half ≤ **50 ms** (`engine_density.rs:232-248`) |
| `last < bound` | **required** (`sp_a_fleet.py:63`) | not checked (`grid_sim.rs:43-60`) | not checked |
| Grid | 320 ms | **160 ms for engine and ORT** (`suite_engine.rs:104`, `suite_r1_z.rs:48`), 320 ms for R0 (`suite_r1_z.rs:176`) | 320 ms default; `--max-chunks 2` per chat, default 1 (`engine_density.rs:38`) |
| Steps | — | `max_chunks` = 1 (NativeEngineBackend never sets it) | 2 chunks/step per §3 (CLI flag) |
| Weights | per worker | shared `Arc` across worker threads | **separate copy per process** |
| Duration | per run | 10 s default (`suite_cmd.rs:29`) → about 31 ticks, so p99 ≈ max | 60 s |
| CPU | process | process RUSAGE_SELF minus observer thread | **one thread** (`/proc/thread-self/schedstat`, `engine_density.rs:64-81`) |

Some of these differences make the engine look worse (drain included, per-chunk weighting, stricter growth rule, unshared weights). Others favour it (no `last` check, 320 ms grid with 2-chunk steps while ORT in bench-suite gets 160 ms and 1-chunk steps). The net sign is unknown. The comparison is not like for like either way.
- **The "same harness" ORT claim is false:** STATUS says "ORT on the same harness did not sustain 32 streams per CCX", but `engine_density` can only run `ultra_engine::Engine`. That claim cannot be true as worded.
- **Fix:**
  - Pick **one** density harness for every density row. The natural choice is bench-suite's `grid_sim`, which already has an engine path (`run_sp_a_native_engine_shared_workers`).
  - Make grid, `max_chunks`, duration, packet size, and the A1 predicate (bound, growth rule, `last` check, drain treatment) CLI-pinned values recorded in the receipt, identical across systems.
  - Demote `engine_density` to a development tool, or make it call the same `sustains_lateness`.
  - Re-measure R0 in Rust on H0. Do not reuse the Python/xbabe0 192.

### B3. C-density needs a Nemotron density row, but no tool can produce one, and §4.4 leaves Nemotron out
- **The claim:** §5 C-density reports "a ratio against R0 and against Nemotron".
- **The protocol:** §4.4 lists the density systems as "R0, R1-dense, R1-MR, and ORT R1". Nemotron is not among them.
- **The code:** `run_nemotron_candidate` hard-codes `density: None, latency: None, sustained_probe: None` (`suite_runners.rs:110-112`). The NeMo adapter is a cold-load, single-stream batch process (`suite_runners.rs:71-73`).
- **Fix:** do one of the following, and fix the choice before the tag:
  - drop the Nemotron density ratio from C-density;
  - implement and dress-rehearse a paced multi-stream Nemotron path under the same A1 harness, and add Nemotron to §4.4.

### B4. Nothing can produce a qualifying row, and the full-machine run cannot have a disjoint observer
- **bench-suite cannot qualify:** `ultra bench-suite --qualified` always returns an error: "bench-suite emits exploratory receipts only" (`suite_cmd.rs:83-93`). `engine_density` hard-codes `"exploratory": true` (`engine_density.rs:250`). The G6 qualifying window currently has no tool that can emit a qualifying receipt.
- **The full-machine run and the observer rule cannot both hold:**
  - §3 requires the observer on "a CCX disjoint from the service";
  - §4.4 requires a full-machine run with 16 CCX workers;
  - H0 has exactly 16 L3 domains;
  - `affinity.rs:70-92` and `suite_manifest.rs:157-182` reject an observer in any service L3 or SMT domain.
  So a 16-CCX service leaves no legal observer CPU, and `ultra-measure` fails closed.
- **Fix:**
  - Either define "full machine" as 15 CCX plus 1 observer CCX, or state that the full-machine row uses an in-service observer whose CPU is reported and subtracted.
  - Implement the qualifying path: lease, freeze-tag check, profile gate.
  - Run both in the A5 dress rehearsal.

### B5. The DEV selection panel is not wired into any tool, so G1/G2 will run on the 6-speaker continuity panel
- **No consumer:** `grep` finds no reader of `dev_selection_panel.json` / `ultra.dev-panel/v1` outside `dev_panel.py`.
- **What the tools actually take:**
  - `engine_eval` takes `offset=0, limit=400` (`engine_eval.rs:223`), which is first-400 dev-clean, 6 speakers;
  - `engine_density` takes the first 400 (`engine_density.rs:160`);
  - bench-suite and the trainer eval take offset/limit too.
- **The conflict:** FREEZE §1 says the continuity panel is "Historical comparability only, not selection", yet G2's "quality gate (≤ +0.10 vs FP32) on 400 DEV" and the trainer's "dev-clean400" curves run on exactly that panel.
- **Fix:** add `--panel <manifest>` (uid list plus sha check) to `engine_eval`, the trainer evaluate path and bench-suite, and require it for every G1/G2 decision. Record the panel `manifest_sha256` in every selection output.

### B6. The C-density and C-cost claims are not pre-registered in any testable sense
§5 names quantities but fixes none of the following:
- **Which R1:** dense, MR, or "the better of the two after seeing results".
- **The scale:** per-CCX or full machine.
- **The search:** the N grid (step size), the number of repeats per N, and the rule when N passes and N−1 fails. A/A noise is +25% (PLAN_TO_V1 §5).
- **Multiple workers:** whether every worker in a CCX or machine must pass, and in the same run. `engine_density` is one process; the 4-process orchestration and all-pass aggregation behind 136/CCX are not in the repo.
- **Cost conditions:** the cost statistic (median of k ABAB blocks?), the 5% CPU rerun tolerance from BENCH_PLAN §0a, and the stream count, pacing and warm/cold state at which C-cost is measured.

**Selection effect:** "increase N until A1 fails" with one run per N reports a maximum over noisy trials, which is biased upward. Choosing among {R1-dense, R1-MR} × {per-CCX, machine} × {vs R0, vs ORT, vs Nemotron} after the results are in is a forking-paths problem. Holm cannot repair it, because the "declared family" (§4.2) is never declared.

**Fix:** pre-register in §5:
- the primary claim: one system, one scale, one baseline;
- a fixed N ladder;
- "claimed density = largest N on the ladder where every worker passes A1 in ≥ 3 of 3 alternating repeats, then confirmed by the 1 h run at that N";
- the same procedure for every baseline;
- the cost statistic with its block design and tolerance;
- the Holm family listed explicitly;
- everything else labelled descriptive.

### B7. Density is reported at very different accuracy levels, with no framing fixed in advance
R1-MR is at 12–15% dev-clean WER and R0 at 2.85% (PLAN_TO_V1 §1.3). "×N streams vs R0, with the WER of each" puts an easy-to-quote ratio beside a 4–5× worse error rate. FREEZE C-density also contradicts BENCH_PLAN §0a ("R1-MR is excluded from the initial freeze"; "the primary formal comparison is R1 vs R0 only"). PLAN_TO_V1 says the BENCH_PLAN *protocol* still applies.

**Fix:**
- Settle the conflict explicitly in FREEZE.
- Pre-register that density is presented only as a WER × density frontier, or as density at a stated WER, never as a standalone ratio in titles or summaries.
- Carry BENCH_PLAN §0a's margins, the 5% rerun tolerance and the two 1 h runs into FREEZE, or state that they are dropped.

---

## Should-fix

### S1. Cost scope is not uniform across R0, ORT R1, engine R1 and Nemotron
- **Same label, different meaning:** bench-suite gives R0 sherpa, ORT R1 and the engine all the label `cost_scope = "in_process_engine_warm_sp_a"` (`suite_r1_z.rs:105,232`, `suite_engine.rs:161`). For R0 and ORT that label is wrong. The value is the process RUSAGE_SELF delta after load: warm, model load excluded.
- **Nemotron is measured differently:** it is `post_prep_process_tree_service_cold_load` (`suite_runners.rs:71-73`), a single-stream process tree **including model load**.
- **`ultra-measure` includes load too:** it charges spawn-to-exit with RUSAGE_CHILDREN (`procfs.rs:388`). The cold-load tax is material on DEV400: R0 0.1372 cold vs 0.1362 warm on the bridge, and 0.078 on the later build. The headline table mixes these sources.
- **`engine_density` is narrower still:** its `cpu_s_per_audio_s` is one thread only and excludes load and any pool threads (`engine_density.rs:64-81,230`).
- **The protocol is silent:** FREEZE says only "whole-service CPU-s per audio-s".
- **Fix:**
  - Define the scope in §3: process tree, warm or cold, load included or excluded, frontend included, stream count, paced or unpaced.
  - Measure all four systems with that one scope via `ultra-measure` (or the bridge) in the same window.
  - Report cold and warm as separate columns.
  - Correct the R0/ORT `cost_scope` labels.
  - Forbid `engine_density`'s per-thread figure in C-cost.

### S2. The INT8 gate is a point threshold too small to resolve on 7.7k words
- **The size of the threshold:** +0.10 pp on ~7.7k words is about 8 word errors.
- **The noise:** when INT8 and FP32 disagree on k utterances (the earlier continuity check found 33/400 differing), a rough paired SD is ≈ sqrt(k) × 1.5 / 7700 ≈ 0.11 pp.
- **What that means:** the gate has roughly coin-flip power against a true +0.10 regression and passes a true +0.2 a non-trivial share of the time. On 6 speakers the bootstrap also has only 6 clusters, and `bootstrap.rs` flags "sparse" only when there are fewer than 2.
- **Fix:**
  - Run the gate on the stratified panel (730 utterances, 73 speakers, ~14k words).
  - Gate on the one-sided 95% upper bound of the paired cluster bootstrap (`tail_at_least(0.10)`) rather than the point delta.
  - Keep the ≥ 98% same-precision token-identity continuity check as a separate gate.
  - Raise the bootstrap sparse threshold to, for example, fewer than 20 clusters.

### S3. Engine settings that change density can be set without leaving a record
- **Unrecorded environment variable:** `engine_density.rs:148-153` reads `ULTRA_ENGINE_STEM_KINDS` from the environment and records only a count.
- **Silent runtime quantisation:** `engine_density.rs:145-147` and `engine_eval.rs:204-205` quantise weights at runtime when no prepared artifact exists. FREEZE §3 says "No runtime weight quantisation". The output only shows `int8_prepared_sha256: null`.
- **Missing identities:** the output records no executable sha, git commit, ISA, effective affinity, host load, npz sha, utterance ids or RSS (`engine_density.rs:249-264`). `engine_eval` does record `executable_sha256`.
- **Mislabelled model:** `NativeEngineBackend::bundle_id` labels every model "ultra-engine-dense" (`native_engine.rs:92`), S_MR included.
- **Fix:**
  - Fail closed on unprepared INT8 and on any `ULTRA_ENGINE_*` environment variable in timed modes.
  - Record all knobs and identities in the output.
  - Pin `int8_stem`, `act7`, `fp32_kinds`, `max_chunks` and the grid per finalist in CANDIDATES.lock.

### S4. Fleet worker placement is assumed, not checked against topology
- **Hard-coded and unchecked:** `native_engine_fleet.rs:35-41` keeps CPUs below 64 (an H0-specific constant) and splits them evenly into workers (`:59-72`). It never checks that a worker's CPUs share one L3 domain.
- **Silent pin failures:** `pin_current_thread` ignores errors (`:23-32`).
- **Observer validation fails open:** in exploratory runs, observer disjointness is checked with topology failing open (`suite_manifest.rs:62`).
- **What a "worker" is:** in bench-suite a worker is **one thread allowed to float over 4 CPUs**, so one core of compute per CCX. In the engine's per-CCX runs it was 4 pinned processes, which is B1 again.
- **Fix:** verify worker CPU sets against `cache/index3/shared_cpu_list`, fail on pin errors, record effective per-worker affinity, and state SMT sibling policy: siblings 64–127 of service cores idle or excluded.

### S5. The sustained 1 h run in §4.5 does not exist as specified
- **What bench-suite does:** its "sustained" probe runs **1 stream** (`suite_engine.rs:151-157`), not "the claimed density".
- **What `engine_density` lacks:** an RSS series and a backlog series.
- **Fix:** add a time-series receipt (backlog steps, lateness and RSS per 10 s) to the chosen density harness, and run it at the claimed N with all workers.

### S6. PUBLIC-TEST is already non-pristine, and LibriSpeech test reads are not mechanically sealed
- **Prior reads:** `LOOKS.jsonl` already records several test-clean/test-other reads (R0 smoke, Nemotron full splits, P05 contrast). They are reconstructed after the fact, with times unknown.
- **No seal for LibriSpeech test:** only the OOD loader (`python/ultra/ultra/ood.py`) enforces the freeze tag and the one-read rule. Rust bench-suite and `engine_eval` read `test-*.npz` with no check and no LOOKS entry.
- **FREEZE says neither:** §4.2 states "one read each, logged before reading" and mentions neither point.
- **Fix:**
  - Disclose the prior reads in FREEZE §4.2 and in the frontier report. This matters especially for Nemotron and R0, whose test WERs were already known when they were chosen as baselines.
  - Add a Rust-side guard on PUBLIC-TEST partitions, mirroring `ood.py`: freeze-tag ancestor, look-id env, append to LOOKS.

### S7. Identities in FREEZE §1 are incomplete
- **Truncated hashes:** the DATA.lock manifest hash `3e21c0e4…bff5` and the continuity workload `fb8241b9…` are truncated.
- **No scorer binary hash:** the scorer is "at the freeze commit" without one.
- **Unpinned density workload:** the npz, offset, utterance order and the per-worker start offset are not pinned. `engine_density` uses stride 37 within one process; the fleet uses `w*137`.
- **Fix:** give full sha256 values, the `ultra-score` executable sha, and a density workload manifest.

### S8. Per-CCX results must not be extrapolated to the machine
- **Clock speeds differ:** a per-CCX run on an otherwise idle host gets higher boost clocks than an all-core run.
- **Baselines differ too:** R0's 192 loaded only 16 cores; an engine full-machine run loads 64.
- **Fix:** state in §5 that machine-level claims come only from the measured machine run, never from per-CCX × 16, and record the frequency governor and boost state in the manifest.

---

## Notes

- **N1.** `engine_density`'s lateness math is correct for the causal frontend. Frame `i` needs samples up to `160i+400` (`trainer_mel.rs:4-5`) and a chunk needs feature frame `4·t1−4` (`rnnt.rs:107`), matching `engine_density.rs:114`. The final padded chunk is capped at EOF.
  - The earlier e1-review findings are fixed here: per-chunk deadlines; accepted, processed and unprocessed audio accounted; drain lateness kept.
  - Warm-up is **not** excluded, which is conservative.
- **N2.** Shared audio across streams and processes does not flatter `engine_density`. Each process has private weights and activations, with no shared pages, and the engine has no input-dependent cache. Identical sequences across the 4 processes (same stride and order) are harmless, but the density workload should still offset workers as the fleet does.
- **N3.** `processed_audio_s = frames × 0.04` (`engine_density.rs:229`) counts the padded tail frames of each final chunk, which understates CPU per second very slightly.
- **N4.** The scorer, bootstrap and normaliser look sound:
  - paired by id with ref, speaker and dataset equality enforced (`score.rs:131-150`);
  - stratified cluster resampling, a +1-corrected tail and Holm (`bootstrap.rs`);
  - the normaliser hash pinned in a test (`normalise.rs:152`) and matching FREEZE.
  The OOD seal (`ood.py`) has tests for missing tag, non-ancestor and second read.
- **N5.** I found no leakage path in the DEV panel:
  - selection uses only uids and speakers, never texts;
  - panel speakers and uids are disjoint from test-clean and test-other;
  - CMVN comes from training batches and INT8 calibration from train-clean-100 / train-other-500;
  - references are loaded only after inference (`engine_eval.rs` "References are read only after inference"; `load_npz_audio_only` never touches `texts`).
- **N6.** Per-CCX engine runs used unshared weights per process (4 copies), which works against the engine on an L3-bound CCX. Fixing B1 by moving to the shared-weight fleet may *raise* the engine's number. That still has to be measured.
- **N7.** FREEZE §4.3 "≥ 5 ABAB blocks for each finalist pair" has no decision rule or statistic. Adopt BENCH_PLAN §0a's median and 5% tolerance, or state that timing is descriptive.

## Minimum set to close before `bench-freeze-v1`
B1–B7 resolved in FREEZE text and code. S1 (cost scope), S2 (gate on the panel with an interval), S3 (fail-closed knobs) and S6 (disclosure) are needed for the claims to stand. A5's dress rehearsal on xbabe2 should run the exact §4 script end to end, including a qualifying receipt and the observer placement for the full-machine run, before H0's window is booked.
