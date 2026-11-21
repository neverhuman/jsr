# Independent re-review: FREEZE.md revision and review fixes (2026-11-21)

**Reviewer:** the same fresh-context reviewer as `2026-11-21-independent-freeze-review.md`.
**Scope:** read-only review of the following:
- `docs/ultra/FREEZE.md` (17338f3);
- the engine fixes merged in cc9d0fb: `engine_eval`, `engine_density`, `engine_gate`, `run_info`, `panel`;
- the runtime-lane fixes in 30e2554, merged as 14f4bc3: `grid_sim`, `placement`, the three fleets, `suite_cmd`, `suite_ladder`, `suite_panel`, `suite_engine`, `suite_r1_z`, `suite_runners`, `suite_manifest`, `build.rs`.

I ran nothing heavier than `grep` and file reads. Each item is judged on what the code does, not on what commit messages or FREEZE say.

## Summary

| Item | Verdict | One line |
|---|---|---|
| B1 cores per CCX | **Mostly resolved** | Every fleet now defaults to 4 workers per CCX through one placement path. "1 worker = 1 core" is still not enforced (see S4). |
| B2 one harness, one A1 | **Partial** | Single harness, single predicate, uniform grid: done. The FREEZE lateness definition does not match the code. "2-chunk steps for every system" is engine-only. |
| B3 Nemotron density | **Resolved** (with caveat) | Removed from the claim. The "implied ceiling" in §4.4 contradicts §3's no-extrapolation rule. |
| B4 qualifying path / observer | **Partial** | `--qualified` and the 15-CCX machine now exist. The clean-build and freeze-ancestry checks can be bypassed. Identity pins are not checked. |
| B5 panel wiring | **Partial** | engine_eval is correct. bench-suite never verifies the manifest hash, and its Nemotron/offline rows silently ignore the panel. The trainer panel loader is unverified. |
| B6 pre-registration | **Partial** | A ladder and repeats exist, but the claimed-N rule is non-monotone, which is the upward bias the review flagged. Bounds, margins and the selection statistic are not fixed. No ABAB runner exists. |
| B7 framing / BENCH_PLAN conflict | **Resolved** | WER is printed beside density, and the deviation is recorded in DECISIONS.md. |
| S1 cost scope | **Unresolved for Nemotron** | See the dedicated answer below. |
| S2 INT8 gate | **Partial** | Panel plus bootstrap: done. The +0.50 pp upper limit is lax, the tool exits 0 on FAIL, and the continuity gate was dropped. |
| S3 knobs / env / runtime quantisation | **Resolved** | Environment knobs refused, identities recorded, qualifying mode refuses runtime quantisation. |
| S4 placement | **Partial** | Verified pinning and L3/SMT checks added. Topology checks still fail open without an observer, and workers can silently run unpinned. |
| S5 sustained run | **Partial** | It runs at N with a series, but the series has no backlog (always `None`), and "RSS" is the running peak. |
| S6 PUBLIC-TEST | **Partial** | Disclosure added. The guard is an env-var-non-empty check in `--qualified` only, with no LOOKS append and no one-read rule. |
| S7 identities | **Unresolved (deferred)** | Hashes are still truncated, and there is no scorer binary hash. |
| S8 extrapolation / clocks | **Partial** | Text fixed. The governor and boost state are not recorded, and the Nemotron implied ceiling is an extrapolation. |

New issues found in this round: **N1–N8** below. N1 (non-monotone ladder), N2 (gate exit code) and N3 (stale dirty flag) should be fixed before the tag.

---

## The coordinator's specific question: is `in_process_warm_rusage_self_post_load` the same scope for Nemotron?

**No.** Three things differ.

1. **Nemotron's label never changed.** `suite_runners.rs:77-80` still sets `("rust-nemo-speech-adapter-v1", "post_prep_process_tree_service_cold_load")`. Only the in-process runners (`suite_engine.rs:202`, and `suite_r1_z.rs` for ORT and R0) were relabelled. The commit message's "uniform cost_scope … across all runners" is false for Nemotron and for the offline rows (`suite_runners.rs:172`, `in_process_offline_recognizer`).

2. **What is measured differs, not just the label.**
   - Nemotron's CPU is `(RUSAGE_SELF + RUSAGE_CHILDREN)` after the child exits, minus a baseline taken before spawn (`ultra-adapters/src/nemotron.rs:318-323`, `nemotron_proc.rs:15-31`). That covers the child's **model load, the `--warmup` pass** (`ultra-bridge/src/nemotron.rs:282` shows the flag is used), inference, and the adapter's own parsing and sampling.
   - The in-process rows are `RUSAGE_SELF` deltas taken **after** load, via `RealClock::new()` after `NativeEngineBackend::load…`. They exclude load and warm-up. They include the in-thread `/proc` TreeSampler; only fleets subtract an observer.
   - Nemotron also runs as a file-mode batch over a child process, not through the paced grid.

3. **FREEZE claims something the code does not do.** §3 says "Whole process tree, the same scope for R0, ORT-R1, engine-R1 and Nemotron. Nemotron also gets a cold row." No warm Nemotron row exists. The adapter reports `load_ms` as wall time, so load CPU cannot be subtracted afterwards.

**Fix:** pick one of the following, and fix the choice in FREEZE before the tag.
- **(a)** Make every cost row a cold child-process tree measured by `ultra-measure`. Add a "load-only" control run per system (same binary, zero audio, or the first utterance only) and report `warm = tree − load-only` with the control's CPU beside it.
- **(b)** Keep the in-process warm rows. Give Nemotron a warm figure from a two-length design: run N and 2N utterances in one child each; the CPU difference divided by the audio difference is the warm marginal cost. Label it as such.

Either way, until one of these exists, FREEZE §3/§5 must not put Nemotron in the C-cost comparison. Today it is a cold-inclusive descriptive row with a different label.

---

## Blocking items (from the first review)

### B1. Equal cores per CCX: mostly resolved
- `--density-workers` defaults to 4 and `--full-machine` sets 60 (`suite_cmd.rs:33,66`).
- The R0 (`fleet_inproc.rs:123-137`), ORT (`ort_r1_fleet.rs:84-98`) and engine (`native_engine_fleet.rs:51-65`) fleets all share `physical_first_cpus`, `verify_worker_l3_and_smt` and `pin_current_thread_verified`.
- **Remaining:**
  - Workers get `phys_aff.len() / workers` CPUs each. A run under `taskset -c 0-7` with 4 workers gives each worker **2 cores** for one thread. Nothing enforces FREEZE §3's "4 single-threaded workers on that CCX's 4 physical cores".
  - Require `phys_aff.len() == workers`, so each worker gets exactly one core, at least under `--qualified`.
  - See S4 for the unpinned fallback.
- **Ordering bug:** `--full-machine --density-workers 4` silently reverts to 4 workers, because the last flag wins.

### B2. One harness and one A1: partial
**Done:**
- One `simulate_grid` and one `sustains_lateness` for every system.
- `--grid-ms` defaults to 320 for every system.
- Drain ticks are now included.
- Accepted, processed and unfinished audio are accounted (`grid_sim.rs` diff).
- `engine_density` is labelled a development tool (`engine_density.rs:9-12`; schema v3 with `"tool": "development (not an evidence harness)"`).

**Not done or misdescribed:**
- **The lateness definition does not match FREEZE.**
  - FREEZE §3 says "capture deadline per chunk = arrival of its last required sample + processing budget". The code still computes **one tick-level value per tick**, `done_t − deadline(oldest unprocessed packet)`, and pushes that same value `total_ready_decoded` times (`grid_sim.rs`, the `for _ in 0..total_ready_decoded` loop).
  - That is per-tick lateness weighted by the number of stream-steps, not per-chunk lateness. "Processing budget" is undefined anywhere.
  - Oldest-packet lateness is never smaller than chunk-ready lateness, so the code is the more conservative of the two. FREEZE should say what the code does, or the code should change. The pre-registered rule must match the implementation exactly.
- **"2-chunk steps for every system" is not true.**
  - `max_chunks` exists only in `EngineSpAConfig` and `NativeEngineBackend` (`suite_ladder.rs:148`).
  - `R1SpAConfig` (ORT) and `SpAConfig` (R0/sherpa) have no such field (`grep max_chunks crates/ultra-stream/src` hits only the native-engine files). ORT and sherpa process their ready chunks as separate batched calls.
  - Fusing two chunks into one weight pass is a legitimate engine capability. FREEZE §3 should say "2-chunk steps where the runtime supports it (ultra-engine); ORT and sherpa decode every ready chunk each tick". It should not say "for every system".

### B3. Nemotron density: resolved, with one caveat
§4.4 and §5 now exclude Nemotron from C-density. The "implied ceiling from the single-stream cost row" is an extrapolation of exactly the kind §3 forbids, and it uses a cold-inclusive cost (see S1). Keep it out of the density table, or footnote it as "not measured".

### B4. Qualifying path and full-machine observer: partial
**Done:**
- `--qualified` preflight: explicit observer with fail-closed L3/SMT disjointness, a lease file named `BENCH_LEASE.host`, refusal of `ULTRA_ENGINE_*`, freeze-tag ancestry, and ≥ 300 s per step (`suite_cmd.rs:113-144`, `suite_manifest.rs:202-273`).
- The full machine is 15 CCX.

**Holes:**
- **Stale dirty flag (N3).** `crates/ultra-cli/build.rs` reruns only on `Cargo.lock`, `ultra-cli/src`, `HEAD`, the branch ref and `packed-refs`.
  - An uncommitted edit to `crates/ultra-stream/src/grid_sim.rs` (the A1 code itself) rebuilds and relinks `ultra` without rerunning `build.rs`.
  - `ULTRA_BUILD_GIT_DIRTY` then stays `false`, and `--qualified` accepts a binary carrying uncommitted harness code.
  - `ultra-engine/build.rs` has the same pattern (it watches `index` and its own `src`).
  - **Fix:** watch every workspace crate's `src` (or emit `rerun-if-changed` for the whole workspace), or compute dirtiness at runtime against the embedded commit and refuse a mismatch.
- **Freeze ancestry is checked against the wrong commit.** `verify_freeze_tag_ancestor` runs `git merge-base --is-ancestor <tag> HEAD` in the **current working directory at run time**, not against the binary's embedded `ULTRA_BUILD_GIT_COMMIT`.
  - A binary built before the tag passes if it is run from a checkout that contains the tag.
  - **Fix:** check `tag` against `build_commit`, and require `build_commit == HEAD` of the run-time checkout or record both.
- **Identity pins are not checked.** FREEZE §4.1 says the preflight checks "pinned identities", but nothing compares the loaded artifacts with `CANDIDATES.lock`. Either add `ultra candidate validate --verify-files` to the preflight, or remove the phrase.

### B5. Panel wiring: partial
- **engine_eval: resolved.** `ultra_engine::panel` recomputes `manifest_sha256` from canonical JSON (`panel.rs:29-41`) and verifies each split's cache sha. `--qualifying` requires `--panel`.
- **bench-suite: verifies the cache, not the manifest.** `suite_panel.rs:67-72` only checks that `manifest_sha256` is 64 characters long. The file's actual sha is computed and then discarded (`let _file_sha`, line 112).
  - An edited uid list with the old hash passes, and the old hash is written into `frontier.json`.
  - **Fix:** recompute it exactly as `ultra_engine::panel` does, ideally by sharing that function.
- **bench-suite: Nemotron and offline rows ignore the panel (N4).**
  - With `--panel`, `ctx.limit` becomes `None` (`suite_cmd.rs:187-191`). `run_nemotron_candidate` uses `ncfg.limit = ctx.limit` (`suite_runners.rs:64`) and `run_offline_candidate` uses `limit: ctx.limit` (`:152`). Neither uses `selected_indices`.
  - Those rows therefore run the **entire split** while R0, ORT and the engine run the panel. Their job manifests still record `panel_manifest_sha256`.
  - **Fix:** pass `selected_indices` to both, or refuse those candidates when a panel is given.
- **Trainer: the loader is unverified.** `python/ultra/train/evaluate.py:23-52` accepts four shapes. It checks no schema, manifest hash or cache hash, and it tolerates missing uids for list-shaped input.
  - Its `splits` branch accepts any split name, `test-clean` included. That opens an unlogged PUBLIC-TEST read path.
  - **Fix:** use one verified loader, refuse non-`dev-*` splits, and record the manifest hash in the output.

### B6. Pre-registration: partial
**Done:**
- `--density-ladder start:end:step` and `--density-repeats 3`, with candidate order alternating per repeat (`suite_ladder.rs:286-299`).
- FREEZE fixes step 2, 3 repeats and a 4-claim Holm family, and restores the 5% rerun tolerance.

**Not done:**
- **The claimed-N rule is non-monotone (N1).**
  - `compute_claimed_n` returns the largest N at which all repeats pass, even when a smaller N failed. The unit test asserts this: N = 6 fails one repeat, N = 8 passes, claimed = **8** (`suite_ladder.rs:356-369`).
  - This is the max-over-noisy-trials bias the review warned about.
  - **Fix:** claimed N = the largest N such that **every** ladder value ≤ N passed all repeats; stop at the first failure. Pre-register that rule in §4.4 and invert the unit test.
- **The ladder is not fixed, and the budget does not fit the window.** FREEZE gives the step but not the start or end.
  - At 300 s per step, a 20:60:2 ladder is 21 × 3 repeats × 4 systems × 300 s ≈ **21 h per scale**, and there are two scales.
  - The qualifying window is a few hours, so the bounds will in practice be chosen from exploratory results. That is acceptable only if they are written into FREEZE before the tag.
  - **Fix:** pin start, end and the step-time budget per system and scale, or use a pre-registered bracketing rule (coarse exploratory knee, then a fixed ±3-step qualifying ladder).
- **Holm needs a test for each claim it covers.** "The Holm family is the 4 claims" only makes sense if each claim has a pre-registered test and α.
  - C-density is a descriptive ratio with no test.
  - C-cost covers two R1 systems.
  - C-quality has no margin or α. The +0.10 non-inferiority margin from BENCH_PLAN §0a is gone, and "unless the interval supports it" fixes neither an interval nor a level.
  - **Fix:** list each hypothesis (null, statistic, margin, α), and count the family accordingly (C-cost dense, C-cost MR, C-quality clean, C-quality other, …). Mark density as descriptive, outside Holm.
- **The finalist selection rule is not fixed.** §2's "selected on the DEV selection panel" and "best of D2MR-final / D3MRb-final" give no statistic: pooled 730, dev-clean and dev-other macro, final step or best step, tie-break.
- **There is no timing-block runner.** §4.3's five ABAB blocks, median and bootstrap CI have no implementation; the only "ABAB" in the CLI is the ladder ordering. A bootstrap CI over 5 block values is nearly degenerate.
  - **Fix:** pre-register the median with min/max (or a sign test across blocks) and implement or script the block runner before the window.

### B7. Framing and BENCH_PLAN conflict: resolved
FREEZE §5 requires the WER gap in the headline sentence, and DECISIONS.md records the R1-MR deviation and restores the 5% tolerance and the two 1 h runs.

---

## Should-fix items (from the first review)

- **S1, cost scope: unresolved for Nemotron.** See the dedicated section above. Two smaller points:
  - The in-process single and batch rows include the in-thread TreeSampler's `/proc` reads (`native_engine_bank.rs:347-356`). That is uniform across in-process systems, but FREEZE §3 says observer CPU is reported separately, which holds for fleets only.
  - Per-worker `sim.cpu_seconds` in the fleets uses process-wide `RUSAGE_SELF` (`grid_sim.rs` `RealClock`). Never report it per worker. The fleet aggregate (process − observer) is correct.

- **S2, INT8 gate: partial.** `ultra-engine-gate` correctly requires both runs on the same panel and checkpoint, pairs by id, uses the speaker-cluster bootstrap per split, and flags strata with fewer than 20 clusters. Three problems remain:
  - **The upper limit is lax.** It is +0.50 pp. Together with point ≤ 0.10, it still lets a true ≈ +0.2–0.3 pp regression pass a fair share of the time. With ~14k words and a 73-speaker panel, an upper bound ≤ +0.25 pp is achievable for a near-identical INT8. Pick the number on DEV before the tag and state the reasoning.
  - **The tool exits 0 on FAIL (N2).** `engine_gate.rs:196-216` returns `ExitCode::SUCCESS` whenever the comparison computed, including `pass: false`. A script that checks only the exit status treats a failed gate as passed. **Fix:** exit non-zero on `pass == false`.
  - **The continuity gate is gone.** The ≥ 98% same-precision token-identity check from the first review (and the earlier e4 finding) is no longer in FREEZE §5. Restore it as a separate gate, or record why it was dropped.

- **S3, knobs and environment: resolved.**
  - `run_info::refuse_engine_env`, `check_qualifying` and the `identity` block (executable sha, commit, dirty flag, ISA, affinity, every knob) are in place.
  - bench-suite `--qualified` refuses `ULTRA_ENGINE_*`.
  - `bundle_id` now says multirate or dense.
  - Caveat: the dirty flag can be stale (N3).

- **S4, placement: partial.**
  - **Done:** `pin_current_thread_verified` reads the mask back and fails on mismatch, and `verify_worker_l3_and_smt` checks single-L3 workers, SMT collisions and observer disjointness.
  - **Unpinned fallback:** when `phys_aff.len() < workers`, every `worker_cpus` entry stays empty. Verification then skips empty sets (`placement.rs:179-181`) and pinning an empty set is a no-op (`placement.rs:81-83`). The workers run **unpinned and the run passes**. This is the case where someone asks for 4 or 60 workers on a smaller cpuset. **Fix:** fail when `phys_aff.len() < workers`.
  - **Fail-open topology:** topology verification is fail-closed only when an observer is given (`observer_cpus.is_some()` at each call site). Make it always fail-closed under `--qualified`.

- **S5, sustained run: partial.**
  - **Done:** the sustained run now uses `--sustained-workers/--sustained-streams` (default = density settings) and writes `{slug}.sustained.series.jsonl`.
  - **Not done:** the fleet series is the observer's `FleetSeriesPoint`, whose `max_worker_backlog_ms` is hard-coded `None` (`placement.rs:59`). Its `rss_bytes` is the running **peak** from `sampler.summary()`, not the current RSS (`placement.rs:53-58`).
  - The per-worker `SustainedSample` series with backlog exists in `SimOutcome.series` but is never written.
  - **Fix:** record current RSS/PSS, merge each worker's backlog series, and write both.

- **S6, PUBLIC-TEST: partial.**
  - **Done:** disclosure in §4.2.
  - **Not done:** the guard (`suite_cmd.rs:132-140`) only checks that `ULTRA_SEALED_READ` is non-empty, and only under `--qualified`. It appends nothing to LOOKS.jsonl and does not enforce one read.
  - Exploratory bench-suite, `engine_eval --npz test-*.npz` and the trainer's panel path (B5) can all read test with no record.
  - **Fix:** mirror `ood.py`: tag ancestor, look-id, LOOKS append, refusal of a second read for the same (system, split). Apply it to every binary that can load `test-*`, regardless of mode.

- **S7, identities: unresolved (deferred).** FREEZE §1 still has the truncated `3e21c0e4…bff5` and `fb8241b9…`, and no `ultra-score` executable sha. §6 defers them to the freeze commit. Acceptable only if the tag commit is checked for them.

- **S8, extrapolation and clocks: partial.** The §3 text is fixed. Two gaps remain:
  - The bench-suite job manifest does not record the scaling governor or boost state. `inventory.rs:95` reads the governor but it is not in `SuiteHostSnapshot`.
  - The Nemotron implied ceiling (B3) contradicts the no-extrapolation rule.

---

## New issues

- **N1. Non-monotone claimed N.** See B6. It is in code and asserted by a unit test (`suite_ladder.rs:356-369`). **Should-fix before the tag.**
- **N2. `ultra-engine-gate` exits 0 when the gate fails.** See S2. **Should-fix.**
- **N3. The build dirty flag is stale for edits outside `ultra-cli/src`, and the freeze-ancestry check uses run-time HEAD.** See B4. Together these defeat `--qualified`'s clean-build guarantee for the harness code itself. **Should-fix.**
- **N4. With `--panel`, Nemotron and offline rows run the full split.** See B5. Rows in one `frontier.json` silently cover different utterance sets.
- **N5. bench-suite records a panel hash it never verified.** See B5.
- **N6. Any error in one ladder point aborts the whole ladder and writes no `density_ladder.json`.** Examples: a worker failing to drain (`native_engine_fleet.rs:152-157` returns `Err`) or a backend error. All earlier points are lost. Write each point as it completes (append-only JSONL), then the summary.
- **N7. `--full-machine` and `--density-workers` depend on argument order.** See B1. Make them mutually exclusive.
- **N8. The trainer panel loader accepts `test-*` splits.** See B5 and S6.

## Minimum before `bench-freeze-v1`
1. **Cost and lateness text:** settle the Nemotron cost scope (S1 a or b) and correct FREEZE §3's cost and lateness wording to match the code.
2. **Ladder rule:** make claimed N monotone (N1) and pin the ladder bounds and budget.
3. **Gate exit code:** make the gate exit non-zero on FAIL (N2), and fix the gate's upper limit and continuity rule.
4. **Build provenance:** fix the build-dirty and freeze-ancestry checks (N3).
5. **Panel:** apply `--panel` to every row type, and verify the manifest hash in bench-suite (B5, N4, N5).
6. **Placement:** fail closed on `phys_aff.len() != workers`, and always under `--qualified` (S4).
7. **Hypotheses:** list each hypothesis with margin and α, and drop density from Holm (B6).
8. **Test reads:** add a test-read guard to every binary (S6).

Everything else can be labelled in the frontier report.
