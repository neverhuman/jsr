# Independent freeze review, pass 3 (2026-11-21)

**Reviewer:** the same fresh-context reviewer as the first review and the re-review.
**Code reviewed:** `origin/main` at **b29080c**. That merge includes runtime-lane e8ee325, engine 32967c6 (merged as 6a845a9) and FREEZE revision 2 (e0da750). I also read the trainer fix 02755be, which is still on `lane/trainer` and not yet on main.
**Method:**
- read-only, with files extracted via `git archive origin/main`;
- line numbers refer to `origin/main`;
- nothing was built or run.

Items fully resolved in pass 2 (B3, B7, S3) are not repeated here.

## Verdict table

| Item | Pass 2 | **Pass 3** | Evidence / remaining gap |
|---|---|---|---|
| **N1** non-monotone claimed N | new | **Resolved** | `suite_ladder.rs:126-143` breaks at the first failed rung, and the unit test is inverted (`:77`). The candidate stops early after a failure. |
| **N2** gate exit 0 on FAIL | new | **Resolved** | `engine_gate.rs:207-221`: exit 1 on FAIL, 2 on error. |
| **N3** stale dirty flag | new | **Resolved** | `ultra-cli/build.rs` watches every `crates/*` directory (Cargo scans directories recursively), plus `index`, `HEAD`, `logs/HEAD` and the workspace manifests. `ultra-engine/build.rs:37-41` does the same. |
| **N4** panel ignored by Nemotron/offline | new | **Resolved** | Offline rows now take `selected_indices` (`suite_runners.rs` diff; `contrast.rs` `load_selected_audio`). Nemotron refuses `--panel`. Minor: it refuses only when its turn comes in the candidate loop, so earlier rows are computed and then lost, and no `frontier.json` is written. |
| **N5** unverified panel hash | new | **Resolved** | `suite_panel.rs` delegates to `ultra_engine::panel::Panel::read` (canonical recompute) and refuses non-`dev-*` splits. |
| **N6** ladder aborts on error | new | **Resolved** | Each point, errors included, is appended to `density_ladder.points.jsonl`. An error counts as a failed rung (`suite_ladder.rs:214-244`). |
| **N7** flag-order dependence | new | **Resolved** | `suite_cmd.rs:127-134` makes the two flags mutually exclusive. |
| **B1** one worker per core | mostly | **Mostly resolved** | `assign_worker_cpus` gives exactly one physical core per worker and refuses too few cores (`placement.rs:87-116`). `--qualified` requires an exact 1:1 count (`suite_cmd.rs:176-183`). **Gap N9:** the 4 workers of a "per-CCX" run are not required to share one L3. |
| **B2** lateness matches FREEZE | partial | **Partial** | The code now stamps every decode step's own completion time, but still measures against the **oldest unprocessed packet of the tick** (`grid_sim.rs:331,342,352`), not "arrival of the chunk's last required sample + processing budget" (FREEZE.md:50). FREEZE.md:43 still says "2-chunk steps **for every system**". This is conservative, but the pre-registered text still does not describe the code. |
| **B4** tag ancestry and pins | partial | **Partial** | Ancestry is now checked against the binary's build commit, and the build commit must equal HEAD (`suite_manifest.rs:233-271`), so that part is resolved. The `CANDIDATES.lock` check (`:275-309`) only re-hashes the in-repo YAML manifests, which the clean-build check already covers. It requires just **one** match. It does not verify that the bundles and weights loaded in the run are the pinned ones. R1-dense and R1-MR have **no pins at all** (`CANDIDATES.lock:34-36`). |
| **B5** panel everywhere | partial | **Resolved on main for bench-suite and engine; trainer pending merge** | 02755be `evaluate.py:26-61` recomputes the manifest hash, refuses non-`dev-*` splits and checks the cache sha and every uid. **Not on main yet.** It records only the panel basename, not `manifest_sha256` (`:112`). |
| **B6** pre-registration | partial | **Partial** | Fixed ladder (`4:48:4`, 120 s rungs, 3 repeats, stop at first failure, 3 × 300 s confirmation), a Holm family of tested claims, and a G1 rule are now in FREEZE §4.4/§5. Remaining gaps: (a) `--qualified` does not force the preset; (b) a failed confirmation drops one rung without testing it; (c) step-4 granularity biases ratios at low N; (d) a top-of-ladder pass is not flagged as censored; (e) the C-cost test cannot be decided as specified. Details below. |
| **S1** Nemotron cost scope | unresolved | **Resolved by exclusion** | FREEZE §3/§5 exclude Nemotron from C-cost; its row keeps `post_prep_process_tree_service_cold_load` (`suite_runners.rs:82-90`) as context only. ABAB refuses non-in-process candidates (`suite_abab.rs:259-261`). README marks the 0.472 with † for a different scope, which is acceptable. |
| **S2** INT8 gate | partial | **Partial (decision, not bug)** | Exit code fixed. The upper limit is still +0.50 pp, and the ≥ 98% same-precision token-continuity gate is still absent from FREEZE §5. Both are owner decisions; record them as such. |
| **S4** placement fail-closed | partial | **Resolved** (except N9) | No unpinned fallback; `verify_worker_l3_and_smt(…, true)` always fails closed (`placement.rs:114`); all three fleets use `assign_worker_cpus` (`fleet_inproc.rs:124`, `ort_r1_fleet.rs:83`, `native_engine_fleet.rs:49`). |
| **S5** sustained series | partial | **Resolved** | Current RSS, PSS and thread count, plus the real maximum worker backlog per 10 s interval (`placement.rs:15-80`), wired into all three fleets. Note N11 on observer intrusiveness. |
| **S6** PUBLIC-TEST reads | partial | **Partial** | `suite_looks.rs` requires `ULTRA_SEALED_READ` in every mode, refuses a second (candidate, split, tag) read, and appends to LOOKS before inference. Holes are listed below. |
| **S7** identities | deferred | **Unresolved (deferred)** | FREEZE §1 is still truncated; §6 now lists the scorer binary hash. It must be checked at the tag commit. |
| **S8** governor and boost | partial | **Resolved** | Recorded in the host snapshot and job manifest (`suite_manifest.rs:41-88`). The Nemotron implied ceiling has been removed from FREEZE §4.4. |
| ABAB runner | missing | **Partial** | `suite_abab.rs` exists. It does not compute the pre-registered test (N10). |

**Count:** 13 resolved, 6 partial, 1 deferred. New issues: N9, N10, N11, and N12 (doc consistency).

---

## Details on the partial items

### B2: the lateness text still does not describe the code
- **What the code measures:** `grid_sim.rs:331` sets `oldest = deadline_s(k_dec)`, the deadline of the first packet not yet offered at this tick. Each decode step's lateness is `step_done_t − oldest` (`:342-352`).
- **What FREEZE says:** FREEZE.md:50 describes a per-chunk capture deadline plus an undefined "processing budget".
- **Direction:** oldest-packet lateness is never smaller than chunk-ready lateness, so the code is the conservative choice and the same for every system.
- **Fix (text only):**
  - Replace FREEZE.md:50-51 with "lateness of a decode step = its completion time − capture deadline of the oldest packet not yet decoded at that grid tick; one sample per stream advanced".
  - Replace FREEZE.md:43 with "320 ms grid; every ready chunk is decoded each tick; ultra-engine fuses up to 2 chunks per weight pass, ORT and sherpa decode them one call per chunk".
  - FREEZE.md:55 ("Run length 300 s per step") also contradicts §4.4 (120 s rungs and 300 s confirmation).

### B4: the identity pin check verifies the wrong things
`verify_candidates_lock_pins` passes if any one `manifest:` / `manifest_sha256:` pair matches. It hashes YAML files inside the repo, which the build-commit check already covers, and it never inspects the artifacts the run loads.
- **Fix:** in `--qualified`, look up each requested candidate in `CANDIDATES.lock`, refuse candidates with no entry, and compare the loaded identity with the pinned values: bundle `manifest_sha256` for ORT, `weights.int8.safetensors` sha plus `arch.json` sha for the engine, model file shas for sherpa. The engine finalists need entries before the tag.

### B6: what remains open in pre-registration
- **(a) `--qualified` does not force the preset.** `--density-ladder 4:48:2`, `--density-repeats 1` and `--ladder-confirm-seconds 0` are all accepted under `--qualified` (`suite_cmd.rs:86-89,154-166,226`), and the run is still labelled "qualified". **Fix:** under `--qualified`, require `--density-ladder freeze`, repeats = 3 and confirmation = 300 s.
- **(b) A failed confirmation drops one rung without testing it.** `suite_ladder.rs:322-327` drops to N − 4, which then rests only on 120 s runs. FREEZE §4.4 wrote this rule, so it is pre-registered. It is still weaker than confirming the lower rung. **Fix:** confirm the dropped rung too, or label the claim "confirmation failed at N; N − 4 unconfirmed".
- **(c) Step 4 is coarse at low N.**
  - R0 sits around 12 streams per worker. With step 4 the reported value is the bottom of a bracket up to 4 wide, so it can understate true capacity by up to about 25–30% at R0's level and about 10% at the engine's (around 34).
  - The density *ratio* therefore inherits a bias that depends on where each system's knee falls relative to the grid.
  - **Fix:** report each claim as a bracket [N, N + 4) and compute the ratio from both ends. Alternatively, pre-register a finer step below 24.
- **(d) A pass at the top rung is not flagged as censored.** If a system passes 48, the claim is "≥ 48", but `CandidateLadderResult` has no flag for that. **Fix:** add `censored_at_top: bool` and print "≥".
- **(e) The C-cost test cannot be decided as specified (N10).** FREEZE §5 pre-registers a one-sided paired bootstrap over 5 ABAB blocks at α = 0.05 inside a Holm family of 4. See N10.
- **Time budget:** expect about 3 h per scale for the ladder with confirmation. That assumes early stops at roughly R0 16, ORT 8, dense 28 and MR 36. Two scales, the two 1 h sustained runs, ABAB blocks and the quality reads all have to fit the window. Rehearse the full timing on xbabe2.

### S6: holes in the PUBLIC-TEST guard
1. **No freeze-tag check before the freeze.** In exploratory mode nothing checks that `bench-freeze-v1` exists. A pre-freeze test read is allowed. It is logged under the tag string, which then **blocks the real post-freeze read** of that (candidate, split). **Fix:** require the tag to exist and be an ancestor of the build commit for any PUBLIC-TEST read, in every mode.
2. **The ledger location is unanchored.** The LOOKS path is relative to the current directory and overridable with `ULTRA_LOOKS_PATH` (`suite_looks.rs:32-39`). Running from another directory, or setting the variable, creates a fresh ledger and bypasses the one-read rule. **Fix:** resolve the path from the git top-level and refuse the override in `--qualified`.
3. **Aliases bypass the one-read rule.** The key is the raw candidate string, so `nemotron-560` and `nemotron_en_q8_live_560` count as different candidates. **Fix:** key on the resolved loaded identity.
4. **A failed run still burns the read.** The read is logged before the out-dir, npz, panel and Nemotron-with-panel checks run (`suite_cmd.rs:190-197`). A run that fails afterwards consumes the one read. Probably acceptable as conservative, but write a matching `aborted` record.
5. **Two other entry points read test with no guard.** `ultra-engine-eval --npz …/test-clean.npz` and `python -m train.evaluate --split test-clean` both read test unguarded. **Fix:** refuse `test-*` in both unless the same guard passes.

---

## New issues (pass 3)

- **N9. Per-CCX co-location is not enforced.**
  - FREEZE §3 defines the density unit as 4 workers on **one CCX's** 4 cores.
  - `assign_worker_cpus` checks one core per worker, SMT collisions and observer disjointness. It never checks that the 4 workers share one L3 domain.
  - Under `--qualified`, `taskset -c 0,4,8,12` with 4 workers passes, which gives each worker its own 16 MiB L3. On a workload the project itself describes as L3-bound, that **overstates per-CCX density**.
  - **Fix:** for per-CCX runs, require every worker's `cache/index3/shared_cpu_list` to be identical. For `--full-machine`, require exactly 4 workers per L3 domain across 15 domains.
- **N10. The pre-registered C-cost test cannot be computed or decided.**
  - `suite_abab.rs:89-150` reports a two-sided 95% percentile CI of the **median** paired difference. It reports no one-sided p-value, so Holm cannot be applied.
  - With n = 5 blocks, the bootstrap is anti-conservative. If all five differences share a sign, every resample does too, and the "p" collapses to about 1/10,001 whatever the block-to-block noise.
  - An exact sign test on 5 pairs has a minimum one-sided p of 1/32 ≈ 0.031. That can never pass the first Holm step (0.05 / 4 = 0.0125).
  - **Fix:** pre-register ≥ 8 blocks with an exact sign test (minimum p = 1/256), or a paired t-test on log CPU ratios with n ≥ 5. Implement the p-value in `abab_timing.json`.
  - Also: the baseline is simply `candidates[0]` (`suite_abab.rs:304-312`). **Fix:** require R0 to be named explicitly, or resolve it by id.
- **N11. The observer may perturb the service.** Every 25 ms it reads `smaps_rollup` for PSS across the whole tree (`procfs.rs:237`) during 1 h runs. That takes the service process's mmap lock. The CPU cost is on the observer CCX, but the lock contention lands on the service. **Fix:** sample PSS every 1–10 s and RSS (from `status`) at 25 ms. Alternatively, show with an A/A run that the overhead is negligible.
- **N12. FREEZE has internal inconsistencies to clean up before the tag:**
  - §3 lines 43, 50 and 55 versus §4.4 (see B2);
  - §5 lists C-density/C-cost/C-quality twice, once in the Holm block and again in the older bullet list;
  - §6 still lists the re-review items as open.

## Minimum before `bench-freeze-v1`
1. N9 (L3 co-location for per-CCX runs) and B6(a) (`--qualified` forces the frozen ladder, repeats and confirmation).
2. N10: fix the C-cost test (block count and test) and emit a p-value.
3. B4: pin and verify the actual loaded artifacts for every finalist, R1-dense and R1-MR included.
4. S6 items 1 and 5: tag check for all PUBLIC-TEST reads, and guards in engine_eval and the trainer.
5. The FREEZE text fixes (B2 and N12), and merging 02755be.

Everything else (B6 b–d, S2 thresholds, N11, S6 items 2–4) can be fixed or explicitly labelled in the frontier report.
