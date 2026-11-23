# ULTRA v3: major release plan (2026-11-23)

## Context
At 05:07Z the owner stopped all jobs and asked for a major release built on the external designs in `tips/v3/` and on a deep analysis of our WER errors:
- **Goals:** the lowest WER, the fastest first token and the lowest watts.
- **Training:** world-class across xbabe0–xbabe9 with cross-node weight sharing.
- **Dashboard:** updated.
- **Research waves:** documented in incrementing `research/H*/` folders that keep results but never models.
- **Coordination:** frequent reads and writes of the shared `AGENT_CHAT.md`. Writes resume the moment this plan is approved; planning mode blocks them now.

**What was studied:**
- All 11 `tips/v3` files: one duplicate; `Review and scores.md` scores the nine designs and confirms ten code defects; the two final plans were read in full.
- Research digests of the frontier, breakthrough, always-on and omega specs.
- Runtime-lane's live audit and full-dev error breakdown.
- Reviews from the trainer, eval, bench-open, kd-cache, alwayson-bench and paper lanes.

**The designs agree on an order:** fix the instrument; take the zero-training wins; gate the encoder in silence; compress the LM; add sequence-level training; adopt staged low-bit kernels; spend the saved watts on capacity. Each change is a gated arm with seed pairs.

**Our newest results revise the designs:**
- The delay penalty works. 0.005 is adopted: 242 → 202/206 ms on two seeds with no WER loss.
- Low-latency arms reach 92–105 ms.
- A1 already ran native DDP with bf16 communication, at ~85% per-rank efficiency.
- Our noise spread is 0.087/0.094 pp, not 0.57.

**Owner decisions (05:4xZ):**
1. **Codex** keeps the canonical scorer (H43) and the engine-side correctness fixes (H44). It moves to a fresh worktree on current main and never commits the main checkout's 2,707 staged deletions.
2. **Data:** Libri-Light with R0 labels, LibriSpeech LM text to train the predictor, and MFA alignments in training are all approved. Results are labelled "semi-supervised", "+text" and "MFA-anchored".
3. **Power:** pausing the owner's own scheduled timers on xbabe0 while TIMING.lock is held, a per-thread `uclamp` pacing option, and an owner-supplied metered target device are all approved.

### Where we stand: B0 = A1 step 49,000 (the run's final checkpoint), DEV only, exploratory
| Axis | B0 now | v3 targets (promotion minimum / stretch) |
|---|---|---|
| WER, full dev pooled, INT8, Rust scorer | 8.272% (EMA greedy 8.302; clean 4.48 / other 12.37; 640 ms 7.703; beam 4 8.223) | ≤ 7.86% (−5% rel) / 7.45% (−10%) / ≤ 6.62% (−20%), now plausible with Libri-Light, the commit pass and capacity. "+LM" is a separate row |
| First token, end-of-first-word clock | 258 ms (onset clock 564 ms) | Default mode ≤ 210 ms with no WER loss. Low-latency mode ≤ 133 ms at 80 ms chunks with WER within spread. Both clocks named, plus correct-by-deadline curves |
| Watts per always-on stream | 0.35 W CPU-reported (core register); 0.035 CPU-s per audio-s in speech and silence alike | Silence ≤ 2% of ungated. About half on a 15%-speech stream. Paced with `uclamp` −20–35% J. **Device watts on the owner's metered target** |
| Installed bytes | 284 MB (the INT8 bundle still needs the FP32 file) | ~57 MB self-contained bundle |

### What our errors are (A1-49000, 160 ms, full dev)
**Mix:**
- 8,797 edits: 81% substitutions, 9% deletions, 9% insertions.
- dev-other holds 72% of the substitutions.

**Substitution causes:**
- Function-word swaps, 20.5% (in↔and 197, a↔the 133). The predictor sees only 2 tokens, so an LM or text-trained predictor tables can fix these.
- Near-phonetic, 21.4%.
- Morphology, 14.0%.
- Acoustic (edit distance ≥3), 27.2%. The worst 10% of speakers cause 20.6% of dev-other errors; more and harder data addresses this.

**Where they fall:**
- Utterances under 5 words have 2.6× the WER.
- First words have +15% WER.
- 12% of insertions come before the first word.
- Unstressed function words make up half of deletions and insertions.

**Headroom:**
- 640 ms removes 646 edits (−7.3%, 80% of them substitutions): the commit pass.
- A per-utterance oracle over A1, F1 and H22ctl gives 6.48%. That is not attainable, but it shows headroom for soups, sequence KD and MWER.
- The LM cut WER by 16% relative on H22ctl.
- A 5% gain is about 5× the best recipe tweak we have measured, so v3's WER comes from data, the commit pass, the LM, KD/MWER and capacity.

## Research waves (`research/H43…`, numbers increment; results kept, never models)
| Wave | Folder | Owner(s) |
|---|---|---|
| H43 | `H43_v3_baseline_and_measurement`: canonical scorer and ledger; two latency clocks, correct-by-deadline; statistics; B0 receipt set | **Codex** (scorer / ledger); eval-lane (metrics, statistics, B0); bench-open (timing) |
| H44 | `H44_v3_correctness_fixes`: confirmed defects, each with a failing witness first | **Codex** (engine side); trainer (training side) |
| H45 | `H45_zero_training_wins`: soups / post-hoc EMA, task-arithmetic bundle, commit pass, CTC rescoring / MBR | eval + trainer; runtime-lane (commit pass in the engine) |
| H46 | `H46_lm_and_text`: LM gain on B0, compact LM ladder, text-trained predictor tables (+text) | lm-lane + runtime-lane + trainer |
| H47 | `H47_always_on_sleep_gate`: one-pass replay, oracle gate, skip-gate simulation, Rust gate, standard always-on benchmark, `uclamp` pacing | runtime-lane + alwayson-bench + bench-open / energy |
| H48 | `H48_latency_modes`: A1L default model, low-latency mode, onset peek, MFA-anchored latency loss | trainer + eval |
| H49 | `H49_sequence_training_and_kd`: MWER (canonical scorer), sequence KD on candidate unions, frame KD with the fixed guard | trainer + kd-cache |
| H50 | `H50_fleet_training`: A100 island tuning, sampler stratification, DiLoCo proof → islands, elastic ops, dashboard v3 | trainer + lead |
| H51 | `H51_engine_efficiency`: minimal bundle, bounded output, small-M kernels, staged PQ | runtime-lane + bench-open |
| H52 | `H52_libri_light_semi_supervised`: download, hash, speaker / book overlap check, R0 labels on GPU, confidence filtering, a data-scaling gate | kd-cache + trainer + eval |
| H53 | `H53_v3_trunk_and_capacity`: the v3 trunk (adopted recipe + Libri-Light), then a larger model at fixed gated watts, on the full fleet | trainer |
| H54 | `H54_v3_release_qualification`: final matrix, metered-device watts, quiet timing / power holds, soak, paper, tag `v3.0.0` | all; paper-lane for docs |

**Folder contract:**
- `README.md`: question; pre-registration (hypothesis, arms, seeds, gates, kill rules, compute in R), committed before any run; changes (PRs, commits); protocol; results; decision.
- `configs/`: run configs.
- `results/`: metrics JSON / JSONL, score summaries, latency and power receipts.
- `MANIFEST`: checkpoint and data paths with sha256, never weights.

**CI guard (H44):**
- `.gitignore` adds `*.safetensors`, `*.onnx`, `*.ulm`, `*.bin` and `*.npz` under `research/`.
- A test fails on any file over 5 MB in `research/`.

**Memory note:** save this convention after approval.

## Phase 0: restart safely (first hour after approval)
1. **Kill the frozen leftovers by PID.** Check each command line first, then SIGCONT so the queued SIGTERM lands, or SIGKILL; never by pattern. Do this before anything restarts, because the frozen guards hold stale PGIDs.
   - trainer guards: x7 1116004 / 1116009, x8 853917, x9 745982;
   - H0: gate pgid 2249925, launcher 1887941 and its rsync, mirrors 1874483 / 2184004 / 2427399 / 2427402, burst_watch2 2218710 / 2218714, guard_ceiling 3700905;
   - x7: the frozen arm-21b trainer, pid 1115853, which holds 15 GB of GPU memory.

   Then run `ps` on every host and post the STOPPED lines, including kd-cache's text.
2. **Post the approved plan, the owner's rulings and the lane assignments in `AGENT_CHAT.md`,** including the Codex split. From then on, every lane reads the chat at each step and posts start / result / stop lines.
3. **Preserve unmerged work:**
   - copy `/dev/shm/eval-lane/export/arm{14,19}-ema/` to the data drive;
   - PR the trainer's 4 commits on `lane/trainer-h41d` (keys default to off);
   - PR bench-open's entry-14 receipts and its own `stash@{0}`, leaving `stash@{1}`, which belongs to another lane;
   - fold paper's refresh-7 commits into the v3 branch;
   - PR kd-cache's scipy patch; the low-memory patch only after its 40-min byte check;
   - keep runtime-lane's VNNI draft as a draft;
   - Codex re-bases its work onto current main in its own worktree.
4. **De-duplicate the defect work before anyone writes more.** At 05:39Z runtime-lane opened **PR #72** (`feat/v3-wave0-defect-closure`, +287 −32, 13 files on clean main e61121e). It covers B1–B10:
   - engine: `beam.rs`, `decoder.rs`, `engine.rs`, `model/mod.rs`, `pool.rs`, `native_engine.rs`;
   - training: `data.py`, `kd.py`, `model.py`, `mwer.py`, `trainer.py`;
   - scorer: `ultra/wer.py`;
   - tests: `test_wave0_defects.py`.

   It overlaps both Codex's work (scorer, engine fixes) and the trainer's (training-side fixes plus its lane commits). As the H43 / H44 owner, Codex reviews #72's engine and scorer parts; trainer-lane reviews the training parts against its 4 lane commits. Per defect, the better implementation merges and the other is dropped. Codex then rebases its local work onto the result, in its own worktree on main. One owner per file from then on: runtime-lane takes no further H44 work and moves to H47 / H51.
5. **Pin one commit with a tree-hash check** in every launcher on every host.
6. **Deep error analysis.** The remaining read-only error-analysis agent's report goes into H43's error slices and is posted in the chat when it lands. Its scope: the competitor-win categories on the panel (what R0 / Parakeet get right that we miss), rare words and OOV, word position, and speaker concentration.
7. **xbabe0:**
   - Install the owner-approved hook that stops the owner's scheduled timers while TIMING.lock exists. It waits for any running unit to finish before a hold starts and restarts the timers when the lock is released. It logs every pause.
   - `open_turn.py` honours gap requests, so there is no ssh polling.
   - Re-queue 14b (B0 ×3, ~13 min) and 14d + 24 (low-latency mode, ~8 min). Cut entry 17.

## Phase 1: freeze B0 and fix the instrument (H43, H44; days 1–2)
- **Canonical scoring (Codex, H43).**
  - Rust `ultra-eval` is the authority; `ultra/wer.py` becomes an exact port.
  - A rule-table sha256 parity test plus a shared fixture: the panel must give 857 / 93 / 131 / 14,448.
  - Monitor, MWER and trainer numbers use the port. Old Python numbers are re-scored, not mixed in.
- **Metrics (eval-lane, H43).**
  - Onset and end-of-first-word clocks, from MFA times for all 5,567 dev utterances.
  - First-correct-word with coverage; correct-by-deadline at 150 / 300 / 500 ms, provisional and stable; revision burden; endpoint delay.
  - Scored on full dev with runtime-lane's unpaced trace mode. Paced product claims come only from quiet xbabe0 or the owner's metered device.
- **Statistics (H43).**
  - Seed pairs (1234 / 4321) compared against the comparator's pair; both seeds must agree.
  - Holm correction over ≤4 arms per wave, or select on half the dev speakers and confirm on the other half.
  - Fix non-inferiority margins before looking; report INCONCLUSIVE when unresolved.
  - Adoption needs a gain above 2× the four-run spread (0.17 pp).
  - From now on, screen bundles with a 2^(7−4) fractional factorial plus fold-over, so real 0.1-pp effects are not missed.
- **B0 receipt set (H43).** Read-only manifest:
  - Have: full dev at 160 / 320 / 640 ms, beam 4, INT8.
  - Add:
    - quiet timing (14b);
    - core-register watts joined from the power log;
    - both clocks on full dev;
    - A1-49000 on the 27.56 h stream, with no reset and with reset;
    - the 10 h of non-speech;
    - error slices.
- **Defect fixes (H44).** Small PRs, each with a failing witness first; all cross-reviewed.
  - *Codex, engine side:*
    - 7-bit blank certificate off;
    - beam reset clears predictor and LM context (`beam.rs:69,111`);
    - final-text prefix check plus a revision event (`native_engine.rs:365-374`);
    - bounded incremental detokenisation (`engine.rs:631`);
    - remove the `pool.rs` spin-wait;
    - a forced endpoint close never emits text;
    - minimal self-contained INT8 bundle, hashed once (284 → ~57 MB; runtime-lane reviews);
    - Python beam and Rust predictor context documented as fixed at 2.
  - *Trainer, training side:*
    - KD guard: tokenizer hash, `verify_emitted_path(emitted, ex.rnnt)`, sample-count match, exact speed lookup, and a cache metadata side file (the existing 126 files stay valid);
    - chunk legality (`model.py:341,351`);
    - an explicit offline flag;
    - global normalisation of z-loss, CR-CTC KL and balancer;
    - consistent stream-context shift across heads.
  - *Order:* Codex's engine fixes merge before runtime-lane's gate (H47) touches `engine.rs`.

## Phase 2: zero- and low-training wins (H45, H46; days 1–4, in parallel)
- **H45 averaging.**
  - Soups and post-hoc EMA within A1's basin (snapshots 40k–49k, raw and EMA, plus A1L). Add `--key ema` to `average.py`; `diloco.greedy_model_soup` exists.
  - Select on half the speakers, confirm on the other half.
- **H45 task-arithmetic bundle.**
  - Arms 16 / 17 / 18 plus full CR-CTC.
  - First check that control + τ10 + τ11 reproduces arm 12. Then one seed-pair confirmation.
  - Caveats: pred_ctx 4 needs Rust support, and left 256 doubles the K/V cache.
- **H45 commit pass**, the top measured WER lever.
  - At an endpoint, the same weights re-decode the segment at chunk 16, with the LM and optional CTC rescoring + MBR. "Final" rows are reported separately from provisional text.
  - Probe on existing dumps first (hours), then runtime-lane's engine path (~2 weeks).
  - Measure final-text latency and the extra CPU, paid only at endpoints and funded by the sleep gate.
- **H46 LM.**
  - Re-measure the full 5-gram + LODR gain on B0 with held-out weights (H22ctl's was −16% rel).
  - Require 730 / 730 identical outputs between v1 and v2 packs.
  - Byte ladder at 0.5 / 2 / 8 MiB: pruned n-gram vs a small neural LM vs a residual student. Promote at ≥70% of the gain, ≤5% CPU and ≤8 MiB.
  - **+text:** hashed 2–4-token predictor tables (zero-init projection) trained with a text-only loss on the LibriSpeech LM corpus, aimed at the 20% function-word swaps. Rust predictor-context support is part of this item.

## Phase 3: always-on power (H47; days 1–7)
- **One-pass replay (quick win).**
  - Today's replay re-encodes one chunk per call (`engine.rs:562-625`), so a 1 s replay streams the weights 6 times. Use `with_max_chunks` to replay in one batched pass (~7–9 ms instead of ~35 ms), with a one-call-equals-chunked test.
  - First confirm on quiet xbabe0 that today's reset policy costs ~1.6× speech CPU in silence (15-min assay).
- **Oracle speech gate (offline):** bounds the savings and the restart damage.
- **Runtime-lane's skip-gate simulation** (pre-registered):
  - wake before the 1 s ring overflows on ≥99.8% of 5,567 onsets;
  - stream WER within noise with the skip emulated in the real engine;
  - CPU measured on xbabe0.
- **Rust gate behind a flag.**
  - Ring ownership upstream of `accept_i16`; states Listening / Waking / Active / Draining / Recovery.
  - Each segment exactly equals an isolated decode (`EngineStream::restart`, `ultra-alwayson-check`).
  - Batched wake replay; an ungated twin for every comparison.
  - The designs' warm-pause and suppressed-emission variants are left out of v3.0.
- **Capture and pacing.**
  - One wake per chunk instead of per packet (50 → 6.25 wakes/s).
  - Sleep-depth polling.
  - An engine option `--uclamp-max` (per-thread `sched_setattr`, owner-approved), measured paced and unpaced, for the energy and latency trade-off.
- **Standard always-on benchmark** (`gen_ao_bench.py`):
  - duty 0 / 5 / 20 / 50 / 100% × onset rate 1 or 6 per minute × silence / pink / white backgrounds;
  - torture sets T1–T4 (fricative and quiet onsets, after noise, at −20 / −40 dB);
  - 30 h of non-speech;
  - first-word recall after sleep;
  - CPU on speech vs non-speech, on xbabe0 only;
  - memory slope under live-caption reading.
- **Gates:**
  - non-speech CPU ≤ 2% of ungated;
  - ≥ 2× lower incremental energy at 5% duty;
  - first-word recall after sleep ≥ continuous − 0.5 pp;
  - false words ≤ 0.1/h at the upper bound;
  - gate overhead ≤ 5% at 100% speech.
  - Our differentiator is first-word recall at equal silence CPU, because Moonshine already sleeps at 0.01 CPU-s per audio-s.
- **Finish the competitor always-on rows** (~77 CPU-h on xbabe1 / xbabe2 under the trainer-throughput rule).

## Phase 4: training waves (H48, H49; week 1)
R = one 8k-step replay at a 650 s batch on one A100 (≈33 min, ≈1,340 audio-h).

| Step | What | Gate | Compute |
|---|---|---|---|
| H48a | Resume A1L from step 5,000 (B0 + dp 0.005) | Within spread of B0 at 4 / 8 / 16; first token reported | 0.375R |
| H48b | Low-latency fine-tune from A1L: chunks {2,4,8,16} + dp 0.01, or 0.0075 if the guard fails; seed pair; arms 20 / 20-r's verdict first | ≤ 133 ms end clock at 80 ms; WER at 4 / 8 / 16 within spread; otherwise ship a separate low-latency model, stated plainly | 2R |
| H48c | MFA-anchored latency loss (owner-approved): MFA on train, then an anchored hinge or valid-path tilted loss on word ends, on top of dp 0.005 | ≤ 133 ms at 80 ms with no WER loss beyond spread | MFA ≈ CPU-hours; 2R |
| H48d | Speculative onset peek (engine): run 1–3 frames ahead for a preview replaced by the committed step | −40 to −60 ms at 160 ms; ≤ 25% retracted; committed text unchanged | days |
| H49a | MWER with the canonical scorer and word-edit counts, against a continuation control | > 2× spread, both seeds | 3R |
| H49b | Sequence KD on candidate unions: R0 N-best + student N-best + reference, ≤ 12 per utterance, force-scored. Needs a new R0 N-best / forced-score script on GPU (< 1 A100-h for 960 h); 20k-utterance pilot with an oracle-headroom report first | Stop below 1 spread of headroom; otherwise > 2× spread | 4R |
| H49c | Frame KD with the fixed guard and the full 0.9 / 1.0 / 1.1 cache (R0 fp32 on GPU, 1–2 A100-h; H22 amended first). Low prior | > 2× spread; first token no worse than +20 ms | 4R |

## Phase 5: Libri-Light semi-supervised data (H52; week 1 → 2)
- **Get and lock the data.** Download Libri-Light `small` (577 h) and `medium` (~5.2k h, ~320 GB) to the data drive. Hash each archive into `docs/ultra/CANDIDATES.lock`; files stay untrusted until hashed.
- **Overlap check before any use:** verify that no Libri-Light speaker or book overlaps LibriSpeech dev or test, using the official exclusion lists plus our own speaker and book-id check. Log it in `LOOKS.jsonl`. The test sets stay sealed.
- **Segment:** cut utterances from Libri-Light's VAD segments.
- **Label with R0, fp32 batched on A100s.** This is the same R0 GPU script as H49, run on idle A100s before the trunk starts; about 5–6 A100-h for medium.
- **Filter by confidence:** R0 posterior and blank margin, agreement with A1 / A1L, and length limits. Keep per-utterance confidence for loss weighting.
- **Data-scaling gate.** A matched-compute replay comparing 960 h vs 960 h + medium pseudo-labels (seed pair). Proceed only on a gain > 2× spread.
- **Then `large`.** Take a large subset (10–20k h first; the full ~52k h, ~3 TB, does not fit most hosts) to the A100 hosts' local disks: xbabe7 834 GB, xbabe5 777 GB and xbabe6 1.2 TB free.
- **Labels and loaders.** Results carry the "semi-supervised (R0 labels)" label. Loaders read NVMe-local shards, never the USB drive.

## Phase 6: fleet training architecture (H50; week 1, in parallel)
- **A100 island (6 GPUs: x5 ×2, x7 ×2 under the caps, x8, x9).**
  - Keep native DDP + bf16 comm + global batches.
  - Tune NCCL sockets, checked with nccl-tests (x5↔x8 is 23.5 Gbit/s, 0.2 ms).
  - Auto-solve `ddp_rank_weights` from heartbeat step times every 500 steps.
  - Stratify chunk mode and length bucket across ranks; add gradient-noise-scale telemetry.
  - Elastic torchrun; async checkpoints every 500 steps; world-size-invariant sampler; drop a straggler slower than 1.5× the median for 5 min.
  - Never SIGSTOP a rank.
  - Target ~14.5k audio-h/h at the 0.87 cap (A1 ran 11.1k at world 5).
- **DiLoCo islands:** 4090 (xbabe1), 3090 (xbabe2), 3090 (H0), A10 (xbabe6, once the other project's CPU load clears).
  - Wire the existing `diloco.py` and the dev-gated `ops/diloco_sync.py`.
  - Coordinator on H0. A bf16 pseudo-gradient (~116 MB) every H = 250–500 steps, sized to 10–15-min rounds.
  - Token-weighted outer Nesterov (lr 0.7, momentum 0.9), k-of-n quorum, staleness weight, cosine outlier guard; promote only on a dev check.
  - Expected +2.6 A100-equivalents, ~+25–30% effective.
- **Proof before production (pre-registered):**
  1. Measure on-prem↔cloud bandwidth and latency (iperf3, nccl-tests) and set up a relay; host names do not resolve between sites today.
  2. Lookahead (`lookahead_k`, DiLoCo's single-worker case).
  3. D0 (2-A100 sync) vs D1 (2 single-A100 islands, H = 250), equal compute, seed replicate. D1 must be non-inferior within the spread.
  4. D2 adds a 3090 island and measures time to a target WER.
  5. Islands join production only after D1 passes.
- **Outside the gradient path:**
  - xbabe0: the quiet timing and power host; no GPU.
  - The Mac: NEON engine verification and CPU eval.
  - Both run CPU eval and verification.
- **CPU lanes on training hosts:** the trainer's audio-h/h wins. Ramp in quarters and back off if the trainer slows by more than 10%. H0's memory bandwidth is already contended.

## Phase 7: engine speed (H51; weeks 1–3)
In order of value per effort:
1. Minimal bundle and single hashing (H44).
2. Bounded output: ~8% CPU after 24 h at chunk 4, ~30% at chunk 1.
3. Staged PQ / pqscan, the only lever above 1.2× on Zen 2. 1–2-bit codes (7–14 MB) fit in L3, which is decisive for the low-latency mode.
   - Real-shape microbenchmarks including the table build.
   - TRAIN activation traces.
   - Per-layer sensitivity.
   - 2-bit QAT, one layer group at a time.
   - Native integration with cross-ISA tests.
   - Quiet xbabe0 comparison.
   - Stop if the table build or quality repair erases the gain.
4. Small-M kernels (≤10% at chunk 4, 1.2–1.5× at chunk 1).
5. VNNI / ARM i8mm, only for the owner's target device.

Deprioritised (memory-bound): Winograd / multiply-saving tiles, shape-specialised kernels, new front-end features, DeltaNet / TDT rewrites.

## Phase 8: v3 trunk and capacity (H53; weeks 2–4)
- **v3 trunk.** B0's architecture + dp 0.005 + every H45 / H46 / H48 / H49 winner, trained on 960 h + the Libri-Light pseudo-labels that passed H52. It runs on the full fleet: the A100 island, plus DiLoCo islands if proven. The run's final checkpoint becomes the new release candidate.
- **Capacity model.** Once H47 measures the gated duty savings, train a larger dense model (or an S_MR-v2 class with enough high-rate detail) sized so that gated average CPU ≤ B0 ungated, with an active real-time margin ≥ 10×.
  - Engine support (export, bundle, cross-ISA identity) is verified first.
  - About 115–120R plus the Libri-Light exposure.
  - Gate: better than the v3 trunk by > 2× spread at equal gated watts and within the first-token budget.

## Phase 9: release v3.0.0 (H54)
- **Final matrix:** B0 continuous; B0 + sleep; new weights continuous; new weights + sleep; + compact LM; + PQ / exact kernels. Each against original B0 and its own control; effects are not added across rows.
- **Qualification:**
  - quiet xbabe0 timing and power holds with the owner's timers paused (idle 3 / X 10 / Y 10 / idle 3 min, ABBA, 3 holds per pair; the gated matrix ≈ 15 h overnight);
  - **whole-device watts on the owner's metered target device**, by duty level;
  - a 24 h paced soak;
  - 30 h of non-speech;
  - competitor rows;
  - all on one artifact.
- **Docs (paper-lane):**
  - Tag `v3.0.0` (annotated); a GitHub release with the PDF, hero, manifests and checksums.
  - Fig. 1 with three of our points (standard, low-latency, +LM), each with same-run watts.
  - An always-on panel: device watts by duty, false words per hour with bounds, competitors.
  - Abstract order: cost, correct-by-deadline on both clocks, gated always-on energy, the silence bound, the endless stream, the honest accuracy line, +LM, protocol, data labels.
- **Claims to correct now:**
  - "0.35 W" is core-register incremental energy;
  - installed bytes are 284 MB until the bundle lands;
  - drop the memory-bus sentence;
  - every latency figure names its clock;
  - the silence result carries its ~0.3 per hour bound;
  - the LM row leads with its held-out gain.

## Dashboard v3 (`python/ultra/tools/fleet_dashboard/`; PR in week 1, extended as waves land)
- **Fleet:**
  - all 10 hosts, including other projects' load, labelled as such;
  - per-rank compute, comm and throttle time;
  - island topology;
  - global audio-h/h for the A100 island plus the islands.
- **DiLoCo panel:** round, workers, delta L2 norm, cosine agreement, staleness and quorum, accepted / rejected with the dev WER change.
- **Runs and gates:**
  - per run, the WER curve with S / D / I (canonical scorer);
  - latency on both clocks;
  - each wave's arms with PASS / FAIL / INCONCLUSIVE.
- **Operations:**
  - Libri-Light labelling progress;
  - cap compliance (GPU 5-min mean ≤ 90%, CPU ≤ 80–90%);
  - CPU-lane throughput-guard state;
  - the xbabe0 TIMING queue and lock owner, and the timer-pause state.

## Verification
- **Waves:** each `research/H*` pre-registration is committed before its runs; results go in `results/` with hashes; CI rejects model files and files over 5 MB in `research/`.
- **Pushes:** `just required` before every push; audit ≥ 85 with 0 caps; pytest-light in CI.
- **Defects and scoring:** every defect fix has a witness test that fails before and passes after; the scorer parity test (857 / 93 / 131 / 14,448) is green.
- **Baseline and data:** the B0 manifest reproduces; the Libri-Light overlap check is logged before first use.
- **Fleet and gate:** the D0 / D1 DiLoCo proof is posted before any island trains; the sleep gate passes its pre-registered criteria against an ungated twin.
- **Release:** the final matrix on one artifact; paper tests (MANIFEST committed, no `.log` files, Fig. 1 on page 1); the dashboard shows every host busy within the caps.

## Timeline (estimates; recalibrated from measured throughput)
| When | What |
|---|---|
| Day 0 | Phase 0; Codex H43 scorer + H44 engine fixes start; trainer H44 fixes; H45 soups / commit-pass probe; H46 LM re-measure; H47 one-pass replay + oracle gate + simulation; H50 network measurement, Lookahead, D0 / D1; A1L resume; 14b / 14d timing; Libri-Light download starts |
| Days 1–3 | Defects merged; B0 receipts frozen; soups / commit-pass / LM / bundle verdicts; low-latency fine-tune; R0 GPU script (KD caches, sequence-KD pilot, Libri-Light labels); MFA on train; sleep-gate prototype |
| Days 3–7 | MWER / KD / MFA-latency arms; Libri-Light data-scaling gate; DiLoCo islands if proven; minimal bundle merged; PQ microbenchmarks; dashboard v3 |
| Week 2 | v3 trunk on the full fleet; sleep gate qualified; compact LM / +text tables |
| Weeks 2–4 | Capacity model; PQ integration; metered-device qualification; v3.0.0 |
