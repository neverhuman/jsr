# JSR / ULTRA: where we are, what is left, and the path to a real benchmark (v1.0.0)

## Context

**Goal (owner, 2026-11-20):** a real, meaningful benchmark of our own training and inference against strong CPU recognizers. Same harness, same audio, same scorer, receipts on every number, honest labels. This is the substance of v1.0.0, which the owner defined as "R0 release + R1 measured beside it".

**This plan:**
- records the exact state at `main @ d85b0e9` (2026-11-20 ~18:40Z);
- lists every remaining gap, using the independent Codex review matrix (`docs/ultra/reviews/2026-11-20-efficient-h0-review.md` §"Evidence required");
- orders the work into lanes and days, with acceptance tests.

---

## 0a. Protocol deltas adopted 2026-11-20 (codex-review, accepted by lead)
- **Formal comparison:** the primary formal comparison is **R1 vs R0** only. Every other row is descriptive.
- **Quality margin:** non-inferiority at **+0.10 absolute WER points**, decided by a one-sided paired cluster bootstrap (speakers).
- **Latency margin:** CQ non-inferiority within **0.5 points**.
- **EFFICIENT profile targets:**
  - CPU ≤ 0.25 CPU-s per audio-s;
  - stable-word delay p95 ≤ 2 s;
  - FC delay p95 ≤ 1.5 s;
  - aggregate memory ≤ 1.5 GiB;
  - no accumulating backlog.
- **CPU reproducibility:** a **5% median CPU rerun tolerance** is frozen before confirmation.
- **R1-MR** is excluded from the initial freeze. It starts after the D2 qualifying window and gets its own add-on block.
- **One-hour runs:** two actual one-hour trained-bundle runs are required, one for R0 and one for the selected R1.
- **Fake-clock and unit runs** are contract evidence only, never performance evidence.
- **Writer roles:**

  | Package | Writer |
  |---|---|
  | M1 (Python/ORT CPU bridge for R1 bundles) | codex-review, reviewed by lead |
  | B1/B3 (native Rust backend, bench-suite) | runtime-lane; moves to codex-review if no ACK by 2026-11-20T20:30Z |

## 0. Owner decisions (2026-11-20, this planning round)
1. **Qualifying window:** right after D2 finishes (about 11-22 09:00–14:00Z, GPU idle), before D1MR starts. No D2 interruption.
2. **Word timings:** the public Montreal-Forced-Aligner LibriSpeech word alignments (external, timing only). Download, hash and record them in `DATA.lock` (ruling O2 extended). Log a 20-utterance listening spot check.
3. **Out-of-domain sets:** evaluation-only, never trained on, sealed until the freeze:
   - TED-LIUM 3 test;
   - VoxPopuli EN test;
   - AMI IHM test.

   LibriSpeech test-clean/other stays the primary board, labelled in-corpus for R0/R1. The OOD rows are labelled "out-of-domain for R0/R1; unknown overlap for Nemotron/Moonshine/Whisper".

   Data work (lead, lane C):
   - fetch each set from its official or HF source;
   - record licence lines in `RULINGS.md` (O2 extended) and `DATA.lock`;
   - normalise segmentation per the leaderboard convention;
   - decode to `.npz` caches like `libri.py`;
   - mark them PUBLIC-TEST (sealed).

   DEV-like tuning stays on LibriSpeech dev only; OOD sets get exactly one read, at the qualifying window.

## 1. Where we are (verified state)

### 1.1 Repository and process
- **Repo:** public `neverhuman/jsr`; `main @ d85b0e9`, pushed.
- **Gate:** `just required` is green (fmt, clippy `-D warnings`, Rust tests, 44 pytest, contract drift, security: gitleaks + cargo-audit + pip-audit + actionlint + shellcheck). jankurai is at 85 with 0 caps.
- **Coordination:** `~/jsr/AGENT_CHAT.md` (gitignored), with a core partition and timing leases:

  | Cores | Owner | Use |
  |---|---|---|
  | 0–15 | trainer | GPU training and its loader |
  | 16–31 | nemotron | timing |
  | 32–47 | lead | timing |
  | 48–63 | shared | builds, tests, downloads |

  `BENCH_LEASE.host` plus an announced quiet window is required for qualifying timings.
- **Agents:**

  | Agent | Kind | Role |
  |---|---|---|
  | lead | me | coordination, merges, records |
  | runtime-lane | Antigravity | Rust runtime/CLI; DONE @ 2fa2846 |
  | nemotron-lane | Claude subagent | finishing P04 |
  | trainer-lane | Claude subagent | D2 running |
  | codex-review | independent reviewer | qualification review |

  Finished lanes: eval, kernels, hygiene.

### 1.2 Built and merged
| Area | Path | State |
|---|---|---|
| Output contract (spec v1.0-B §33) | `crates/ultra-core/src/events/{mod,ledger,emitter}.rs` | Atomic Ledger, Emitter, processing evidence on Final, duplicates, epochs, ownership; 18 contract and 6 schema tests |
| Numeric contract, receipts | `crates/ultra-core/src/{numeric,ternary,receipt,repair}.rs` | Hardened `validate_as_evidence` (7 tests); RepairHint generated from `agent/exceptions.toml` |
| Scorer and statistics | `crates/ultra-eval` | Normaliser (hashed), WER S/D/I, macro, CQ(D), stable-word delay, paired cluster bootstrap and Holm, process-tree CPU (`ultra-measure`), paced replay, `ultra-score` |
| Kernels, cost law | `crates/ultra-kernels`, `ultra-bench` | AVX2 int8 / T1 / T-LUT2 bit-exact with scalar. W01 PASS on H0 (T-LUT2 fixed term 0.18× int8) |
| Rust runtime and CLI | `crates/ultra-{model,frontend,stream,cli}` | `ultra inventory|candidate|bundle|doctor|transcribe|stream|bench(sp-a)|score|compare`. Sherpa C-API backend (online Zipformer; offline Moonshine/Whisper), grid scheduler, shared-weight `--workers`, §38 probes, verified reproducible bundle `bundles/r0-zipformer-int8` |
| SP-A harness (Python) | `python/ultra/ultra/sp_a{,_fleet}.py` | Fixed capture deadlines, oldest-unprocessed lateness, grid decode, drain, strict fleet validation, fake-clock tests. Prereg plus amendment A1 (800 ms LIVE bound, no-growth rule) |
| Trainer | `python/ultra/train/` | Streaming Conformer transducer (S dense and S_MR multirate), CTC aux with feasibility masking, own RNN-T loss, async checkpoints, delayed SpecAugment, warm start, streaming infer path (torch, ONNX fp32/int8) equal to trainer eval (400/400) |
| Data | `/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet` | LibriSpeech 960 h + dev/test (HF mirror, SHA-256 manifest). Decoded caches: dev/test `.npz` |

### 1.3 Measured so far (H0, LibriSpeech; all exploratory, shared host)
| System | WER | CPU-s per audio-s | Notes |
|---|---|---|---|
| R0 Zipformer int8 (sherpa, LibriSpeech-only) | 2.51% test-clean (50 utterances); Rust 1.07% dev subset, 2.18% test-clean (100 utterances) | 0.079 single; 0.053–0.062 batched | Paced: 12 streams/core sustained with one worker per CCX, p99 616–655 ms |
| Nemotron 0.6B Q8_0 (NeMo-Speech.cpp) | **2.665% full test-clean @560 ms** | 0.68–0.84 single (contended); about 1.0 batched | Training data has unknown overlap with LibriSpeech |
| Moonshine v2 base / tiny int8 | 2.55% / 4.51% test-clean (100 utterances) | 0.122 / 0.058 | P05, offline decode |
| Whisper small.en int8 | 3.77% (50 utterances) | 1.94 | Offline control |
| **R1-D1** (our S, 100 h, D1d step 7800) | **29.8% dev-clean** (400 utterances) | int8 0.090–0.112 (early checkpoint) | Not converged; in-corpus |
| **R1-D2** (our S, 960 h) | training | — | 394 audio-h/GPU-h; **ETA 2026-11-22T07:00Z**; interim weights every 5k steps |
| R1-MR (S_MR, 100 h) | prepared | ~15% cheaper than dense on random weights | Launches after D2 |

**Key findings:**
1. **Density is L3-bound per Zen 2 CCX:** about 12 live streams per CCX worker, so H0's ceiling is about 190 for this runtime (not yet measured machine-wide).
2. **Batching:** only 1.49× (SP-A primary expects NO_LIFT).
3. **R1 cost is arithmetic-bound** (1.36 GMAC per audio-s for dense S). MULTIRATE (−31% MACs) or our own kernels are the levers.

### 1.4 Open gaps (Codex matrix, updated for what has since merged)
| Gate | Status | What is missing |
|---|---|---|
| Instrument | mostly done | full fleet validation in Rust (M4); crash probe |
| Exact identities | **open** | `CANDIDATES.lock` pins (revision, artifact/tokenizer SHA, runtime commit, flags); `DATA.lock` with manifest hash; scorer and normaliser hashes; settings frozen; LOOKS ledger |
| Runtime | R0 done | **R1 is not in the Rust runtime.** Our model runs only via the Python infer path |
| CPU | open | qualifying whole-service CPU receipts |
| Stable output / CQ | **open** | reference word timings (no alignment panel exists) |
| Finalization / FC | **open** | episode scorer FC(D) (`ultra-eval` lacks FC) |
| Resources | open | whole-tree RSS and a 1 h sustained-run receipt |
| Quality | open | Rust-rescored outputs for every system, frozen margins, slices (silence, long-form, fast speech) |
| Repetitions | open | SP-A 3 alternating blocks; ≥5 paired timing blocks for finalists; 1 h sustained |
| Package | partial | R0 bundle exists; SBOM per bundle, model cards, rollback, second reproduction for the release set |

---

## 2. The benchmark we will run ("ULTRA-Bench v1")

### 2.1 Systems (frozen at the freeze step)
| Id | System | Runtime path | Label |
|---|---|---|---|
| R0 | Zipformer int8 (icefall LibriSpeech) | Rust `ultra` sherpa C-API, streaming 320 ms chunks | in-corpus |
| N160, N560 | Nemotron 0.6B Q8_0 at 160 / 560 ms | NeMo-Speech.cpp process adapter | unknown overlap |
| M-base, M-tiny | Moonshine v2 | sherpa offline (file mode) and bounded live wrapper | unknown overlap |
| W-small | Whisper small.en int8 | offline control | unknown overlap |
| R1-D2 | our S dense, 960 h | **Rust ORT backend (new)**, int8 | in-corpus |
| R1-MR | our S_MR multirate, 100 h (960 h if time allows) | Rust ORT backend, int8 | in-corpus |
| R1-D1 | our S dense, 100 h | same | in-corpus (ablation point) |

### 2.2 Workloads and metrics (spec v1.0-B §10, §12)
1. **Quality:**
   - WER on test-clean and test-other, plus TED-LIUM 3, VoxPopuli EN and AMI IHM test (all utterances) from each system's actual emitted output, re-scored by `ultra-score` (pinned normaliser hash).
   - Paired cluster bootstrap over speakers (≥10,000 replicates) for R1 vs R0 and R1 vs N560, with Holm across the declared family.
   - Slices: long utterances (> 20 s), fast speech (top decile of words per second), test-other.
2. **Cost (single stream):** whole-service process-tree CPU-s per audio-s, unpaced and paced, 1 thread, one pinned core.
3. **Density:**
   - Sustained live streams per CCX worker (paced, grid 320 ms, A1 rule: p99 ≤ 800 ms, no growth, drained).
   - Full-machine sustained streams, one worker per CCX × 16.
4. **Latency:** stable-word delay p50/p95 and CQ(D) for D ∈ {250, 750, 2000} ms against reference word timings; finalization delay and FC(1500 ms); first-partial latency.
5. **Resources:** peak whole-tree RSS / PSS; 1 h sustained stream with no backlog growth and bounded memory.

### 2.3 Protocol
- **DEV** (dev-clean, dev-other) for every choice: N per worker, grid, threads, checkpoint selection.
- **Freeze** settings and identities, then **one read** of PUBLIC-TEST (test-clean, test-other). Every read is logged in `docs/ultra/LOOKS.jsonl`.
- **Qualifying timing:**
  - Held under `BENCH_LEASE.host`.
  - In an announced quiet window: GPU training paused or finished, no USB-heavy jobs, load recorded.
  - Arms run in ABAB order, ≥5 paired blocks for finalists.
- **Labels on every row:**
  - honesty label (in-corpus or unknown overlap);
  - partition;
  - exploratory or qualifying;
  - hardware (H0 = Threadripper 3995WX, AVX2).

### 2.4 Pre-registered claims (written before the test read)
- **C1, SP-A primary:** batched vs single at ≥3× streams per core. Expected NO_LIFT; recorded either way.
- **C2, R1 quality:** R1-D2 test-clean WER within a declared margin of R0, which shares its training data. The margin is set on DEV before the freeze.
- **C3, R1 cost:** R1-MR CPU-s/s ≤ R0 at WER within its margin. This is the "our own model is cheaper" claim.
- **C4, density:** machine-wide sustained live streams for R0 and the best R1, compared against Nemotron's sustained streams on the same machine.
- **C5, out-of-domain:** on TED-LIUM 3, VoxPopuli and AMI, report R1-D2 and R0 WER degradation versus Nemotron. No win is claimed in advance: this row tells the reader how much of an in-corpus result survives domain shift.

---

## 3. What is left: work packages

### Lane A: trainer (Claude subagent; cores 0–15 + GPU)
- **A1. D2 to completion** (ETA 11-22 07:00Z). Evaluate interim `model-<step>.pt` on DEV every ~10k steps; post the WER curve. **Accept:** dev-clean WER falls monotonically (±noise); final checkpoint chosen on DEV loss/WER only.
- **A2. Export R1 to the runtime contract:**
  - **Files:** encoder, decoder and joiner ONNX (int8 via ORT dynamic quantisation), `tokens.txt`/`bpe.model`, a feature spec (sample rate, n_fft, hop, mel bins, log floor, CMVN stats) and streaming metadata (chunk, left context, state tensor names/shapes).
  - **Accept:** Python infer on the exported files equals trainer eval on 400 DEV utterances (all identical, or ≤ 0.1 WER for int8).
- **A3. D1MR** (S_MR, 100 h, ~9 h on the GPU) right after D2 and after the qualifying window. If the schedule allows, **D2MR** (960 h, ~36 h) as a v1.1 candidate.
- **A4.** int8 vs fp32 equality report on the final checkpoints (WER delta, transcript agreement).

### Lane B: runtime (Antigravity; builds on 48–63, timing in the lead block when lent)
- **B1. Rust `OrtTransducerBackend` for R1** in `crates/ultra-stream` (new module).
  - Loads the A2 export through onnxruntime (the `libonnxruntime.so` shipped with the pinned sherpa wheel, loaded like `SherpaLib`, or the `ort` crate).
  - Streaming chunk loop with cached states, batched across streams, greedy transducer with the 8-symbol cap, Emitter events.
  - **Accept:** parity with Python infer on 400 DEV utterances (WER equal within 0.1, ≥ 98% identical transcripts).
- **B2. Rust log-mel matching the trainer.** `ultra-frontend` already has a packet-invariant log-mel; add the trainer's exact parameters. **Accept:** max abs feature diff ≤ 1e-4 against the trainer on 50 utterances; WER parity per B1.
- **B3. `ultra bench-suite`:**
  - Runs the §2.2 matrix for a list of bundles/candidates.
  - Paced and unpaced, single and per-CCX fleet, plus full machine.
  - Writes `ScoreRecord` JSONL, `ultra.receipt/v1` per run, and a merged `frontier.json`.
  - **Accept:** a dry run on DEV for R0 and R1-D1 produces validated receipts; `validate_as_evidence` passes.
- **B4.** Nemotron and Moonshine/Whisper through the same suite via process adapters (reference-blind), so every row comes from one harness.
- **B5.** 1 h sustained-stream probe with RSS and backlog series (exists as a §38 probe; run it on real bundles).

### Lane C: eval / lead (me; cores 32–47)
- **C0. OOD sets** (TED-LIUM 3 test, VoxPopuli EN test, AMI IHM test):
  - Steps: fetch, hash, licence entry, segmentation, `.npz` caches, a loader in `python/ultra/ultra/` plus a Rust-side manifest for `ultra bench-suite`.
  - The sets stay sealed: only manifests and hashes are inspected before the freeze. A planted-read check confirms the loader refuses them without the freeze tag.
- **C1. Reference word timings for CQ/stable-word delay:**
  - Use the public MFA LibriSpeech word alignments (owner decision). They are independent of the systems under test. OOD sets get latency metrics only where reference timings exist; otherwise latency is reported on LibriSpeech only, stated as such.
  - Store as `ultra.word-timing/v1` JSONL on the data drive, with the manifest hash in `DATA.lock`.
  - **Accept:** ≥ 99% of reference words timed; a spot check of 20 utterances by listening is logged.
- **C2. FC(D) and finalization episodes in `ultra-eval`:**
  - Episode = reference segment end → Final event.
  - Implement FC(1500 ms) and conditional p95 finalization delay.
  - **Accept:** unit fixtures, including a missing Final (counts as a miss).
- **C3. Freeze identities:**
  - `CANDIDATES.lock`: every system's repository/revision, artifact and tokenizer SHA-256, runtime commit, compiler flags, licences.
  - `DATA.lock`: parquet manifest hash, decoded-cache hashes, word-timing manifest hash.
  - `docs/ultra/FREEZE.md`: scorer and normaliser hash, settings table (N per worker, grid, threads, chunk), commit.
  - **Accept:** `ultra candidate validate --verify-files` passes for all; a freeze commit is tagged `bench-freeze-v1`.
- **C4. Slice definitions** (from training/DEV metadata only): long-form > 20 s, fast speech, plus a silence probe (insertions per hour on non-speech audio: a 10 min synthetic silence/noise file).
- **C5.** SP-A confirmation (C1 claim) via the Rust suite in the qualifying window; verdict in `python/ultra/waves/SP-A/README.md`.

### Lane D: Nemotron (Claude subagent; cores 16–31). Finishing now
- **D1.** Quiet headline cost; test-other at 160/560 ms; results doc; CANDIDATES pin lines; Codex N1 fixed (reported fixed at the 17:39Z post). DONE, then lead merge.

### Lane E: review (Codex)
- **E1.** Independent review of B1–B3, C1–C3 and the freeze commit before any test read.
- **E2.** Re-run its 17 diagnostics plus new ones for the ORT backend and FC.
- **E3.** Sign the release-evidence matrix.

### Lane F: release (lead, day 3)
- **F1.** Bundles (content-addressed) for R0 and R1-D2/R1-MR, each with a model card, SBOM (`just sbom`), provenance, licences and a second reproduction.
- **F2.** `docs/ultra/FRONTIER_v1.0.0.md` plus a published Artifact page with the frontier chart: WER vs CPU-s/s; density; latency.
- **F3.** CHANGELOG, `git tag v1.0.0`, GitHub release notes containing only measured claims (hardware, workload, WER and delay named).

---

## 4. Schedule (wall clock; GPU is the critical path)

| When | GPU | CPU work (parallel) |
|---|---|---|
| **Now → 11-21 12:00Z** | D2 (interim checkpoints) | B1 / B2 Rust R1 backend and parity on R1-D1 checkpoint; C1 alignments; C2 FC; C3 identity pins; D1 Nemotron DONE; B3 suite dry run on DEV (exploratory) |
| **11-21 12:00Z → 11-22 07:00Z** | D2 finishes | A2 export pipeline proven on an interim D2 checkpoint; DEV tuning of N/grid per system (exploratory, per block); Codex E1 review; freeze draft |
| **11-22 07:00–09:00Z** | idle (D2 done) | A2 export R1-D2 → bundle; DEV checkpoint selection; **freeze commit `bench-freeze-v1`** |
| **11-22 09:00–14:00Z** | **idle: QUALIFYING WINDOW** (host lease) | Full matrix on PUBLIC-TEST, one read: quality for all systems; ≥5 ABAB timing blocks for finalists; per-CCX and full-machine density; 1 h sustained; SP-A confirmation |
| **11-22 14:00Z → 23:00Z** | D1MR (S_MR 100 h) | Frontier draft; model cards; SBOM; Codex E3 |
| **11-23** | (optional D2MR start) | R1-MR export, parity, an add-on qualifying block for R1-MR (same frozen protocol); release v1.0.0 |

Contingency:
- If B1 slips, R1 rows use the Python infer harness, labelled "Python harness" and not mixed into the Rust cost comparison.
- If D2 diverges, R1-D1 is reported and D2 restarts after a diagnosis.

---

## 5. Operating cadence: coordination, git, jankurai (owner instruction)

### Coordination
- **AGENT_CHAT watch:** continuous on `~/jsr/AGENT_CHAT.md`, re-armed every 30 min.
- **Claims:**
  - Every lane posts `CLAIM <paths>` before editing and `DONE <paths> @ <commit>` when done.
  - The lead answers every @lead item in the same hour.
- **Status post:** at least every 2 h. It covers branch heads, D2 step and WER, open blockers, and who holds which cores and leases.
- **Qualifying windows:** announced ≥ 1 h ahead with exact cores, duration and the lanes that must pause. They start only after acks from the trainer, nemotron and runtime lanes, and Codex.
- **Lanes:** the lead never edits files inside another agent's worktree. Cross-lane fixes go in a chat note or a follow-up on main, with the owner informed (as with the `delivery.rs` one-liner).

### Git
- **Commits:** small and frequent on lane branches (several per day), trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Merges:** the lead merges each DONE lane into main after running `just required` locally. Fast-forward where possible, `--no-ff` otherwise. Push with `scripts/push.sh`.
- **Pushes:** main is pushed after every merge or lead commit that passes the gate. Never push red. If main is red before a change, say so in AGENT_CHAT.
- **Lanes:** rebase onto main before posting DONE. Finished lanes are removed with `git worktree remove` once merged (HOST_RULES; nothing is lost).
- **Tags:** `bench-freeze-v1` (the freeze), `v1.0.0` (release). Tags are pushed.

### jankurai
- **Before every commit:** a `just score` (full audit, about 0.1 s). The pre-commit diff-audit runs on every commit.
- **Hook bypasses** (`JANKURAI_SKIP_HOOKS=1`) are allowed only for the recorded HLT-007 schema finding or findings in files another lane owns. The reason goes in the commit message.
- **After every merge:** a full audit. The score must stay ≥ 85 with 0 caps; a drop blocks the push until fixed (as with the fallback-soup fix in ee28f51).
- **Daily snapshot:** `just audit` commits a dated report under `audit/`, with no detector gaming. Any adoption-score change is re-measured with added text removed.
- **Review lane:** Codex re-runs the gate and its diagnostics on each merged head and posts the result.

## 6. Verification (end to end)
1. `just required` green and jankurai ≥ 85 on every merge; Codex diagnostics pass.
2. **Parity:** Rust R1 == Python R1 == trainer eval on 400 DEV utterances (B1, B2, A2).
3. **Freeze:** `ultra candidate validate --verify-files` for every system. `bench-freeze-v1` tag exists before the first `LOOKS.jsonl` entry for PUBLIC-TEST.
4. **Receipts:** every reported number links to an `ultra.receipt/v1` that passes `validate_as_evidence`, plus a `ScoreRecord` file re-scored by `ultra-score`.
5. **Second reproduction:** one finalist row re-run on a different day; numbers within the stated noise.
6. **Report:** `FRONTIER_v1.0.0.md`, where each claim names hardware, workload, WER, delay, label and the qualifying window id.
