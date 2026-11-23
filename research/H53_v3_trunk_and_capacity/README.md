# H53: the v3 trunk and the capacity model

**Status:** PRE-REGISTERED by h53-trunk, written 2026-11-23 06:47-07:27Z, before any H53 run, H53 decode or H53 result existed (P1's seeds were made and checked for exactness at 07:20Z; no P1 arm has started). claude-lead lands it as `research/H53_v3_trunk_and_capacity/README.md`. Amendments are dated and written before the runs they affect. Every number is **exploratory, DEV only**; the test sets stay sealed. Weights are never committed; `MANIFEST` lists paths and sha256 only.

**Owners:** h53-trunk writes specs, configs, checks and the depth-growth tool. **trainer-lane launches every run.** h50-fleet owns NCCL / topology, h52-librilight the Libri-Light data, kd-cache the teacher labels, eval-lane the statistics, Codex the engine and scorer. GPU slots come only from fleet-sched's queue.

## Question
1. **Trunk.** Trained on the fleet with every adopted recipe change and every data source that has passed its gate, can B0's architecture beat B0 under codex-lead's 06:03Z release rule, and can it deliver a release candidate every few hours instead of once at the end?
2. **Capacity.** Which larger model keeps its sleep-gated always-on CPU at 15% speech at or below B0's **ungated** CPU, with an active real-time margin of at least 10x? Does it beat the trunk by more than 2x the spread at matched exposure?

## State at registration (sources in AGENT_CHAT.md)
- **B0** = A1 step 49,000, EMA (dense S: d 384, 16 blocks, 6 x 64 heads, FFN x4, conv 15, left context 128, pre-norm + final LN, R0 BPE-500 on both heads, stateless predictor context 2, joiner 320 tanh, pruned RNN-T + 0.3 CTC). Canonical native full DEV **8,773 / 106,054 edits = 8.272%** (S/D/I 7,042 / 894 / 837). PyTorch EMA greedy 8.302 / 7.961 / 7.703 at chunks 4 / 8 / 16. Quiet xbabe0: **0.0351 CPU-s per audio-s, 28.45x RT**, 74.4 MB RSS, 0.348 W core register (bench-open 06:21Z). First token 258 ms on the end-of-word-0 clock (100 clips). B0's history: H22ctl (28.8k audio-h from scratch), then A1 (87.6k audio-h, re-warm to 1e-3, chunk sampling {4,8,16}, one cosine).
- **H41 scoreboard:** adopted: chunk sampling {4,8,16}. Not adopted: Muon, adaptive clipping, CR-CTC full and half, the arm-8 MWER fine-tune, chunks {1,2,4,8,16}, stream-context augmentation, arms 12 / 12b, predictor context 4, left context 256, stronger SpecAugment.
- **delay_penalty 0.005:** **not adopted for the WER default** (lead 06:51Z). A1L (B0 + 8k replay steps + dp 0.005) is measurably worse than B0 at chunks 4 / 8 (+0.124 [+0.050, +0.200], +0.112 [+0.033, +0.193]), and on the confirmation half the result is INCONCLUSIVE. Penalty and re-warm are confounded; A1C (amendment 23, B0 + the same replay at dp 0) separates them. The trunk-recipe question is amendments 21 + 22: arms 14 / 14-r / 14-s3 against the chunk arm / 19 / 19-s3, 3 vs 3.
- **Not yet decided:** H45 soups / commit pass (−0.60 pp final text at chunk 16, offline); H46 LM; H49 sequence KD / MWER; H48c MFA-anchored loss; H52 Libri-Light gate. Small is verified (526.5 VAD h, 7.6% excluded by book overlap); medium's pass 1 ends ~07:55Z.
- **Fleet:** the co-tenant project holds x7 x2, x8 and xbabe6 (never touched). Ours: x5 x2 and x9 (A100), xbabe1 4090, xbabe2 and H0 3090s, the Mac. On-prem GPUs cannot join a cloud DDP job: the WAN carries 110-250 Mbit/s, and one FP32 gradient is 232 MB. DiLoCo stays blocked (PR #73).
- **NCCL** (h50-fleet 06:50Z, codex-lead ruling 06:39Z / 06:50Z): production is synchronous FP32 NCCL DDP with the tuned profile frozen as **intra-node `NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3`** (3.0 → 16.9 GB/s) and **cross-host `NCCL_NET_GDR_LEVEL=SYS`** (RDMA without GDR 3.4 → 6.3 GB/s with GDR). The `NCCL_IB_*` and socket knobs are dropped: they are inert here. BF16 communication buys nothing once the transport is fixed, so it stays off.

---

# Part A: the v3 trunk

## A1. Model and init
- **Architecture: B0's, unchanged** (`--preset S`, R0 BPE-500 `zipformer-en-2023-06-26/bpe.model` sha256 `c53433de…`, predictor context 2). This keeps the native engine, the INT8 bundle, the LM (BPE-500 tokens) and the KD caches valid as they are.
- **Init:** B0's raw weights from **rank 0's** checkpoint, `xbabe5:~/jsr-data/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt` (sha256 `2a3dd9fb…`), with a fresh optimizer. x9 holds a verified copy at `~/jsr-data/seeds/A1/ckpt-00049000.pt` (`2a3dd9fb…`; A1C inits from it). The same-named file in x9's A1 run dir is A1's rank-3 checkpoint (sha256 `9e8bba41…`): do not use it; the launcher's `INIT_SHA256` check refuses it. The EMA restarts from these weights. The init is B0 whatever the penalty decision; A1L's 8k steps are not worth inheriting (it is worse than B0 at chunks 4 / 8).

## A2. Recipe (adopted changes only)
A1's recipe: pruned RNN-T (range 5) + 0.3 CTC, AdamW (betas 0.9 / 0.98, wd 1e-2), clip 5, SpecAugment on, speed perturbation 0.9 / 1.0 / 1.1, qkv cap 1.5x the step-100 norms with guard 2.0x, **chunk sampling {4,8,16}**, `eval_chunks=8,16`.
- `delay_penalty` = **0.005 only if amendments 21 + 22 (3 vs 3 seeds) pass non-inferiority at chunks 4 / 8 / 16; otherwise 0.** The fast-first-token model is then a short fine-tune from each release candidate (the A1L / H48b recipe), reported as its own row.
- **H49 sequence KD / MWER, H48c MFA-anchored loss, an LS-only cooldown:** each enters **only as a cooldown-recipe arm** (A4), compared against a plain cooldown from the same branch point. A method is adopted into the release cooldown only by its own wave's gate (> 2x the four-run spread, both seeds, confirmation half). At most two methods are confirmed on two paired seeds, per codex-lead.
- **Nothing else changes.** No optimizer, augmentation or architecture change rides along with the trunk.

## A3. Data and sampling, by segment
A segment change happens only at a branch point (A4), as a **new run** that inits from that branch point's raw weights. Each new run gets a fresh optimizer, a 500-step warmup back to the same stable LR, and an EMA that restarts. Changing the data mix in place would scramble the sampler cursor.

| segment | data | sampling | starts |
|---|---|---|---|
| **S1 = "v3-a"** | LibriSpeech 960 h (281k utts, human labels) | uniform over row groups (= proportional to hours) | when A6's launch gates close |
| **S2** | 960 h + Libri-Light **small** with R0 labels after H52 filtering (~0.5-0.6k h) | proportional (LS ≈ 60% of exposure) | at the first branch point after H52's small data pilot PASSES |
| **S3 = "v3-b"** | 960 h + small + **medium** (~5.7k h before filtering) | **LS upweighted to ≈ 1/3 of exposure**: weight 3 on the LS splits, 1 on Libri-Light | at the first branch point after H52's medium gate PASSES |

- **The H52 gate must test exactly this mix.** For h52-librilight and trainer-lane: the medium data-scaling arm should use LS weight 3 : LL weight 1, or the trunk adopts whatever mix the gate tested.
- **Labels:** R0 1-best token ids from kd-cache's `r0-fp32-gpu-v1` convention. They are in the trunk's own tokenizer (R0 BPE-500), so no re-tokenisation is needed. Filtering (R0 confidence, blank margin, agreement with A1 / A1L, length ≤ 18 s) is H52's, frozen before the gate. Per-utterance loss weights stay off in the trunk; confidence weighting would be its own arm.
- **Overlap:** only shards whose overlap verdict (speakers and books against dev / test) is logged in `LOOKS.jsonl` are used. h52's small verdict: 178 files excluded. Results carry the label **"semi-supervised (R0 labels)"**.
- **Duplicates:** the 315 small files (80.7 h) that h52 flagged as probably the same recordings as 960 h chapters stay excluded, as trainer-lane asked for the H52 pilot (06:55Z). Otherwise "+LL" partly re-weights existing training audio.
- **Loader requirements (trainer-lane):**
  - trainer-lane's `extra_splits` (PR 2): an extra parquet split with pseudo-label text, a mixing weight and a `conf` column. The trunk uses its mixing weight for the LS 3 : LL 1 ratio in S3. Fallback: list each LS split three times (each LS row group then appears 3x per epoch, with identical speed perturbation within an epoch);
  - a test that the per-epoch counts match the weights;
  - shards read from NVMe-local disk on every rank (x5 `~/jsr-data/librilight/shards/`, mirrored to x9 and to x7 / x8 when they join), never from the USB drive.

## A4. Schedule: WSD-stable trunk, exposure-indexed EMA, branch points, cooldowns, soups
**Batch, sized for the owner's 80-90% VRAM target (lead 06:58Z), with a fixed LR rule:**
- **Per rank: A1's shape, 1,300 s / 288** (75-84% of 40 GB measured on A1). In the first 200 steps trainer-lane may raise it once, to at most 1,400 s / 310, to land in 80-90%; that value is then fixed for the run.
- **Global = per rank x world** (`ddp_global_batches=true`): 3,900 s on 3 ranks (x5 x2 + x9), 7,800 s on 6.
- **LR and EMA follow the global batch:** LR = 4.3e-4 x √(global / 3,900), and EMA d = exp(−global / (3,600 x 3,600)).

  | world | global budget / max batch | LR | EMA d |
  |---:|---|---:|---:|
  | 2 (x5 x2, lead 07:21Z) | 2,600 s / 576 | 3.51e-4 | 0.999799 |
  | 3 | 3,900 s / 864 | 4.30e-4 | 0.999699 |
  | 4 (x5 x2 + x7 x2) | 5,200 s / 1,152 | 4.97e-4 | 0.999599 |
  | 6 | 7,800 s / 1,728 | 6.08e-4 | 0.999398 |
- **The world size changes only at a 500-step checkpoint restart** (x7 / x8 joining or leaving), with the rule applied and the change posted. A fixed 3,900 s global batch would leave 6 ranks at 37% VRAM, below the owner's band.
- **H50 and the trunk are separate** (codex-full-fleet 07:23Z): the trunk's large-batch windows do not substitute for the H50 ladder (750 s / 160 global). The trunk reports its own throughput, comm share and negotiated transport. They are information, not H50 verdicts.

**LR, warmup-stable (WSD's W and S):**
- `lr=4.3e-4 warmup=1000 total_steps=10000000 min_lr_ratio=0.02`. The existing cosine over 10^7 steps is flat to within 0.03% over the first 100k steps, so **no trainer change is needed** for the stable phase.
- 4.3e-4 is half of A1's peak, √-scaled from 5,200 s to 3,900 s (1e-3 x √0.75 / 2). The basis is A1's own curve: at LR ≈ 6.7e-4 (step 20k), A1's EMA was within ~0.11 pp of its final full-dev WER (PyTorch EMA 8.408 vs 8.302).
- It is fixed now and not tuned.

**EMA, exposure-indexed:** time constant **τ = 3,600 audio-h**, i.e. A1's 0.9995 at 6,500 s per step.
- Per step, d = exp(−B / (3600 τ)) with B the global budget in seconds: **d = 0.99970 at 3,900 s**, and 0.99950 at 650 s with τ = 361 audio-h, the value the H41 / A1L cooldowns already use.
- Written as a computed `ema_decay` per run, so no trainer change is needed. A `ema_tau_h` key that derives d from the live global audio would be cleaner (optional, trainer-lane).

**Branch points (BP):** the first 500-step checkpoint past every **10k audio-h** of logged real audio (≈ 10,000 steps at 3,900 s). They are exposure-indexed, so a world-size change does not move them. The raw + EMA checkpoint is kept permanently, with its sha256 in `MANIFEST`. A copy goes to a second cloud host, and to the data drive in one sequential pass.

**Cooldowns: the "D" of WSD.** They branch off; the trunk keeps going.
- **Fast cooldown (FC), on one GPU, one per BP.** Init = the BP's raw weights.
  - **Schedule:** the H41 replay machinery (`h41_replay.sh`: 650 s / 144, EMA 0.9995, 8,000 steps) with the LR started at the trunk's own stable LR √-scaled to 650 s: `lr=1.76e-4 min_lr_ratio=0.1705`, i.e. 200 warmup steps (rebuilding Adam moments) to 1.76e-4, then a cosine to 3e-5. ≈ 1R.
  - **Why 1.76e-4:** it is 4.3e-4 x √(650 / 3,900), and the same at 6 ranks, since LR ∝ √global. The trunk is at η_s, not annealed, so A1L's 2.476e-4 re-warm (≈ 6.1e-4 trunk-equivalent) would overshoot first. A1C will show what such a re-warm costs.
  - It runs on xbabe1 / xbabe2 / H0 or any idle A100.
  - **Purpose:** an annealed candidate ~1-2 h after each BP, for the scoreboard, and the matched base for cooldown-recipe arms.
  - **Caveat:** A1L shows that a short small-batch re-warm from an annealed model can cost ~0.1 pp. FC candidates are therefore compared with each other and with B0, but the release candidate comes from a release cooldown.
- **Cooldown-recipe arms (CA).** Each is an FC from the same BP with one change:
  - CA-KD (H49b's sequence-KD term);
  - CA-MWER (H49a, as a fine-tune after the FC);
  - CA-ANCHOR (H48c);
  - CA-LS: the cooldown on LibriSpeech only, i.e. finishing on human labels after Libri-Light segments.

  Each is paired against the plain FC (same BP, same batches, seed 1234; seed 4321 for the pair). The batch effect cancels in the pair.
- **Release cooldown (RC), on the island at the full global batch.** It inits from the chosen BP's raw weights and uses the adopted cooldown recipe: 200 warmup steps to the stable LR of its global batch (4.3e-4 x √(global / 3,900)), then a cosine to 2e-5 over **D = 15% of the steps since the segment's start, at least 5,000 steps** (≥ 5.4k audio-h). EMA as the trunk.
  - **The trunk pauses at that BP while an RC runs, then resumes from the BP checkpoint.** Two RCs per segment at most.
- **Cooldown soups.** Two FCs or RCs from the same BP, seeds 1234 and 4321, uniformly averaged (EMA weights). Greedy soups follow H45's protocol (select on the selection half, confirm once).
  - A soup replaces its best member only if it is better on the selection half and holds on the confirmation half.

## A5. Chunks, seeds, exposure, checkpoints
- **Chunk modes {4,8,16}** for the trunk, FCs and RCs; evals at chunks 4 / 8 / 16. The 80 ms mode is H48b's fine-tune, not a trunk change.
- **Seeds:**
  - **trunk seed 7919**, not used by any selection run so far. This is the protected independent final seed of codex-lead's rule.
  - CA arms and soups use 1234 / 4321.
  - The RC that makes a release candidate uses 7919.
- **Exposure budget:**
  - S1 ≤ 40k audio-h (≈ 41k steps; ~6 h on 3 A100s at ~6.8k audio-h/h, ~3 h on 6).
  - If S2 / S3 are not ready by then, S1 continues only while FCs keep improving; otherwise the island goes to the capacity branch or the CA arms (decision at the 40k BP, posted in the chat).
  - S3: 100k audio-h (~15 effective epochs of the LL mix) with BPs every 10k audio-h.
- **Checkpoints:**
  - `ckpt_every_steps=500` (atomic; model, EMA, optimizer, scheduler, RNG, data cursor; the trainer's async saver);
  - `weights_every=2500` (eval-only snapshots, kept);
  - BPs kept permanently;
  - rolling checkpoints keep the trainer default (last 3).
  - `log_every=25` (h50-fleet's window protocol); dev-clean400 every 2,500 steps, **monitor only, never a decision**.
- **Code:** one pinned commit: trainer-lane's merged H44 PR (per-objective DDP denominators), plus its PR 2 (rank telemetry), plus patch 0001 (`encoder_blocks`, default off).
  - Every node checks the `LC_ALL=C` content sha256 of `python/ultra/**.py` and a clean tree before start, and records the commit, tree hash, every `NCCL_*` variable, GPU UUIDs and torch / NCCL versions (`config.rank<R>.json`).

## A6. Gates
**Launch gates (all must hold; `tools/h53_preflight.sh` prints the list):**
- **G1** the H44 PR is merged and pinned;
- **G2a (single node, world 2 on x5):** the tuned intra-node profile is frozen (SHM memcpy; h50-fleet's x5 DDP probe: exposed FP32 comm 36% → 5%). h50-fleet's exact-sum / FP32-parity check of that profile passes, and the launcher captures the negotiated SHM transport.
- **G2b (multi-node, world ≥ 3 across hosts):** additionally the H50 ladder's T2X (and T4 / T6 when used) pass their registered scaling, exact-sum and FP32-parity checks with `NET/IBext_v10 … GDRDMA` captured per rank. The single-node trunk does not wait for the cross-host ladder; a multi-node restart does;
- **G3** amendments 21 + 22 have decided the trunk's `delay_penalty`;
- **G4** the init checkpoint digest matches;
- **G5** the 960 h parquet manifest matches on every node;
- **G6** every chosen GPU has foreign memory < 15% and fleet-sched's reservation;
- **G7** run dirs and mirrors are off H0's root disk;
- **G8** the CA arms' pre-registrations exist before any CA arm runs.

**A trunk candidate supersedes B0 as the release-candidate weights only if all of these hold** (codex-lead 06:03Z, adopted 06:03:48Z):
1. Canonical full DEV (5,567 utterances), native engine, prepared INT8, greedy, 160 ms, pinned `ultra-score`. The paired speaker-cluster bootstrap (10,000 resamples) against B0 gives a **pooled 95% upper bound < 0**, and **split upper bounds ≤ +0.20 pp** (dev-clean, dev-other).
2. **Selection and confirmation:**
   - candidates are ranked on dev_split_v2's selection half only;
   - at most one candidate per segment, and at most four in all, are scored on the confirmation half, with Holm over those looks;
   - it must show upper bound < 0 there.
3. **Latency, end-of-first-word clock (eval-lane's pipeline):** median no higher than B0's; paired p95 ≤ +20 ms; first-word coverage, first-word error and correct-by-300 ms within 0.50 pp.
4. **Cost and parity:**
   - quiet xbabe0 CPU-s per audio-s ≤ 1.05x B0 (same architecture: catches export regressions);
   - INT8 engine − PyTorch within ±0.05 pp pooled;
   - chunk 16 (the commit pass's operating point) no worse than B0's chunk 16 beyond +0.20 pp upper.
- **The release (H54) bar is separate:** ≥ 10% relative, **≤ 7,895 edits**, for the whole artifact (weights + commit pass + any in-bundle LM).
  - The trunk alone is **not expected** to reach it on 960 h: expect 0 to −2% for S1. The candidates that could reach it come from S3 plus the cooldown recipe plus capacity, combined with the commit pass.

**Kill and stop rules:**
- divergence: the train-loss EMA rule after step 5,000, once the re-warm has settled (A1 used 8,000 for its 1e-3 re-warm, A1L 1,000 for a smaller one); or a dev-clean400 eval worse than the best one by more than 3.0 pp (`diverge_wer_points=3.0`);
- qkv guard 2x;
- a non-finite loss stops the run.
- **Not stops:** per-rank efficiency below 0.75 for 30 min pages h50-fleet; a cap breach engages the trainer's throttle (`ddp_util_target=0.87`), never SIGSTOP.

## A7. Compute (R = one 8k-step 650 s replay on one A100 ≈ 0.55 A100-h ≈ 1.4k audio-h of S)
| item | R |
|---|---:|
| S1 ≤ 40k audio-h | ≤ 29 |
| each FC | 1 |
| each RC | ~4 |
| S3 100k audio-h | ~71 |
| **total for S1 + S3 with 8 FCs and 2 RCs** | **≈ 116R** |

---

# Part B: the capacity model

## B1. Constraint and cost model (`tools/capacity_cost.py` → `results/capacity_sizing.json`)
**Cost law** (`docs/ultra/costlaw/W01_H0.md`, Zen 2 AVX2 int8, weights streamed from DRAM):
- a GEMM pass over W weights for F frames costs W (51 ps + 14 ps x F);
- runtime-lane's profile puts `encoder.gemm_i8` at 88-91% of B0's step;
- the law predicts B0 at 0.037 CPU-s per audio-s against 0.0351 measured.

So, calibrated on B0: **c(K) = c_B0 (0.10 + 0.90 G(K) / G(B0))**, where G is proportional to encoder weights for dense models at one chunk size. Multirate middle blocks (12.5 Hz, F = 2) cost 26% less per weight.

**The constraint, in a measured form:** r = c_K / c_B0 ≤ **r_max = min(0.1 / c_B0 = 2.85, (1 − φ) / (g15 − φ))**.
- g15 = gated B0 CPU at 15% speech ÷ ungated, measured by H47 on the real gate;
- φ = non-speech floor ÷ ungated (≤ 0.02 by H47's gate).
- The first term is the 10x active real-time margin on the quiet xbabe0 core.
- The gating term binds only if g15 > 0.36.

| model | inference params | CPU ratio r | CPU-s per audio-s | RT margin | gated at 15% ÷ B0 ungated (g15 = .19 / .24 / .35 / .50) |
|---|---:|---:|---:|---:|---|
| S = B0 (d384 x16) | 56.4 M | 1.00 | 0.0351 | 28.5x | 0.19 / 0.24 / 0.35 / 0.50 |
| **S-24** (d384 x24) | 83.6 M | 1.44 | 0.0506 | 19.8x | 0.27 / 0.34 / 0.50 / 0.71 |
| **S-32** (d384 x32) | 110.9 M | **1.88** | 0.0661 | **15.1x** | 0.34 / 0.43 / 0.64 / **0.92** |
| M-16 (d512 x16) | 99.1 M | 1.69 | 0.0593 | 16.9x | 0.31 / 0.39 / 0.58 / 0.83 |
| M-18 (d512 x18) | 111.2 M | 1.89 | 0.0662 | 15.1x | 0.34 / 0.43 / 0.64 / 0.93 |
| M-20 (d512 x20) | 123.3 M | 2.08 | 0.0731 | 13.7x | 0.37 / 0.48 / 0.71 / **1.02 (fails)** |
| M_MR = L3 (d512, 4/10/2) | 100.1 M | 1.45 | 0.0507 | 19.7x | 0.27 / 0.34 / 0.50 / 0.71 |

**Where the g15 scenarios come from:**
- 0.19: the eval-lane stream's items (mean 6.8 s), 1 s pre-roll replayed in one batched pass at ~1/4 cost, 1.5 s hangover;
- 0.24: AO-bench items (mean 3.0 s);
- 0.35: adds false wakes;
- 0.50: the approved plan's "about half on a 15%-speech stream".

**h47-sleep's measurements (07:12Z) place g15 low:**
- The ideal word-onset oracle O3 decodes 49.9% of a 38.1%-speech stream (awake ÷ speech = 1.31), with no restart damage against C1.
- Scaled to 15% speech that is a ≈ 0.20, or ≈ 0.18 with a batched pre-roll. So **g15 ≈ 0.20-0.22 for an ideal gate**, and a real gate sits higher.
- At g15 = 0.22, S-32's gated CPU at 15% speech ≈ **0.40x B0's ungated CPU**, and the binding constraint is the RT margin.
- **"B0 ungated" is taken conservatively as 0.0351** (speech, no reset). The deployable ungated always-on B0 is C1 (reset-with-replay). It costs 1.41x on non-speech (A0), so ≈ 0.048 CPU-s per audio-s at 15% speech. Against that the margin would be 1.38x larger.
- **S-40 (r ≈ 2.32, 12.3x RT) passes too if g15 ≤ 0.44.** It stays out until the owner's target device shows ≥ 10x RT for it.

**The decision uses measurements, not this table:**
- r from the xbabe0 timing of the grown-at-init export (B3);
- g15 from H47 on B0 with the real gate, interpolated linearly between the AO-bench 5% and 20% duty cells (a gate's awake fraction is affine in duty for a fixed item mix), or from a duty-15 cell (asked of alwayson-bench: `duty15_r1_{Q,N20}`).

## B2. Choice: **S-32, grown exactly from B0 / the trunk; fallback S-24**
- **Robust to the plan's pessimistic g15 = 0.50:** r = 1.88 ≤ 2.04, with RT 15.1x. M-20 and anything larger fail that scenario.
- **Exact warm start: function-preserving depth growth** (`tools/grow_depth.py`).
  - Each new block is inserted after an original block. Its internals are copied from that block, so their norms are on the trained scale and the qkv cap binds as before. Its four residual output projections (ff1.w2, att.out, conv.pw2, ff2.w2, weights and biases) are **zero**, so with pre-norm modules and no post-block norm the block is an exact identity.
  - **The grown model computes bit for bit what B0 computes** (offline, chunks 4 / 8 / 16 and the streaming ChunkEncoder; 11 witness tests pass on both A1L's tree (d5f231e) and trainer-lane's (fd0f64f), including a checkpoint-format round trip through `train.checkpoint.load`, `tests/test_grow_depth.py`, ~1 min on CPU).
  - Training then starts at B0's WER instead of from scratch. A from-scratch d512 model would need ~150R (1.77x S's compute per audio-h) just to reach B0's 116k audio-h of exposure.
  - Zero output projections get gradients at step 1 (tested), as in ReZero / Fixup and the growth operators of Shen et al. 2022, bert2BERT and LiGO.
- **Everything downstream stays valid:** same tokenizer (LM, KD caches, R0 labels), same dense engine path, same predictor context 2.
- **Not chosen, and why:**
  - **M_MR / L3:** same CPU as S-24, but a spm512 tokenizer (LM / KD / labels would all need a second copy). Multirate also costs accuracy at fixed width: F1 (S_MR) trails H22ctl by ~0.37 pp at similar exposure.
  - **d512 dense:** no exact warm start.
  - **V2 / unet3, GQA, ReLU², dwsep stem, QK-RMSNorm, LayerScale, predictor context 4:** not in the engine (B3).

## B3. Engine support (status at registration; questions to Codex / runtime-lane in the chat)
| item | status |
|---|---|
| Dense n_blocks 24 / 32 | `model/mod.rs` bounds n_blocks ≤ 256 and has no fixed block count on the production path (only bench examples hard-code 384 / 16). Expected to load as is |
| Prepared INT8 | The quantizer uses scale `max(absmax/127, 1e-8)`, so the new blocks' all-zero rows give zero codes, zero scale and zero wsum. **A grown-at-init export should give transcripts bit-identical to B0's INT8** (the residual stream adds exact zeros) at exactly S-32's CPU cost |
| Cross-ISA identity | Same kernels. Re-verify NEON vs AVX2 token identity on the panel for the grown model (the H11 method; the Mac is free) |
| Predictor context | 2, the only value the engine accepts. `export.py` hard-codes `"context": 2` |
| **Hazard (found here)** | `arch_json.rs` `RawEncoder` ignores unknown fields, and `export.py` writes the full `EncoderConfig`. A *dense* model trained with `kv_heads`, `ffn_act=relu2`, `stem_type=dwsep`, `qk_rmsnorm`, `norm_affine=False`, `layer_scale_init` or `pred_ctx=4` would export and **load silently wrong**; only family ≠ dense / multirate is refused. **Ask:** the engine refuses non-default V2 fields (or `deny_unknown_fields`), and `export.py` refuses them and `pred_ctx ≠ 2` (Codex flagged context 2 at 06:32Z) |
| Bundle | The H44 minimal self-contained bundle must take any n_blocks. S-32 INT8 is ~111 MB of weights, and the K/V cache at 32 blocks is 12.6 MB fp32 per stream |
| Commit pass (#76) / sleep gate (#77) | They use `Model` generically. Confirm there is no 16-block assumption |

**Readiness action, before any training:**
1. trainer-lane or eval-lane runs `grow_depth.py --weights ema --blocks 32` (and 24) on B0 = A1-49000.
2. eval-lane exports, prepares INT8 and checks **engine transcripts == B0's INT8 transcripts** on the 730 panel. This is the engine-support proof.
3. bench-open times the export on xbabe0 (3 reps, the same harness as 14br), giving the true r.
4. The Mac checks NEON identity.

## B4. Expected gain (stated with its uncertainty)
**Our evidence:**
- Same family, F1 (S_MR, 57 M) → L3 (M_MR, 100 M): 9.32 → 8.82 pooled at ~30k audio-h, **−5.3% relative for 1.77x params**. Confounded: different tokenizers (BPE-500 vs spm512) and batch / LR; one seed each.
- Across families, H22ctl (S) → L3: −0.12 pp (−1.3%), interval touching 0.
- Implied power-law exponent α ≈ 0.10 in WER ∝ N^−α.

**Literature (non-streaming LibriSpeech 960 h, no LM):**
- Conformer M → L (3.9x): −9% clean / −14% other (Gulati et al. 2020), α ≈ 0.07-0.11;
- Zipformer M → L (2.3x): −7% / −3% (Yao et al. 2024), α ≈ 0.04-0.09.
- Capacity pays more as data grows (acoustic scaling laws, Droppo & Elibol 2021). Libri-Light pseudo-labels add 6-7x data.

**Estimate** for S-32 (1.98x encoder params), trained to matched exposure on the same data:

| data | expected | ~90% range |
|---|---|---|
| 960 h | −3 to −5% relative (~0.25-0.4 pp) | ≈ −0.5% to −7.5% |
| with Libri-Light | upper end or beyond | |

S-24: about 60% of that.
- **In a 1R pilot the new capacity is barely used:** expect 0 to −1.5%, often INCONCLUSIVE under the 2x-spread rule (0.174 pp).
- **Bottom line:** P1 screens for safety, capacity use and cost. **It does not decide adoption**; the trunk-scale branch does.

## B5. P1: the matched capacity pilot (pre-registered; launch-ready)
**The control already exists:** **C0 = A1C**, trainer-lane's H41 amendment 23 (07:04Z, x9).
- A1C is B0 + 8,000 replay steps at dp 0, on A1L's pinned tree (99bce400…). It is also H48a's unconfounded control and H48c's replay-only control.
- So P1 adds only the growth arms. They run on **the same tree plus patch `0001a`** (`encoder_blocks`, default off; it applies cleanly to x9's `trainer-a1l` worktree, `git apply --check`). The witness tests show the default leaves `model_config_for` unchanged.

**The growth arms copy A1C exactly:**
- the H41 replay protocol from **B0's raw weights**: 200 warmup steps to 2.476e-4, cosine to 3e-5 at step 8,000;
- 650 s / 144, seed 1234, EMA 0.9995;
- `chunk_modes=4,8,16 eval_chunks=8,16`, **`delay_penalty=0`**;
- guard 0.87. The only change is the init (grown seed) and `encoder_blocks`.
- **Batch:** pinned at A1C's 650 s / 144 for comparability, which is the owner-rule exception "paired arms keep their pinned batch", recorded as the capacity reason.

| arm | model | init | R | GPU memory at 650 s (A100) |
|---|---|---|---:|---|
| **C0 = A1C** (control) | S (16 blocks) | A1-49000 raw | (running) | ~15 GB (37%) |
| **G32** (primary) | S-32 (`encoder_blocks=32`) | `grow_depth --blocks 32 --weights model` of A1-49000 | ~2.0 | ~26 GB allocator peak (≈ S at 1,300 s), ~72-80% on nvidia-smi: a whole A100 |
| **G24** (secondary) | S-24 (`encoder_blocks=24`) | `grow_depth --blocks 24` | ~1.5 | ~22 GB (~55-60%): a whole A100 (co-tenancy gives no throughput, trainer-lane 07:18Z) |

- Neither arm fits a 24 GB card at the pinned batch: S-24 needs ~92% there.

**Decision metric:** canonical full-DEV EMA pooled at chunks 4 / 8 / 16, each arm against C0, with eval-lane's paired bootstrap. Margins are the registered four-run spreads 0.087 / 0.094 / 0.360. Holm over the two arms.

**Readings (fixed now):**
- **HARM:** point > 0 and upper bound > margin at chunk 4 or 8. Growth hurts at this horizon: no phase 2, and the capacity branch waits for S3 and gets its own pilot there.
- **PROMISING:** point ≤ −0.087 pp at chunk 4, Holm-adjusted bound < 0, and no worse than the margin at 8 / 16. Then phase 2 (seed 4321 for C0 and that arm, +3R), and the trunk-scale branch (B6) starts at the next BP that has resources.
- **NEUTRAL:** anything else. The branch is deferred to S3 and runs there as a matched arm against the trunk.

**Reported, not gated:**
- the train-loss gap on identical batches (G − C0, steps 6,001-8,000: transducer and CTC);
- new-block qkv norms;
- step time and peak memory;
- first-token latency (100 clips);
- r on xbabe0.

**Kill rules:**
- step-2,000 dev-clean400 EMA chunk 4 above **6.163**, the fixed threshold A1C was registered with, so the rule does not depend on when A1C runs;
- raw chunk-4 WER > 20% at any eval;
- the trainer's divergence and qkv guards.

**Total:** 3.5R (G32 + G24; the control is A1C) + 3R (phase 2, conditional: seed 4321 for A1C and the promising arm).

**Done by trainer-lane at 07:20-07:23Z** (on x5, from rank 0's `2a3dd9fb…`):
- P1 tree = A1L's tree + 0001a, content sha256 **1254ca23…** (worktree `trainer-p1`), with the 3 + 11 witness tests passing there;
- seeds `A1-49000_grow32_model.pt` **4eaa6c07…** and `A1-49000_grow24_model.pt` **df2b9565…**, each reproducing B0's encoder output exactly (max |diff| 0.0) offline and at chunks 4 / 8 / 16 on the real checkpoint;
- **G32 runs on x9 after the A1C resume (~08:33Z, lead 07:21Z); G24 on the next free A100.**

The command used (for the record; `tools/p1_prepare.sh` wraps it with the checks):
```
sha256sum ~/jsr-data/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt   # must be 2a3dd9fb…
cd ~/jsr/.agent/worktrees/trainer-a1l/python/ultra && for n in 32 24; do
  PYTHONPATH=$PWD ~/jsr-data/venvs/train/bin/python /path/to/h53-trunk/tools/grow_depth.py \
    --src ~/jsr-data/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt \
    --out ~/jsr-data/seeds/H53/A1-49000_grow${n}_model.pt --blocks $n --weights model; done
```
`--weights ema` gives the grown-at-init exports for B3 (B0 = A1-49000's EMA).

**Launch** (A1L's launcher, unchanged):
```
INIT_FROM=~/jsr-data/seeds/H53/A1-49000_grow32_model.pt RUN_NAME=H53_P1_G32_s1234 \
  h41_replay.sh ~/jsr-data <cpus> <gpu> P1_G32 chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty=0 encoder_blocks=32
```

## B6. The trunk-scale capacity branch
- **Init:** at a BP, the trunk's raw weights are grown to S-32. The branch continues with the trunk's segment config plus `encoder_blocks=32`, same global batch, same data order, same seed (7919).
- **The trunk's own continuation from that BP is the matched control.** Both get the same release-cooldown recipe, and they are compared at matched exposure.
- **Memory:** S-32 at 1,300 s per rank does not fit 40 GB, so the branch needs **6 A100s at 650 s per rank** or **activation checkpointing**. Ask to trainer-lane: `act_ckpt=true` wraps each block in `torch.utils.checkpoint` (preserve RNG), with a witness test that grads are identical; ~+30% compute. The alternative is gradient accumulation.
- **Gate** (the plan's, with codex-lead's rule):
  - better than the trunk by > 2x the spread (0.174 pp) on full DEV, both seeds where run, and on the confirmation half;
  - **gated watts at 15% speech ≤ B0 ungated, measured** (r on xbabe0 x g15 from H47);
  - first token within the budget (paired p95 ≤ +20 ms);
  - INT8 parity and cross-ISA identity hold.
- **Compute:** ≈ 1.97x S per audio-h (+30% with checkpointing): about 140-180R for 100k audio-h, matching the plan's "115-120R plus the Libri-Light exposure".

---

# Part C: launch readiness
**Checklist** (✔ done; ◻ open, with its owner):

| # | item | state | owner |
|---|---|---|---|
| 1 | Trunk spec, segments, gates (this file) | ✔ | h53-trunk |
| 2 | Configs `configs/v3_trunk_s1.set`, `fast_cooldown.set`, `release_cooldown.set`, `p1.set` | ✔ | h53-trunk |
| 3 | Launcher `tools/h53_trunk.sh`: tuned profile, tree pin, init digest, foreign-memory check, transport capture | ✔ draft | trainer-lane adopts into `train/ops/` |
| 4 | Preflight `tools/h53_preflight.sh` (G1-G8) | ✔ draft | h53-trunk |
| 5 | Depth-growth tool + 11 witness tests (both trees) | ✔ | h53-trunk |
| 6 | Patch 0001 `encoder_blocks` (+3 witness tests; applies to fd0f64f; trainer.py 976 → 977 lines) for the trunk's tree, and **0001a** for A1L's tree (d5f231e; +3 tests; applies to x9's `trainer-a1l`) for P1 | ✔ | trainer-lane applies |
| 7 | H44 PR merged and pinned (G1) | ◻ (gate running) | trainer-lane / v3-integrator |
| 8 | Tuned profile frozen; ladder T2L / T2X pass; exact-sum / FP32 parity (G2) | ◻ | h50-fleet / trainer-lane |
| 9 | Amendment 21 → the trunk's dp (G3) | ◻ ~07:15Z | eval-lane / lead |
| 10 | Grown-at-init exports: engine == B0 transcripts; xbabe0 r; NEON identity | ◻ | eval-lane, bench-open, Mac |
| 11 | Engine refuses unknown V2 fields; export refuses them and pred_ctx ≠ 2 | ◻ | Codex / trainer-lane |
| 12 | LL split dirs + LS weighting in the loader; shards NVMe-local on every rank | ◻ | trainer-lane / h52 |
| 13 | H52 small / medium gates use the trunk's mix | ◻ | h52 / trainer-lane |
| 14 | g15 measured by H47 (duty-15 cell or 5 / 20 interpolation) | ◻ | h47-sleep / alwayson-bench / bench-open |
| 15 | `act_ckpt` (or grad accumulation) for the 3-GPU capacity branch | ◻ | trainer-lane |
| 16 | GPUs: x5 x2 + x9 after H48b / 14-s3 / data pilot / ladder; x7 x2 + x8 only if the co-tenant project releases them (< 15% foreign memory for ≥ 3 min) | ◻ | fleet-sched |

**Blockers for the trunk launch:** 7, 8, 9 (and 16). **For P1:** 6 (in the pinned tree) and a GPU slot. **For the trunk-scale capacity branch:** P1 not HARM, 10, 14, and 15 or 6 A100s.

## Files (this draft folder; results only, never weights)
- `README.md` (this file);
- `configs/*.set`;
- `tools/grow_depth.py`, `tools/capacity_cost.py`, `tools/h53_trunk.sh`, `tools/h53_preflight.sh`;
- `tests/test_grow_depth.py`;
- `patches/0001-encoder-blocks-override.patch`;
- `results/capacity_sizing.{json,md}`;
- `MANIFEST` (checkpoint and data paths with sha256, filled as runs happen).
