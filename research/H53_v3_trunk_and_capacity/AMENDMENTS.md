# H53 amendments (appended to README.md when landed; each written before the runs it affects)

## Amendment 1 (2026-11-23 ~07:30Z): S1 may start now on x7 x2 on A1L's pinned tree
Written after the owner's 07:23Z decision freed x7 GPU0 + GPU1, x8 and xbabe6, and before any H53 trunk run, decode or result existed.

**At that time:**
- x7's two A100s were empty (0 MiB, 0%, read-only check);
- x5 x2 was reserved for S1 at ~08:17-08:20Z, waiting for PR #80 (G1).

**Change to G1, for segment S1 only.** S1 may run on a pinned tree whose code paths for the trunk's recipe are numerically identical to #80's: **A1L's tree, `python/ultra` content sha256 99bce400…** (d5f231e + its local diff), the tree of A1L, A1C and P1's control.
- **Why it is equivalent here:** #80's training changes touch only objectives and paths the trunk recipe does not use.
  - per-objective DDP denominators for z-loss / CR-CTC KL / the balancer (all weight 0 here);
  - stream-context shifts (off);
  - KD cache identity and the exact-path guard (KD off);
  - multirate chunk legality (dense model);
  - an explicit offline eval mode (the trunk evaluates at chunks 4 / 8 / 16, never 0);
  - a startup chunk-legality check ({4,8,16} is legal for dense S).
- The RNN-T and CTC global normalisation under DDP is the one A1 trained with for 49,000 steps on that tree.
- All 40 keys of `configs/v3_trunk_s1.set` exist in d5f231e's `TrainConfig` (checked).
- **From S2 on** (a segment boundary is a new run), the trunk pins #80's merged commit (+ PR 2 for `extra_splits`), as the README says.

**Placement:** S1 on **x7 GPU0 + GPU1, world 2** (one node, the G2a intra-node SHM-memcpy profile), with the world-2 row: `budget_s=2600 max_batch=576 lr=0.0003511 ema_decay=0.999799`, seed 7919.
- **Init:** B0 = rank 0's `ckpt-00049000.pt` (2a3dd9fb…), copied from x5 to x7 with its `.sha256` marker and re-hashed there.
- **The tree on x7:** replicated as on x5 / x9 (d5f231e + the local diff, content hash re-verified as 99bce400), in a clean worktree.

**What x5 x2 does instead:** the H50 ladder at #80's merge (T1 x5 g0, T2L x5 x2, T2X x5 g0 + x8), so the ladder-vs-trunk conflict goes away.
- **Multi-node trunk (world 4 = x5 x2 + x7 x2, or 6):** a restart at an S1 checkpoint or at the S2 boundary, after G2b (the ladder's T2X / T4 pass), on #80's commit.

**Unchanged:** everything else in README Parts A-C.

### Placement note to Amendment 1 (2026-11-23 07:36Z, before S1 starts; lead 07:28:47Z / 07:29:48Z / 07:33:16Z)
- **All of S1 stays on tree 99bce400**, including the world-4 multi-node restart (lead ACK 07:29:48Z). Trees switch only at S2 (a new run on #80's merged commit).
  - trainer-lane committed the replica locally as 562dd31d (content 99bce400), so `h53_trunk.sh`'s default clean-tree check holds.
  - `REQUIRE_CLEAN=0` (launcher sha e783ea45) stays available for uncommitted replicas.
- **Placement (lead 07:33Z):** S1 starts on **x5 x2 at ~08:15Z** (world 2, the same world-2 row). Meanwhile x7 x2 runs P1's control A1C (GPU1) and P1 G24 (GPU0), and x8 runs P1 G32 (since 07:30:15Z).
  - **World-4 restart on x5 x2 + x7 x2 at ~08:25Z**, when A1C / G24 end, with the world-4 row (`budget_s=5200 max_batch=1152 lr=0.0004965 ema_decay=0.999599`), after G2b.
- **Still open:** codex-lead's G1-equivalence ruling (needed by ~08:10Z). Without it, S1 waits for #80's merge.

## Amendment 2 (2026-11-23T08:56:43Z, trainer-lane for H53; lead ruling option (A), 08:5xZ; written before the world-4 restart)
- **LR across world sizes in S1:** S1 keeps **lr = 3.511e-4 at every world size**. The trainer on tree 99bce400 restores the optimizer and the LambdaLR state on resume, so `base_lrs` and `initial_lr` stay at world 2's value whatever `--set lr` says. The world-4 wrappers therefore pass `lr=0.0003511`, so config.json records the LR that actually applies. **The √-rule (A4) applies from S2 on** (new runs), and to cooldowns.
- **Batch and EMA at world 4:** `budget_s=5200 max_batch=1152` (A1's 1,300 s / 288 per rank) and `ema_decay=0.999599` (exposure-indexed, τ = 3,600 audio-h). Both are read from the config each step, so they take effect.
- **Cursor:** with `ddp_global_batches`, the restart re-packs the current window with the new budget, and the saved `Position.batch` indexes the new groups. **Up to one window** (96 row groups, ≈ 9.6k utterances, ≈ 0.08% of S1's 40k audio-h) may be partly repeated or skipped. The restart receipt logs the exact position and the repeated / skipped utterance counts, recomputed offline from both packings.
- **Unchanged:** tree 99bce400 / commit 562dd31d on every node, init chain (B0 2a3dd9fb…), seed 7919, dp 0, A1 default NCCL environment (a1env). Rollback to world 2 if per-GPU audio-h/h over the first 30 min is < 85% of world 2's (~2,490).
- **For S2+:** a trainer PR moves `base_lrs` / `initial_lr` to cfg.lr and the cursor to the next window boundary when the batch changes at resume, with witness tests (option B).

## Amendment 3 (2026-11-23T09:04:26Z, trainer-lane for H53; lead ruling 09:03Z; written BEFORE any H52 pilot result exists)
**Context:** eval-lane's 09:01Z readings: A1C − B0 fails at every chunk (a B0 continuation on 960 h with a 2.476e-4 re-warm at 650 s is +0.08 / +0.14 / +0.19 pp). S1 (960 h only) is now only the production-shape scaling measurement plus a placeholder; its checkpoints are not candidates. The world-4 S1 restart runs for ~30 min of H50 windows.

**S2, fixed now:**
- **Launch condition:** only if H52's small pilot PASSES under the registered gate (§6 Amendment 1: chunk 4 primary). Otherwise there is no S2-small, and the island goes back to the lead.
- **Init:** **B0's raw weights**, rank 0's `ckpt-00049000.pt` (2a3dd9fb…), not any S1 checkpoint. Fresh optimizer; the EMA restarts from those weights.
- **Code:** main at **90b7f4d** (#80 merged: the H44 fixes, `extra_splits` from #87, the exact KD frame rule). python/ultra content sha256 **3780b451…** (LC_ALL=C rule). Clean worktrees on every node, one PIN_COMMIT. The resize fix (#99) is not needed, since S2 is a new run.
- **Shape:** **world 4** (x5 ×2 + x7 ×2), `budget_s=5200 max_batch=1152`, `ddp_global_batches`, a1env (until the tuned profile is re-qualified with a DDP-init case), `workers=20`.
- **LR: a small re-warm.** Peak **lr = 3.511e-4** at the 5,200 s global batch.
  - Under A4's √-rule that equals **1.24e-4 at 650 s**: half of A1C's harmful 2.476e-4 re-warm, and 0.71× h53's 1.76e-4 fast-cooldown value. It is also exactly the LR S1 is actually running.
  - `warmup=1000`, then flat: the existing cosine with `total_steps=10000000 min_lr_ratio=0.02` is WSD-stable. Release cooldowns branch from S2 by A4.
- **Rest of the recipe:** EMA τ = 3,600 audio-h (`ema_decay=0.999599`), seed 7919, **dp 0** (amendment 22), chunk modes {4,8,16}, `eval_chunks=8,16`, the rest of `v3_trunk_s1.set` unchanged.
- **Data: the H52 pilot's winning mix.** For P that means 960 h + Libri-Light small (R0 labels, h52's final shards: 50 shards, 22 GB, 99,113 utterances, 378.7 h; the 315 duplicates excluded) as `extra_splits=<abs dir>` at natural share, no conf floor.
  - The directory **must have the same basename on every node**, because the data manifest and the global batch order depend on it: `~/jsr-data/librilight/shards/r0_small_tl` (on x7 already; on x5 a symlink or copy, sha256-checked).
- **Transfer caveat:** the pilot forked H22ctl 60k with the H41 replay (2.476e-4 at 650 s), not B0. Its P − C difference is a data-mix effect at that fork; S2 adopts the mix, not the pilot's absolute numbers.
- **Low re-warm by default** for every B0 fork (cooldowns, recipe arms): at most the √-equivalent of 1.76e-4 at 650 s, unless the arm pre-registers a reason.
- **Gates:** A6 as written (paired against B0 on full dev; one confirmation-half look per segment with Holm; latency and CPU guards).

### Amendment 3 addendum: the B0 bridge (2026-11-23T09:05:39Z, lead 09:05Z correction; before any bridge or H52 result)
The pilot forks H22ctl-60k, not B0, so its evidence does not transfer by construction. **S2's full launch waits for the pilot AND this bridge.**
- **Arms**, one A100 each, the same seed, **1234** (7919 stays the protected final seed):
  - **S2c** = B0 + 960 h only;
  - **S2p** = B0 + 960 h + Libri-Light small, the pilot's mix: `extra_splits=~/jsr-data/librilight/shards/r0_small_tl`, natural share, no conf floor.
- **Recipe = S2's, at one rank's share, with a 1R annealed end so the EMA endpoint is comparable:**
  - init B0 raw (2a3dd9fb…), fresh optimizer, EMA restarted;
  - **per-rank batch 1,300 s / 288**;
  - **peak lr 1.756e-4**, which is S2's 3.511e-4 at 5,200 s √-scaled to 1,300 s (≙ 1.24e-4 at 650 s, the low re-warm);
  - 200 warmup steps, cosine to 3e-5 at step **4,000** (`min_lr_ratio=0.1708`): 4,000 × 1,300 s ≈ 1,444 audio-h = 1R;
  - **`ema_decay=0.999`** (the replay family's τ ≈ 361 audio-h at this batch; S2's 3,600 audio-h τ would leave a 1R run's EMA mostly B0);
  - dp 0, chunk modes {4,8,16}, `eval_chunks=8,16`, `eval_every=1000`, checkpoints every 500.
  - Tree: main 90b7f4d (3780b451…).
- **Reading** (eval-lane registers it before either arm starts): EMA, full dev, pooled, canonical scorer, S2p − S2c, paired speaker bootstrap.
  - **PASS:** chunk 4 point ≤ −0.087 pp (one four-run spread, since the arms are single seeds) and one-sided 95% upper bound < 0; chunks 8 / 16 non-inferior at 0.094 / 0.360.
  - Otherwise FAIL or INCONCLUSIVE. S2 launches only on pilot PASS **and** bridge PASS.
  - Also reported: S2c − B0, the low re-warm's own cost against A1C's.
- **Kill rules:** raw chunk-4 WER > 20% at any eval; the divergence guard.
- **Cards:** fleet-sched's pick. x8 + x9 after H48c-b's and 17-r's decodes (~09:50Z), or x7 ×2 after the pilot.

## Amendment 4 (2026-11-23T09:08:55Z, trainer-lane for H53; lead 09:08Z after the independent audit; written BEFORE A1C's latency is measured)
**S2's delay penalty, as a commit-pass-aware decision.** The runtime shows provisional text at chunk 4 and commits final text at chunk 16, so the penalty is judged on what each pass is for.
- The evidence is B0's own lineage: **A1L** (B0 + 8k replay at dp 0.005) against **A1C** (amendment 23: the same at dp 0, same tree, data and seed).
- **dp = 0.005 in S2 if and only if all three hold:**
  - **(a) Final text:** chunk-16 full-dev WER, A1L − A1C, one-sided 95% upper bound **≤ +0.10 pp** (paired speaker bootstrap, EMA, pooled, canonical scorer).
  - **(b) First token:** chunk 4, end-of-first-word clock, full dev, eval-lane's paced pipeline. A1L is **≥ 40 ms earlier than A1C** on the mean.
  - **(c) Provisional text:** chunk-4 full-dev WER, A1L − A1C, upper bound **≤ +0.15 pp**.
- **Otherwise dp = 0**, and amendment 22's not-adopted stands.
- **Order:** this rule replaces amendment 22 only for S2's commit-pass deployment. It is fixed now, before eval-lane measures A1C's latency.
  - (a) / (c) come from the existing full-dev decodes: A1L (eval-lane's) and A1C (x7, 08:35Z, DONE).
  - (b) needs A1C's traces, which eval-lane runs next.
- If S2 adopts dp under this rule, the B0 bridge (Amendment 3 addendum) runs at the same dp, so the bridge evidence matches S2's recipe.

## Amendment 5 (2026-11-23T09:51:52Z, h53-trunk recording the lead's 09:49:48Z decision; written BEFORE any B0-bridge result and before any S2 evaluation exists)
**What it supersedes:**
- the A3 addendum's launch condition ("pilot PASS and bridge PASS");
- the lead's 09:43Z interim line ("S2 on pilot PASS alone, bridge as kill-check").

**Readings at writing** (eval-lane 09:47:49Z, accepted as registered):
- H52 pilot = **INCONCLUSIVE, provisional**: −0.096 pp [upper −0.033] at chunk 4, both seeds better, under the −0.17 bar. No adoption claim is made from it.
- Amendment 4 = **met**, so **dp 0.005 in S2** (blind first-token −76.5 ms).

### S2 runs as the PROVISIONAL main train
- **Why** (the lead): the release question is ~4,190 h of R0 labels at full exposure on B0, which no 1R test settles, and the island's alternatives produce nothing for the final model.
- **Pin: main 90b7f4d, re-confirmed** (python/ultra content 3780b451…).
  - Everything S2 needs is there: #87's `extra_splits`, `delay_penalty`, the H44 fixes.
  - The changes on main since then are default-off or not imported by S2's recipe: #93 MFA anchors (off), #100 FFNN-LM trainer (separate scripts), #86 VRAM guard (separate), plus engine and docs.
  - The bridge arms run on the same tree. Clean worktrees, one PIN_COMMIT on every node.
- **Init:** B0 raw, rank 0's `ckpt-00049000.pt` (2a3dd9fb…); x7 uses its sha-checked copy. Fresh optimizer, EMA restarts.
- **World 4 (x5 ×2 + x7 ×2):**
  - `budget_s=5200 max_batch=1152 ddp_global_batches=true`;
  - `lr=0.0003511 warmup=1000 total_steps=10000000 min_lr_ratio=0.02` (WSD-stable, the low re-warm, ≙ 1.241e-4 at 650 s);
  - `ema_decay=0.999599` (τ 3,600 audio-h, A2);
  - `workers=20`, NCCL a1env, seed 7919, chunk modes {4,8,16}, **`delay_penalty=0.005`**;
  - the rest of `configs/v3_trunk_s1.set` unchanged.
- **Data:** 960 h + Libri-Light **small v1.1** (h52 09:48Z re-finalisation, labels v1.1 only):
  - `extra_splits=/home/ubuntu/jsr-data/librilight/shards/r0_v11/librilight.small`, natural share, no confidence floor;
  - **the same absolute path, and so the same basename, on x5 and x7**, with the manifest sha256 checked on both nodes before launch;
  - the v1 directory (`r0_small_tl` / `shards/r0/`) is not used by S2.
- **Gain augmentation:** not pinned or tested at 90b7f4d (trainer-lane 09:47Z), so it **joins at the medium boundary (S3)** (Amendment 6).
- **Launch receipts**, posted within 10 min of the start:
  - per node: PIN_COMMIT, content hash, init sha256, shard-manifest sha256;
  - the start record: step 0, Position, world 4, rank → host / GPU UUID map, all DDP checksums equal;
  - the first logged step: loss and audio-h/h;
  - 5-min telemetry: per-rank step / comm / data-wait, GPU busy / VRAM, the negotiated NCCL transport.
- **The B0 bridge (S2c x9 / S2p x8) is EVIDENCE ONLY.** It is read by its registered rule and reported (the small-data effect on B0 at 1R, dp 0). It is **not** a kill switch, and not matched to S2 (dp 0.005).

### S2's own gates (eval-lane registers them; paired against B0 on canonical full dev)
- **(a) Safety, at every branch point** (S2's and S3's):
  - measured on the BP's own EMA weights, chunk 4, pinned PyTorch evaluator, greedy;
  - **point estimate > +0.20 pp worse than B0 (8.302 EMA greedy, chunk 4) → stop at the next checkpoint**, kept, not deleted;
  - B0 stays the candidate, and the lead re-plans.
- **(b) Decisive, at BP-m:**
  - **S2/S3 must beat B0 at chunk 4** (one-sided 95% upper bound < 0), with chunks 8 / 16 non-inferior by the registry margins (0.094 / 0.20).
  - It is read on **BP-m's fast cooldown** (the annealed endpoint, recipe below), because B0 is the end of a full anneal: A1's stable-phase EMA sat ~0.11 pp above its annealed end. BP-m's EMA reading is reported beside it.
  - **FAIL → stop S2/S3 and re-plan; B0 stays the candidate.**
- **(c) Adoption** into a release candidate: by A6 (confirmation half, Holm, latency / CPU guards), at the release cooldown.

### Branch-point schedule (exposure-indexed; "exposure" = logged real audio seconds summed over the segment's train records)
- **S2-BPk** = the first 500-step checkpoint past k × 10,000 audio-h of S2.
  - S2-BP1 ≈ 1.1 h after launch at world 4 (~8.8k audio-h/h), or ≈ 0.8 h at world 6 (~13k).
  - **S2-BP1 is the base of the fast-cooldown factorial** (FACTORIAL.md).
- **The S3 boundary** is the first S2 window-boundary checkpoint after Amendment 6's conditions hold. Those include **S2 ≥ 10,000 audio-h**, so S2-BP1 always exists before medium joins.
- **S3-BPk** = the first S3 checkpoint past k × 10,000 audio-h of S3.
- **BP-m := S3-BP1:**
  - ≥ 1R (1,444 audio-h) of exposure after medium joins, as the lead requires;
  - ≈ 0.9 epoch of medium at Amendment 6's 0.5 pseudo-label cap.
- **The fast cooldown for gate (b)** runs from BP-m's raw weights with S2/S3's own recipe:
  - dp 0.005, the trunk's gain setting, no factorial factors;
  - 650 s / 144, 8,000 steps, 200 warmup steps to 1.241e-4, cosine to 2.12e-5, EMA 0.9995, seed 1234;
  - on the fastest free 24 GB card (4090, ~50 min), ahead of the factorial's queue.
  - An A100 is used only if the lead lends one; no other A100 launches.

### World 6 (x8 + x9 join when the bridge ends, ~10:45Z): two paths, fixed now
- **W6-A, the resize path, if #99 is merged with v3-reviewer's fix and its tests pass by then:**
  - S2 stops at a checkpoint that lands **exactly on a window boundary** (as S1's 09:45Z restart did; zero cursor perturbation);
  - it restarts at **world 6** with `budget_s=7800 max_batch=1728 lr=0.0004300` (√-rule; #99 moves base_lrs / initial_lr to cfg.lr) and `ema_decay=0.999398` (τ 3,600 audio-h);
  - workers 20 (~83% CPU on x8 / x9's 30 vCPUs), a1env;
  - the checkpoint is copied to x8 / x9 sha-checked; all 6 DDP checksums must agree in the restart receipt.
  - **This restart is the one allowed pin change inside S2:** 90b7f4d → the commit that merges #99. Conditions:
    - every change between them is default-off for S2's recipe (#93, #100, #86, engine / docs);
    - #99's witness tests pass on that commit (bit-exact resume when nothing changes; refuse a mid-window shape change unless explicit);
    - the new pin is hash-checked on all 6 nodes.
  - Rollback to world 4 if per-GPU audio-h/h over the first 30 min is < 85% of world 4's.
- **W6-B, otherwise:**
  - x8 + x9 join at the **S3 boundary**: S3 starts at world 6 as a new run, with the world-6 row at warmup 500;
  - until then they finish the bridge decodes and serve as main-train producers (label checks, KD), **with no other training launches**.

### Amendment 5 note: the world-5 row (2026-11-23T09:53:09Z, lead 09:51:17Z: launch with every staged A100, x9 included; written before any S2 result)
- **World 5 = x5 ×2 + x7 ×2 + x9:** `budget_s=6500 max_batch=1440 lr=0.0003925 ema_decay=0.999499`.
  - LR is 3.511e-4 × √(6500 / 5200), still ≙ 1.241e-4 at 650 s, so the re-warm stays low.
  - EMA is exp(−6500 / 3600²) (τ 3,600 audio-h).
  - Everything else as in Amendment 5's world-4 row.
- **If x9 is not staged at fire time** (pin, small v1.1 at the same path, B0 copy), S2 launches at world 4 and x9 joins at the medium boundary. The launch is not delayed for x9.
- **Rank layout:** nproc per node (2, 2, 1). The start record must show global ranks 0-4 unique with the expected host / GPU UUID map; torch's elastic agent assigns ranks from the gathered local world sizes. Fallback: A1's one-torchrun-node-per-GPU layout (nnodes 5, nproc 1).
- **Later world changes follow the same √ / exposure rule.** World 6 is `budget_s=7800 max_batch=1728 lr=0.000430 ema_decay=0.999398`, at the medium boundary or by #99's resize.

## Amendment 6 (2026-11-23T09:56:37Z, h53-trunk; the lead's 09:44Z item 3 / 09:49:48Z item 4; written BEFORE medium is finalised (h52's trigger ~10:40Z) and before any S2 result)
### Naming (supersedes Amendment 5's "S3")
The lead calls the whole main train "S2" (B0 + dp 0.005 + 960 h + small → + medium) and reserves **S3** for Libri-Light large tier-A (labels ~12:30Z+). So:
- **S2a** = the small segment (launched now);
- **S2b** = the medium segment (a new run at the boundary below);
- **BP-m := S2b-BP1** (the first S2b checkpoint past 10,000 audio-h of S2b).
- Amendment 5's gates and schedule read with these names.

### The S2a → S2b boundary
S2b starts as a **new run** at the first S2a checkpoint that lands **exactly on a sampler-window boundary** after ALL of these hold:
1. medium is finalised by h52's pre-registered finaliser, **v1.1 labels only** (U00320 re-labelled);
2. the sealed manifest sha256 is published by h52 and checked on **every** island node, at **one absolute path (one basename)** on all nodes;
3. medium's speaker / book overlap verdict against dev / test is logged in `LOOKS.jsonl`, with excluded files listed;
4. **S2a ≥ 10,000 audio-h** (S2a-BP1 exists and is kept as a branch point);
5. the S2b pin's tests pass, and the S2b launch line is posted.

**New run, new pin allowed:**
- main at the boundary, provided every change since 90b7f4d is either default-off for S2b's recipe or part of it on purpose (gain augmentation, #99);
- clean worktrees on all nodes, LC_ALL=C content hash recorded.

**Init:**
- S2a's boundary checkpoint, raw weights; fresh optimizer; EMA restarts (τ 3,600 audio-h);
- **500 warmup steps back to the stable LR of S2b's global batch** (√-rule), then WSD-stable (`total_steps=10000000 min_lr_ratio=0.02`);
- seed 7919, chunk modes {4,8,16}, **dp 0.005** (Amendment 4);
- **world 6 if all six A100s are free** (`budget_s=7800 max_batch=1728 lr=0.000430 ema_decay=0.999398`), otherwise world 5 (`6500 / 1440 / 0.0003925 / 0.999499`).

**Gain augmentation joins here** if it missed the launch:
- trainer-lane's pre-registered distribution (≈ 0 to −35 dB plus mild jitter, waveform scaled before fbank), on all sources;
- its clean-level effect is guarded by gates 5(a) / (b) at full dev 0 dB (the factorial and its run 1′ are cut to v3.1, lead 09:5xZ).

### The medium mix
**Pools:**
- LibriSpeech 960 h (human labels);
- Libri-Light small v1.1 (378.7 h; h52's re-finalised count when posted);
- Libri-Light medium v1.1 (h52's finalised hours H_m, ≈ 3.8k h by the lead's ~4,190 h total).
- Only h52's pre-registered filters apply. The A1-agreement rule is dropped for medium (lead 09:03:45Z). No confidence floor (as the pilot and S2a).

**Cap: Libri-Light (R0 pseudo-label) share = 0.5 of the audio,** i.e. LibriSpeech ≥ 0.5. Why 0.5:
1. **Tested-regime extrapolation ≤ 2×.** The only measured share is the pilot's natural share (Libri-Light ≈ 28% of audio): −0.096 pp at chunk 4, both seeds, upper < 0, provisional. That is also S2a's share. 0.5 is a 1.8× step; medium's natural share (~80% Libri-Light) would be a ~2.9× step into teacher-dominated training.
2. **Label-noise budget.** R0's canonical full-dev WER is 2.82 / 8.08% (eval-lane 07:29Z), so its token error on Libri-Light is plausibly ~3-6% before h52's filters. At a 0.5 share, erroneous targets are ≤ ~1.5-3% of a step's tokens: below B0's own dev error (4.48 / 12.37%), so the human half dominates where R0 and the references disagree.
3. **LibriSpeech is memorised** (B0's TRAIN 1-best 0.234%, h49 08:28Z). Its marginal value is as an anchor, not as new information, so its share can fall from S2a's ~72% to 50% while every step stays anchored.
4. **Teacher conventions are watched, not trusted.** R0 is a 320 ms-chunk Zipformer. Gates 5(a) / (b) at every branch point, plus first-token latency on each fast cooldown, catch any drift toward R0's timing.

**Within the Libri-Light half,** small and medium are sampled at natural share by row groups (both come from h52's finaliser in one shard layout, so row-group share ≈ hours share). Small is not upweighted: S2a has already given it ~7 epochs.

**Implementation at the pin** (`extra_splits` / `extra_base_weight`, data.py):
- `extra_splits=<small v1.1 dir>,<medium v1.1 dir>` with no per-split weights;
- `extra_base_weight` = the LibriSpeech share **in row-group units**, set so the LibriSpeech share of *audio* is 0.5. That is extra_base_weight = a_LL / (a_LS + a_LL), with a_X = the mean audio seconds per row group of source X, computed from the manifests (rows per row group × mean duration) before launch. Both values and the result are recorded in the launch receipt.
- **The share is enforced per epoch** (row-group draws) and inherited by batches through 96-row-group windows. **Strict per-batch stratification is not available at the pin.**
- **So the per-batch share is audited:** a receipt computed offline from the deterministic sampler (seed, manifest) for the first 2,000 steps. Targets:
  - the median per-batch Libri-Light audio share is 0.45-0.55;
  - its 95th percentile is ≤ 0.65.
  - If the audit fails, window_groups is raised at the next window boundary, recorded.

**Exposure:**
- S2b runs WSD-stable until the release cut (Amendment 7);
- branch points every 10,000 audio-h (S2b-BP1 = BP-m). At the 0.5 cap, each 10k audio-h is ≈ 1.2 epochs of Libri-Light (~4.2k h) and ≈ 5.2 of LibriSpeech;
- the cap stays 0.5 for all of S2b; any change is a new amendment at a branch point.

## Amendment 7 (2026-11-23T09:58:07Z, h53-trunk; the lead's 09:5xZ wrap-up: "a release checkpoint this afternoon"; written before any S2 result)
### Scope changes (the lead 09:53Z)
- The S2 fast-cooldown factorial and its run 1′ are **cut to v3.1**, so FACTORIAL.md is parked unrun.
- Branch points now serve only gates 5(a) / (b) and the release cut:
  - **every S2a / S2b BP:** gate (a) on its EMA (eval-lane's GPU evals on the 24 GB cards; the release decoder on xbabe0);
  - **BP-m = S2b-BP1:** also gate (b), read on BP-m's fast cooldown (Amendment 5's recipe, on the 4090, ~50 min + decodes);
  - no other fast cooldowns are planned.

### The release cooldown (RC): one annealed branch from the last main-train checkpoint
- **Base:** the cut checkpoint (below), raw weights. Its EMA is kept for reference. New run `H53_RC1_s7919`.
- **Where:** **the whole island** at the main train's world size, so world 6: `budget_s=7800 max_batch=1728` (world 5: `6500 / 1440`). The main train pauses at the cut, resumable bit-exactly; there is no other A100 use.
- **LR path:** a fresh AdamW with **200 warmup steps back to S2b's own stable LR** (4.30e-4 at world 6; 3.925e-4 at world 5), so the stable phase is not re-warmed, then the trainer's **cosine to a floor of ≈ 2.2e-5** (`min_lr_ratio=0.051`).
  - The floor is A1's 2e-5 end at 6,500 s, √-scaled; B0 is that endpoint.
  - WSD's decay shape is approximated by the existing cosine. No new schedule code at the pin.
- **Length D = max(10,000 audio-h, 0.15 × E_main)**, with E_main = logged exposure since B0 (S2a + S2b). At world 6, 10k audio-h ≈ **5,130 steps ≈ 45 min** (6 × ~2,350 audio-h/h per GPU, measured at world 4). `total_steps` is set from D and the measured audio per step at the cut.
- **EMA:** τ_RC = 0.25 × D (= 2,500 audio-h at D = 10k, the fast-cooldown family's ratio). `ema_decay = exp(−B / (3600 τ_RC))` = **0.999134** at 7,800 s (0.999278 at 6,500 s).
- **Recipe otherwise = S2b's, unchanged:**
  - the 0.5-capped mix;
  - dp 0.005;
  - gain augmentation as in S2b;
  - chunk modes {4,8,16};
  - **seed 7919** (the protected final seed);
  - checkpoints every 500 steps; evals are monitor-only.
  - **No other change rides in the anneal:** an LS-only finish or CR-CTC were untested factorial arms, now cut.
- **The candidate:**
  - the RC endpoint's **EMA and raw** weights are both scored on the selection half (dev_split_v2), chunk 4, canonical;
  - **EMA is the candidate unless raw is better there by ≥ 0.05 pp**;
  - the candidate gets **one** confirmation-half look under A6 (Holm across RC looks, ≤ 2):
    - against B0: pooled upper < 0, splits ≤ +0.20 pp;
    - latency: lower median, paired p95 ≤ +20 ms (dp 0.005 should make it lower);
    - xbabe0 CPU ≤ 1.05×;
    - INT8 engine − PyTorch within ±0.05 pp.
  - The release decoder (commit pass at chunk 16 + beam 16 + M5 / LODR, per the lead's v3.0.0 definition) is then measured for the H54 bar (**≤ 7,895 edits** for the whole artifact).

### When to cut
- **Precondition:** gate (b) PASSED at BP-m. If (b) FAILs, there is no RC from this train: S2 stops, and **B0 ships** (the lead's rule).
- **Cut point:** the first S2b checkpoint **exactly on a window boundary** at or after **max(T_cut, the time gate (b) is read PASS)**.
  - **T_cut = 14:00Z by default** (the lead may move it).
  - The RC (~45 min), export and INT8 parity, release-decoder full dev (~45-60 min on xbabe0 / the 24 GB cards) and the A6 look then give **a candidate ≈ 15:30-16:00Z**.
- **If (b) is still pending at T_cut,** S2b continues until it reads, then cuts at the next window boundary (PASS) or stops (FAIL).
- **Expected exposure at a 14:00Z cut:**
  - S2a ≈ 10k audio-h;
  - S2b ≈ 35-40k audio-h (world 6 from ~11:15-11:30Z), ≈ 4-5 epochs of Libri-Light at the 0.5 cap.
  - So E_main ≈ 45-50k, and D = max(10k, 0.15 E_main) = 10k (≤ 7.5k would be the 15%).
- **A second RC (RC2) is optional and pre-registered:**
  - only if RC1 has been scored and time remains;
  - S2b resumes from the cut and RC2 branches ≥ 20k audio-h later with the same recipe;
  - the better of RC1 / RC2 on the selection half takes the remaining confirmation look (Holm, ≤ 2);
  - at most two RCs (README A4).

### Amendment 3 addendum 2 (2026-11-23T10:07:27Z, trainer-lane; before S2p s1234 has any result): bridge seed 2
- x9 is free (S2c s1234 and its decodes finished 10:04Z). The bridge reads 1 vs 1, which is weak for a kill-check, so each arm gets a **second seed, 4321**, with the same recipe and tree:
  - **S2c s4321** on x9 now;
  - **S2p s4321** on x8 after S2p s1234 (~10:25Z).
- **Reading:** eval-lane's registered bridge rule, applied to the **seed-pair means** once all four arms exist. Until then the s1234 pair is reported alone and is not decisive by itself.

## Amendment 6.0a (recorded 2026-11-23T12:2xZ by trainer-lane for h53-trunk; the lead's ruling of 10:20:04Z, made before any S2b data or result): gain augmentation is DROPPED from v3.0's training
- **Supersedes:** Amendment 5's "gain augmentation joins at the medium boundary", Amendment 6's "Gain augmentation joins here if it missed the launch", and Amendment 7's "gain augmentation as in S2b".
- **So:** S2b = S2a's recipe + medium (this file's Amendment 6), nothing else. RC1 inherits S2b's recipe without gain augmentation.
- **Why** (the lead):
  - (1) The engine-side warm AGC recovers most of the quiet-input loss.
  - (2) With the factorial cut there's no matched control, and gain at S2b would confound gate (b).
  - (3) Gain code wasn't written or pinned at 90b7f4d.
- Quiet-mic robustness ships as the engine's opt-in warm-AGC flag. Gain-aug training is deferred to v3.1.

## Gate (b) Amendment 1 (cross-reference; the lead's 11:20:26Z design, registered by eval-lane at 11:23:12Z before S2b-BP1 existed)
- **Two-stage gate (b) at BP-m = S2b-BP1, Bonferroni across stages** (each one-sided 97.5%):
  - **Stage 1:** S2b-BP1's stable-phase EMA vs B0, canonical full dev (decode only). PASS iff chunk 4's upper < 0 and chunks 8 / 16 are non-inferior by the registry margins. Stage-1 PASS = gate (b) PASS.
  - **Stage 2, only if stage 1 doesn't pass:** Amendment 5's fast cooldown from BP-m's raw weights on x9 alone. S2b pauses at an exact window boundary meanwhile and resumes at world 6 from it.
- **Registry:** `h53-S2-gate-b-amend1-stage{1,2}-c{4,8,16}`, entries_sha256 176ca6fb…; plan `h43/families/h53-gate-b-amendment-1.json` (54ce5206…).
- It answers the data-locality blocker (cloud → on-prem measured at 16 MB/s, so the 24 GB cards can't hold S2b's ~280 GB mix in time) without taking an A100 from S2b.

## Amendment 6.1 (2026-11-23T12:2xZ, trainer-lane; the lead's ACK 12:16:33Z): S2b's window_groups = 96, a disclosed deviation from the 11:19Z audit rule, for throughput
- **The audit rule** (posted 11:19Z before medium was sealed; lead ACK 11:17:33Z): the smallest of {768, 1024} that passes median 0.45-0.55 / p95 ≤ 0.65; if neither passes, 768 when its p95 is within 0.01 of the best of {768, 1024, 1536}, else the best.
- **The audit on the sealed medium** (fast_audit.py, the pin's packing, exact per-row FLAC frames, first 2,000 global batches, bw 0.527689):

| window_groups | pooled LL audio share | median | p5 | p95 |
|---:|---:|---:|---:|---:|
| 96 | 0.4999 | 0.4973 | 0.218 | 0.830 |
| 768 | 0.4990 | 0.4887 | 0.361 | 0.681 |
| 1024 | 0.4992 | 0.4890 | 0.366 | 0.666 |
| 1536 | 0.4995 | 0.4876 | 0.374 | 0.664 |

  - No value passes. The rule chose **1536**, and S2b first fired at 1536 at 12:02:10Z.
  - p95 levels off because packing sorts each window by duration and the sources' length profiles differ: LibriSpeech has 49.7% of its audio at 14-16 s, while Libri-Light has 12.5% at 17-20 s.
- **Throughput at 1536** (12:06:18-12:13Z, steps 25-496):
  - Mean 0.88 s per step against S2a's 0.54 s at 96, i.e. **~30-35% slower**. Forward and backward + comm were normal; every rank's step time inflated whenever the loader built its next window.
  - Each rank packs a 1536-group window from ~151k rows' headers (~33 GB of row bytes as Python objects) inside the training process. Trainer RSS was 67 GB per rank. Logged data wait stayed 2-7%.
  - The first window took 3m51s to load.
- **Decision:** S2b was stopped at ckpt-00000496 (12:13:00Z, by recorded PID), and the attempt was renamed `H53_S2b_medium_s7919_wg1536_aborted` (kept; **its 496 steps are not counted**). S2b **re-fired fresh at 12:15:40Z with window_groups 96**, from the same init (S2a ckpt-00010516, 0f2a089b…) and with everything else identical.
  - At 96 the pooled share (the cap this amendment enforces) and the median are on target. Only the per-batch spread is wider: p95 0.830 / p5 0.218.
  - S2a ran 96 at world 6 at a steady ~13k audio-h/h.
- **Measured at 96 after the re-fire:** 25-step walls of 13.2-14.2 s, 13.0-13.5k audio-h/h, data wait 4-6%.
- **BP exposure is measured from the 12:15:40Z attempt only.** S2b-BP1 = the first 500-step checkpoint past 10,000 audio-h of that attempt.

## Amendment 8 (2026-11-23T14:2xZ, trainer-lane; the lead's 14:13:25Z plan; written BEFORE the large tier-A manifest exists and before any S3 data or result): S3 = the trunk + Libri-Light large tier A
### Scope
- **v3.0.0 tags on RC1's candidate.** S3 and RC2 feed the next release.
- Right after RC1 ends (~14:50Z), **the trunk (S2b) resumes at world 6 from its cut ckpt-00010818**, exactly: same pin, config and window_groups 96. It trains until the S3 switch.
### Data: large tier A, a disk-budget subset
- h52's large tier A, v1.1 labels only, under h52's pre-registered filters (as for medium: amendment 4's NaN rule, no A1-agreement rule).
  - It is finalised by h52's finaliser in medium's shard layout (2,000 rows per shard, 100-row row groups, bucketed global shuffle). Large cut 15:30Z, sealed manifest ~15:45Z.
  - The overlap verdict against dev / test is logged in LOOKS before use (as Amendment 6 condition 3).
- **The subset:** S3's large = the longest prefix, in manifest order, of h52's sealed shard list whose total size fits **min over the four island nodes of (free − floor)**, measured at staging time.
  - Floors: 60 GB on x5 (the finaliser host), 20 GB on x7 / x8 / x9 (trainer checkpoints only).
  - The finaliser's bucketed global shuffle makes any prefix a uniform random subset of the cut.
  - The S3 manifest is the first K lines of h52's SHA256SUMS; its own sha256 is posted. Path `~/jsr-data/librilight/shards/r0_v11/librilight.large.s3/` on all four nodes.
- **Disk at writing** (x8 / x9 hold 497 GB, with medium 200 GB, small 22, LibriSpeech 57 and their run dirs):

| node | free now | floor | budget | + superseded v1-small copies (never read by S2 / S3) |
|---|---:|---:|---:|---|
| x5 | 212 GB | 60 | 152 | + 22 (`shards/r0`) |
| x7 | 512 | 20 | 492 | + 44 (`r0`, `r0_small_tl`) |
| x8 | 65 | 20 | 45 | + 22 (`r0_small_tl`) |
| x9 | 76 | 20 | 56 | — |

  - So x9 sets the subset at **~55 GB ≈ 950 h of large** (at medium's ~58 MB per audio-h). Superseded copies are removed only after their survivor is checked.
  - More room on x8 / x9 needs the lead's decision. The candidates there are old research run dirs (x9: ~50 GB of D2* / H41* / H48* / H53 bridge runs, plus `A1_S_r0bpe_960h` = A1 rank 3's non-canonical copy, 11 GB). They are deleted only after a canonical copy is confirmed, and never on my own call.
  - Alternatively S3 runs at world 4 on x5 / x7 (budget ~150-170 GB ≈ 2.6-2.9k h), with x8 / x9 released.
  - The choice is recorded before the switch.
### The mix (extends Amendments 6 and 6.1)
- **LibriSpeech = 0.5 of the audio (cap unchanged).** Within the Libri-Light half, small + medium + the large subset run at their natural share by row groups.
- **extra_base_weight = a_LL / (a_LS + a_LL)**, with a_LL = the row-group-weighted mean audio seconds per row group over the three Libri-Light sources. It's computed from every row's FLAC frames (`dur_table.py` / `bw_compute.py`, as for S2b) and recorded in the launch line.
- **window_groups = 96** (Amendment 6.1). The per-batch audit (`fast_audit.py`, first 2,000 batches) is run and recorded, with no window_groups remedy (6.1's throughput measurement).
### Staging (one writer: trainer-lane; data wait ≤ 5%)
- From the finaliser host to x5, pre-copying completed shards as they land (as for medium), then x5 → x7 / x8 / x9 over the cloud LAN.
- **Rate limit:** ≤ 250 MB/s per target, the copy processes pinned to the spare cores at idle IO priority. **A guard pauses the copy while the trunk's logged data wait is > 5% on two consecutive records** and resumes below it.
- `sha256sum -c` against the S3 manifest on all four nodes; each node's file set must equal the manifest's.
### The switch
- At **the first trunk window boundary after all four nodes verify the S3 manifest** (`boundary.sh`), **S3 starts as a new run, `H53_S3_large_s7919`:**
  - init = that boundary checkpoint's raw weights; fresh AdamW; 500 warmup steps back to 4.30e-4, then WSD-stable; EMA restarts (0.999398, τ 3,600 audio-h);
  - seed 7919, dp 0.005, chunk modes {4, 8, 16}, world 6, the 0.5-capped mix above;
  - pin 90b7f4d (unchanged: everything S3 needs is there).
  - The launch line is posted before the fire, the receipts after.
### Branch points, gates, RC2
- **S3-BPk** = the first 500-step checkpoint past k × 10,000 audio-h of S3. Gate (a) on each BP's EMA (the stop rule is unchanged).
- **RC2 (the registered rule, unchanged):**
  - only after RC1 is scored;
  - from a trunk window-boundary checkpoint ≥ 20,000 audio-h after RC1's cut (S2b + S3 exposure);
  - Amendment 7's cooldown recipe on the trunk's then-current mix;
  - the better of RC1 / RC2 on the selection half takes the remaining A6 look (Holm, ≤ 2 looks). For the next release.

### Amendment 8 addendum 1 (2026-11-23T15:1xZ, trainer-lane; the lead's ACK of 15:0xZ; before any S3 data or result)
- **The move guard is baseline-relative.** It pauses while the trunk's last 2 logged records BOTH exceed max(5%, baseline + 0.5 pp) data wait, or its rate is below 0.95 × baseline; baseline = the median of the 40 records before the first move. The reason: the resumed trunk sat at ~5.65% with nothing moving, so a fixed 5% line would never let a move start.
- **The S3 launch is the data-wait fix point (cap ≤ 5%).** The launch line sets **`workers=24` per rank** (was 20), or whatever brings every rank ≤ 5%, provided x5 / x7 stay ≤ ~80% CPU with two trainers each (they run at load ~8-10 of 60 cores now).
  - workers is part of the world-6 row and is recorded in the launch line. It changes loader parallelism only: the ordered loader output is identical.
  - S3's first telemetry reports per-rank data wait. The trunk isn't restarted for this.
