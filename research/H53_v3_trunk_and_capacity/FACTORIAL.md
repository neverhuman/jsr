> **PARKED, 2026-11-23 09:53Z: CUT TO v3.1 by the lead's research wrap-up (owner directive). Not registered, not run.** Kept as the v3.1 starting design. The lead's 09:32Z run-1′ ruling (gain OFF control; clean-level guard: 0 dB point ≤ +0.087 and one-sided 95% upper ≤ +0.20) is recorded here for v3.1, unapplied. v3.0.0's gates are AMENDMENTS 5-7.

# H53 S2 fast-cooldown factorial: four slow-lever factors in 8 runs, resolution IV, fold-over to the full 2^4

**Status:** PRE-REGISTRATION DRAFT by h53-trunk, 2026-11-23 ~09:20-09:45Z (lead's 09:07Z audit actions, item 3; resumed 09:08Z). It builds on trainer-lane's draft `prereg/S2_FC_factorial_DRAFT.md` (09:09Z); the changes and reasons are in §10. **No launches.** It is frozen, then registered by eval-lane, **before S2's first branch point (BP1) exists**. No arm, decode or result of this design exists at writing time.

**Owners:**
- h53-trunk: the design (this file);
- trainer-lane: the factor code, then the launches;
- eval-lane: the estimator and the registry entries;
- h46-lm: the "+text" module;
- fleet-sched: the cards.

---

## 1. What this decides, and what it cannot
- **It decides the release-cooldown recipe:** which of four changes are switched on in the final anneal that produces a v3 release candidate from S2. Every factor is a change applied *during a cooldown*, and is judged on the annealed endpoint.
- **It does not decide trunk-phase recipes.**
  - B (regularisation) is a slow lever (tips/v3 FRONTIER WER-1: "they hurt early and pay late"). A 1R cooldown is biased against it.
  - So B gets a second reading (per-token NLL, §5). If its WER is inconclusive but NLL moves the right way, it is flagged **PROMISING-SLOW**: a candidate for a trunk-phase test (S3 or the next trunk), never adopted from this screen.
- **Not a screen for capacity.** S-32 growth stays out (§9). The registered P1 consequence is "capacity waits for S3 with its own pilot" (eval-lane 09:01Z, lead 09:03Z). Within this design it would also double or quadruple the cost of the arms that carry it, and they would need A100s.

## 2. Base and cooldown recipe (the low re-warm)
- **Base:** S2's BP1 raw weights (the first 500-step checkpoint past 10k audio-h of S2), sha256 recorded in `MANIFEST` before any arm starts. Each arm gets a fresh optimizer, and its EMA restarts from those weights.
  - **Fallback, fixed now:** if S2 does not launch (H52 pilot or B0 bridge fails), the same design runs from **B0's raw weights** (2a3dd9fb…) on 960 h. Its control arm is then the bridge's S2c recipe √-translated to 650 s.
- **Cooldown recipe, identical in every arm except for its factor settings:**
  - the H41 replay machinery (`h41_replay.sh`) at **650 s / 144 per step, 8,000 steps** (≈ 1,444 audio-h = 1R of exposure);
  - **200 warmup steps to 1.241e-4**: S2's 3.511e-4 at 5,200 s, √-scaled to 650 s. This is the "low re-warm" of Amendment 3, i.e. no re-warm relative to S2's own LR;
  - **cosine to 2.12e-5** (`min_lr_ratio=0.1709`), the same √-scaled path as the B0 bridge (1.756e-4 → 3e-5 at 1,300 s). So the factorial's control follows the bridge's schedule in √-LR terms. trainer-lane's draft floor of 3e-5 is replaced to keep that equivalence;
  - **EMA 0.9995** (τ ≈ 361 audio-h at 650 s, the replay family's, as in the bridge);
  - S2's data mix (960 h + Libri-Light small via `extra_splits`, natural share, the same basename on every host), chunk modes {4,8,16}, `eval_chunks=8,16`, checkpoints every 500 steps;
  - **seed 1234 in every arm**: identical sampler order and identical SpecAugment / chunk streams wherever the factor does not change them;
  - `delay_penalty` per factor C (0 or 0.005).
    - If S2's stable phase itself runs dp 0.005 (Amendment 4's blind criterion (b)), C is read as "keep or drop dp in the release cooldown".
- **Batch:** 650 s / 144 is pinned for comparability across card types (A100 and 24 GB cards). This is the owner-rule exception "paired arms keep their pinned batch", recorded as the capacity reason.
- **Tree:** one pinned commit: main + the A / B / D factor code, all off by default.
  - Each factor's witness test shows the off level bit-identical to the plain fast cooldown.
  - `check_init_compatible` must accept the two config changes that keep the function: B's `layer_drop` (identity in eval mode) and D's new zero-initialised tables (`predictor.text.*` missing in BP1). h46-lm's integration patch already initialises with `strict=False` for D.

## 3. The four factors (− = off; + = on)

| | factor | + level, fixed now | cost per step | code (owner) | engine at deployment |
|---|---|---|---|---|---|
| **A** | trained full-context commit mode | **dual-mode with lattice consistency** (tips/v3 BREAKTHROUGH §7.4, T3b), detailed below | ×2.1 | to write (trainer-lane), ~1 day per the T3 build list | none for chunk 16. A chunk-0 commit tier needs a segment-level encoder mode (engine chunk ≤ 64 today) |
| **B** | regularisation parity (augmentation + stochastic depth) | **lhotse-strength SpecAugment** plus **stochastic depth** (FRONTIER WER-1), detailed below | ×1.05 | to write (trainer-lane); the reference code is in tips/v3 `train_snippets_ref.py` (`spec_augment_v2`, `drop_path`) | none (eval identity) |
| **C** | delay penalty | `delay_penalty=0.005` (− = 0) | ×1.0 | exists | none |
| **D** | "+text" predictor tables | h46-lm's design (amendment 3, lane commit 6cf33ac / code 20b064d), detailed below | ×1.1 | h46-lm's module + integration patch; trainer-lane's hook (~8 lines + 4 fields) | predictor context 4 + ~3.2 MB int8 tables (Codex). **Export must refuse it until then:** the engine silently ignores unknown arch fields (h53's 07:1xZ hazard) |

**A, dual-mode with lattice consistency:**
- Two encoder passes on the same augmented batch:
  - **small** chunk drawn per step from the trunk's {4, 8, 16};
  - **large** = chunk 0 (full context).
- Both get the transducer + CTC losses, weighted ½ / ½.
- A **symmetric** KL between the two passes' joiner distributions at every node of the large pass's pruned band, **λ = 0.3**.
- **Not a CTC-posterior KL, and not one-way.** The NVIDIA 2026 result cited there found CTC-head consistency degrades the streaming transducer and a one-way offline teacher is the weaker direction.
- To stay inside memory, each 650 s batch runs as two 325 s halves with both graphs alive (the `two_view.py` pattern), with a one-graph-equivalence witness test. Refused together with `cr_ctc_*`.

**B, regularisation parity:**
- **lhotse-strength SpecAugment** (`specaug_recipe=lhotse`): time warp 80; up to 10 time masks of ≤ 100 frames, together ≤ 15% of the utterance; applied with p = 0.9; frequency masks unchanged (2 × ≤ 27). This against today's ~3.3-4.0% masked, no warp.
- **Stochastic depth:** `layer_drop=0.1`, drop-path on each of the four residual branches of block l with p_l = 0.1 (l+1)/16, inverted scaling, training only.
- **Why bundled:** both target the measured over-fit (B0's TRAIN 1-best 0.234% vs dev-clean 4.48%), as WER-1 bundles them. If B passes, a follow-up splits them.

**D, "+text" predictor tables:**
- Exact ids of the last 2 / 3 / 4 tokens, mod primes 16,381 / 16,369 / 16,363, index three 64-d tables. A **zero-initialised** 192 → 320 projection adds them to `pred_p`, so a D+ arm starts exactly at BP1's function.
- **Text loss weight 0.3** (h46's T1): next-token cross-entropy through the frozen joiner at the mean encoder vector ē, on the leak-free OpenSLR 11 stream (sha256 8e249254…).
- Gradients reach only the tables and projection from the text loss. The audio losses train everything as before.

**The 2^(4−1) leaves no free column, so these are out of this screen** (reasons in §9):
- H48c anchor;
- H49b sequence KD (STOPPED 09:10Z);
- `synth_aug` + input-gain augmentation;
- S-32 growth.

**Alternates, fixed now** (decided per factor, before that factor's first arm starts):
- **A:** if T3's code is not merged and witnessed by BP1 + 2 h, the A+ arms still wait for it; the BP1 checkpoint does not expire. The A− arms start at BP1, and the analysis waits for all 8.
  - If it is still not witnessed by BP1 + 4 h, **A becomes A' = T3c**: a second supervised pass at chunk 0 with λ = 0 (dual-mode without the KL), and the report says so.
- **B and D:** if not witnessed by BP1, the column becomes a **dummy factor** (both levels identical). Its "effect" then measures pure noise, and the design keeps its other three factors at full precision. No untested change is substituted.

## 4. Design: 2^(4−1), resolution IV, control inside the design, hardware blocks orthogonal to every main effect
**Generator D = ABC (I = ABCD).**
- Main effects are aliased only with three-factor interactions.
- Two-factor interactions are aliased in pairs: AB = CD, AC = BD, AD = BC.
- The interesting AB pair (consistency × augmentation; the CR-CTC paper found stronger masking helps a consistency-regularised model) is paired with CD (dp × text, implausible), so it is readable in practice. AC (dual-mode × dp, both touch emission timing) is paired with BD (implausible), also readable.

| run | A | B | C | D | block (AD) | R (cost) | role |
|---|:-:|:-:|:-:|:-:|:-:|---:|---|
| **1** | − | − | − | − | 1 | 1.00 | **the plain fast cooldown = the control** (also S2-BP1's scoreboard candidate) |
| 2 | + | − | − | + | 1 | 2.31 | |
| 3 | − | + | − | + | 2 | 1.16 | |
| 4 | + | + | − | − | 2 | 2.21 | |
| 5 | − | − | + | + | 2 | 1.10 | |
| 6 | + | − | + | − | 2 | 2.10 | |
| 7 | − | + | + | − | 1 | 1.05 | |
| 8 | + | + | + | + | 1 | 2.43 | all four |
| **1r** | − | − | − | − | 1 | 1.00 | the control at **seed 4321**: a direct read of run-to-run noise on this base (WER and NLL), and the control of the confirmation stage |

- **Blocks:** the block generator is **AD (= BC)**: block 1 = runs 1, 2, 7, 8 (+1r) on the A100s; block 2 = runs 3-6 on the 24 GB cards.
  - Each main effect has two + and two − runs in each block. So a hardware difference cannot bias any main effect; it is confounded only with the AD = BC pair (dual-mode × text and regularisation × dp, both implausible).
- **Every two-factor "+ +" combination with the other two off is in this half** (runs 2-7), and so is all-four (run 8). So the combination of any two passing factors is directly measured.
- **Fold-over (8 more runs) if ≥ 2 factors pass** (§6). It is **the complementary half, I = −ABCD**, made by reversing the sign of **one** factor (D = −ABC).
  - Reversing *all* signs would map this resolution-IV half onto itself, since ABCD has even length.
  - Blocks are again by AD: the complementary runs with AD = + go on the A100s. Each main effect is again balanced within each block (checked).
  - Together with this half it makes the **full 2^4**: all main effects and the 2FIs separately estimable (AD stays confounded with the hardware blocks; BC becomes estimable), with the training-noise variance of each main effect halved. The 3FIs + 4FI give the internal error.

## 5. Responses (eval-lane; everything at step 8,000, EMA weights)
1. **WER:** canonical full dev (5,567 utterances), pinned evaluator, greedy, at **chunks 4 / 8 / 16 and chunk 0** (offline mode from #80), pooled + dev-clean / dev-other.
   - **Chunk 16 is the commit pass's final text; chunk 4 is the provisional text.**
2. **Per-token NLL (second metric):** the exact full-lattice transducer NLL (`train.transducer.rnnt_nll`, real joiner) summed over full dev ÷ reference tokens, at chunks 4, 16 and 0.
   - It excludes the delay penalty, KD, text and consistency terms. D+ arms use their own predictor (with tables).
   - Pooled per arm; contrasts bootstrapped over speakers.
3. **First-token latency, chunk 4, end-of-first-word clock, full dev.**
   - D+ models cannot be exported to the engine until it supports the tables, so **all 9 arms are measured by one method: the PyTorch streaming decode's emission frames on the design clock.**
   - The method is validated on run 1 against eval-lane's engine trace (per-utterance first-token times must agree up to INT8 effects) before it is used for any contrast.
4. **Reported, not gated:**
   - revision burden: provisional chunk-4 vs final chunk-16 word edits per 100 words, as in H45's commit probe; A is expected to lower it;
   - function-word substitutions (D's target class, eval-lane's error categories);
   - **the gain-stress panel** (eval-lane's −20 / −30 dB sets, chunk 4) for every arm, since B's augmentation may interact with level robustness (lead 09:19Z);
   - per-step cost and peak memory.

## 6. Decision rules and registered margins (fixed now; eval-lane registers them before any arm runs)
**Estimator:**
- Effect of a factor on a response = mean over its four + runs − mean over its four − runs (runs 1-8; run 1r is not in the contrast).
- Interval: SE_total = √(σ_run² / 2 + SE_boot²).
  - **σ_run = 0.042 pp**: the registered H41 four-run spread at chunk 4 (0.087) ÷ d2(4) = 2.059, used at every chunk. This is conservative: the chunk-sampled triples of amendment 22 show 0.016 / 0.032 / 0.032 at chunks 4 / 8 / 16.
  - SE_boot = a speaker-cluster bootstrap of the same contrast (10,000 resamples, seed 20261122) on per-utterance edits averaged within each side.
  - After a fold-over (16 runs) the training term is σ_run² / 4.
- Run 1r is reported against σ_run as a check (|run1 − run1r| / √2). It does not change the registered σ_run.

| factor | primary response | screen PASS (all required) | guards (must hold for PASS) |
|---|---|---|---|
| **A** | chunk-16 WER (final text) | one-sided 95% upper bound of the effect < 0; NLL effect at chunk 16 < 0 | chunk 4 upper ≤ **+0.087 pp** (one four-run spread) |
| **B** | chunk-4 WER | upper bound < 0; NLL effect at chunk 4 < 0 | chunk 16 upper ≤ **+0.20 pp** (lead 09:07Z gates cleanup (b)) |
| **C** | first token, chunk 4 | mean ≤ **−40 ms** and its 95% upper bound < 0 (Amendment 4's blind criterion (b), lead 09:09Z) | chunk 16 upper ≤ **+0.20 pp**; chunk 4 upper ≤ **+0.20 pp** (the registry's runtime-change product tolerance, 06:14:30Z) |
| **D** | chunk-4 WER | upper bound < 0; NLL effect at chunk 4 < 0 | chunk 16 upper ≤ **+0.20 pp** |

- **Multiplicity:** Holm across the three accuracy primaries (A, B, D) at one-sided α = 0.05. C's latency criterion is a trade-off rule outside that family. All guards are read at α = 0.05 without adjustment, which is the stricter direction for guards.
- **PROMISING-SLOW** (not a PASS; nothing is adopted):
  - a factor's WER screen fails;
  - but its NLL effect is < 0 at both chunk 4 and chunk 16, each with a 95% bound < 0 using the larger of SE_boot and the half-normal pseudo-error (Lenth) of the 2FI contrasts.
  - → it becomes a candidate for a trunk-phase test.
- **What follows from the count of PASSES among A-D:**
  - **≥ 2 PASS → fold-over.** The 16-run estimates replace the screen's. A factor stays in only if it still passes with the 16-run estimates, and the 2FI between passing factors is reported.
  - **Exactly 1 PASS → no fold-over.** Resolution IV already keeps its main effect clear of 2FIs.
  - **0 PASS:** the release cooldown is the plain fast-cooldown recipe (run 1's), and B / D are re-read for PROMISING-SLOW.
- **Confirmation, before anything enters the release cooldown** (codex-lead: "confirm at most two methods on two paired seeds"):
  - run 1r (the control at seed 4321) + the **combination of at most two passing methods at seed 4321**;
  - **adopt if the seed pair (1234 and 4321 together) has the combination − control upper bound < 0 at each method's primary**, with all guards.
  - The release cooldown itself then runs on the island at the full global batch, at **seed 7919** (the protected final seed), and goes to the A6 / H54 gates.
- **Deployment conditions, outside this screen:**
  - **D** also needs the engine's predictor-context-4 path with exact parity to PyTorch, and **≤ +5% service CPU and ≤ +8 MiB RSS** on xbabe0 (the registered H46 compact-LM budget). Until then a D-trained model is not a release artifact.
  - **A's chunk-0 mode** is used only if a segment-level commit tier is built. Its chunk-16 gain needs nothing new.
- **Kill rules, per arm:**
  - step-2,000 dev-clean400 EMA chunk 4 above **BP1's own dev-clean400 EMA chunk-4 value + 1.0 pp**, a threshold fixed at BP1 before any arm;
  - raw chunk-4 WER > 20% at any eval;
  - the trainer's divergence / qkv guards.
  - A killed arm makes its factors' contrasts unreadable. The factor is reported HARM, and the arm is re-run once only if the kill is a documented guard false-positive.

## 7. Compute (R = one 8,000-step 650 s replay of S on one A100 ≈ 0.55 A100-h ≈ 1,444 audio-h) and placement
**Per-arm multipliers:**
- A ×2.1: two passes, plus the band joiner for the KL, plus the halves overhead;
- B ×1.05: the time warp in the loader, partly offset by the skipped layers;
- C ×1.0;
- D ×1.1: text batches through the predictor and the frozen joiner.

| set | R | A100-h |
|---|---:|---:|
| half fraction (runs 1-8) | **13.3** | **7.3** |
| + run 1r | 1.0 | 0.55 |
| decodes: 9 arms × chunks 4 / 8 / 16 / 0 + NLL | — | ≈ 2.7 (any card) |
| fold-over (conditional) | 13.3 | 7.3 |
| confirmation (one combination arm at seed 4321; run 1r already exists) | ≤ 2.4 | ≤ 1.3 |

**Memory at 650 s:**
- every arm fits a 24 GB card: S ≈ 13-15 GB; A+ uses the halves trick, so its peak ≈ one 650 s graph; B, C and D add < 1 GB;
- on the A100s that is ~37% VRAM: the pinned-batch exception, recorded.

**Placement options** (fleet-sched picks at BP1; nothing is reserved by this file):

| plan | cards | wall time (half + 1r) | note |
|---|---|---|---|
| **P-A (default)** | **block 1 on x8 + x9** (A100): x8 = run 2 → run 1 → 1r; x9 = run 8 → run 7. **Block 2 on the 24 GB cards**: 4090 = run 6, 3090 (xbabe2) = run 4, 3090 (H0) = run 3, A10 (xbabe6) = run 5 | **≈ 2.5-3 h**, set by the 3090 carrying run 4 (2.2R) | needs the on-prem cards free of Libri-Light medium labelling (R0 ETA ~10:25Z), and S2's mix staged on each host's local disk (small: 22 GB; never the USB drive) |
| P-B (cloud only) | block 1 + run 1r on x8 (7.8R), block 2 on x9 (6.6R) | ≈ 4.3 h (+ decodes) | no on-prem dependency |
| P-C | block 1 on x8 + x9, block 2 on two island A100s lent by the trunk (world 4 → 2 for ~2.5 h) | ≈ 2.4 h (x8 with runs 2 + 1 + 1r is the long pole) | saves ~0.3-0.5 h over P-A and costs ~2.5 h of half the trunk. Not recommended, since the trunk is the final run |

- **Order inside a card:** A+ runs first (they are the long ones), so the slowest runs start earliest.
- **If A's code is late:** the A− runs (1, 3, 5, 7, 1r) start at BP1 and the A+ runs (2, 4, 6, 8) follow; all of them branch from the same BP1 checkpoint.

## 8. Freeze, deadlines, registry
- **Freeze:** this file (with its sha256 posted in the chat) and eval-lane's registry entries, before BP1's checkpoint exists.
- **Code deadline:** A / B / D merged into one pinned commit with witness tests (off = bit-exact; on = the defining property each), by BP1. The alternates in §3 apply otherwise.
- **eval-lane registers:**
  - σ_run = 0.042 pp;
  - the estimator (§6);
  - the per-factor primaries, PASS rules and guard margins (0.087 / 0.20 / 0.20 / −40 ms);
  - the Holm family {A, B, D};
  - the PROMISING-SLOW rule;
  - the fold-over and confirmation rules;
  - the block structure (AD);
  - the latency-method validation on run 1.

## 9. Not in this screen, and why
- **S-32 growth.** The registered P1 consequence keeps capacity for S3's own pilot.
  - S-32 at 650 s needs ~26 GB (A100 only), doubles the cost of its arms, and ×4 with A.
  - As a 5th factor the clean design is 2^(5−1)_V (16 runs) at ≈ 40R (≈ 22 A100-h; it is a full factorial in A / B / D / E, so the cost is (1 + 2.1)(1 + 1.05)(1 + 1.1)(1 + 2.0)). That is costed here in case the lead overrides; it is not recommended.
- **H48c MFA anchor.** Its own wave is reading now (H48c-a / b).
  - It is a second latency lever whose interaction with C would alias in an 8-run design.
  - Its anchors exist only for the 960 h part of S2's mix.
  - It is a candidate for the fold-over's place in a later screen if H48c passes.
- **H49b sequence KD.** STOPPED by its pre-registered rule (h49-seqkd 09:10:46Z; B0 has memorised TRAIN, and the teacher adds no headroom).
- **`synth_aug` + ±20 dB input gain** (trainer-lane's proposal for B). It is a robustness lever for the gain-stress panel, not the over-fit remedy WER-1 specifies. It is a candidate for a later screen.

## 10. Changes from trainer-lane's 09:09Z draft (`prereg/S2_FC_factorial_DRAFT.md`)
| draft | this file | why |
|---|---|---|
| 2^(7−4) resolution III, 7 factors, + a separate R0 control | **2^(4−1) resolution IV, 4 factors, the control inside the design** (run 1) | the lead's factor list (09:08Z); G is STOPPED; F aliases with dp; main effects stay clear of 2FIs at the same 8 runs |
| B = augmentation parity, C = stochastic depth, separate | **one factor B** (WER-1's bundle) | the lead's list; saves a column; split later if it passes |
| A: one-way KL into chunk 16, λ = 0.1 | **symmetric per-node joiner KL, λ = 0.3, small pass from {4,8,16}, large pass at chunk 0** (T3b) | BREAKTHROUGH §7.4's evidence; keeps every trunk chunk mode trained |
| floor 3e-5 at 650 s | **2.12e-5** | the bridge's √-equivalent path |
| PASS: chunk-4 effect ≤ −0.087 pp (point) | **one-sided 95% bound < 0 with SE_total (registered σ_run + bootstrap), Holm over A / B / D; per-factor primaries; guards** | more power for real ~0.1 pp effects at the same false-pass control; final text is chunk 16; dp is a latency trade-off |
| 9R | **13.3R + 1R** (A ×2.1, D ×1.1 included) | costed per arm |
| no blocking | **hardware blocks on AD** | lets 24 GB cards run half the arms without biasing any main effect |

## 11. Gain / level augmentation (lead 09:19Z, after the gain-stress trigger): two cases, fixed now
The lead ruled that random input gain (≈ 0 to −35 dB plus mild jitter; trainer-lane pre-registers the distribution) **enters S2**, with a clean-level guard. Which design runs depends only on whether it is in S2's first segment. That is known before BP1, so this does not choose after seeing data.

**Case 1: gain augmentation is in S2's stable phase.**
- The design above runs unchanged, with gain augmentation **on in every arm**: it is part of S2's recipe, like the data mix.
- The gain-stress panel is reported per arm (§5).
- **The lead's clean-level guard is read outside the factorial, as h53's reading of "its matched control":**
  - run 1 (S2-BP1's plain fast cooldown) against the **B0 bridge's S2p** endpoint (B0 + 960 h + small, same low re-warm, no gain augmentation);
  - EMA, full dev, chunk 4, 0 dB;
  - pass if the one-sided 95% upper bound of run 1 − S2p ≤ **+0.087 pp** (one four-run spread);
  - eval-lane registers it before BP1.
  - It is a no-harm guard on clean audio (S2 has more exposure *and* the augmentation), not an isolated effect of the augmentation.

**Case 2: gain augmentation is not in S2's first segment.** It becomes **factor E** (− = off, + = trainer-lane's pre-registered distribution, in the cooldown only), and the design becomes **2^(5−1), resolution V, generator E = ABCD (I = ABCDE)**: the full 2^4 in A-D with E set by the generator.
- **16 runs, every main effect and every 2FI clear. No conditional fold-over is needed.**
  - A resolution-V half cannot be split into two readable 8-run stages: each stage would carry a 2-letter word. So all 16 run before any reading.
- **Blocks by ABC (= DE):** hardware is confounded only with a 3FI and the text × gain 2FI.
- **Cost:** Σ over the full 2^4 of A 2.1 × B 1.05 × D 1.1 (C and E free) = 3.1 × 2.05 × 2 × 2.1 = **26.7R ≈ 14.7 A100-h**. That equals this design's half + fold-over, so the budget ceiling does not change.
- Wall time ≈ 5 h on P-A's six cards.
- **E's readings:**
  - **clean-level guard** = E's main effect on 0 dB chunk-4 WER, one-sided 95% upper bound ≤ +0.087 pp;
  - **benefit** = E's main effect on the −20 / −30 dB panel (one-sided 95% upper bound < 0);
  - NLL reported.
  - E enters the release cooldown only if both hold. E is outside the A / B / D Holm family, since it is required by the trigger rule rather than screened.
- The PASS rules, guards, confirmation and deployment conditions for A-D are as in §6, with the 16-run training term σ_run² / 4.
