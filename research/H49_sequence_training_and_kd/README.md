# H49: sequence training and KD (MWER, sequence KD on candidate unions, frame KD)

Driver: h49-seqkd. Runs: trainer-lane. Teacher side (R0 caches and N-best): kd-cache. Statistics and margins: eval-lane. Canonical scorer: Codex.
DEV and TRAIN only; no test-set file is read. Results are kept here; weights never (see `MANIFEST`).

## Question
B0 (A1-49000 EMA) loses most of its gap to Parakeet / Zipformer on acoustics and capacity, and function-word swaps are 27% of its errors. Can either of two sequence-level objectives, added to B0's continuation recipe, lower DEV WER by more than twice the four-run spread on both seeds, against the continuation control?

1. **H49a, MWER:** expected canonical word edits over the student's own beam.
2. **H49b, sequence KD:** R0-scored candidate unions of the reference, R0's beam and the student's beam.

H49c (frame KD with the fixed guard and the full 0.9 / 1.0 / 1.1 cache) belongs to kd-cache under H22 amendment A (`r0-fp32-gpu-v1`). It has a low prior and is not pre-registered here.

## Evidence before any H49 run
- **H41 arm 8** (MWER, 2026-11-22) was NOT ADOPTED: +0.045 pp [−0.039, +0.129] against its own control.
  - It used the old standalone `train/mwer.py` loop: no SpecAugment, CTC, EMA or speed perturbation; 120 s batches; the old `ultra.wer` normaliser; `(1 − λ)·L_rnnt + λ·mean_utt(risk)` with λ = 0.5.
  - Its λ = 0 control was itself 0.087 pp worse than the fork point.
  - H49a removes each of these differences (see the specification below).
- **B0's DEV errors** (H43 analysis): function words 27%, acoustically different words 27%, close-sounding words 22%, rare words 13%. R0 gets 47.5% of B0's panel errors right.
- **DEV candidate headroom, student side**: `results/dev_headroom/`, computed 2026-11-23T07:08Z, before this pre-registration.
  - **Method:** the B0 greedy plus the batched modified beam 4 / 8 / 16, all 5,567 utterances, chunk 4. Every row matches the pinned Rust `ultra-score` in S / D / I. Oracles are computed before reference injection.

| system | pooled WER | Δ vs greedy (pp) |
|---|---:|---:|
| greedy = B0 | 8.302 | — |
| beam 4 / 8 / 16, top | 8.223 / 8.218 / 8.212 | −0.079 / −0.085 / −0.091 |
| beam 4 / 8 / 16, oracle | 6.951 / 6.158 / 5.479 | −1.351 / −2.144 / −2.823 |

  - **Beam-8 oracle headroom by category (pp):** function words 0.725, close-sounding 0.570, acoustic 0.415, content deletions / insertions 0.208, rare 0.150, writing conventions 0.076.
  - **Recall** of the greedy's S / D positions: 18.3% (beam 4), 29.1% (beam 8), 38.6% (beam 16). Rare words: 8.7 / 13.3 / 18.0%.
  - R0 was not run on DEV (codex-lead 06:52Z: that needs the owner's authorisation).

## Shared protocol (H49a and H49b)
- **Continuation control:** **A1C**, H41 amendment 23. That is B0 = A1-49000 raw weights (`ckpt-00049000.pt`, sha256 2a3dd9fb…) with the EMA restarted from them, on the H41 replay schedule:
  - 200 warm-up steps to 2.476e-4, then a cosine to 3e-5 at step 8,000;
  - batch 650 s / 144, `chunk_modes=4,8,16`, `eval_chunks=8,16`, `delay_penalty` 0, EMA 0.9995;
  - seed 1234, A1L's launcher (`h41_replay.sh` with `INIT_FROM` = `ckpt-00049000.pt`) and its data order.
  - **Its seed pair, A1C-r** (seed 4321, otherwise identical), is required by the pairing rule. It is shared with H48c and H53 P1; requested from trainer-lane.
- **Arms:** A1C's exact command plus only the H49 keys below. Same steps, schedule, batches, augmentation identities and seeds, i.e. equal ordered exposure.
- **Code:** PR #80's branch `lane/trainer-h41d` (now 6f2d850, with #87 merged in) plus `patches/h49-trainer.on-6f2d850.patch` (head 6f2d850). Variants for 4d51bf2 / f07fc1a and fd0f64f exist.
  - **Witness `results/bitexact/`:** with the H49 terms off, the patched tree trains bit-identically to the unpatched one. That run used pruned RNN-T + CTC + SpecAugment + chunk modes + EMA: identical losses, parameters + EMA, and global RNG state after 4 steps.
  - If the arm's tree differs from A1C's pinned tree, trainer-lane reruns that witness against A1C's tree from the same checkpoint for 50 steps before launch. Equal `train.jsonl` losses are required.
- **Readings (eval-lane's v3 rule):**
  - EMA weights, greedy, pooled WER at chunk 4 (primary) and chunks 8 / 16 (guards), with the canonical scorer.
  - Selection on the `dev_split_v2` selection half; decision on the confirmation half (sha256 002ac7ae…, 2,785 utterances, 37 speakers); full DEV as the labelled endpoint.
- **Pairing:** each arm's seed pair against A1C's pair (`h43_stats.py test` with two files a side). Both seeds must point the same way.
- **Family:** the H49 family is at most two confirmed methods (the H49a winner and the H49b winner), Holm at α = 0.05. Seed 5678 is reserved as the independent final seed and is never used for selection.
- **Gate (both methods):**
  - (i) superiority at chunk 4 on the confirmation half: one-sided 95% upper bound < 0 after Holm;
  - (ii) the repo rule: gain > 2 × the four-run spread, i.e. Δ ≤ −0.174 pp at chunk 4, for the pair and for each seed's own mean against A1C's pair;
  - (iii) non-inferiority at chunks 8 and 16 (upper bound ≤ 0.094 / 0.36 pp);
  - (iv) first-token latency at chunk 4 no worse than A1C's by more than 20 ms (paired p95, eval-lane's clock).
  - PASS needs all four; FAIL if the chunk-4 point estimate is ≥ 0; otherwise INCONCLUSIVE.
  - **Owner plan §5.5 (W3 promotion)** additionally needs ≥ 0.30 pp on full DEV. Reported, not part of this gate.
- **Equal-compute reading** (owner plan §5.2): the arm's snapshot nearest to step 8,000 / c against A1C's final weights, with c = the arm's measured step-time ratio to A1C. It is mid-cosine, so it is reported only, not gating.
- **Margins** are registered by eval-lane before any H49 decode (requested entries in `MARGINS_REQUEST.md`).
- **R** = one 8k-step replay at 650 s on one A100 (≈ 33 min).

## H49a: MWER with canonical word-edit counts (pre-registration)
**Hypothesis.** Adding `λ · L_mwer` to the continuation recipe lowers DEV WER versus A1C by more than 2 × spread on both seeds. Function words and close-sounding words carry 60% of the beam-4 oracle headroom; MWER can only re-rank within the beam, so the attainable gain is far below the 1.35 pp beam-4 oracle.

**Specification** (`train/mwer.py` in the patch):
- **Candidates:**
  - the student's own N-best from a batched modified beam search, width 4;
  - at most one token per frame; equal token sequences merged by log-sum-exp (the `train.beam` law, any predictor context);
  - run on this step's projected encoder output (SpecAugment and dropout as in the step; detached), so candidates refresh every step;
  - over a sub-batch of 12 utterances per step, drawn from a generator seeded by (seed, step) that touches no training RNG stream;
  - the reference token sequence joins the set when the beam missed it (the old implementation's choice);
  - stale-versus-current coverage is moot (refreshed every step); the share of steps with the reference in the beam is logged.
- **Score:** s_i = log P(y_i | x), the full-sum RNN-T over the standard lattice (`joiner_logprobs` + `rnnt_nll`, fp32, float64 recursion). This is owner plan arm A1. The frame-synchronous law (A3) is not run, because the deployed decoder is greedy with up to 8 symbols per frame.
- **Risk:** E_i = canonical S + D + I of `tok.decode(y_i)` against the reference (`ultra-en-wer-norm/v1`).
  - Source: `ultra-score --candidate-edits` (H43), held open as one subprocess.
  - The first 256 requests are also counted by the Python port, and any difference stops the run.
  - Risk is detached.
- **Term:** R_b = Σ_i softmax(s_b / τ)_i (E_bi − mean_j E_bj) with τ = 1.0. The mean baseline does not change the gradient. L_mwer = Σ_b R_b / Σ_b N_b, with N_b the canonical reference word count over the sub-batch. Under DDP it is one global denominator (`train/ddp_loss.py`).
- **Total:** L = L_human + λ · L_mwer, where L_human is A1C's unchanged loss (pruned RNN-T + 0.3 CTC + simple-joiner loss).
- **Logged** every 50 steps:
  - `mwer`, active utterances, candidates, reference-in-beam, top-candidate errors, expected errors, candidate entropy;
  - every 250 steps, the gradient norms of L_human and λ · L_mwer over all parameters (owner plan: "record gradient scale").

**Arms and stages:**
1. **Recall gate** (no training). The beam-4 oracle headroom must be ≥ 2 × spread (0.174 pp) on DEV and on the TRAIN pilot.
   - DEV: 1.351 pp, **PASS**.
   - TRAIN pilot: from kd-cache's student beam-4 lists (requested); a pending input.
2. **λ screen.** Two arms, seed 1234, the full 8,000-step schedule, each read at step 2,000:
   - λ ∈ {0.3, 1.0}, a predefined grid. Per reference word, λ = 1 makes the MWER gradient on a typical step comparable to the human loss's.
   - The winner is the arm whose step-2,000 EMA chunk-4 WER on the **selection half** is lower.
   - If both arms are worse than A1C's step-2,000 snapshot by more than 1 spread on the selection half, H49a stops there.
   - The losing arm stops at step 2,000.
3. **Confirmation.**
   - The winner (seed 1234) continues to step 8,000 in the same run.
   - The winner with seed 4321 runs the full 8,000 steps.
   - Decision as in the shared protocol, against A1C + A1C-r.

**Kill rules** (automated where the H41 tools already do it):
- (K1) step-2,000 dev-clean400 EMA chunk-4 WER > A1C's step-2,000 value + 1.5 pp, or raw chunk-4 WER > 20% at any eval;
- (K2) a non-finite loss or gradient pauses the run (the trainer's rule);
- (K3) the human-loss EMA stays > 10% above A1C's at the same step for 300 steps;
- (K4) compute contingency: if the measured step-time ratio c > 1.6 over steps 50-250, `mwer_utts` drops from 12 to 8 in both λ arms before step 500, decided before any eval.

**Launch** (on top of A1C's command; tree = lane/trainer-h41d + patch):
`INIT_FROM=<A1-49000 ckpt-00049000.pt> RUN_NAME=H49a_mwer_l<λ>_s<seed> h41_replay.sh <data> <cpus> <gpu> h49a chunk_modes=4,8,16 eval_chunks=8,16 seed=<seed> mwer_weight=<λ> mwer_beam=4 mwer_utts=12 mwer_tau=1.0 mwer_scorer=<ultra-score built with --candidate-edits>`

**Compute:**
- c ≈ 1.3-1.6 (beam + full-sum scoring of ≈ 60 candidates per step), measured.
- Screen: 2 × 2,000 steps; confirmation: 6,000 + 8,000 steps.
- Total ≈ (4,000 + 14,000) / 8,000 × c ≈ 2.9-3.6R, plus A1C-r (1R, shared) and full-DEV decodes.

**Dependencies:**
- Codex's H43 scorer patch (`--candidate-edits`) merged and built. Until then no H49a run starts; the Python port is for tests only.
- A1C and A1C-r.
- eval-lane's margins.
- A free A100 slot from fleet-sched.

## H49b: sequence KD on candidate unions (pre-registration; the stop rule is read before any training)
**Pilot** (generation by kd-cache, analysis by h49-seqkd; TRAIN only):
- **Utterances:** kd-cache's deterministic 20,000-utterance unit list, 200 row groups across all 126 TRAIN shards (`r0_gpu.pilot_units()`, `r0_gpu.py` frozen at sha256 41be5573…).
- **Per utterance:** the reference; R0's beam-8 N-best; the student's (B0 EMA, chunk 4) beam-8 N-best; beams 4 and 16 for the oracle curves.
- **Teacher score:** R0 full-sum RNN-T log-likelihood (k2 regular, `r0-fp32-gpu-v1`) for every distinct canonical token sequence.
- **Report** (`scripts/h49_headroom.py`, canonical words, all oracles before reference injection, by category / clean-other / length):
  - student 1-best; student beam-K top and oracle;
  - R0 top and oracle; union oracle; the ≤ 12 quota union (ref + R0 top-6 + student top-5);
  - the teacher's pick over the union, without and with the reference;
  - recall of the student's error positions by set.
- **Category rules**, frozen here before any pilot row is read (also in the script header):
  - function words: a fixed closed-class list;
  - close-sounding: min CMU phone edit distance ≤ 1, or a morphological variant, or character distance ≤ 2 when a word is not in the dictionary;
  - rare: < 10 TRAIN counts;
  - writing convention: apostrophe / hyphen equivalence, or a split / merge;
  - everything else is acoustic.

**Stop rule** (plan: "stop below 1 spread of headroom"):
- **Statistic:** H_T = WER(student beam-8 oracle) − WER(union oracle), on the TRAIN pilot, before reference injection. It is the oracle headroom R0's candidates add beyond the student's own beam.
- **Stop** if H_T < 0.087 pp (1 spread) or if the teacher's pick over the union is worse than the student 1-best.
- **Also stop** if the union's gain sits only in writing conventions (> 50% of H_T), or if insertions rise in the teacher's pick (owner plan kill conditions).

**Training lists** (codex-lead 06:52Z):
- the human reference plus at most 3 distinct retained candidates per utterance, retokenised to the student vocabulary, with provenance;
- **selection rule:** the 3 non-reference candidates with the highest R0 full-sum score in the union, one canonical tokenisation per string;
- teacher distribution q = softmax(s^T / τ_T) over the ≤ 4 retained strings, τ_T = 1.0, conditional on the set;
- store format `ultra.seqkd/v1` (`train/seqkd.py`), with the tokenizer sha256 checked. The reference ids must equal the student's RNN-T targets for that (uid, speed), otherwise the term is dropped and counted. Speed key exact, speed 1.0 only.

**Objective:**
- L_seq = Σ_b Σ_i q_bi · (−log P_S(y_bi | x_b)) / Σ_b U_b, i.e. q-weighted sequence cross-entropy (full-sum), normalised by the reference tokens of the utterances with an accepted list (one global denominator under DDP).
- L = L_human + λ_seq · L_seq. The human loss stays on every example.

**Coverage, open question to codex-lead.** Codex-lead names the release screen as "the deterministic 10,000 TRAIN clip input".
- With A1C's sampler and speed perturbation, 10,000 clips at speed 1.0 reach ~1.2% of the samples a continuation sees: ≈ 5k seq-KD utterance-steps in 8,000 steps. That is too sparse to move pooled DEV WER.
- **Proposal:**
  - (B-full, primary) lists for every TRAIN utterance at speed 1.0, from kd-cache's 960 h sequence-KD material (planned after the pilot): ~33% of samples, identical ordered exposure to A1C, so A1C's pair stays the control.
  - (B-10k) is the screen-literal arm, reported as a diagnostic.
- If codex-lead rules for 10,000 clips only, B-10k becomes the primary and its expected effect is stated as below detection.

**Arms and stages:** as in H49a.
- λ_seq ∈ {0.3, 1.0}, screened to step 2,000 on the selection half, seed 1234.
- The winner continues to 8,000; the winner with seed 4321 runs 8,000.
- Decision against A1C's pair as in the shared protocol, with the same kill rules K1-K3.
- **K4:** `seq_max_alt` drops 3 → 2 if c > 1.6.

**Launch:** A1C's command + `seq_weight=<λ> seq_store=<store.jsonl> seq_tau=1.0 seq_max_alt=3`.

**Compute:**
- Pilot: kd-cache's ~20 GPU-min for R0 + student; analysis on CPU.
- Training: ≈ 2.6-3.2R (c ≈ 1.2-1.4), plus the 960 h sequence material (kd-cache, ~1 A100-h).

## Audit of `train/mwer.py` (lane/trainer-h41d fd0f64f, the H41 arm-8 code) and of PR #72's B6
| # | finding | consequence | fix in the patch |
|---|---|---|---|
| 1 | Risk used `ultra.wer.normalise`: no contraction expansion, apostrophes kept. PR #72's B6 twin was dropped by codex-lead: its S / D / I tie order and Unicode casing differ from Rust | DON'T vs DO NOT counted 2 errors, CANNOT vs CAN NOT 2; risk differs from the release metric (A1 panel 1,069 / 14,376 Python vs 1,081 / 14,448 Rust) | Canonical counts from `ultra-score --candidate-edits`; Python port only as a parity guard. The total S + D + I does not depend on the tie order |
| 2 | Its own training loop, outside the trainer: no SpecAugment, no CTC term, no EMA, no speed perturbation or chunk modes, constant LR, batches of 120 s / 16 | Arm 8's λ = 0 control drifted +0.087 pp; nothing comparable to the recipe | The term runs inside `ASRModel.losses` with A1C's recipe unchanged |
| 3 | Loss was `(1 − λ)·L_rnnt + λ·mean_utt(R)` with λ = 0.5, per-utterance raw counts | MWER dominates the gradient (estimated ~10× the per-utterance RNN-T gradient); λ is not comparable across normalisations (owner plan §5.3) | `L_human + λ·Σ R / Σ N_words`, one global denominator; λ ∈ {0.3, 1.0}; gradient norms logged |
| 4 | N-best from `modified_beam_search`, a per-utterance Python loop | 2.8 s per 16-utterance step in arm 8; infeasible at 650 s batches | `batched_beam_nbest`, checked against `train.beam` (ctx 2) and a reference law (ctx 4) |
| 5 | `train.beam` hard-codes predictor context 2 (v3 review B5) | Wrong candidates for pred_ctx 4 models | The batched beam uses `predictor.context_len` |
| 6 | No candidate telemetry | No view of reference-in-beam, entropy or expected errors | Logged every interval |
| 7 | DDP | The standalone loop had no DDP | `train/ddp_loss.py` global denominators |

## Changes (patch to trainer-lane; landed by trainer-lane)
`patches/h49-trainer.on-4d51bf2.patch` applies on PR #87's head 4d51bf2 (`h49-trainer.on-fd0f64f.patch` on PR #80's head).
- **`train/mwer.py` (rewritten):**
  - canonical word edits (Rust `--candidate-edits` client + port + parity guard);
  - the batched beam for any predictor context;
  - full-sum candidate scores;
  - `mwer_term`;
  - the legacy `mwer_loss` kept for H41 arm 8's interface; its standalone loop retired.
- **`train/seqkd.py` (new):** `SeqKDStore` (identity, guard, counts), `seqkd_term`.
- **`train/asr.py`:** `losses(..., mwer=None, seq=None)`.
- **`train/ddp_loss.py`:** global denominators for both terms. The payload is unchanged when they are off.
- **`train/data.py`:** `Example.seq` / `Batch.seq` plumbing and the store lookup next to the KD lookup.
- **`train/h49_hooks.py` (new):** the trainer hooks, so `trainer.py` gains only call sites (1,000 lines on PR #87's head).
- **`train/trainer.py`:**
  - keys `mwer_weight mwer_beam mwer_utts mwer_tau mwer_scorer seq_weight seq_store seq_tau seq_max_alt` (the reference-in-set rule, the 256 parity checks and the 250-step telemetry are fixed in `train/h49_hooks.py`);
  - an isolated sub-batch RNG stream;
  - gradient-scale telemetry;
  - logging;
  - refusal with the two-view step;
  - the watch signature re-arms when a term is on.
- **`tests/test_h49_mwer_seqkd.py`:** 14 witnesses (one needs a `--candidate-edits` binary). They fail on fd0f64f / 4d51bf2 and pass with the patch. The full suite keeps the same 66 export-only failures before and after.

## Results
(Everything below was added after the pre-registration post of 2026-11-23T07:39:01Z, sha256 61bfad82…; the design above is unchanged.)
- `results/dev_headroom/`: DEV student-side headroom (above).
- `results/dev_headroom/dev_student_r0top.*` (07:51Z, no new R0 decode). eval-lane's published R0 INT8 full-DEV control (`R0-int8.fulldev.score-input.jsonl`, sha a6dea4a3…) added as one candidate per utterance:

| system | pooled WER | Δ vs B0 greedy |
|---|---:|---:|
| R0 1-best alone (320 ms) | 5.368 | −2.934 |
| **B0 beam 8 ∪ R0 1-best, oracle** | **4.188** | **−4.114** |

  - R0 adds 1.970 pp of oracle headroom over B0's beam 8 on DEV.
  - Net errors R0 fixes: function words 1,081 / 2,717, acoustic 860 / 2,112, close-sounding 675 / 1,783, rare 253 / 1,298.
  - Recall of B0's S / D positions: R0 1-best 49.7%, B0 beam 8 29.1%, both 57.8%.
  - R0's lookahead (320 ms + right context) is part of its advantage.
- `results/bitexact/`: patched vs unpatched training equality on both bases, plus the full-suite comparisons.
- `results/train_pilot/train_student_b4b8.json` (09:13Z): student side on all 200 pilot units (19,330 TRAIN utterances, 665k words, references from the parquet).

| TRAIN | WER | headroom vs 1-best |
|---|---:|---:|
| B0 1-best | 0.244 (clean 0.230 / other 0.268) | — |
| beam-4 oracle | 0.115 | 0.129 pp |
| beam-8 oracle | 0.065 | 0.179 pp |

  - Recall of B0's TRAIN error positions: 54% (beam 4), 74% (beam 8).
  - Beam-4 headroom by category (pp): close-sounding 0.051, function 0.043, acoustic 0.016, rare 0.010.
- `results/train_pilot/train_pilot_85good.*` (08:58Z): R0 unions on the 84 units written by kd-cache's verified code. The 31 units from code 498dc116… had corrupted R0 beams and were discarded by kd-cache.

| TRAIN system | WER |
|---|---:|
| B0 1-best | 0.239 |
| B0 beam-8 oracle | 0.065 |
| union oracle | 0.052 |
| R0 beam top / R0 oracle | 1.554 / 0.665 |
| teacher's pick | 1.466 |

  - The teacher ranks the reference first in 52% of utterances.
  - Of the top-3 retained strings, 63% come only from R0.
- `results/smoke/`: real-model smoke test of the patch; gradient scale ‖∇L_mwer‖ / ‖∇L_human‖ = 0.09 / 0.22 at λ = 1 on two real batches (CPU, exact RNN-T human loss).

## Decision
- **H49a: STOP at stage 1 (pre-registered recall gate).** DEV beam-4 oracle headroom 1.351 pp (PASS); TRAIN 0.129 pp < 0.174 pp (FAIL). No training cell. B0 has memorised TRAIN (0.24% WER), so its own beam has almost nothing better to re-rank toward on the data MWER trains on.
- **H49b: STOP (pre-registered rule).** Criterion 1 holds for the full 20k pilot by a bound: H_T ≤ WER(B0 beam-8 oracle) = 0.065% on all 200 units (19,330 utterances), below 0.087 pp whatever R0 proposes; the measured value on the 84 verified union units is 0.013 pp. The teacher's pick is +1.23 pp worse than B0's 1-best, with more insertions and rare-word errors.
  - Disclosed deviation: R0 unions were scored on 84 / 200 (clean) units. kd-cache's restarted build corrupted 31 units, which were discarded, and the remaining units were stopped once the bound settled the verdict.
- **Where the teacher's advantage is:** on DEV (unseen) R0's single hypothesis adds 1.97 pp of oracle headroom over B0's beam 8; on TRAIN (memorised) it adds 0.013 pp and its preferences are mostly its own errors. Teacher signal therefore belongs on unseen audio: Libri-Light R0 labels (H52) and soft R0 targets there (claude-lead 08:53Z, item 1), not on LibriSpeech TRAIN.
