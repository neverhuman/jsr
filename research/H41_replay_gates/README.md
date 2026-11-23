# H41: replay gates for training changes

**Status:** pre-registered on 2026-11-22 at about 19:55Z, before any H41 arm produced an eval. Protocol set by the lead (owner-approved plan, 19:42Z).

## Why
Two identical 100 h from-scratch runs differ by about 0.57 pt on dev-clean400, so a 100 h gate cannot tell a real gain from run-to-run spread. A replay gate forks one trained model late in its schedule and replays the end of training under each change, with a repeated control to measure the spread.

## Protocol (same for every arm)
- **Fork:** H22ctl `model-00060000.pt` (dense S, R0 BPE-500, step 60,000 of 79,800; sha256 prefix `d12336677f3946d9`), raw weights. The EMA restarts from the fork.
- **Schedule:** 200 warmup steps to 2.476e-4 (H22ctl's LR at step 60,000), then a cosine to 3.0e-5 at step 8,000.
- **Data and batches:** 960 h with speed perturbation 0.9/1.0/1.1, budget 650 s, max batch 144 (fits 24 GB on every host), seed 1234. Every arm sees identical batches unless the arm says otherwise. 8,000 steps are about 1,444 audio-h.
- **Recipe:** H22ctl's otherwise: pruned RNN-T (range 5) + 0.3 CTC, AdamW, clip 5, EMA 0.9995, SpecAugment on, qkv cap at 1.5× the step-100 norms.
- **Launcher:** `python/ultra/train/ops/h41_replay.sh`.

## Arms
| arm | host | change from control |
|---|---|---|
| control | xbabe1 (RTX 4090) | none |
| A/A | xbabe2 (RTX 3090) | `seed=4321` (different batch order and augmentation, same fork) |
| Muon | xbabe6 (A10) | `optimizer=muon muon_momentum=0.95`, same LR schedule, no separate LR tuning |
| chunk sampling | xbabe7 GPU1 (A100), after L3 ends | `chunk_modes=4,8,16` (per-step attention chunk of 160/320/640 ms; SpecAugment masks stay the control's) |

The arms run on different GPU models. The A/A arm therefore measures seed and hardware variation together.

## Measurements
- **Decision metric:** codex-review scores each arm's step-8,000 weights (raw and EMA) on the FULL dev-clean and dev-other sets, with a paired bootstrap against control.
- **Monitoring only:** the trainer's dev-clean400 greedy WER every 2,000 steps.

## Decision rule (fixed now)
- **Spread:** the A/A spread is the absolute difference between the A/A arm and control on the same full-DEV metric.
- **Adoption:** a change is adopted only if it beats control by more than twice the A/A spread.
- **Chunk sampling:** its expected gain is at larger chunks, so it is compared with control decoded at the same chunk (4, 8 and 16). It is adopted if it beats control by more than twice the spread at chunk 8 or 16 and is not worse than control by more than the spread at chunk 4.
- **Kill rule:** an arm stops early if its dev-clean400 WER at step 2,000 is more than 1.5 pp worse than control's at step 2,000.
- A change that does not pass is recorded as not adopted. It is not retried with different settings under this pre-registration.

## Amendment 1 (2026-11-22, about 20:20Z): two more arms, and the spread becomes a spread of three
Written before either new arm started. At this time no arm had finished and no full-DEV score existed. The trainer's monitoring evals (dev-clean400) existed for control up to step 4,000, A/A at step 2,000 and chunk sampling at step 2,000.

| arm | host | change from control |
|---|---|---|
| adaptive clipping | xbabe1 (RTX 4090), after control ends | `grad_clip_mode=adaptive clip_k=2.0 clip_window=200` (the H23e setting), seed 1234, identical batches |
| A/A-2 | xbabe2 (RTX 3090), after A/A ends | `seed=9876` (a third control-like run) |

- **Chunk-sampling arm host:** it runs on H0's RTX 3090 (free when D2 ended), not on xbabe7 GPU1 as the table above planned. xbabe7 GPU1 became A1's fifth rank.
- **Spread, replacing the definition above:** the spread is the range (largest minus smallest) of the three control-like runs (control, A/A, A/A-2) on the same full-DEV metric.
- **Adoption rule, unchanged in form:** a change is adopted only if it beats control (seed 1234, the run with identical batches) by more than twice that spread.
- Everything else in the protocol is unchanged.

## Amendment 2 (2026-11-22, posted in AGENT_CHAT at 20:46Z): arm 7 (CR-CTC, full form) and arm 8 (minimum-WER fine-tune)
Written and posted in the chat before either arm started; committed here afterwards because a gate was running in the worktree. At this time control and A/A had finished (their weights were with eval-lane, no full-DEV score posted), and the clipping, A/A-2, Muon and chunk arms were running.

### Arm 7: CR-CTC, full form (xbabe1, after the clipping arm)
- **Change from control:** `cr_ctc_weight=0.2 cr_ctc_full=true`. Same fork point, schedule, batch and seed as control.
- **What it does:** two SpecAugment views of every batch. Each view gets the CTC loss and the transducer loss, and a consistency term pulls the two views' CTC posteriors together (KL in both directions, targets detached). View A's masks are control's masks; SpecAugment settings are unchanged.
- **How it runs:** the two views run one after the other and their gradients accumulate, because both graphs at a 650 s batch do not fit 24 GB. A unit test checks that this equals the one-graph gradient.
- **Cost:** about 2.3× control's compute per step (three forward passes, two backward passes).
- **Two readings, both reported:**
  - **Per step:** the arm's step-8,000 weights against control's step-8,000 weights. This is the pre-registered decision metric, with the same rule as every arm.
  - **Per GPU-hour:** the arm's step-4,000 snapshot (about 1.15× control's whole GPU time) against control's final weights. The arm is mid-cosine there, so a loss on this reading is confounded by the unfinished schedule; a win would be strong evidence.
- **Earlier evidence:** the H26 runs used a form with only one supervised view and were level with their control within noise (see `../H26_kd_audit_smbr_and_cr_ctc/README.md`).

### Arm 8: minimum-WER fine-tune (xbabe2, after A/A-2) and its own control (xbabe6, after Muon)
- **Code:** runtime-lane's `train/mwer.py` from PR #35 (head c3c8846), unmodified. It is a separate short training loop, not the trainer: AdamW at a constant LR, exact transducer loss, no CTC term, no SpecAugment, no speed perturbation, no EMA.
- **Fork point:** H22ctl's FINAL raw weights (step 79,800, sha256 prefix `2626b96c22dfcca5`), not step 60,000.
- **Settings:** 2,000 steps, lr 2e-5, budget 120 s / max batch 16, 960 h, seed 1234, beam 4, `tau` 1.0. Loss `(1 - lam) × transducer + lam × expected word errors over the model's own beam-4 hypotheses`.
- **Arms:** `lam=0.5` (minimum-WER) and `lam=0` (its control: the same loop, data order and optimiser without the term).
- **Decision:** the minimum-WER arm against its own control on full DEV with the paired bootstrap. It is adopted if it beats that control by more than twice the spread of the three control-like H41 runs. That spread comes from a different protocol, so it is a yardstick here, not this arm's measured run-to-run spread.
- **Step-time contingency, fixed now:** if the measured step time puts 2,000 steps beyond 3 hours, both the arm and its control are cut to the same smaller step count that fits 3 hours, decided before any eval of either.
- **Launcher:** `python/ultra/train/ops/h41_mwer.sh`.

## Amendment 3 (2026-11-22, posted in AGENT_CHAT at 22:14Z): arm 9 (CR-CTC, half form)
Written and posted in the chat before arm 9 started (22:15Z); committed here afterwards.

- **Arm 9:** CR-CTC, half form (the PR #32 form), on xbabe2 beside arm 8. `cr_ctc_weight=0.2 cr_ctc_half=true`; same fork point (H22ctl step 60,000), schedule, batch (650 s / 144) and seed 1234 as control and as arm 7.
- **What it does:** two SpecAugment views of every batch. Both views get the CTC loss, and the consistency KL (both directions, targets detached) pulls their CTC posteriors together at weight 0.2. **The transducer loss is on view A only.** View A's masks are control's masks.
- **How it runs:** like arm 7, one view at a time with gradients accumulated; view B runs only the encoder and the CTC head. A unit test checks the accumulated gradient against the same loss in one graph.
- **Cost:** about 1.8× control's compute per step (arm 7 is about 2.3×).
- **Two readings, both reported:**
  - **Per step (the decision metric):** step-8,000 weights against control's step-8,000 weights, full dev, paired bootstrap, adopted only above 2× the spread of the three control-like runs.
  - **Per GPU-hour:** its step-4,500 snapshot against control's final weights, confounded by the unfinished cosine (as for arm 7).
- **The question arms 7 and 9 answer together:**
  - arm 7 passes and arm 9 does not: the gain needs the second transducer view;
  - both pass: the cheaper half form suffices;
  - neither passes: CR-CTC is not adopted at this operating point.
- **Placement:** xbabe2, its own guard at 0.87, CPUs 76-99. Arm 9 yields if the card's 5-minute mean passes the guard's target; arm 8's guard is CPU-only so arm 9's load does not pause it. Arm 9 shares the card with arm 8, so its wall time differs from control's, but its steps and batches do not; the per-GPU-hour reading uses its own measured step time.

## Amendment 4 (2026-11-22, posted in AGENT_CHAT at 22:45Z): arm 10 (low-latency chunk sampling)
Written and posted in the chat before arm 10 started (22:46Z); committed here afterwards. At that time eval-lane had posted its verdict adopting the chunk-sampling arm ({4,8,16}).

- **Arm 10:** `chunk_modes=1,2,4,8,16` (40 / 80 / 160 / 320 / 640 ms attention chunks, one drawn per step from its own RNG stream; SpecAugment masks stay control's), `eval_chunks=1,2,8`. Otherwise identical to the chunk arm: fork point H22ctl step 60,000, schedule, batch 650 s / 144, seed 1234, xbabe1 (RTX 4090).
- **Why:** the owner's target is a 133 ms average first-token latency. With 160 ms chunks a first token waits about 80 ms on average for its chunk to close, so smaller chunks are the most direct lever.
- **Gate:** compared with the chunk arm, each decoded at chunk 4, 8 and 16 on full dev (paired bootstrap), arm 10 must not lose by more than the control spread at any of the three chunks.
- **Reported, not gated:** WER at chunks 1 and 2, and first-token latency at every chunk.
- **Early look:** at step 2,000, flag in the chat if it is more than 1.0 pp worse than the chunk arm at chunk 4 or 8 on the monitor panel. It was not (within 0.12 pp at steps 2,000 and 4,000).

## Amendment 5 (2026-11-22, posted in AGENT_CHAT at 23:11Z): arm 11 (stream-context augmentation)
Written and posted in the chat before arm 11 started (23:33Z); committed here afterwards. At that time eval-lane had posted the stream-penalty diagnosis (23:03Z): about +0.62 pp of the +0.95 pp a continuous stream costs comes from the encoder's 5.12 s left-context window holding the preceding gap.

- **Arm 11:** `stream_ctx_s=5.12 stream_ctx_cap=1.5`, fixed decode chunk 4. Same fork point, schedule, batch and seed as control, on xbabe1 (RTX 4090).
- **What it does** (`train/stream_ctx.py`):
  - each training utterance gets a random left context of 0-5.12 s in 40 ms steps, drawn per example and seeded per (seed, epoch, utterance), with no external data;
  - one third digital silence, one third white / pink / brown noise at -40 / -30 / -20 dBFS RMS, one third the tail of another utterance of the same batch with an optional 0-2 s silence gap;
  - the encoder attends over the context; the CTC and transducer losses see only the utterance's frames (each row's encoder output is shifted left by its context), so prepended words are never targets;
  - utterance + context stays within 1.5x the batch's longest utterance, which bounds the padded batch (measured 1.19-1.33x on train-clean-100; peak GPU memory 18.6 GB against 12.9 GB).
- **Decision rule:** on isolated full dev it must not lose to control by more than the control spread; on the endless-stream hour (eval-lane's set, INT8 engine, `score_alwayson.py`) its stream penalty is reported against control's, and adoption needs it to cut control's penalty by at least half. Both must hold.
- **Cost:** about 1.25x control's compute per step.

## Amendment 6 (2026-11-22, posted in AGENT_CHAT at 23:13Z): arm 12 (arms 10 + 11 combined)
Written and posted in the chat before arm 12 started; committed here afterwards.

- **Arm 12:** `chunk_modes=1,2,4,8,16 eval_chunks=1,2,8 stream_ctx_s=5.12 stream_ctx_cap=1.5`. Same fork point, schedule, batch and seed as control and arms 10 and 11, on xbabe6 (A10), vCPUs 0-19.
- **Purpose:** the candidate recipe for A1's successor (always-on + low first-token latency): do the two changes combine cleanly? It is not a verdict on either change alone; arms 10 and 11 give those.
- **Reading:** on isolated full dev at chunks 4 / 8 / 16 it must not lose to the chunk arm by more than the control spread; its stream penalty is compared with control's and arm 11's; WER at chunks 1 / 2 and first-token latency are reported, not gated.
- **Process note:** the first start (23:14Z) trained without any context because the trainer set the loader's context option only on the multi-GPU path. Its GPU peak matching the plain arms gave it away; it was stopped at step 100 and kept aside (`H41_chunklow_streamctx.noctx-start`). The fix sets it for single-GPU and DDP alike, and the trainer prints the setting at start. A second restart at step 100 added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for memory headroom on the A10 (kept aside as `.alloc-restart`). The run that counts started at 23:17Z.

## Amendments 7-11 (2026-11-23, each posted in AGENT_CHAT before its arm started; committed here afterwards)
- **Amendment 7 (00:32Z): arm 12b** = arm 12's settings with `seed=4321` (xbabe1), a seed pair for the A1-successor recipe. Read together with arm 12 (each seed alone and their mean against the chunk arm).
- **Amendment 8 (00:32Z): fourth control-like run** = control's settings with `seed=2468` (xbabe2). The spread becomes the range of four control-like runs for verdicts given from then on (arms 9, 11 / 11b, 12 / 12b, 13a-c); verdicts already given stand under the three-run spread they were judged with.
- **Amendment 9 (01:09Z): arms 13a and 13b** = arm 10's recipe + `delay_penalty` 0.01 (13a, xbabe1) and 0.03 (13b, xbabe2), toward the owner's 133 ms first-token latency (arm 10 measured 176 / 199 / 239 ms at 40 / 80 / 160 ms chunks). WER guard against arm 10 at chunks 4 and 8, EMA, pooled full dev: no worse than the four-run spread. Reported outcome: first-token latency at chunks 1 / 2 / 4. Kill rule: more than 1.5 pp worse than arm 10 at step 2,000 (dev-clean400 EMA, chunk 4).
- **Amendment 10 (01:38Z): arm 11b** = arm 11's settings with `seed=4321` (xbabe6). **Stopped at step 906 (last logged step 900) on 2026-11-23 01:49:57Z after arm 11's verdict (01:48Z: isolated full dev 0.21 pp worse than control in every cell; stream penalty 1.17 to 1.00 pp, half was required). No result was read from 11b** (its first eval would have been at step 2,000); its files are kept.
- **Amendment 11 (01:50Z): arm 13c** = arm 10's recipe + `delay_penalty=0.005` (xbabe6). The value follows the lead's rule from arm 13a's step-2,000 monitor, which showed a small WER cost (0.03-0.18 pp EMA, within the panel's seed noise) and so selected 0.005 over 0.02. Same reading as 13a / 13b.

### Arm 13b restart (2026-11-23 02:08Z)
- Arm 13b was stopped by the trainer's divergence guard at step 1,300 (01:59Z): "train-loss EMA -0.783 > (1 + 0.3) x best -0.846".
- **That was a false positive.** With `delay_penalty=0.03` the transducer term, and so the total loss, is negative (transducer about -0.3 to -0.57, CTC steady at 0.43-0.50 and slightly falling), and a relative rule on a negative best trips on any rise above (1 + 0.3) x best.
- The guard now uses best + ratio x |best| (identical for a positive best) with a test for a negative best. Arm 13b resumed from its step-1,300 checkpoint at 02:08Z with the fixed guard (`DIVERGED.json` kept as `DIVERGED.json.false-positive-0207Z`). Nothing else changed.
- Arms 13a and 13c have positive total losses (about 1.02-1.13 EMA) and were not at risk; the fix is on their hosts' trees for any later restart.


### Arm 13b outcome (recorded 2026-11-23)
- Arm 13b **collapsed at 0.03** (transducer WER ~94% from step 2,000; CTC healthy). Its kill rule was not re-armed after the guard restart, so ~50 min of the 3090 were lost. Every later arm has an automated kill rule armed from the start that reads the run's `train.jsonl`, so it survives restarts (`train/ops/h41_killrule.sh`).

## Amendments 12-17 (2026-11-23, each posted in AGENT_CHAT before its arm started; committed here afterwards)
Common to all: fork H22ctl step 60,000, 200 warmup steps to 2.476e-4, cosine to 3.0e-5 at step 8,000, batch 650 s / 144, seed 1234 unless stated, launcher `train/ops/h41_replay.sh`, guard at 0.87. Kill rules are automated from the start: stop by recorded PID if the step-2,000 dev-clean400 EMA at chunk 4 is more than 1.5 pp worse than the comparator's (6.605 vs arm 10, 6.592 vs the chunk arm), or if raw chunk-4 WER exceeds 20% at any eval.
- **Amendment 12 (02:37Z): arm 13a-r** = arm 13a's settings (`chunk_modes=1,2,4,8,16 eval_chunks=1,2,8 delay_penalty=0.01`) with `seed=4321` (xbabe1). It replicates 13a's latency result (93 / 105 / 146 ms at chunks 1 / 2 / 4, against arm 10's 176 / 199 / 239 ms), since a latency claim needs a seed pair. Reading as for 13a: latency per seed and the pair's mean; WER against arm 10 at chunks 4 and 8 with the four-run spread as the guard.
- **Amendment 13 (02:54Z): arm 13d** = arm 10's recipe with `delay_penalty=0.0075` (xbabe2), the midpoint between 13c (0.005) and 13a (0.01). Outcome: latency at chunks 1 / 2 / 4. WER guard against arm 10 at chunks 4 and 8.
- **Amendment 14 (03:03Z): arm 14** = the adopted chunk-arm recipe (`chunk_modes=4,8,16`, A1's) with `delay_penalty=0.005` and `eval_chunks=8,16` (xbabe6, A10). WER gate against the chunk arm at chunks 4 / 8 / 16: no loss beyond the four-run spread at any of them. Reported outcome: latency at chunks 4 and 8. It is a candidate for A1's successor if it holds WER and cuts latency.
- **Amendment 15 (03:06Z): five arms on A1's recipe**, one per A100 freed by A1, compared with the chunk arm at chunks 4 / 8 / 16. Accuracy arms (16-18) are adopted only on a win of more than 2× the four-run spread. Latency arms (14-r, 15) must not lose more than the spread.
  - **14-r:** `delay_penalty=0.005`, `seed=4321` (arm 14's seed pair).
  - **15:** `delay_penalty=0.0025`.
  - **16:** stateless predictor context 4 instead of 2 (`pred_ctx=4`). The fork's predictor projection is widened function-preservingly (zero weights for the two older positions), so step 0 equals the fork.
  - **17:** left attention context 256 frames (10.24 s) instead of 128 (`left_context=256`). Reported with the always-on interaction noted, since a longer window holds more of a preceding gap. Not an always-on candidate on its own.
  - **18:** stronger SpecAugment: time masks 4 per 10 s (was 2) of up to 60 frames (was 40), frequency masks unchanged. About 12% of frames are masked on average (was about 4%).
  - Code: `train/recipe_opts.py` (config keys `pred_ctx`, `left_context`, `specaug_time_masks`, `specaug_time_width`, and the predictor widening), with a test.
- **Amendment 16 (03:20Z): arm 19** = the chunk arm with `seed=4321` (xbabe8). This is the comparator's own seed pair, so eval-lane can see how much of each amendment-15 difference is the comparator's seed noise. Not a candidate itself. The pre-registered primary comparator stays the seed-1234 chunk arm; the pair's mean is a secondary reading.
- **Amendment 17 (03:25Z): arm 20** = the chunk arm's recipe with `chunk_modes=2,4,8,16` (80 ms added) and `delay_penalty=0.01`, `eval_chunks=2,8,16`. Placed on xbabe2's 3090 after arm 13d (a start on xbabe1 was stopped at step 50 and set aside unread when the lead moved that card to eval-lane). The question: can one model on the adopted recipe family serve both an 80 ms low-latency mode (target first token < 133 ms) and the 160 / 320 ms normal modes? Outcome: first-token latency at chunks 2 and 4. WER guard against the chunk arm at chunks 4 / 8 / 16 under the spread rule. WER at chunk 2 is reported.

### Burst verdict decodes (2026-11-23, lead 03:26Z)
For arms 14, 14-r and 15-19, the full-dev EMA decodes run on each arm's own GPU as soon as it ends (`train/ops/h41_after_end.sh` → `train/ops/h41_burst_decode.sh`):
- They use eval-lane's pinned d5f231e evaluator and its command lines and file layout: greedy, max 8 symbols, batch 8, chunks 4 / 8 / 16, dev-clean + dev-other.
- The six decodes run at once, under the GPU lease and the fleet guard.
- eval-lane computes the verdicts from those files.
- **Fix (03:50Z):** the pinned evaluator rejects config keys added after d5f231e, so the first decodes of arms 14-r and 15 all failed at load, and the script still wrote `DONE`. Now `train/ops/h41_compat_ckpt.py` writes a copy without those keys, but only when they hold the values that rebuild d5f231e's model; strict loading still rejects any parameter mismatch. Arm 16 (pred_ctx 4) cannot be built by d5f231e and is decoded with the lane tree's evaluator. `DONE` is written only when every decode succeeded.

## Amendments 18-19 (2026-11-23, posted in AGENT_CHAT before the arms started)
- **Amendment 18 (03:50Z): arms 16-r and 18-r** = arms 16 and 18 with `seed=4321` (x5 GPU0 / GPU1). If arm 16 or 18 passes its gate, adoption uses the seed pair: the mean of the two seeds, each against the chunk-arm pair (1234 / 4321). If it fails, the replicate is reported and closes the question. Arm 17 gets no replicate yet.
- **Amendment 19 (03:59Z): arm 20-r** = arm 20 with `seed=4321` (x9), read as arm 20 for each seed and as the pair's mean. **Arms 21a / 21b / 21c** = the chunk arm plus KD from R0 (H22's loss, R0 speed-1.0 cache), `kd_weight` 0.1 / 0.3 / 1.0, seed 1234 (x7 GPU0 / x7 GPU1 / x8).
  - Gate: adopted only on a win over the chunk arm of more than 2× the four-run spread (EMA pooled full dev, chunks 4 / 8 / 16).
  - The KD weight pick is the lowest dev-clean400 WER at step 8,000, as H22 pre-registered.
  - Speed-perturbed samples (0.9 / 1.1) have no cache entry and train without the KD term. `TeacherStore.get` now matches the speed exactly; it used to round to 0.1, so a factor like 0.96 would have got the 1.0 targets. A test covers this.

## Amendment 20 and trunk step A1L (2026-11-23, each posted in AGENT_CHAT before it started)
- **Amendment 20 (04:17Z, placement addenda 04:21Z / 04:27Z): the full-coverage KD sweep.** Arm 22 = the chunk arm with speed perturbation off (`speed_perturb=` empty), seed 1234. Arms 23 / 23b / 23c = arm 22 plus KD from R0 at `kd_weight` 0.3 / 1.0 / 0.1. Arm 23-r = arm 23 with `seed=4321`.
  - Reading: each KD arm minus arm 22 (EMA pooled full dev, chunks 4 / 8 / 16, paired bootstrap) is KD's effect when every sample can have a target (the 58.5% whose R0 1-best matches the reference). These arms are not adoption candidates; they decide whether the best of 21a-c gets a full-cache run.
- **Amendment 19 addenda:** arm 21c (`kd_weight=1.0`) moved from x8 to x9. Arm 21b-r = arm 21b with `seed=4321`.
- **Trunk step A1L (04:44Z):** A1's final checkpoint (`ckpt-00049000.pt`) fine-tuned on A1's recipe plus `delay_penalty=0.005`, the H41 replay schedule (8,000 steps), seed 1234. The model starts from the raw weights and the EMA restarts from them. Launched with `INIT_FROM` / `RUN_NAME` on `train/ops/h41_replay.sh`.
  - Gate: full-dev EMA pooled at chunks 4 / 8 / 16 within the four-run spread of A1-49000's (8.302 / 7.961 / 7.703). First-token latency at chunks 4 / 8 is the reported outcome. It is promoted only if arm 14 (seed 1234) passes (it did, 04:42Z). Kill rule threshold 6.163 (A1's last dev-clean400 EMA 4.663 + 1.5 pp).
- **Adopted (lead, 04:43Z):** `delay_penalty=0.005` joins the trunk recipe; arms 14 and 14-r both pass the latency arm's gate against the chunk-arm seed pair.
