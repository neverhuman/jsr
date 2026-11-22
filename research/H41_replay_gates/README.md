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
