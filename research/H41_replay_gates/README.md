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
