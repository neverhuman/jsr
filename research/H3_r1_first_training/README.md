# H3: First training of our own streaming transducer (R1), 100 h

**Status:** FAILED attempts, then fixed. All attempts are kept in `docs/ultra/EXPERIMENTS.jsonl`.

## Question
Can our own streaming Conformer-transducer (S preset: d_model 384, 16 blocks, 160 ms chunks, 5.12 s left context) learn from LibriSpeech train-clean-100?

## Setup
- **Trainer:** `python/ultra/train/`, with its own RNN-T loss, CTC auxiliary loss, chunked causal encoder and async checkpoints (c6455ef, 6e6c6e9).
- **Hardware and data:** an RTX 3090 on xbabe3; train-clean-100; dev-clean 400 utterances for evaluation.

## Results

| Run | Change | Outcome |
|---|---|---|
| D1 attempt 1 | post-norm S | **Collapsed:** CTC flat at 6.18, 100% deletions (FAILED_TRAINING) |
| D1b | pre-norm S (0ea0742) | Stuck on the unigram plateau (FAILED) |
| D1c | lr 1e-3, SpecAugment from step 0 | Still stuck (FAILED) |
| **D1d** | SpecAugment delayed to step 4000 (11d527d) | **Learned:** dev-clean 84.7% @1.4k → **29.8% @7.8k** |

## What went wrong
- **Post-block LayerNorm made the 16-block stack collapse** to near time-invariant output.
- **SpecAugment from step 0** kept the model on the plateau.
- The first attempt was initially labelled "instrument invalid". It was relabelled as an honest training failure, because no instrument bug was ever demonstrated.

## Consequence
- Pre-norm plus delayed SpecAugment became the recipe.
- The R1-D1 checkpoint (step 7800) became the first R1 artifact for the export, bridge and engine work.

## Pointers
- Commits: c6455ef, 0ea0742, 11d527d, c0a610a.
- Run dirs: `/mnt/xbabe3-external/jsr/runs/train/D1*/`.
- Checkpoint `D1d.../ckpt-00007800.pt`, sha b3344342….
