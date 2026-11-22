# Distributed training

Start with [trainer PR #10][trainer-pr] for the DDP trainer and the
[operations scripts](../../python/ultra/train/ops/README.md) for seed transfer,
real-audio validation, and fixed-rank launch. The trainer repository path is
`python/ultra/train/`. Record experiments and their evidence in the
[research index](../../research/README.md).

## Shared checkpoint and data

D2c continues the D2 step-60k/65k/70k weight average with pruned RNN-T loss
on the three LibriSpeech training splits. Its seed scored 10.964% pooled
full-DEV WER with beam 4 and the Rust normalizer. The first step-2,500
D2c checkpoint scored 6.994% raw and 6.564% EMA on the 400-utterance
DEV-clean greedy screen. Use a matched full-DEV run before comparing those
weights against the seed. The [inference protocol](FREEZE.md) defines the
separate workload used for qualified inference results.

The exact sampler builds the same ordered global batches on each rank.
Each rank reads its own portion, the trainer weights the loss by global token
counts, and synchronized gradients drive the same optimizer update on every
GPU. A fixed-global-batch parity run uses `ddp_global_batches=true`,
`ddp_native=true`, `budget_s=750`, and `max_batch=160`. Preserve checkpoint,
tokenizers, corpus manifest, sampler cursor, and recipe when comparing worker
counts. A larger global batch changes audio per update and needs its own
convergence curve.

## Validation and contribution

- Run `just pytest` to exercise the trainer and dashboard tests. Run
  `just required` before pushing. Use [the operations guide](../../python/ultra/train/ops/README.md)
  to run the bounded probe on real audio and compare its utterance-order and
  final model digests.
- An FP32 Gloo probe with six ranks across four hosts assembled disjoint
  real-audio batches and returned identical final model digests across ranks.
  A matched native DDP pilot on two A100s in one host reported about 2,300
  audio hours per wall-clock hour against about 1,950 on one A100. With
  the global budget increased from 750 to 1,500 audio seconds, the two-GPU
  native pilot reported 2,826 audio hours per wall-clock hour under the
  utilization cap. Track the larger batch as its own recipe and measure
  cross-host scaling directly.
- For multi-host throughput reports, include the global and per-rank audio
  budgets, gradient collective time, input stalls, padding, and steady-state
  rate. Track WER against cumulative audio and elapsed wall time on the same
  DEV split and decoder. Compare the resulting checkpoint with the original
  seed before promoting the recipe.
- Use the [dashboard guide](training-dashboard.md) to check training progress
  and identify each WER by run, step, and evaluation cohort.

[trainer-pr]: https://github.com/neverhuman/jsr/pull/10
