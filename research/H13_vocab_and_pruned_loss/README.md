# H13: Output vocabulary and pruned RNN-T loss

**Status:** `ONGOING`. H13a is training; H13b–d are pre-registered and queued for one GPU, run one after another.

## Questions
1. Does a 512-piece SentencePiece model shared by the RNN-T and CTC heads train as well as, or better than, 4096/1024 at equal exposure, while making the loss cheaper?
2. Does the pruned RNN-T loss (k2, icefall-style simple joiner) match the exact loss in final quality at equal exposure, and how much cheaper is it per step?

## Pinned settings

### H13a: 960 h, GPU (xbabe2 RTX 3090), running since 2026-11-21 23:55Z
- **Model and loss:** preset S (dense), from scratch, `rnnt_pieces=512 ctc_pieces=512` (SentencePiece-512 sha256 `4e213d44e53c22dceeb22dcfc14b8ae3adf790c6fd4a043606c63902858b2dba`), exact loss.
- **Schedule:**
  - `seed=1234 budget_s=600 max_batch=128 lr=1e-3 warmup=2000 total_steps=102000 clip=5`;
  - `specaug_start=4000 speed_perturb=0.9,1.0,1.1 ema_decay=0.9995`;
  - `qkv_cap_from=30000 qkv_cap_ratio=1.5 qkv_guard_ratio=2.0 diverge_after=5500`.
- **Reference:** D2 (S, 4096/1024, exact loss), compared at equal audio-hours.
- **Known differences from D2:** D2 was warm-started from D1d-7800 with warmup 500, SpecAugment from step 0, and no speed perturbation.

### H13 matched 100 h controls (one GPU, run one after another, same host)
Identical for all arms:
- preset S from scratch, train-clean-100;
- `seed=1234 budget_s=300 max_batch=64 lr=1e-3 warmup=2000 total_steps=24000` (about 20 epochs; ~2,000 audio-h of exposure) `clip=5 specaug_start=4000`, no speed perturbation, `ema_decay=0.9995`;
- dev-clean400 every 2000 steps; the 730-utterance panel (greedy + beam 4) at the end.

| Arm | RNN-T loss | RNN-T pieces | CTC pieces |
|---|---|---:|---:|
| H13b | exact | 4096 | 1024 |
| H13c | pruned (`prune_range=5 prune_warmup=2000`, simple-loss scale 0.5, lm-only 0.25) | 4096 | 1024 |
| H13d | pruned (same) | 512 | 512 |

- **Comparisons:** H13b vs H13c isolates the loss. H13c vs H13d isolates the vocabulary under the pruned loss. The comparison is made at equal steps (that is, equal audio-hours, because the batches are identical) and reports wall time per step.
- **Decision rule:** pruned is acceptable if dev-clean400 at the end is within +0.3 pp of exact (H13c vs H13b). 512 is preferred if it is no worse than 4096 by more than 0.3 pp and is cheaper.

## Validation already done (H12 §3, train/ops/pruned_validate.py, tests/test_train_pruned.py)
- **Full band:** k2's pruned loss equals our exact loss, values and gradients (float64 ≤ 1.3e-13; model-level float32 ≤ 1e-5).
- **Occupation-based band of 5** on real D2 lattices: relative loss gap 5e-7. On CPU the loss alone is 10.3× faster. GPU smoke test (V 513): 0.33 → 0.22 s/step.

## Status (2026-11-22 ~02:45Z)

**H13a** (xbabe2) is at step 12,150 (epoch 1), 544 audio-h/h at guard duty 0.75.

| Step | dev-clean400 greedy | Panel dev-clean greedy / beam 4 | Panel dev-other greedy / beam 4 |
|---:|---:|---:|---:|
| 5,000 | 33.28 | 34.33 / 30.99 | 48.40 / 44.45 |
| 6,922 (end of epoch 0) | 26.97 | — | — |
| 10,000 | 20.98 | (auto-posted) | (auto-posted) |

- D2 reached 21.9 at 5k and 17.3 at 10k on dev-clean400 at the same 600 s batch, but D2 started from D1d-7800 (100 h, trained), so the early curves are not comparable.
- The equal-exposure comparison against D2 is meaningful from mid-run onward, and at the end: averaging the last 3 snapshots, decoded with the frozen beam-4 setting (H12 §5).

**H13b–d: the CPU route was rejected.** A probe on xbabe2 ran all three arms at once with 24 threads each. After 11 minutes no arm had finished 2 steps. Each process used about 1.3 cores, because the serial float64 RNN-T recursion, the k2 CPU kernels and the featurisation dominate, so that is < 1 audio-h/h per run. The arms will run one after another on the first free GPU, probably H0's 3090 when D2 ends at 102k; each takes ~2,000 audio-h, about 3–5 GPU-hours.

## Addendum (Codex, 2026-11-22)

The GPU implementation smoke used dense S with vocabulary size 513 and a BF16 batch of sixteen 10-second inputs. The 0.33 to 0.22 second step change was accompanied by a 4.2 to 3.7 GB reduction in peak memory. The queued equal-exposure experiments carry the accuracy comparison. [Trainer implementation and smoke record](../H12_accuracy_diagnostics/README.md)
