# H16: bounded beam search in the Rust engine (B6), plus finalist prep and a left-context sweep

**Status:** `EXPLORATORY`.
- B6 passes its accuracy and identity acceptance.
- The CPU cost is measured only on a contaminated host, so it is indicative.
- Code: `lane/engine` a985f2a (beam); `research/H15_*` covers the earlier engine wave.

## B6: beam search

**What it is.** `Engine::with_beam(k)` and `--beam k` in eval and density (recorded as `"beam"`). The default `k = 1` is today's greedy decoder, so the default path and the continuity fixtures are unchanged.

**Algorithm.** A port of `train/beam.py::modified_beam_search`:
- frame-synchronous, at most one token per frame;
- BLANK-padded 2-token context;
- f32 log-softmax;
- top-k over hypotheses × vocabulary;
- equal sequences merged by f32 logaddexp.

**Determinism and batch invariance.**
- Ties break by score descending, then hypothesis, then token.
- The final pick is the first of equal maxima, matching Python's `max`.
- Each stream's hypotheses are their own joiner rows (the int8 GEMM quantises per row), so the search is batch-invariant. The test `batch_invariance::int8_beam4_is_batch_invariant` covers B = 1/3/5/16/32, multi-chunk steps included.

**Results on the 730-utterance DEV panel, D2MR-final, H0.**

| | FP32 WER | INT8 WER |
|---|---:|---:|
| greedy (max_symbols 8) | 15.864 | 15.933 |
| **beam 4** | **14.431** (−1.43) | **14.438** (−1.50) |

- Per split, FP32 beam 4 gives dev-clean 10.255 and dev-other 20.404, i.e. −1.3 / −1.6. That agrees with trainer H12's full-dev beam gain.
- **INT8 gate with beam 4: PASS.** Delta +0.007 pp, one-sided 95% upper bound +0.150 pp.

**Token identity against `train/beam.py`.**
- **End to end: 730/730 identical** token sequences. Engine FP32 beam 4 (streaming engine encoder) against beam.py on the trainer's chunk-masked encoder, run per utterance (`runs/engine/b6-ref2`). Corpus WER is equal under the Python scorer (14.413). The engine's own scorer reports 14.431; same tokens, different normaliser.
- **Decoder only: 50/50 identical.** `examples/beam_replay` feeds torch encoder projections straight into the engine beam. This also holds with the follow-up fixed-order SIMD log-sum-exp.

**CPU cost.** xbabe0, cores 76/84. These runs are **contaminated**: about 6 unpinned `train.evaluate` processes of another lane started during them (load 62–72). Totals are unusable. The decoder phase is better isolated:

| | Decoder phase (cpu-s/s) |
|---|---:|
| greedy | 0.0015–0.0018 |
| beam 4 | 0.0051–0.0055 (+0.0035) |

- Of the beam's decoder cost, 0.0025 was bookkeeping, mostly the scalar exp over 4 × 4097 logits per frame.
- The follow-up replaces it with a fixed-order SIMD sum (`kernels::sum_exp_shifted`). It is bit-identical across scalar, AVX2 and NEON, with a test. It also skips candidates that cannot enter the top k.
- Expected cost: +5–10% of single stream before that fix, less after. A clean measurement is owed.

## D2 average of the last three snapshots (45/50/55k)

- **What.** `train.average` of the three snapshots (input shas verified, output sha `10a581a4…`), converted and prepared on xbabe0.
- **Panel INT8 gate: PASS.** FP32 11.663 → INT8 11.704, delta **+0.042 pp**, upper **+0.140 pp**. For comparison, the single 45k snapshot was 14.168.

## B3 left-context sweep (D2MR-final, INT8, panel)

- **Setup.** The left context is changed in `arch.json` only; the weights are the same, hard-linked, and there is no retraining. The middle rate uses half.
- **Validity.** WER is valid. The CPU numbers come from the same contaminated xbabe0 window, so only the attention-phase ratios are indicative.

| left | WER (clean / other) | Attention phase vs 128 | RSS at 16 streams |
|---:|---|---:|---:|
| 128 | 15.933 (11.561 / 22.187) | 1.00 | 316 MiB |
| 64 | 16.023 (11.549 / 22.422), **+0.09** | about 0.70 | 283 MiB |
| 32 | 16.611 (12.043 / 23.146), +0.68 | about 0.45 | 267 MiB |

- **Reading.** Left 64 is a cheap density knob: +0.09 pp, attention −30%, K/V state halved. It needs its own INT8 gate and a clean density run.
- Left 32 needs retraining to be usable.

## Timing caveats

- Every xbabe0 CPU number here was taken while other lanes' unpinned evaluation jobs ran (load 20 → 70).
- H0 has the same problem with the h458 jobs and the trainer evaluations.
- Clean timing waits for the xbabe2 qualifying window, or a host with its compute pinned away from the measurement CCXs.

## Raw outputs

- H0 `/mnt/xbabe3-external/jsr/runs/engine/`:
  - `b6-panel-bin-b6-wip-001417/` (engine beam panel and gate);
  - `b6-ref2/` (beam.py panel reference);
  - `b6-ref/` (decoder-only dump and replay);
  - `d2-avg3-45-50-55/`.
- xbabe0 `~/jsr-engine-data/runs/`:
  - `x0-a-4523c2f-004145` (D2 average gate, left-context sweep);
  - `x0-b6-a985f2a-004757` (beam CPU).
