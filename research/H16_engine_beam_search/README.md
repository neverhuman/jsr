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

## Addendum (2026-11-22 04:30Z): SIMD log-sum-exp in the beam, and finalist gates

**Change** (c56c6e7). The beam's per-row log-sum-exp now uses `kernels::sum_exp_shifted`.

- It is a fixed-order sum of the engine's exp. Element `i` goes to lane `i % 8`, and the lanes reduce in one fixed tree.
- Scalar, AVX2 and NEON produce bit-identical sums. The test `kernels::tests::sum_exp_shifted_is_bit_identical_across_isas` covers lengths 1 to 4097, tails included.
- Candidates whose score is below the current k-th best are skipped before insertion. Ties still go through the deterministic order, so selection is unchanged.

**Identity.**

- Panel, D2MR-final, beam 4: tokens are 730/730 identical to a985f2a in FP32 (WER 14.431) and INT8 (14.438).
- Decoder-only replay against `train/beam.py`: still 50/50.
- The engine's beam is now one arithmetic on every ISA. Given the portable log-mel frontend (G4), x86 and ARM beam transcripts are expected to agree PCM to tokens. A merged-main cross-ISA panel run is still to come.

**Finalist gates** (panel, INT8 prepared in qualifying mode vs the FP32 control). All pass.

| Model | Search | Host | FP32 | INT8 | Delta / upper 95% |
|---|---|---|---|---|---|
| D2 avg 55/60/65k | greedy | x86 | 11.344 | 11.261 | −0.083 / +0.033 |
| D2 avg 55/60/65k | greedy | ARM | 11.344 | 11.261 | −0.083 / +0.033 |
| D2 avg 55/60/65k | beam 4 | x86 | 10.673 | 10.638 | −0.035 / +0.056 |
| D2 avg 45/50/55k | beam 4 | x86 | 10.860 | 10.901 | +0.042 / +0.145 |
| D2MR-final left 64 | beam 4 | x86 | 14.590 | 14.646 | +0.055 / +0.189 |
| D2MR-final left 96 | greedy | x86 | 15.885 | 15.933 | +0.048 / +0.171 |
| D2 avg 45/50/55k left 64 | greedy | x86 | 11.953 | 11.932 | −0.021 / +0.083 |

**Artifact.** The D2 avg 55/60/65k prepared INT8 artifact (source checkpoint sha `24d7106d…`, `weights.int8.safetensors` sha `8170357b…`) is on xbabe0 and on H0 for runtime-lane's ladder.

**Still open.** The beam's CPU cost on a clean host remains unmeasured. Every timing in this wave was taken while other lanes' unpinned jobs were running.
