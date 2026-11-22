# H12: Accuracy diagnostics: the multirate padding fix and where D2's errors come from

**Status:** `EXPLORATORY`. Both measurements are complete. The fix ships in every MR run from now on.

## Questions
1. The batched training path zeroed padding only *after* pairing 25 Hz positions, so a padded state could leak into the pair projection (CPU-ASR spec §4.1). Was that a real bug, and did it cost the shipped MR checkpoint (D2MR-final) any WER?
2. Where do the dense D2 model's errors come from? This should decide the next accuracy wave (spec §10.1). The candidate sources are:
   - search: greedy vs beam;
   - missing future context: trained chunk vs larger chunks vs full attention;
   - utterance length or edge words: per-duration and first/last-word breakdowns.

## Setup
- **Code:** lane/trainer.
  - Fix: `ac30d53`, in `train/model.py` `Encoder.forward`, multirate.
  - Tooling: `d240ed8`:
    - `train.evaluate` flags `--batch/--order/--beam/--chunk/--left-context/--full-context`;
    - `train/beam.py`, icefall-style modified beam search with one token per frame;
    - `train/diagnose.py`;
    - scripts `train/ops/mrpad_impact.sh` and `train/ops/h12_diag.sh`.
- **Checkpoints:**
  - D2MR-final: S_MR, 960 h, `ckpt-00050000.pt`, sha256 in `D2MR_S_MR_960h_copy/ckpt-00050000.sha256`.
  - D2 snapshot: S dense, 960 h, `model-00050000.pt`, sha256 `9a4097bc…a6a0b`, trained chunk 4 = 160 ms, left context 128.
- **Data:** full LibriSpeech dev-clean (2,703 utterances) and dev-other (2,864). FP32 on CPU.
  - Fix impact: H0, cores 16-45, guarded.
  - D2 diagnostic: xbabe4 M3 Ultra CPU, nice 10.
- **Scoring:** provisional `ultra.wer` (Python). `ultra-eval` stays the scoring authority.

## 1. Multirate padding before pairing

**The bug.** `Encoder.forward` zeroed nothing before `pair`.
- In a padded batch, the last valid 25 Hz position of an odd-length utterance (encoder length 29, from input lengths 113–116) was paired with a padded position.
- The blocks leave padded positions nonzero, because of biases and padded queries attending to themselves.
- The streaming `ChunkEncoderMR` and `ultra-engine` both pair an odd final row with zeros. The engine packs each stream's own rows into a zero-filled `pair_in` (`crates/ultra-engine/src/encoder/mod.rs`). So batched training disagreed with inference.

**The fix.** `x.masked_fill(pad[..., None], 0)` before pairing.
- New test (`test_multirate_batched_junk_padding_matches_single`): every input length from 105 to 120, batched next to a longer utterance with ×50 junk padding, must equal the utterance run alone.
- It fails on the old code and passes on the fixed code.

**Impact on D2MR-final** (WER %, full dev):

| Code | Batch / order | dev-clean | dev-other |
|---|---|---:|---:|
| old | 8, length-sorted (as the finisher scored it) | 11.088 | 22.900 |
| old | 8, shuffled (much more padding) | 11.082 | 22.868 |
| fixed | 8, length-sorted | 11.097 | 22.902 |
| fixed | 8, shuffled | 11.097 | 22.902 |
| either | 1 (no padding) | 11.097 | 22.902 |

- With the fixed code, batched results equal batch 1 exactly in every configuration.
- The old code changes 16 of 2,703 dev-clean and 42 of 2,864 dev-other hypotheses, and only by ±0.03 pp.
  - Batched old-code numbers are *slightly better* than the per-stream truth. The model was trained with the leak and makes tiny use of the extra input.
- **The deployed (per-stream) WER is the fixed / batch-1 row.** The finisher's GPU numbers (11.086 / 22.898) differ by ≤ 0.02 pp.
- **Decision:**
  - No corrective fine-tune of D2MR is needed. R4T-960 (ternary QAT from D2MR-final) continues.
  - All new and resumed MR runs use the fix. D3MRc relaunched fresh with it; D4M resumed with it at step 5200.
  - D2MR finalist selection was re-run entirely on the fixed code; the earlier mixed-code rows are kept as `selection_mixedcode.jsonl`.

## 2. D2 error sources (D2 model-50000)

| Decode / attention | dev-clean | dev-other |
|---|---:|---:|
| greedy, trained chunk 4 (160 ms) | 8.650 | 20.160 |
| **modified beam 4**, same encoder | **7.998 (−0.65)** | **18.529 (−1.63)** |
| greedy, chunk 8 (320 ms, untrained) | 8.467 (−0.18) | 19.653 (−0.51) |
| greedy, chunk 16 (640 ms, untrained) | 8.264 (−0.39) | 19.259 (−0.90) |
| greedy, full attention (untrained) | 9.456 (+0.81) | 20.007 (−0.15) |

**Breakdown, greedy at chunk 4** (`train/diagnose.py`):
- **Error mix:**
  - dev-clean: S 3,745, D 534, I 427.
  - dev-other: S 7,773, D 1,613, I 885.
  - Substitutions dominate.
- **Duration:** no length trend. dev-clean WER by 0–5 / 5–10 / 10–15 / 15–20 / 20+ s is 9.2 / 8.6 / 8.3 / 8.9 / 8.3%. dev-other is flat at 19.7–20.7%.
- **Word position:**

  | Position | dev-clean error (deletion) | dev-other error (deletion) |
  |---|---:|---:|
  | first word | 8.8% (1.15%) | 19.0% (4.7%) |
  | interior | 7.7% (1.0%) | 18.3% (3.1%) |
  | last word | 9.5% (**0.04%**) | 21.2% (0.9%) |

  - Last words are substituted, not dropped.
  - No insertions occur after the last reference word.
  - The first word of dev-other has elevated deletions (4.7%). That is a start-of-utterance effect, not an end one.
- **Beam 4 vs greedy:** lowers errors at every position, mostly fewer substitutions and deletions (dev-other S 7,773 → 7,225, D 1,613 → 1,277). Insertions rise slightly (427 → 440 clean, 885 → 938 other).

**What this says:**
1. **Search error is large**: 0.65 pp on dev-clean and 1.63 pp on dev-other, about 8% of all errors, recoverable with no retraining. Greedy decoding with a 2-token stateless predictor leaves real accuracy on the table. A bounded native beam in the engine is the cheapest accuracy lever (spec §11.1), at a CPU cost to measure.
2. **Future context is worth something, but only if the model is trained for it.**
   - Larger chunks help even without training (−0.39 / −0.90 at 640 ms).
   - Full attention, which the model never saw, hurts dev-clean.
   - A training-time full-context branch (dual-mode, spec §10.5) would let the model actually use future context. The untrained chunk-16 gain is a lower bound on what lookahead is worth. Whether more latency is acceptable is a product decision.
3. **Utterance length and final-word handling are not where the errors are.** The gap to R0 is spread over interior words and mostly substitutions. That points to capacity, exposure and data (epochs, pruned loss for more of them, teachers), plus decoding context, rather than a boundary bug.

## 3. Follow-ups (xbabe0, cores 0-47)

**D2 model-55000, full dev:**

| D2-55k | dev-clean | dev-other |
|---|---:|---:|
| greedy | 8.073 | 19.734 |
| beam 4 | 7.496 (−0.58) | 18.380 (−1.35) |

The beam gain persists at a later checkpoint.

**Checkpoint averaging** (`train.average`, 730-utterance DEV selection panel, manifest `76fead74…`):

| Checkpoint | panel dev-clean (400) | panel dev-other (330) |
|---|---:|---:|
| model-55000 | 8.167 | 18.923 |
| avg of 45k, 50k, 55k (last 3) | **7.754 (−0.41)** | **17.279 (−1.64)** |
| avg of 35k–55k (last 5) | 7.931 (−0.24) | 17.279 (−1.64) |

Averaging the last 3 snapshots is free at inference and recovers about as much as beam 4. The two are likely to stack, but that has not been measured.

**H13 preparation:**
- **SentencePiece 512.** Same recipe as the 1024/4096 models. `librispeech_train_bpe512.model`, sha256 `4e213d44e53c22dceeb22dcfc14b8ae3adf790c6fd4a043606c63902858b2dba`.
- **Pruned RNN-T loss** (`train/ops/pruned_validate.py`, k2 `1.24.4.dev20250714+cpu.torch2.7.0`):
  - **Exactness (float64, mixed lengths, including U = 0):**
    - k2's full regular `rnnt_loss`, and `rnnt_loss_pruned` at full range, both equal our `rnnt_nll`.
    - Values agree to ≤ 7.1e-15.
    - Gradients w.r.t. enc_p, pred_p, W_out and b_out agree to ≤ 1.3e-13.
  - **Approximation** on 16 D2-55k dev-clean utterances (mean T 122, U 19.6), with prune ranges taken from the exact lattice's occupation:
    - relative loss gap 2.1e-6 at s_range 3 and 4.7e-7 at s_range 5 and 8;
    - gradient cosine 1.000000 for all four tensors.
    - These ranges are the *best case*. In training, the ranges come from a jointly trained simple joiner (icefall uses s_range 5), so the convergence check still needs a matched run.
  - **Speed** (CPU, batch 8 × T 250 × U 60 × V 4097, forward+backward of the loss alone): exact chunked 7.85 s vs pruned s_range 5 0.77 s, **10.3× faster**. This excludes the simple-joiner range computation. The GPU ratio still has to be measured.

## 4. Addendum: do the levers stack? And beam on the MR finalist (xbabe0)

| Model | Panel dev-clean, greedy → beam 4 | Panel dev-other, greedy → beam 4 | Full dev-clean, greedy → beam 4 | Full dev-other, greedy → beam 4 |
|---|---:|---:|---:|---:|
| D2 model-55000 | 8.167 | 18.923 | 8.073 → 7.496 | 19.734 → 18.380 |
| D2 last-3 average (45/50/55k) | 7.754 → **7.258** | 17.279 → **15.941** | 7.494 → **7.003** | 17.751 → **16.603** |
| D2MR-final (ckpt-50000, fixed code) | 11.566 → 10.233 | 22.090 → 20.413 | — | — |

- **The two levers stack.** On full dev, the last-3 average with beam 4 is **−1.07 / −3.13 pp** below the single snapshot decoded greedily (8.073 → 7.003, 19.734 → 16.603).
- Beam 4 helps the MR finalist too (−1.33 / −1.68 on the panel).
- D2 snapshots 60k and 65k are processed automatically as they arrive (`train/ops/watch_d2_x0.sh` → `xbabe0_d2_step.sh`), with rows posted to AGENT_CHAT, so C1 selection is ready when D2 ends.

## Next (proposed)
- Bounded native beam (width 2–4) in `ultra-engine` as a measured product option: WER vs CPU-s/s on the panel. This is engine lane work.
- Dual-mode fine-tune of D2 (streaming + full-context branch, consistency loss), 20–30k steps, against a control with the same steps. Measure at chunk 4 and at chunk 8.
- **H13a** (running, xbabe2): dense S from scratch with SentencePiece-512 for both heads, the exact loss, EMA and speed perturbation.
- **H13b:** the pruned RNN-T option is now in the trainer (`rnnt_loss=pruned`, `prune_range`, `prune_warmup`). It has a training-only simple joiner and no inference change. Parity test: at full band it equals the exact loss, values and gradients. k2 CUDA 12.8 / torch 2.7.0 is installed on H0 and xbabe2. GPU smoke test (S, V 513, batch 16 × 10 s, bf16): exact 0.33 s/step vs pruned 0.22 s/step, peak memory 4.2 vs 3.7 GB. H13b launches on the next free GPU.
- Use averaging of the last 3 snapshots for D2's final selection, as the D2MR selection already does.
- Weight EMA is already on in D3MRc (`ema_decay=0.9995`).

## Artifacts
- `/mnt/xbabe3-external/jsr/runs/train/D2MR_S_MR_960h_copy/mrpad/`: 10 JSON summaries and per-utterance dumps.
- xbabe0 `~/jsr-data/runs/train/D2_S_960h/x0_eval/` (55k and averaging) and `~/jsr-data/logs/pruned_validate.json`.
- `/mnt/xbabe3-external/jsr/runs/train/D2_S_960h/h12_diag/`: 8 summaries, dumps and `*_breakdown.json`. The originals are on xbabe4 in `~/jsr-train/data/runs/train/D2_S_960h_h12/diag/`.
