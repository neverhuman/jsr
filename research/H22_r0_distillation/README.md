# H22: Distillation from R0 (owner-approved teacher)

**Status:** `PRE-REGISTERED`. The teacher cache is running; the student is not launched.

**Approval and rules** (DECISIONS.md):
- R0, the icefall streaming Zipformer (sherpa-onnx zipformer-en-2023-06-26), is an approved distillation teacher.
- The teacher sees LibriSpeech TRAIN splits only: never dev, never test. `train/teacher_cache.py` refuses non-train splits.
- Every distilled checkpoint is labelled "distilled from R0".
- Checkpoint selection stays on the DEV panel.

## Question
At equal exposure and with an identical recipe, does frame-level distillation from R0's joiner distributions lower the student's DEV WER, and by how much?

## Teacher cache (xbabe0 CPU)
- **Model:** R0 int8 ONNX, chunk 16 / left context 128, from HF `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` at revision `672fbf1b3057…`. sha256 matches H0's copy:
  - bpe.model `c53433de…`;
  - encoder int8 `563fde43…`.
- **Run:** streaming greedy search, one symbol per frame, kaldi-native-fbank (80 mel, dither 0, snip_edges false, high_freq −400) with 0.3 s tail padding.
- **Stored per encoder frame (25 Hz, our dense S's rate):** the top-8 joiner log-probs (float16) with their R0 token ids (int16), and the emitted token.
- **Stored per utterance:** R0's 1-best text.
- **Scope:** all 960 h, at speed factors 0.9 / 1.0 / 1.1, perturbed by the same `resample_poly` and `Fraction(f).limit_denominator(100)` as train.data.
- **Format:** one `.npz` per (shard, speed) plus SHA256SUMS, on xbabe0's local disk (`~/jsr-data/teacher_cache_R0/`).
- **Smoke test** (20 train-clean-100 utterances): R0 train WER 0.92%, single-thread RTF 0.073.

## Student and loss
- **Tokenizer:** R0's bpe.model (500 pieces) for both the RNN-T and CTC heads.
  - Student output index = R0 piece id + 1 for pieces, 0 for blank.
  - The cache keeps R0's own ids (blank 0, piece p), so the map is teacher p → student p + 1 for p ≥ 1, and 0 → 0.
- **Loss:** pruned RNN-T + 0.3 CTC on the ground truth, plus `kd_weight` × KL.
  - The KL is between the teacher distribution, collapsed to {its top-8 tokens, OTHER = 1 − Σ top-8}, and the student's joiner distribution collapsed onto the same 9 buckets.
  - It is evaluated at each frame t on the lattice cell (t, u_t), where u_t = the number of tokens R0 has emitted before frame t.
  - It applies only to utterances whose R0 1-best equals the normalised reference, so that (t, u_t) lies on the ground-truth lattice. Frames beyond either encoder's length are skipped.
- **Weight tuning:** `kd_weight` ∈ {0.1, 0.3, 1.0}, tuned on a short run (3k steps, 100 h). The pick is the lowest dev-clean400 WER, DEV only.

## Matched runs (xbabe5, both A100s, after H13b/c)

| Run | KD | Everything else |
|---|---|---|
| H22-ctl | no | L1 recipe (H21) with R0's tokenizer: dense S, pruned loss, EMA, speed perturbation, qkv cap/guards, budget 1200 s, lr 1.4e-3, seed 1234 |
| H22-kd | yes (`kd_weight` from tuning) | identical |

- **Decision point:** at 10 epochs of 960 h (28.8k steps), compare full dev and the panel with last-3 averaging and beam 4.
  - If KD gains ≥ 0.3 pp pooled, continue H22-kd to 30 epochs. It becomes a candidate labelled "distilled from R0".
  - Otherwise, stop.
