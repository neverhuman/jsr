# H68: scaled neural LM n-best rescoring of the RC2 commit pass

**Status:** EXPLORATORY, DEV selection half S only (2,782 utts, 36 speakers). C half, TEST, HOLDOUT never touched.
Owner: claude lane, WER push 2026-11-26. Bulky outputs: `/mnt/xbabe3-external/jsr/wer-push/H68/`.

## Question
The RC1 commit pass had a 16-best oracle of 3.75% against 5.79% top-1 on S, and the 3.5M L2R+R2L
rescorer (D4, PR #129, merged default OFF) recovered only 21% of that gap (−0.40 pp CV). How much more
of the n-best headroom does a much stronger in-domain neural LM recover on the *shipped* RC2 final?

## Setup (fixed before any rescoring result)
- **N-best:** RC2 EMA (`REGctl-boundary.pt` sha256 fced0f51…), chunk 16, 320 ms zero tail pad, beam 16,
  M5 (`tok5.p00149.trie.bin`) + LODR (`lodr2.train960.bin`) at the shipped weights 0.80 / 0.60
  (`research/H75_shipped_error_anatomy/tools/rc2_nbest.py`, Python FastBeam; xbabe0 CPU).
- **LM data:** OpenSLR-11 `librispeech-lm-norm.txt.gz` (book-ordered), tokenised with the shipped
  `bpe.model` (c53433de…), token = sp_id + 1, 0 = sentence boundary. Lines identical to any DEV
  transcript (dev-clean + dev-other, both halves) are dropped. Every 2000th line is validation.
- **LM:** causal Transformer in the engine's `ULTRTF01` layout (RMSNorm, interleaved RoPE, SwiGLU,
  no biases, tied head) so a winner exports to the existing Rust INT8 rescorer. First rung: d 512,
  8 layers, 8 heads, ff 1536 (27.5M), ctx 256, packed random windows, bf16, AdamW, cosine. L2R and R2L.
- **Rescoring:** `base + a·L2R + b·R2L + c·n_words` over the 16-best; weights picked on one
  speaker-disjoint fold and applied to the other (2-fold CV). Pinned `ultra-score` (d784f8ca…)
  paired against the n-best top-1 and against the shipped `isosyll` final row (H75).

## Gates
- **Keep (research):** CV-rescored S pooled WER below the n-best top-1 with the paired one-sided 95%
  upper bound < 0, and at least 1.5× the D4 3.5M gain (i.e. ≤ −0.60 pp) to justify a larger model.
- **Ship candidate:** the same gain survives Rust INT8 export with commit-pass CPU ≤ +15% (one core,
  paired), else distil / shrink along a size ladder.
- **Kill:** CV gain ≤ D4's −0.40 pp after the first full LM epoch.

## Setup change (recorded before scoring)
The Python FastBeam n-best was replaced by the exact shipped commit pass: codex's native H67/H92
`ultra-dev-worker` (c16, B16, signed M5 0.80/0.60, INT8 release weights) on S PCM padded with 320 ms
of zeros (commit mode itself does not export N-best). Its top-1 reproduces all 2,782 shipped-final
transcripts (paired delta 0.000), so gains below are measured against the real v4.0.0 final row.

## Results (interim: L2R step 14.6k, R2L step 12.4k of 30k)
| row | pooled | dev-clean | dev-other | hard-5 spk | rest-31 | edits (pinned) |
|---|---:|---:|---:|---:|---:|---:|
| shipped final (= n-best top-1) | 5.44% | 2.68% | 8.54% | 15.29% | 3.79% | 2,838 |
| + H68 CV rescoring | **4.89%** | **2.37%** | **7.72%** | 14.33% | 3.35% | 2,552 |

Pinned `ultra-score` paired vs shipped final: macro −0.563 pp, one-sided 95% [−0.688, −0.440].
CV weights (a, b, c, s): fold0 (0.5, 0.3, 1.5, 0.6), fold1 (0.5, 0.6, 3.0, 0.6); full-S-tuned is 2,551
local edits against 2,598 CV, so the selection is stable. 16-best oracle: 1,792 local edits (3.46%).

- Keep gate (≤ −0.60 pp with upper < 0): upper bound passes; point estimate −0.56 is just short at
  half schedule. Final checkpoint decides.
- Rejected side arms: H77 chapter name cache (CV +16 edits); session context for L2R (CV 2,633 vs 2,598);
  M5 subtraction > 0.6 (CV 2,620).
- Cost not yet measured: 27.5M params ≈ 8× the D4 rescorer. Next: `.utfr` INT8 export, Rust parity,
  paired one-core commit CPU, and a size ladder if over +15%.

## Results with landed checkpoints and cost operating points (2026-11-26, 07:40–08:00 MST)
Commit CPU = A2/H141 INT8 commit pass, 147-utt S panel, one core, ABAB×2 (`tools/qual2.sh`). Skip = no
Transformer rescoring when fused-beam top-1 − top-2 ≥ threshold (opt-in `--rescore-skip-margin`);
thresholds and weights by speaker 2-fold CV on S. Panel timing noise is about ±3 pp.

| acoustic final (base) | LM / setting | commit CPU | S pooled | macro Δ [one-sided 95%] |
|---|---|---:|---:|---|
| A2 H104 (5.34) | 27.5M L2R+R2L, no skip | +61.2% | 4.70 | −0.655 [−0.764, −0.554] |
| A2 H104 | 27.5M pair, skip ≥1.48 | +25.0% | 4.82 | −0.531 [−0.638, −0.433] |
| A2 H104 | 27.5M pair, skip ≥0.74 | +15.5% | 5.02 | −0.322 [−0.408, −0.242] |
| A2 H104 | **11.7M L2R, skip ≥3.63** | **+10.8%** | **4.86** | −0.481 [−0.564, −0.400] |
| H141 (5.39) | 27.5M pair, no skip | +63.7% | 4.68 | −0.723 [−0.859, −0.590] |
| H141 | 27.5M pair, skip ≥1.39 | +26.1% | 4.81 | −0.595 [−0.725, −0.467] |
| H141 | **11.7M L2R, skip ≥2.76** | **+11–15%** | **4.85** | −0.552 [−0.681, −0.431] |

- Keep gate (≤ −0.60 pp, upper < 0): met by the unrestricted 27.5M pair on H141 (and −0.655 on A2).
- Ship gate (commit CPU ≤ +15%): met by the 11.7M L2R with the margin skip, at about −0.5 pp.
- Rejected cost levers: L2R top-8 candidate cut (trie already shares prefixes), act7 activations
  (−1 pp CPU), beam 32 (−0.047 pp, n.s., +62% beam CPU).
- What it fixes: common/function words and insertions; nothing on unseen names (see H75).

## Distillation and native-engine confirmation (2026-11-26, 13:00 MST – 2026-11-27, 02:00 MST)
Offline replay (bf16 PyTorch features over the saved 16-best) overstated INT8 engine gains. From here on every
row is the native `ultra-engine-commit` (9fdc75ff) on all 2,782 S utterances (`tools/engine_fullS.sh`,
`tools/engine_grid.sh`), with the rescore weight w and M5-sub s picked by speaker 2-fold CV **over engine
outputs** (`tools/engine_cv.py`), for the lock's own model too. Skip margin 2.762935630679131 throughout.

| rescorer (11.7M L2R .utfr) | recipe | engine-CV S | vs engine lock [one-sided 95%] |
|---|---|---:|---|
| m12_l2r_32k (lock) | CE 32k | 4.86 (folds re-pick 0.6/0.6) | — |
| KD24 (8db9f490…) | KD from 27.5M L2R, 24k | 4.83 at 0.6/0.6 | −0.026 [−0.082, +0.028] |
| **kd12L_t2 (2bce52ea…)** | KD24 + 16k KD from continued teacher (valid 2.463), kd-w 0.7 | **4.79 (0.8/0.8 both folds)** | **−0.071 [−0.118, −0.024]** |
| kd12L_t3 | t2 + 16k more | 4.84 (folds disagree) | −0.021 [−0.100, +0.063] |
| kd12L_t3w9 | kd-w 0.9 chain | 4.84 (folds disagree) | −0.019 [−0.101, +0.071] |

- The locked config's offline 4.84 is 4.86 in the engine (+0.017 [+0.001, +0.033]; 2,730/2,782 identical).
- Quiet-core CPU (xbabe0 core 40, ABAB×6, load ≤0.54 at each start): lock +10.8% commit / +5.6% process;
  kd12L_t2 @0.8/0.8 +11.1% / +6.1%.
- Rejected: 5.2M R2L in the pair (no gain), train-960 transcript fine-tune (+0.024 vs KD24), longer KD
  schedules for L2R alone (no engine gain), KD pairs inside +15% (4.83 at the +15% skip point; 4.74–4.76
  needs about +28% commit).
- Proposal (config only, needs Jepson's OK): `kd12L_t2_16k.utfr` + `--rescore-weight 0.8 --rescore-m5-sub 0.8`.
  Codex's H244 stacked it on c32 lists (4.6287%).
- Seed replicate kd12L_t2s2 (same recipe, seed 51): engine S at the fixed proposal weights 0.8/0.8 = 4.78,
  −0.085 [−0.138, −0.030] vs engine lock; t2s2 − t2 = +0.007 [−0.064, +0.077]. The gain is the recipe's.
