# H261: content-word-substitution reranker on the c32 B16 final (TRAIN-only fit)

**Status:** KILLED by its pre-registered gate (paired upper bound vs H244 not < 0). DEV selection half S only,
scored with the pinned ultra-score (d784f8ca…). claude lane, WER push 2026-11-27.

## Question
codex's H257 anatomy: 79% of the remaining errors are substitutions, mostly frequent content words. Can a
reranker fit only on TRAIN-domain n-best lists recover some of them on top of H244 (H234@2000 c32 B16 +
kd12L_t2 0.8/0.8, wide skip 3.75, 4.6287%)?

## Headroom (`tools/headroom.py`, H244 lists, local Levenshtein)
2,453 final edits, 16-best oracle 1,692. Frequent content-word (train-960 count ≥100) substitutions: 740, of
which 326 have the reference word at that position in some 16-best hypothesis (first correct rank 1–3 mostly).
Rare content words: 770 (171 in the list). Function words: 419 (173). Several "content" errors are US/UK
spelling (HONOR/HONOUR, COLOR/COLOUR) and MAN/MEN.

## Setup
- `tools/nbest_driver.py`: codex's dev_eval worker protocol on an arbitrary PCM npz. It reproduces all 2,782
  H244 n-best lists exactly. H234@2000 INT8, H133 worker, c32 B16, final.ulm 0.8 / LODR 0.6, 320 ms tail.
- TRAIN lists (`tools/build_mls_train*.py`): MLS-en TRAIN split, (speaker, book) in `allowed_train_pairs.tsv`
  (LibriSpeech and MLS-eval overlaps removed), speakers absent from the AM's MLS packs; 9,196 utts / 612
  speakers / 38 h. TRAIN top-1 5.83%, oracle 4.42% (first sample). Row-group reads only; 2 unreadable
  parquets (bad sectors on the xbabe3 data drive) skipped.
- Features (`tools/feats.py` + engine `tf_rescore_nbest` INT8 L2R + kenlm M5): base, L2R, M5, n_words,
  content-word L2R share, n_content, content log-frequency, content OOV count. Baseline replay (INT8 L2R,
  duplicate-text logaddexp as the engine does) matches H244 on 2,774/2,782 finals (Δ −0.002 pp).
- `tools/rerank.py`: linear score, expected-edit loss on routed TRAIN lists, L2 pull toward the H244 weights;
  λ chosen by TRAIN-only speaker 2-fold (held-out edit Δ −7 / −39 / −148 / −119 / −69 for
  λ 1e-2 / 1e-3 / 1e-4 / 1e-5 / 0). Applied once, unchanged, to S at H244's routing.

## Result
| row | S pooled | clean / other | vs H244 [one-sided 95%] |
|---|---:|---|---|
| H244 | 4.63 | 2.31 / 7.23 | — |
| H261, 3.6k TRAIN utts, λ 1e-4 | 4.60 | 2.30 / 7.23 | −0.027 [−0.060, +0.008] |
| **H261, 9.2k TRAIN utts, λ 1e-4** | **4.60** | 2.24 / 7.25 | **−0.028 [−0.065, +0.011]** |

The fit mostly learns a word bonus (n_words +0.55, n_content +0.44; deletions fall, insertions rise); the
content-word LM share gets +0.09. Tripling the TRAIN lists did not move the S effect, so it is not a variance
problem. The available features do not separate the acoustically confusable frequent content words.

## Decision
Kill (gate: upper bound < 0 after the first full fit). Content-word substitutions need acoustic evidence
(e.g. a second-pass acoustic rescoring of the top-k, or the hard-speaker acoustic fixes), not more LM-side
features over the same lists.
