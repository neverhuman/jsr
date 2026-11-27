# H276: H261 refit on H255@2000 TRAIN lists + acoustic top-k feature

Requested by watcher 04:14 after codex H268 showed that the frozen H261 fit (λ 1e-4, fitted on H234 lists) gives 4.5348% when applied to the H255@2000 c32 B16 lists.

## Setup
- TRAIN: the H261 MLS-en TRAIN set: 9,196 utts, 612 speakers, 38 h. Speakers are unseen by the AM, and (speaker, book) pairs come from allowed_train_pairs.tsv.
- These TRAIN utterances were decoded with H255@2000 (ema-int8, weights sha 311c54e6…) into c32 B16 lists, using the H133 worker, final.ulm 0.8 / LODR 0.6.
- Reranker: `../H261_content_word_reranker/tools/rerank.py`, an MWER linear reranker. λ is picked by TRAIN-only speaker 2-fold. Routing is the H244 wide skip 3.75 with engine duplicate-text logaddexp.
- DEV S lists: codex H268 transfer `nbest.jsonl` (H255@2000). The fit was applied once.
- (b) `tools/rnnt_hyp_score.py` computes am_c32, the alignment-summed RNN-T log P(tokens | audio) from the H255@2000 EMA checkpoint under the c32 training mask (chunk 32, left 128).
  - A full-context variant, am_full, was also computed, but it ranked worse on S rows. It was not used in the fit.
  - `tools/rerank_am.py` is rerank.py with am_c32 added as a 9th feature.

## Results (DEV S, pinned ultra-score, one-sided 95% bounds)
| run | TRAIN CV λ | S WER | vs H244 4.6287 | vs H255+t2 | vs frozen H261 |
|---|---|---|---|---|---|
| H255+t2 base (replay) | – | 4.59 | | | |
| frozen H261 fit (= codex H268) | 1e-4 (H234 lists) | 4.53 | −0.096 | | |
| **(a) refit on H255 TRAIN lists** | 1e-5 (Δ −133) | **4.52** | **−0.113 [−0.188, −0.038]** | −0.076 [−0.139, −0.013] | −0.017 [−0.048, +0.015] |
| (b) refit + am_c32 | 1e-5 (Δ −147) | 4.53 | −0.102 [−0.175, −0.030] | −0.065 [−0.117, −0.015] | −0.006 [−0.035, +0.024] |

- (b) vs (a): +0.011 [−0.021, +0.042]. **(b) is KILLED.** The am_c32 weight is 0.75, but the DEV gain is zero, and the feature would add an encoder pass plus an RNN-T lattice per candidate.
- The refit is not significantly better than the frozen fit. Either fit can go to native qualification (codex H275); the frozen one is simpler.
- Fit files on the drive: fit255.json (a) and fit255am.json (b, sha ae896f24…).

## Caveats
- These are offline replays. Engine-native confirmation on 24 S shards is required before any claim (INT8 drift and duplicate aggregation have overstated gains before).
- Libri-Light overlap was not removed from the TRAIN pairs.
- am_c32 compute cost was not measured.

Bulky data: /mnt/xbabe3-external/jsr/wer-push/H276/.
