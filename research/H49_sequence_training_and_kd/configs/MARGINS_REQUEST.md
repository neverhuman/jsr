# H49 margins requested from eval-lane (register before any H49 decode exists)

Same format as amendment 23's entries. dev_split_v2 confirmation half (h43/dev_split_v2.json sha256 002ac7ae..., 2,785 utterances, 37 speakers), EMA, pooled, canonical ultra-score, seed pair vs A1C + A1C-r (both pairings, speakers kept).

| endpoint | margin_pp | kind | note |
|---|---:|---|---|
| h49a-mwer-pair-vs-A1Cpair-superiority-ema-pooled-conf-c4 | 0.0 | superiority | PASS iff one-sided 95% upper < 0 (Holm over the H49 family of at most 2) AND the repo rule delta <= -0.174 pp (2 x the 4-run spread 0.087) for the pair and for each seed's mean vs A1C's pair; FAIL iff point >= 0; else INCONCLUSIVE |
| h49a-mwer-pair-vs-A1Cpair-ni-ema-pooled-conf-c8 | 0.094 | non-inferiority | guard: PASS iff upper <= margin |
| h49a-mwer-pair-vs-A1Cpair-ni-ema-pooled-conf-c16 | 0.36 | non-inferiority | guard: PASS iff upper <= margin |
| h49b-seqkd-pair-vs-A1Cpair-superiority-ema-pooled-conf-c4 | 0.0 | superiority | as h49a |
| h49b-seqkd-pair-vs-A1Cpair-ni-ema-pooled-conf-c8 | 0.094 | non-inferiority | guard |
| h49b-seqkd-pair-vs-A1Cpair-ni-ema-pooled-conf-c16 | 0.36 | non-inferiority | guard |

Selection readings (not decisions, no margin): step-2,000 EMA chunk-4 WER of the two lambda arms on the dev_split_v2 SELECTION half; the lower one continues. Stop if both are worse than A1C's step-2,000 snapshot by more than 0.087 pp on the selection half.
