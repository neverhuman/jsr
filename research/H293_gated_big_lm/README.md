# H293: confidence-gated 27.5M L2R+R2L rescoring on F3 + frozen H261

Watcher request (07:48): run the 27.5M pair (H68 l2r_cur/r2l_cur .utfr) only on low-confidence F3 c32 B16 utterances, on top of frozen H261. Gate threshold chosen on TRAIN. Target DEV < 4.45 at ≤ +11% commit / +5% process cost. Report the curve at gate fractions 10/20/30/50%.

## Method (`tools/gate.py`)
- Candidate score = frozen H261 (H261/fitTT.json) on routed rows (first-pass gap < 3.75), as in H275.
- Gate = margin of that score between the best two unique texts (duplicates merged by logaddexp).
- TRAIN thresholds are quantiles of the margin over routed rows of the F3 TRAIN lists (H286, 9,196 utts).
- Gated rows get + a·L2R27 + b·R2L27 + c·nw. The grid is a, b ∈ {0, 0.1, …, 1}, c ∈ {−1, −0.5, 0, 0.5, 1}; the full-TRAIN best is used, and speaker 2-fold held-out Δ is reported.
- 27M features come from `tf_rescore_nbest` (INT8 engine) on xbabe0.

## Results (DEV S, pinned ultra-score)
| gate (TRAIN quantile) | gated share of all S utts | weights a/b/c | TRAIN held-out Δ edits | DEV S | vs F3 native 4.5272 | est. extra commit |
|---|---|---|---|---|---|---|
| 10% | 6.2% | 0.1/0.1/0.5 | −29 | **4.51** | −0.021 [−0.047, +0.006] | +4–8% |
| 20% | 11.6% | 0/0.2/1.0 | −51 | 4.53 | −0.000 [−0.043, +0.045] | +7–14% |
| 30% | 17.9% | 0/0.2/1.0 | −59 | 4.54 | +0.017 [−0.033, +0.070] | +11–22% |
| 50% | 29.6% | 0/0.2/1.0 | −68 | 4.55 | +0.018 [−0.030, +0.069] | +18–36% |

- **Cost bench:** one pinned xbabe0 core, 174 S utts × 16 cands, host under load from other lanes. t2 11.7M L2R 64–85 ms/utt; 27M L2R 131–156; 27M R2L 516–698. The pair costs about 10× t2.
- **Extra-cost estimate:** gated share × 61–123%. The low end is the H68 qual2 full-pair ratio, the high end this bench. Process cost was not metered.
- **Verdict: null.** Only the 10% gate fits the budget, and it gains −0.02 n.s. On top of t2 + H261 the 27M pair resolves very little; its −0.65 in H68 was against no LM.

Bulky data: `/mnt/xbabe3-external/jsr/wer-push/H293/` (gate.json, S_gate_f*.score-input.jsonl) and `H286/{S,TT}/x_big_*.npy`.
