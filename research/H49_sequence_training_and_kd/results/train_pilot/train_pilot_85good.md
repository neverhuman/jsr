# H49 TRAIN pilot oracle headroom

Utterances: 8026. Canonical words (ultra-en-wer-norm/v1, rule table 988c59ac...).
Sample: {"speakers": 153, "utts_per_speaker_p50_max": [50, 100], "seconds_p10_p50_p90": [6.08, 13.76, 15.68], "hours": 27.79, "by_split": {"train-clean-100": 2492, "train-clean-360": 5534}}
Spread unit: 0.087 pp. Oracles exclude the reference.

| system | WER all | clean | other | headroom pp (all) | spreads |
|---|---:|---:|---:|---:|---:|
| student_1best | 0.239 | 0.239 |  | 0.0 | 0.0 |
| r0_oracle | 0.665 | 0.665 |  | -0.426 | -4.9 |
| r0_top | 1.554 | 1.554 |  | -1.315 | -15.12 |
| student_b8_oracle | 0.065 | 0.065 |  | 0.174 | 2.0 |
| student_b8_top | 0.239 | 0.239 |  | 0.0 | 0.0 |
| teacher_pick_union | 1.466 | 1.466 |  | -1.227 | -14.1 |
| teacher_pick_union12 | 1.465 | 1.465 |  | -1.226 | -14.1 |
| union12_oracle | 0.054 | 0.054 |  | 0.185 | 2.13 |
| union_oracle | 0.052 | 0.052 |  | 0.187 | 2.15 |

Headroom by category (pp of pooled WER, student 1-best errors minus the selection's):

| system | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|
| student_1best | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| r0_oracle | -0.173 | -0.056 | -0.005 | -0.067 | -0.084 | -0.04 |
| r0_top | -0.318 | -0.124 | -0.062 | -0.268 | -0.439 | -0.105 |
| student_b8_oracle | 0.018 | 0.008 | 0.007 | 0.063 | 0.063 | 0.015 |
| student_b8_top | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| teacher_pick_union | -0.277 | -0.125 | -0.062 | -0.272 | -0.387 | -0.103 |
| teacher_pick_union12 | -0.278 | -0.125 | -0.062 | -0.272 | -0.387 | -0.103 |
| union12_oracle | 0.022 | 0.01 | 0.007 | 0.065 | 0.065 | 0.015 |
| union_oracle | 0.022 | 0.01 | 0.007 | 0.066 | 0.067 | 0.016 |

Recall of the student 1-best's S / D positions (some candidate has the reference word there):

| set | all | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 0.522 | 0.4634 | 0.6552 | 0.5217 | 0.4973 | 0.5911 | 0.3768 |
| student_b8 | 0.7366 | 0.6098 | 0.6897 | 0.8696 | 0.7433 | 0.8044 | 0.6232 |
| union | 0.7951 | 0.7439 | 0.7931 | 0.9565 | 0.7754 | 0.8533 | 0.6667 |
| union12 | 0.7854 | 0.7439 | 0.7931 | 0.9565 | 0.7701 | 0.8356 | 0.6522 |

H49b training selection (top-3 non-reference strings by R0 full-sum): {"retained_alts_per_utt": 3.0, "student_only_share": 0.1436, "r0_only_share": 0.6294}

Teacher ranks the reference first: 0.5189

H49b stop rule (pre-registered): {"H_T_pp": 0.013, "H_T_spreads": 0.15, "H_T_ge_one_spread": false, "convention_share_of_H_T": 0.027, "convention_share_below_half": true, "teacher_pick_vs_1best_pp": 1.227, "teacher_pick_not_worse": false, "teacher_pick_insertions_not_up": false, "teacher_pick_rare_not_up": false, "verdict": "STOP"}
