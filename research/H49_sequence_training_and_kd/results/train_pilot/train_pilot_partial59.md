# H49 TRAIN pilot oracle headroom

Utterances: 5559. Canonical words (ultra-en-wer-norm/v1, rule table 988c59ac...).
Sample: {"speakers": 108, "utts_per_speaker_p50_max": [50, 100], "seconds_p10_p50_p90": [6.08, 13.76, 15.68], "hours": 19.38, "by_split": {"train-clean-100": 2492, "train-clean-360": 3067}}
Spread unit: 0.087 pp. Oracles exclude the reference.

| system | WER all | clean | other | headroom pp (all) | spreads |
|---|---:|---:|---:|---:|---:|
| student_1best | 0.234 | 0.234 |  | 0.0 | 0.0 |
| r0_oracle | 0.499 | 0.499 |  | -0.265 | -3.05 |
| r0_top | 1.378 | 1.378 |  | -1.144 | -13.15 |
| student_b8_oracle | 0.061 | 0.061 |  | 0.173 | 1.99 |
| student_b8_top | 0.234 | 0.234 |  | 0.0 | 0.0 |
| teacher_pick_union | 1.298 | 1.298 |  | -1.064 | -12.23 |
| teacher_pick_union12 | 1.297 | 1.297 |  | -1.063 | -12.22 |
| union12_oracle | 0.051 | 0.051 |  | 0.183 | 2.1 |
| union_oracle | 0.048 | 0.048 |  | 0.186 | 2.13 |

Headroom by category (pp of pooled WER, student 1-best errors minus the selection's):

| system | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|
| student_1best | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| r0_oracle | -0.095 | -0.031 | -0.006 | -0.015 | -0.082 | -0.036 |
| r0_top | -0.233 | -0.102 | -0.06 | -0.21 | -0.442 | -0.097 |
| student_b8_oracle | 0.019 | 0.007 | 0.007 | 0.061 | 0.065 | 0.014 |
| student_b8_top | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| teacher_pick_union | -0.191 | -0.104 | -0.061 | -0.216 | -0.397 | -0.096 |
| teacher_pick_union12 | -0.191 | -0.104 | -0.061 | -0.215 | -0.396 | -0.096 |
| union12_oracle | 0.024 | 0.009 | 0.007 | 0.063 | 0.066 | 0.014 |
| union_oracle | 0.024 | 0.009 | 0.007 | 0.064 | 0.068 | 0.014 |

Recall of the student 1-best's S / D positions (some candidate has the reference word there):

| set | all | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 0.5392 | 0.4828 | 0.7778 | 0.6 | 0.5039 | 0.6115 | 0.3409 |
| student_b8 | 0.753 | 0.6379 | 0.7222 | 0.8667 | 0.7519 | 0.8217 | 0.6364 |
| union | 0.8124 | 0.7931 | 0.7778 | 1.0 | 0.7907 | 0.8662 | 0.6591 |
| union12 | 0.8005 | 0.7931 | 0.7778 | 1.0 | 0.7829 | 0.8471 | 0.6364 |

H49b training selection (top-3 non-reference strings by R0 full-sum): {"retained_alts_per_utt": 3.0, "student_only_share": 0.1413, "r0_only_share": 0.6344}

Teacher ranks the reference first: 0.5206

H49b stop rule (pre-registered): {"H_T_pp": 0.013, "H_T_spreads": 0.15, "H_T_ge_one_spread": false, "convention_share_of_H_T": 0.04, "convention_share_below_half": true, "teacher_pick_vs_1best_pp": 1.064, "teacher_pick_not_worse": false, "teacher_pick_insertions_not_up": false, "teacher_pick_rare_not_up": false, "verdict": "STOP"}
