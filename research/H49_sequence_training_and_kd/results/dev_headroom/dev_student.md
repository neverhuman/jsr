# H49 DEV oracle headroom (analysis only)

Utterances: 5567. Canonical words (ultra-en-wer-norm/v1, rule table 988c59ac...).
Spread unit: 0.087 pp. Oracles exclude the reference.

| system | WER all | clean | other | headroom pp (all) | spreads |
|---|---:|---:|---:|---:|---:|
| student_1best | 8.302 | 4.48 | 12.373 | 0.0 | 0.0 |
| student_b16_oracle | 5.479 | 2.401 | 8.757 | 2.823 | 32.45 |
| student_b16_top | 8.212 | 4.474 | 12.192 | 0.091 | 1.04 |
| student_b4_oracle | 6.951 | 3.474 | 10.654 | 1.351 | 15.53 |
| student_b4_top | 8.223 | 4.487 | 12.201 | 0.079 | 0.91 |
| student_b8_oracle | 6.158 | 2.843 | 9.688 | 2.144 | 24.65 |
| student_b8_top | 8.218 | 4.474 | 12.203 | 0.085 | 0.98 |

Headroom by category (pp of pooled WER, student 1-best errors minus the selection's):

| system | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|
| student_1best | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| student_b16_oracle | 0.562 | 0.271 | 0.098 | 0.972 | 0.718 | 0.203 |
| student_b16_top | 0.025 | 0.026 | -0.005 | 0.023 | -0.001 | 0.022 |
| student_b4_oracle | 0.247 | 0.126 | 0.044 | 0.47 | 0.365 | 0.099 |
| student_b4_top | 0.013 | 0.03 | -0.004 | 0.017 | 0.005 | 0.018 |
| student_b8_oracle | 0.415 | 0.208 | 0.076 | 0.725 | 0.57 | 0.15 |
| student_b8_top | 0.026 | 0.024 | -0.005 | 0.027 | -0.008 | 0.02 |

Recall of the student 1-best's S / D positions (some candidate has the reference word there):

| set | all | acoustic | content | convention | function | near | rare |
|---|---:|---:|---:|---:|---:|---:|---:|
| student_b16 | 0.3859 | 0.2997 | 0.3774 | 0.5317 | 0.4642 | 0.5238 | 0.1803 |
| student_b4 | 0.1831 | 0.1359 | 0.1447 | 0.2244 | 0.2223 | 0.2614 | 0.0871 |
| student_b8 | 0.2909 | 0.2216 | 0.2516 | 0.3805 | 0.3477 | 0.4128 | 0.1333 |
