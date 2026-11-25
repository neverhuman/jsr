# H66. Light noise augmentation and a longer clean tail

**Status:** Closed at seed 7919. The clean result triggered the registered stop.

H66 tested whether lighter augmentation and a longer clean tail would recover
clean WER while retaining the held-out noise gain. The recipe kept the
M1-v5 setup except for the preregistered changes. Seed 7919 ran in four-rank
DDP on x5 and x7. Steps 0–15,000 used TRAIN-only MUSAN augmentation
(probability 0.15), gain augmentation (0.10) and EMA decay 0.9997994.
Steps 15,000–19,500 disabled both augmentations, changed EMA decay to
0.9994 and retained the cosine schedule. A3 scored on H0, with terminal EMA
primary and raw weights reported.

Both step-15,000 checkpoints matched their node checksum markers and contained
the optimizer, scheduler, sampler position, EMA and RNG state. Training
resumed at the saved sampler position with the cosine learning rate continuing
across the boundary. Both nodes reached step 19,500. All four ranks exited
with code zero in each phase.

## Registered decision

The DEV selection-half (S) c4 screen requires a clean point delta of
≤ +0.02 pp against signed RC2, and one-sided 95% 5 dB MUSAN-TEST uppers
below zero against both matched PyTorch RC2 and M1-v4. A pass advances
to sequential seeds 7920 and 7921. All three seeds must clear the pooled
clean upper of ≤ 0 pp and both noise uppers below zero. The confirmation
protocol and finalist must be frozen before any DEV confirmation-half decode.

## Results

Terminal EMA c4 on the DEV selection half recorded 3,682 edits over 52,196
reference words (7.0542% WER). Signed RC2 recorded 3,626 edits (6.9469%).
The pooled delta of **+0.1073 pp** exceeded the **+0.02 pp** screen limit.
Raw weights recorded 3,677 edits (7.0446%, +0.0977 pp against signed RC2).
The signed `ultra-score` independently reproduced the counts from the H0
score inputs. The terminal x5 checkpoint scored here has SHA-256
`cf3ba2ccc5dd039ccd21875680e73ed1364e9a5acf0bab067e362b20b4f0477a`.

The report-only pre-tail EMA trajectory used the same selection-half c4
workload.

| Step | Clean WER | 5 dB MUSAN-TEST WER |
|---|---:|---:|
| 10,000 | 6.8971% | 14.9322% |
| 15,000 | 7.0312% | 14.4513% |

Terminal EMA scored 14.4379% WER (7,536 edits) and raw scored 14.4590%
(7,547 edits) on 5 dB MUSAN-TEST. With 10,000 paired speaker-cluster
replicates (seed 20261122), terminal EMA changed WER by **−10.3131 pp**
against matched PyTorch RC2 (24.7509% WER, one-sided 95% upper −9.1486 pp)
and **−11.0660 pp** against terminal M1-v4 EMA (25.5039% WER, upper
−9.8406 pp). Both noise legs cleared their zero upper limit. The clean
miss kept the [signed v3.2.0 package](../../docs/ultra/releases/v3.2.0.md)
as the release for this cycle.

The [evidence checksums](../v3.2.0-closeout.SHA256SUMS) identify the preregistration,
launch configuration, both node-local terminal states, trajectory receipts,
the independent clean screen and paired terminal noise replay.
