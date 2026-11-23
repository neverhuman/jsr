normaliser `ultra-en-wer-norm/v1` (`988c59acad87`), weights `01dbcb5f901d`

| system | dataset | WER % | S | D | I | ref words | utts | silence ins/h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | dev-clean | 4.26 | 285 | 29 | 48 | 8503 | 400 | n/a |
| A | dev-other | 11.86 | 565 | 65 | 75 | 5945 | 330 | n/a |
| A | **macro** | 8.06 | | | | | | |
| A | **pooled** | 7.39 | | | | | | |
| B | dev-clean | 4.94 | 337 | 33 | 50 | 8503 | 400 | n/a |
| B | dev-other | 12.35 | 583 | 70 | 81 | 5945 | 330 | n/a |
| B | **macro** | 8.64 | | | | | | |
| B | **pooled** | 7.99 | | | | | | |

Delta macro WER (B - A): +0.585 points (+7.3% relative), one-sided 95% bounds [+0.358, +0.818], 10000 replicates, seed 20261122
