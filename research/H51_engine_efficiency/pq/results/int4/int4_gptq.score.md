normaliser `ultra-en-wer-norm/v1` (`988c59acad87`), weights `01dbcb5f901d`

| system | dataset | WER % | S | D | I | ref words | utts | silence ins/h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | dev-clean | 4.26 | 285 | 29 | 48 | 8503 | 400 | n/a |
| A | dev-other | 11.86 | 565 | 65 | 75 | 5945 | 330 | n/a |
| A | **macro** | 8.06 | | | | | | |
| A | **pooled** | 7.39 | | | | | | |
| B | dev-clean | 4.90 | 330 | 36 | 51 | 8503 | 400 | n/a |
| B | dev-other | 12.25 | 584 | 65 | 79 | 5945 | 330 | n/a |
| B | **macro** | 8.57 | | | | | | |
| B | **pooled** | 7.92 | | | | | | |

Delta macro WER (B - A): +0.517 points (+6.4% relative), one-sided 95% bounds [+0.298, +0.739], 10000 replicates, seed 20261122
