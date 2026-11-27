# H297: H261 refit on gated wide-beam TRAIN lists (H294 setting)

Watcher request (08:07): codex H294 re-decodes low-margin utterances (TRAIN-30% gate on the B16 margin) at B48 and applies exact H261, for 4.4582 DEV S. Refit H261 (including the t2 weight) on gated B48/B32 TRAIN lists, so the reranker matches the wider candidate set. Target < 4.43.

## Setup
- TRAIN: the H294 gate (margin < 0.6467037) over the H286 F3 c32 B16 TRAIN lists gives 2,759 of 9,196 MLS utterances. They were re-decoded with F3 INT8 at B48 and B32 on xbabe3 (`nbest_driver.py`, BEAM env).
- Features and reranker as in H261/H286, generalised to `NB` candidates. The fit uses TRAIN-only speaker 2-fold CV for λ.
- DEV: codex H294 `mixed/b48-train30` and `mixed/b32-train30` lists. The refit is applied only to gated rows (`routes.json["30"]`, 506 rows); ungated rows use frozen H261 (`tools/apply_gated.py`).

## Results (DEV S, pinned ultra-score)
| arm | CV λ (held-out Δ vs init) | DEV S | vs frozen offline | vs codex native |
|---|---|---|---|---|
| frozen H261, B48 offline | – | 4.47 | – | +0.014 [+0.000, +0.029] (2765/2782 identical) |
| refit B48 (4e9bd5a9…) | 1e-4 (−79) | 4.47 | +0.000 [−0.013, +0.013] | +0.014 [−0.004, +0.032] |
| refit B32 (5c3f7579…) | 1e-4 (−69) | 4.49 | +0.002 [−0.015, +0.019] | +0.006 [−0.010, +0.022] |

- The refit weights stay close to frozen (t2 0.88 vs 0.83).
- Rare-word recovery, H294 b48 vs F3: rare content substitutions fixed 19 / broken 10. Rare substitutions whose ref word is somewhere in the list rose from 170 to 205, but the refit selects no more of them.
- **Verdict: null.** The frozen H261 already transfers to the wide candidate sets.

Bulky data: `/mnt/xbabe3-external/jsr/wer-push/H297/`.
