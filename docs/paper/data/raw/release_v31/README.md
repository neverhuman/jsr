# RC2 (v3.1.0 candidate) confirmatory-look receipts

RC2 is a measured candidate, not a release: it missed registered gates. See `docs/ultra/releases/v3.1.0-rc2.md`.

Every receipt here is kept byte-exact. The score inputs it names under `/dev/shm/rc2_v31/` were later moved to the
durable archive `release/v3.1.0/confirmatory_look/` (34 entries, SHA256SUMS `4cb3ccac…`), which maps as follows:

| Path in the receipts | Durable path |
| --- | --- |
| `/dev/shm/rc2_v31/evals/<mode>.<split>.score-input.jsonl` | `confirmatory_look/inputs/modes/` (panel730 and C-half copies are in `../versus/v310/`) |
| `/dev/shm/rc2_v31/meta/R0-int8.panel730.score-input.jsonl` | `confirmatory_look/inputs/baselines/zipformer_int8.panel730.score-input.jsonl` |
| `/dev/shm/rc2_v31/scores/baselines/v300_*.panel730…` | the **historical B0** checkpoint, mislabelled v3.0.0 at freeze (deviation; see the release record) |

| File | What it is |
| --- | --- |
| `rc2-panel-holm.json` | the Holm family as frozen, against the historical B0 baseline. Kept as history; not a v3.0.0 comparison |
| `rc2-panel-holm-shipped-v300.json` | the same RC2 outputs paired with the shipped v3.0.0 transcripts (`../versus/v300/`); the record of the result |
| `rc2-<mode>.fulldev.score.json`, `zipformer_int8.fulldev.score.json` | full-DEV reports (descriptive: full DEV contains the selection half) |
| `rc2-accuracy.json` | the accuracy matrix (codex) |
| `policies.json`, `rc2-bundle.SHA256SUMS` | frozen decode policies and the RC2 bundle manifest |
| `codex-receipts.SHA256SUMS` | codex's manifest of the paired and full-DEV reports |

The scorer is the pinned `ultra-score` (sha256 `d784f8ca…`), with seed 20261122 and 10,000 speaker-cluster replicates. Bounds are one-sided 95%.
