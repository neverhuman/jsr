# Paired Isosyll v3.0.0 vs Zipformer INT8 on the 730-utterance DEV panel

- Scorer: `ultra-score` sha256 d784f8ca65a0… (`/mnt/xbabe3-external/jsr/h45-wins/bin/ultra-score`), `--seed 20261122 --replicates 10000 --alpha 0.05` (speaker-cluster bootstrap).
- system A = `zipformer-int8.panel730.score-input.jsonl` (eval-lane `r0-fulldev/R0-int8.panel730.score-input.jsonl`).
- system B = `v300-{live,final}.panel730.score-input.jsonl`: exact-isa `score/x6-final-commit.{live,final}.score-input.jsonl` (byte-identical
  transcripts on Zen 2 AVX2 / Xeon AVX2 / AVX-512 VNNI / M3 NEON), utterance ids mapped by panel position (u00000… → LibriSpeech ids);
  ref, speaker, dataset and duration asserted equal on all 730 rows.
