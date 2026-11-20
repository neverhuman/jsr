# Evaluation slices (ULTRA-Bench v1, plan C4). Frozen 2026-11-20 on DEV only

| Slice | Rule | Frozen on |
|---|---|---|
| long_form | audio > 20.0 s | fixed rule; DEV has 102 / 5,567 such utterances |
| fast_speech | reference words per second ≥ **3.5682** | p90 of DEV (dev-clean + dev-other, n = 5,567) |
| test-other | the LibriSpeech "other" test split | split definition |
| silence probe | `/mnt/xbabe3-external/jsr/corpora/probes/silence_noise_10min.wav`: 5 min digital silence, then 5 min Gaussian noise at about −50 dBFS, seed 20261120, sha256 `8e43130d3f52446f000a3d76659e2ee53accd30d4e6cf9d0928ff7c5439dd666` | deterministic generator `ultra.slices.silence_probe` |

**Scoring:**
- **WER per slice:** every system scores each slice on its emitted output, rescored by `ultra-score`.
- **Silence probe:** inserted words per hour and false-transcript episodes per hour, never WER.

**Rule:** thresholds are never refitted on test data. A change needs a new frozen version and a DECISIONS entry.
