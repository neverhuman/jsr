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

## Canonical word alignment ledger

```sh
ultra-score A.jsonl B.jsonl --alignment-ledger alignments.jsonl \
  --receipt-a A.identity.json --receipt-b B.identity.json > score.json
```

Use one score input for a single system or two for a paired comparison. The
ledger path must be new. Each run identity JSON includes
`audio_manifest_sha256`, `artifact_sha256`, `runtime_sha256`, `decoder`, and
`chunk_policy`. The ledger records the score-input digest, scorer executable
digest, normalizer identity, and canonical tie order. Legacy score inputs with
no run identity have a null `run` entry.

Every canonical alignment operation includes zero-based normalized word indexes
with the utterance, split, speaker, and available evidence. Join paired rows by
`id` and `reference.index`. For insertions, `reference_boundary` counts the
reference words consumed before the insertion and identifies leading and trailing
positions. The paired transition totals cover all reference positions. The
alternate tie order supplies per-split S/D/I sensitivity.

Supply optional `ref_words` and `hyp_words` arrays indexed by normalized word.
Each entry can carry `time_s` and `token_span` as two-element half-open intervals,
plus `train_frequency`, `function_word`, `phones`, and `morphology`. Timing
requires a `timing_clock`. To use the pinned reference timing rows, assign their
`words` array to `reference_timing` on the matching score record. Normalization
expansions share the original time span and the expanded sequence must match the
canonical reference. [Reference timing source](data/word_timing.md)

Supply `condition` and `session` for continuous replay. Slice rows pool S/D/I,
matches, and word counts by condition, speaker, duration, word position, and
available lexical annotations. Use frequencies from the canonical TRAIN cohort.
Confusion buckets count annotated substitutions. For `candidates` in decoder
rank order, starting with the selected hypothesis, the ledger records the
earliest rank with an exact aligned match at each reference position. A null
rank with a populated candidate count means the word was absent at that position
in the supplied candidates.

For candidate-risk training, keep `ultra-score --candidate-edits` open on
stdin/stdout. Each JSONL request carries `ref` and a `hyps` array. The scorer
flushes one JSON array of canonical edit counts in candidate order per request.
