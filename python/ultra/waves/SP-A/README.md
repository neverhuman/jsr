INTERIM, 2026-11-20. Screen complete. Paced fleet attempt was INVALID_INSTRUMENT (host contention, load 49). Confirmation not yet run.

# SP-A: stream density, interim results

All numbers below are [H0-MEASURED, shared host]. The model is in-corpus: it was trained on LibriSpeech. The scorer is the provisional Python WER; `ultra-eval` re-scores before any claim.

## Throughput screen
- **Conditions:** unpaced, train-clean-100 first 400 utterances (1.45 h), 1 pinned core, 1 intra-op thread.
- **Column:** "streams/core @100%" = audio-s per CPU-s, an upper bound on paced density.

| Precision | Decode | N | CPU-s / audio-s | Streams/core @100% | WER on train subset (in-training, not a quality number) |
|---|---|---:|---:|---:|---:|
| int8 | single | 1 | 0.0794 | 12.6 | 0.56 |
| int8 | single | 32 | 0.0847 | 11.8 | 0.56 |
| int8 | batch | 8 | 0.0753 | 13.3 | 0.58 |
| int8 | batch | 32 | 0.0624 | 16.0 | 0.56 |
| int8 | batch | 64 | 0.0545 | 18.4 | 0.58 |
| int8 | batch | 128 | 0.0532 | 18.8 | 0.56 |
| fp32 | single | 1 | 0.1198 | 8.3 | 0.58 |
| fp32 | batch | 32 | 0.0794 | 12.6 | 0.58 |

- **Smoke on test-clean** (50 utterances, int8, single): WER 2.51%. (Before tail padding was added it was 8.09%, caused by truncated final words.)

## Reading
- The batching lift saturates at about **1.5×** (N ≥ 64). The pre-registered primary needs ≥ 3×, so it is on track for **NO_LIFT** unless a paced run behaves differently.
- On this runtime, cost per stream is dominated by per-frame compute, not by re-fetching weights. This agrees with W01's c ≈ 14–17 ps/MAC on Zen 2 and the model's roughly 70 MB int8 image.
- For context, Nemotron 0.6B Q8_0 in its own runtime (nemotron lane smoke, 20 utterances): 0.84 CPU-s/s per stream, unbatched. Its training data differ, so this is not a like-for-like quality comparison.

## Next
- Paced fleet on DEV, inside the lead core block 32–47, under the lease.
- Then the one-time confirmation on test-clean and test-other.
