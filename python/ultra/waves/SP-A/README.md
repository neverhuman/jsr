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

## Fleet scaling (exploratory, 2026-11-20 ~15:30Z, lead block 32–47, load 20–35 mostly IO stall)

**Conditions:** unpaced; each worker on its own physical core, SMT sibling idle; 32 batched streams per worker; dev-clean.

| Concurrent workers | Streams per core @100% | Stream-equivalents in total |
|---:|---:|---:|
| 1 | 16.6 | 16.6 |
| 2 | 16.1 | 32 |
| 4 | 12.9 | 52 |
| 8 | 8.0 | 64 |

- **Not frequency:** a repeat with frequency sampled gave 4094 MHz for 1 worker (14.9 streams/core) and 4111 MHz for 8 workers (9.0 streams/core).
- **Likely cause:** contention in shared memory and L3. Each worker streams its own roughly 70 MB int8 weight image. This is the fixed-fetch term of the W01 cost law appearing at fleet scale. Hardware counters are unavailable (`perf_event_paranoid` = 4), so the bandwidth/L3 split is not confirmed.
- **Paced fleets fail to sustain at these densities:** 8, 12 and 16 per core with 8 single-core workers, and 8, 12 and 16 per core with 4 four-core workers. The four-thread workers were worse still (0.20–0.23 CPU-s/s): ORT intra-op threading does not help here.
- **Consequence for the SP-A claim:** about 1,000 real-time streams per H0 workstation is NOT supported for the ORT Zipformer. A density headline needs either a shared-weight multi-stream runtime (one weight image, many streams per process; the runtime lane's Rust batcher), or smaller and lower-bit weights (ternary, as the cost law predicts).
