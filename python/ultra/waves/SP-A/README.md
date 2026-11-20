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

## Mechanism update (runtime lane, Rust shared-weight runner @ 5232d61, exploratory, cores 32–47)
The previous section's DRAM explanation is superseded. The limit is **L3 per CCX**. A Zen 2 CCX is 4 cores sharing a 16 MB L3.

**Evidence:**
- **One worker per CCX scales linearly:** 16.13 streams/core on 1 core, 16.39 per worker-core with 4 workers on cores 32/36/40/44.
- **Paced, one worker per CCX:** 4 × 12 = 48 streams sustained at p99 616–655 ms. Lateness is flat and RSS is 1.18 GiB.
- **Dense packing flattens at about 74 streams in total,** at both 8 cores (2 CCXs) and 16 cores (4 CCXs), even with a single shared weight image.

**Implication:** with this runtime, H0's ceiling is about (16 CCXs) × 12 ≈ 190 live streams. That is a prediction until it is measured across all 16 CCXs in a qualifying window.

**Primary ratio (batch vs single, unpaced, 1 core):** 18.8 / 12.6 = 1.49×, against the pre-registered ≥ 3×. The expected verdict is **NO_LIFT** on the primary, to be confirmed once on the test sets in the qualifying window.

## Instrument caveat (2026-11-20 19:30Z, codex-review)
The Rust runner's sherpa paths derived observation time from accepted samples, and marked samples processed without a decode. The Rust lateness and p99 figures above (one worker per CCX: 616–655 ms) are therefore **INVALID_INSTRUMENT for latency** until runtime fixes land. CPU totals and WER from those runs are unaffected. The Python-harness lateness numbers use the real monotonic clock and are not affected.
