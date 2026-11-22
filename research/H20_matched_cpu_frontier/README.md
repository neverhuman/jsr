# H20 — Common-boundary streaming ASR on CPU

**Status** EXPLORATORY, ongoing. The matched cold process-tree rotation is running. The separately qualified warm cost, paced density, and test WER await the benchmark freeze and registered receipts.

## Question

How do per-stream CPU cost and recognition accuracy change when native CPU recognizers decode the same 730-utterance DEV panel? The same-weight FP32 engine and ONNX Runtime arms isolate runtime effects, and the native FP32 and prepared INT8 arms isolate numerical mode.

## Setup

- The staged 16 kHz mono PCM workload contains 730 utterances and 5,259.5599375 audio seconds. Its ordered audio manifest is SHA-256 `010aa985e8c677d4daf43561dd9600d1ad27429494e88b74c177825a8cbc3963`.
- One recognizer runs at a time on H0 `xbabe3`, AMD Ryzen Threadripper PRO 3995WX, logical CPU 46 with SMT sibling 110 reserved. The observer uses CPU 127 in a separate L3 domain.
- The meter measures user and system CPU for the process tree from recognizer launch through final hypothesis flush and worker exit. The three repetitions rotate arm order. WER uses Rust-normalized edit counts on retained outputs after timing.
- The [evidence manifest](../../docs/paper/evidence.json) records exact builds, model artifacts, tokenizer digests, the workload, and decoder policy. The [benchmark reproduction instructions](../../docs/paper/cold-benchmark.md) cover staging, timing, and scoring.

## Verified progress

The completed DEV evidence at the selection cutoff favored the 50k/55k/60k dense D2 average, at 11.43% pooled WER with greedy decoding and 10.74% with beam 4. D2MR step 50k led the verified multirate greedy candidates at 15.86%. These are checkpoint-selection rows and carry their own transcript and configuration identities in the manifest.

The first completed cold repetition was `cold-moonshine-tiny`, the native Moonshine v0.1.6 quantized Tiny Streaming operating point. All 730 hypotheses were retained. With a 45,398,523-byte model pack, this repetition used 694.150745 process-tree CPU seconds and 203,542,528 bytes peak RSS. It measured 0.1320 CPU-s/audio-s and 9.90% pooled WER, with 6.54% dev-clean and 14.72% dev-other WER. The cost plot uses the median of all three repetitions for each arm.

## Evidence and decisions

Raw evidence for the first repetition is under `target/paper-evidence/cold/round-1/moonshine-tiny/` on H0.

| Artifact | SHA-256 |
|---|---|
| `measurement.json` | `8aac21bd8cfceceaac5f87a33a04a956ea344d080a8ffb0d828ff8af0d4e0679` |
| `score.json` | `89f530982b878dabb74479fb2b9fe3a4eb9a760560566e17df439f8f3c652737` |
| `scores.jsonl` | `fbc1c55162f5ac95472c37c84d0c3e7da0383fa60da79942f014e8a256f0524d` |

Native Moonshine Small Streaming and Nemotron 3.5 Q8 passed the DEV streaming smoke and are in the rotation. The latter uses the English 80 ms profile. Sealed test access and qualifying warm and paced experiments follow the [freeze protocol](../../docs/ultra/FREEZE.md).
