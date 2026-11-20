# Nemotron Speech Streaming EN 0.6B (Q8_0, NeMo-Speech.cpp, CPU): P04 results

**Status: EXPLORATORY. These are not qualifying results.**

**Why exploratory:**
- Timing ran in the nemotron per-block lease (cores 16–31), not under `BENCH_LEASE.host`.
- CPU and RSS come from the Python `os.wait4` instrument, not the Rust whole-service `ultra-measure`.
- No paced latency, CQ or FC was measured.
- WER was scored by Rust `ultra-score` (`ultra-en-wer-norm/v1`). The Python `wer.py` figure is shown for continuity.

**Labels:**
- **Partition:** PUBLIC-TEST (test-clean, test-other). DEV (dev-clean) was used only for the batching screen.
- **Honesty label: unknown overlap.** The model card lists *LibriSpeech 960 h* in training, so LibriSpeech train is in-corpus. The card also lists about 530k h of other corpora (Granary/YTC/YODAS2/LibriLight, MLS-EN, People's Speech and others). Their overlap with the LibriSpeech test speakers or texts is unverified.

**Approval:** owner ruling O2, internal evaluation only.

**Candidate record:** [`nemotron-0.6b-q8.yaml`](nemotron-0.6b-q8.yaml) (`nemotron_en_q8_live_160`, `nemotron_en_q8_live_560`).

## Identities

| Item | Value |
|---|---|
| Model | HF `nvidia/nemotron-speech-streaming-en-0.6b` @ `ebe59e5a817142986528bbbee5dba8db7b38ed50`, `nemotron-speech-streaming-en-0.6b.q8_0.gguf` |
| Model sha256 | `d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d` (equals the HF LFS oid) |
| Tokenizer sha256 | `tokenizer.json` `60dc0361763fa3cd62df60f34fca3e61134676a939967853931db5f3869b2db2` |
| Runtime | `NVIDIA/NeMo-Speech.cpp` @ `e790e463b2913cff9f989a3df224c587f14b4c5c` (nemo-speech 0.3.0, ggml 0.25.1). llama.cpp submodule `bd4f514d…` plus 42 upstream patches. |
| Build | GCC 13.3.0, preset `cpu-asr`, Release `-O3`. `GGML_NATIVE=OFF`, AVX/AVX2/FMA/F16C/BMI2, AVX-512 OFF (objdump finds 0 zmm instructions), CUDA OFF. |
| Effective backend | runtime log: `head=rnnt backend=CPU`. `nvidia-smi` showed no compute context for any run. |
| Geometry (runtime log) | 160 ms: `left=70 center=1 right=1`. 560 ms: `left=70 center=1 right=6`. Encoder frames are 80 ms. |
| Invocation | upstream `nemo-speech bench asr --mode stream --device cpu --asr.backend.threads 1 --asr.streaming.rnnt_right_context R --concurrency C`, under `taskset -c <core>` |
| Adapter | [`python/ultra/adapters/nemotron_adapter.py`](../../../python/ultra/adapters/nemotron_adapter.py). It receives anonymised audio only (`u%05d.wav`) and never sees references. |
| Host | H0 (Threadripper PRO 3995WX, Zen 2, AVX2 without AVX-512). Shared host; the GPU is busy with other jobs and was not used. |

**End-of-stream flush.** `finish()` appends zero mel frames (one chunk plus one shift). The last real frame then sees its full right context, and the RNNT predictor commits pending tokens. Word times are bounded to real audio.

**Truncation check.** Deletions are 10% of errors (139 of 1404 at 160 ms). I inspected the utterances whose last reference word is missing from the hypothesis tail, as a sample from shard 3/4 at 560 ms. All of them were substitutions (proper nouns, compounds), not truncations.

## 1. Quality (WER)

Scores are Rust `ultra-score`, normaliser `ultra-en-wer-norm/v1` (`988c59acad87`). The Python column is the provisional `wer.py` scorer (`corpus_wer`). Output is cased and punctuated; both normalisers lowercase it and strip punctuation.

| Split | Chunk | Utts | Ref words | **WER (ultra-score)** | S / D / I | WER (wer.py) |
|---|---|---:|---:|---:|---|---:|
| test-clean | 160 ms | 2620 | 52,920 | **2.65%** | 1123 / 139 / 142 | 2.737% |
| test-clean | 560 ms | 2620 | 52,920 | **2.58%** | 1087 / 145 / 134 | 2.665% |
| test-other | 160 ms | 2939 | 52,782 | **5.60%** | 2336 / 303 / 317 | 5.707% |
| test-other | 560 ms | 2939 | 52,782 | **5.18%** | 2196 / 259 / 280 | 5.279% |

**Paired differences (560 − 160 ms):** one-sided 95% speaker-cluster bootstrap bounds, 10,000 replicates, seed 20260101.

| Split | Δ WER | Relative | Bound | Clusters |
|---|---:|---:|---|---:|
| test-clean | −0.072 pt | −2.7% | [−0.144, 0.000] | 40 |
| test-other | −0.419 pt | −7.5% | [−0.581, −0.273] | 33 |

The longer right context (480 ms) helps most on harder speech.

**Model card, for reference only:** 2.32% (test-clean) and 4.84% (test-other) at the **1120 ms** chunk, with NeMo's own normalisation. This is not the same evaluation.

## 2. CPU cost, single stream, one thread, one pinned core

| Run | Chunk | Core | Load (1 min) before → after | CPU-s / audio-s | Streams/core at full load | Wall RTF | Peak RSS | Compute per chunk p50 / p95 / p99 |
|---|---|---|---|---:|---:|---:|---:|---|
| **quiet headline** | 160 ms | 22 | 14.9 → 4.7 | **0.709** | **1.41** | 0.710 | 1052 MiB | **104.8 / 138.0 / 151.8 ms** |
| **quiet headline** | 560 ms | 28 | 14.9 → 14.2 | **0.451** | **2.22** | 0.453 | 1055 MiB | **218.2 / 265.9 / 276.5 ms** |

Notes on the quiet headline:
- **Input:** every 8th test-clean utterance (328 utts, 2464.6 s, input-manifest sha256 `518c90c3b387…`). WER on this subset: 2.48% (160) and 2.57% (560), by `wer.py`.
- **What CPU covers:** process-tree user+sys of the `nemo-speech` process from `wait4`. That includes model load (27 ms) and engine warm-up (2.3–3.2 s CPU, under 0.2% of the total).
- **Threads:** 4 observed in the process. All are inside the one-core affinity mask, with the SMT sibling idle.
- **Per-chunk compute** is the runtime's own client-side `chunk_latency_ms`: push plus drain for each chunk, measured by `nemo-speech bench`. At 160 ms, p99 is 152 ms. One stream therefore fits in real time on one Zen 2 core at 1 thread, with only about 5% headroom at p99.
- **Throughput explained:** one 0.6B Q8_0 weight pass (about 700 MB) runs per chunk per stream. That makes the cost dominated by weight streaming. 560 ms chunks amortise it over 3.5× more audio.
- **Real-time only:** a single stream needs only 1/1.41 = 71% (160 ms) or 45% (560 ms) of a core. The runtime's README figure (27 ms per 160 ms chunk, "6× real time" on CPU) uses multiple threads on a different CPU, so it is not comparable.

**Contended runs (not headline):** these are the full-test-clean WER shards. Each was 1 thread on its own core, with 8–12 of our processes running at once and host load 31–49 (trainer benches were then unpinned).

| Chunk | CPU-s / audio-s | Chunk compute mean / p99 (max over shards) |
|---|---:|---|
| 160 ms | 1.43 | 215 / 644 ms |
| 560 ms | 0.74 | 344 / 895 ms |

test-other shards ran 8 of our processes at once on cores 16–30, at host load 6–18. They gave 1.18 (160 ms) and 0.68 (560 ms) CPU-s/s. Those runs measured WER, so treat that CPU figure as self-contended as well.

Contention roughly doubled per-chunk cost, consistent with a DRAM-bandwidth-bound weight stream. An earlier headline attempt that started at load 34 gave 1.27 and 0.85. It is recorded but superseded.

## 3. Batching: several streams per weight pass

**Capability: yes.** NeMo-Speech.cpp has bounded dynamic microbatching (`asr.batching.*`). It is backend-generic and works on CPU.
- **Cache-aware encoder:** concurrent streams are combined into one graph when `mel_frames`, `mask_len` and `tail_dim` match. The mask length is fixed (left 70 plus chunk), so after each stream's first chunk every stream is compatible.
- **RNNT predictor and joint:** these have their own work-conserving batchers.
- **When it engages:** `nemo-speech bench` turns batching on and sizes it whenever `--concurrency > 1`.
- **Threading:** backend compute is serialised. Throughput comes from fewer, wider graphs, not parallel graphs.

**Unpaced screen.**

Setup:
- **Data:** DEV, dev-clean first 128 utts (749 s, mean 5.9 s).
- **Threads and cores:** 1 thread on 1 core per process, cores 20/24/28. Three processes ran at once, one per concurrency level.
- **Load:** 13–21.
- **Not done:** no paced verification.

| Chunk | Streams (C) | CPU-s / audio-s | **Streams/core at full load** | vs C=1 | Peak RSS | Chunk compute p50 / p99 | Utts whose transcript differs from C=1 |
|---|---:|---:|---:|---:|---:|---|---:|
| 160 ms | 1 | 0.773 | **1.29** | 1.00× | 946 MiB | 112 / 173 ms | — |
| 160 ms | 8 | 0.565 | **1.77** | 1.37× | 1078 MiB | 653 / 1131 ms | 2 / 128 |
| 160 ms | 32 | 0.530 | **1.89** | 1.46× | 1761 MiB | 2317 / 4885 ms | 1 / 128 |
| 560 ms | 1 | 0.539 | **1.86** | 1.00× | 951 MiB | 223 / 453 ms | — |
| 560 ms | 8 | 0.456 | **2.19** | 1.18× | 1102 MiB | 1633 / 4086 ms | 0 / 128 |
| 560 ms | 32 | 0.443 | **2.26** | 1.22× | 1805 MiB | 6085 / 16430 ms | 0 / 128 |

How to read this table:
- **What "chunk compute" means at C > 1:** it is one stream's push-plus-drain wall time while it shares one core with C−1 others. That is queueing, not a per-chunk cost.
- **Unpaced only:** with C streams on one core, every stream runs slower than real time. Paced capacity is at most the "streams/core at full load" column and has not been verified.
- **Transcripts:** batched and single-stream transcripts are not bit-identical at 160 ms. The batched Q8_0 matmul path changes the numerics. The paired DEV ΔWER from ultra-score:
  - 160 ms: +0.043 pt at C=8 (bound [−0.133, +0.125]) and +0.085 pt at C=32 (bound [0.000, +0.125]).
  - 560 ms: 0.000 at both levels.

**Reading.** On this CPU path, batching gives only 1.2–1.5× more streams per core, well short of SP-A's 3× bar. The likely mechanism is that per-stream work (cache read/write, attention over the 70-frame left context, conv state, RNNT predictor/joint) does not share weights and dominates once weight streaming is amortised. This has not been profiled yet. The gain is largest at 160 ms, where weight streaming per audio-second is highest. RSS grows by about 26 MiB per extra stream.

## 4. Open items

- **Instrument (codex-review N1 residual):**
  - harness CPU;
  - aggregate service-tree memory and threads;
  - orphan drain.

  These need the Rust `ultra-measure` instrument. The current receipts cover the recogniser process only. Fixed in `9ea46db`: content-verified WAV cache, exact shard-set merge, ScoreRecord rows, GPU-query failure reported as UNAVAILABLE.
- **Qualifying timing:** needs `BENCH_LEASE.host`, five alternating repetitions, and a paced N-stream run.
- **Latency:** CQ(D)/FC(D) and first/stable/final delays were not measured. Partials are available through `--trace`, and word times through the C ABI.
- **Batching:** check bit-exactness against the batched path, and profile the per-stream component (above).
- **CANDIDATES.lock:** lead-owned. The pin lines were posted in AGENT_CHAT.

## Raw data

Everything is under `/mnt/xbabe3-external/jsr/runs/nemotron/`:
- `results.jsonl`: one line per shard run (receipts, argv, runtime commit).
- `*.utts.jsonl`: per-utterance ref/hyp/S/D/I.
- `merged_test-clean.jsonl`, `merged_test-other.jsonl`, `merged_batch.jsonl`: pooled by `nemotron_merge.py`.
- `headline.jsonl`, `batch.jsonl`, `smoke.jsonl`.
- `score_records/`: ScoreRecord JSONL plus the `ultra-score` reports (`rescore_*.json`, `paired_*.json`).

Logs are in `/mnt/xbabe3-external/jsr/logs/nemotron/` (includes `suite_*.log` with uptime before and after). The build log is `logs/nemotron_build.log`.
