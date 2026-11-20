# Wave P05 — Reference-Blind CPU Contrast Baselines (`Moonshine v2` & `Whisper small.en`)

## Scope & Protocol

Per Spec v1.0-B (`P05`) and the adopted H0 timing protocol (`AGENT_CHAT.md`), this wave establishes reproducible, reference-blind, pure-Rust CPU contrast baselines against the R0 streaming Zipformer-Transducer (`sherpa-onnx-streaming-zipformer-en-2023-06-26` int8) using `ultra bench` (`crates/ultra-stream/src/contrast.rs`) over `libsherpa-onnx-c-api.so` (`v2.0.2`, ONNX Runtime `v1.28.2`, `provider = "cpu"`).

- **Timing label**: **Exploratory (shared-host, per-block lease `BENCH_LEASE.runtime`, pinned single logical core `taskset -c 60`, host load ~32–44)**. Qualifying release comparisons require `BENCH_LEASE.host` plus a quiet host window.
- **Reference-blind enforcement**: `ultra_frontend::npz::load_npz_audio_only` reads exclusively `offsets.npy` + `pcm.npy` before inference; reference transcripts (`uids.npy`, `speakers.npy`, `texts.npy`) are loaded only after recognition completes (`load_npz_references`) for `ultra-eval` scoring.
- **CPU-only verification**: Every run inspects `/proc/self/maps` (`verify_cpu_only_process`) and passes hardened `Receipt::validate_as_evidence()`.

## Pinned Candidate Artifacts

| Candidate ID | Manifest | Upstream Repo & Revision | Honesty Label |
| --- | --- | --- | --- |
| `zipformer-int8` (`R0`) | `docs/ultra/candidates/zipformer-en-2023-06-26.yaml` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 @ 4062f0f2dc9f9cbfcdcfbf2ad8fd0ce8b4cc89d7` | `in-corpus` (LibriSpeech 960h train only) |
| `moonshine_v2_tiny_en_int8` | `docs/ultra/candidates/moonshine-v2-en-int8.yaml` | `csukuangfj2/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27 @ d1e6c30921780b8508d04b492dfb3ce8a51605d4` | `fair` |
| `moonshine_v2_base_en_int8` | `docs/ultra/candidates/moonshine-v2-en-int8.yaml` | `csukuangfj2/sherpa-onnx-moonshine-base-en-quantized-2026-02-27 @ 8f4d6c58c03d40bcea40043bb7120a878f2bbef6` | `fair` |
| `whisper_small_en_int8` | `docs/ultra/candidates/whisper-small-en-int8.yaml` | `csukuangfj/sherpa-onnx-whisper-small.en @ d9533f69affd85061aee349af7fea5cb2996dbbe` | `fair` |

## Measured Results (Single-Thread CPU, `threads = 1`, `batch_size = 1`, Core 60, Exploratory Shared-Host)

### LibriSpeech `dev-clean`

| Candidate | Mode | Utterances (`audio_s`) | WER (%) | `S / D / I` (`ref_words`) | `CPU-s / audio-s` (`rtf_cpu`) | Full-Load `streams/core` | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zipformer-int8` (`R0`) | True 320 ms streaming | 100 (`512.29 s`) | **1.770%** | `26 / 1 / 3` (`1695`) | `0.2019` | `4.95` | `159.5 MiB` |
| `moonshine_v2_tiny_en_int8` | Utterance-windowed offline | 100 (`512.29 s`) | `3.186%` | `44 / 5 / 5` (`1695`) | **`0.0454`** | **`22.03`** | `169.8 MiB` |
| `moonshine_v2_base_en_int8` | Utterance-windowed offline | 100 (`512.29 s`) | `2.360%` | `34 / 4 / 2` (`1695`) | `0.0826` | `12.11` | `365.7 MiB` |
| `whisper_small_en_int8` | 30 s padded encoder-decoder | 50 (`232.01 s`) | `2.532%` | `15 / 4 / 1` (`790`) | `3.6814` | `0.27` | `955.4 MiB` |

*(Note: on the larger 200-utterance `dev-clean` slice `1282.11 s` on core 56, `zipformer-int8` achieved `2.591%` WER at `0.2001 CPU-s/s` (`N=1` single) and `0.1048 CPU-s/s` (`9.55 streams/core`, `2.591%` WER) at `N=32` batch — a `1.91x` batching speedup with `0.000 pp` WER delta.)*

### LibriSpeech `test-clean`

| Candidate | Mode | Utterances (`audio_s`) | WER (%) | `S / D / I` (`ref_words`) | `CPU-s / audio-s` (`rtf_cpu`) | Full-Load `streams/core` | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zipformer-int8` (`R0`) | True 320 ms streaming | 100 (`670.57 s`) | **2.177%** | `37 / 2 / 2` (`1883`) | `0.2016` | `4.96` | `166.2 MiB` |
| `moonshine_v2_tiny_en_int8` | Utterance-windowed offline | 100 (`670.57 s`) | `4.514%` | `72 / 8 / 5` (`1883`) | **`0.0576`** | **`17.38`** | `232.2 MiB` |
| `moonshine_v2_base_en_int8` | Utterance-windowed offline | 100 (`670.57 s`) | `2.549%` | `42 / 3 / 3` (`1883`) | `0.1224` | `8.17` | `357.2 MiB` |
| `whisper_small_en_int8` | 30 s padded encoder-decoder | 50 (`328.27 s`) | `3.772%` | `27 / 6 / 2` (`928`) | `1.9376` | `0.52` | `1134.5 MiB` |

## Technical Interpretation

1. **`Moonshine v2` (`tiny` & `base` int8 ORT) vs `Whisper small.en` int8**:
   - Whisper `small.en` int8 pads every utterance encoder pass to a fixed 30-second positional window (`1.94–3.68 CPU-s/audio-s` on 1 Zen 2 core, `~1.0–1.1 GiB` RSS), violating the `EFFICIENT` `<= 0.25 CPU-s/audio-s` budget by `8–15x` on single-core execution.
   - Moonshine v2 scales encoder compute proportionally to actual segment length and uses a merged KV-cached ONNX Runtime graph (`0.045–0.058 CPU-s/s` for `tiny`, `0.083–0.122 CPU-s/s` for `base`, both well inside `0.25 CPU-s/s`), making `moonshine_v2_base_en_int8` a strong bounded-segment / second-pass refiner candidate (`2.36–2.55%` `fair` WER), though it is **not** a frame-synchronous causal streaming transducer (`supports_commit_policy = false`).
2. **`Zipformer-Transducer int8` (`R0`)**:
   - Sustains true 320 ms causal streaming (`Preview` / `Commit` / `Final` / `Gap` via `Emitter` + `Ledger`) inside the `0.25 CPU-s/audio-s` single-stream `EFFICIENT` envelope (`~0.20 CPU-s/s` under heavy host load; `~0.08–0.10 CPU-s/s` batched).

## Artifacts & Reproduction Commands

Artifacts and `Receipt::validate_as_evidence()`-verified receipts are stored under `/mnt/xbabe3-external/jsr/runs/p05/`:
- `dev-clean_100.jsonl`, `dev-clean_whisper50.jsonl`
- `test-clean_100.jsonl`, `test-clean_whisper50.jsonl`
- Per-run `*.scores.jsonl` (`ultra_eval::score::ScoreRecord` rows) and `*.receipt.json` (`ultra.receipt/v1`)

Reproduce any arm from the active workspace:

```bash
cargo build --release -p ultra-cli
taskset -c 60 target/release/ultra candidate validate --config docs/ultra/candidates/moonshine-v2-en-int8.yaml --verify-files
taskset -c 60 target/release/ultra bench \
  --candidate moonshine-v2-base-en-int8 \
  --split test-clean --limit 100 --threads 1 \
  --lease /mnt/xbabe3-external/jsr/BENCH_LEASE.runtime \
  --out /mnt/xbabe3-external/jsr/runs/p05/repro.jsonl \
  --scores-out /mnt/xbabe3-external/jsr/runs/p05/repro.scores.jsonl \
  --receipt-out /mnt/xbabe3-external/jsr/runs/p05/repro.receipt.json
```
