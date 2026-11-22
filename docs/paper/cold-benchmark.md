# Common cold CPU benchmark

The cold comparison uses the frozen 730-utterance DEV panel, one recognizer stream at a time, and three rotations on the same physical core. The meter accounts for recognizer process-tree CPU from launch through model load, final hypothesis flush, and process exit. The Rust scorer runs afterward. The separate warm-cost and paced-density protocols are defined in the benchmark freeze. [Evidence and configuration pins](evidence.json) · [Benchmark freeze](../ultra/FREEZE.md) · [H20 research record](../../research/H20_matched_cpu_frontier/README.md)

## Workload and Rust backend

Run from the repository root. The staging tool writes opaque WAV names in 16 kHz mono PCM16 format and saves scoring metadata outside the WAV directory.

```sh
.venv/bin/python python/ultra/tools/paper_dev_audio.py stage \
  docs/ultra/data/dev_selection_panel.json \
  /mnt/xbabe3-external/jsr/cache/librispeech_pcm \
  target/paper-evidence/panel
cargo build --locked --release -p ultra-bridge --bin ultra-cold-wav
cargo build --locked --release -p ultra-eval --bin ultra-measure --bin ultra-score
```

For a continuation of the recorded rotation, use the exact executable copies pinned in `evidence.json`. Set `PAPER_COLD_WAV`, `PAPER_MEASURE`, and `PAPER_SCORER` to the pinned binary paths if they differ from the runner defaults. Keep each run's audio, model, and executable identities fixed across its three repetitions.

## Native Moonshine

Build against the Moonshine Voice v0.1.6 Linux x86-64 SDK. Set `MOONSHINE_SDK` to its extracted root. Use the Tiny and Small Streaming English bundles in the catalog's `quantized_26_11_02` directories. [SDK release](https://github.com/moonshine-ai/moonshine/releases/tag/v0.1.6) · [Model catalog](https://github.com/moonshine-ai/moonshine/blob/v0.1.6/core/moonshine-model-catalog.cpp)

```sh
mkdir -p target/paper-smoke
c++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I"$MOONSHINE_SDK/include" docs/paper/moonshine_stream.cc \
  -L"$MOONSHINE_SDK/lib" -Wl,-rpath,"$MOONSHINE_SDK/lib" \
  -Wl,-rpath-link,"$MOONSHINE_SDK/lib" -lmoonshine \
  -o target/paper-smoke/moonshine-stream
```

The adapter loads one native streaming model per launch, runs CPU ONNX Runtime in single-thread mode, and saves one final `.txt` hypothesis for each staged WAV. The pinned Nemotron binaries and model packs use the same staged files and their declared streaming profiles.

## Run and verify

Stage all pinned model packs, the meter and the recognizer binaries at the paths in `evidence.json` and `run-cold.sh`. Set `PAPER_SERVICE_CPU` and `PAPER_OBSERVER_CPU` to the allocated service and observer CPUs. The defaults use the recorded xbabe3 allocation. The runner captures commands and hashes before each launch. The aggregator verifies placement, input identity, rotations, runtime and artifact digests, and the retained predictions before emitting cold cost and WER rows.

```sh
flock -n target/paper-evidence/cold.lock bash docs/paper/run-cold.sh
python3 python/ultra/tools/paper_cold_aggregate.py \
  docs/paper/evidence.json target/paper-evidence \
  target/paper-evidence/cold-aggregate.json
```

Export the verified rotation in the manuscript generator's JSONL format (schema v2,
[data/README.md](data/README.md)) once every configuration has three eligible repetitions.
Each configuration in `evidence.json` must declare `"class"`: `streaming` or `whole_utterance`. The output carries the same
configuration IDs, the digest of the staged audio, and the timing and transcript
proofs. Run `just paper` after adding the exported rows.

```sh
mkdir -p docs/paper/data/evidence
python3 python/ultra/tools/paper_cold_aggregate.py \
  docs/paper/evidence.json target/paper-evidence \
  docs/paper/data/evidence/xbabe3-cold-dev.jsonl --format manuscript-jsonl
```

## Open-source opponents (Vosk, Whisper.cpp streaming, onnx-asr Parakeet)

These arms use the same panel, meter and scorer. Each system has its own virtualenv and model directory under `$JSR_DATA/opponents/` (`vosk/`, `whispercpp-stream/`, `parakeet-onnx-asr/`); versions, sources and hashes are in the candidate manifests. [Vosk Large](../ultra/candidates/vosk-en-us-0.22.yaml) · [Vosk Small](../ultra/candidates/vosk-small-en-us-0.15.yaml) · [Whisper.cpp Streaming Base](../ultra/candidates/whispercpp-stream-base-en.yaml) · [Whisper.cpp Streaming Tiny](../ultra/candidates/whispercpp-stream-tiny-en.yaml) · [Parakeet TDT v3 with onnx-asr](../ultra/candidates/parakeet-tdt-0.6b-v3-onnx-asr-int8.yaml)

`run-cold.sh` knows the one-thread arms (`vosk-large`, `vosk-small`, `whispercpp-stream-base`, `whispercpp-stream-tiny`, `parakeet-v3-onnxasr-int8`) and the same-session anchors `jsr-d2e-16457-int8` (our engine) and `r0-zipformer-int8`; their commands are defined once in `opponents/open-arms.sh`. The streaming runners feed 20 ms packets as fast as possible and write the partial results they read to `trace.jsonl` (one JSON object per result: `id`, `audio_ms`, `wall_ms`, `text`, `final`, plus `audio_s`, `chunk_wall_ms`, `cum_wall_ms` for the word-lag harness). "Whisper.cpp streaming" is the Picovoice benchmark's engine: consecutive 3 s chunks, each decoded on its own.

On the shared benchmark host a repetition is started through `opponents/open-run.sh`, which wraps `run-cold.sh` for the one-core budget and repeats its steps with a CPU list for the eight-core budget (`<arm>-t8` = eight intra-op threads, `<arm>-w8` = eight single-thread recognisers side by side). It waits for quiet windows and for room under the host limit of active benchmark cores, records per-CPU busy time (`host.cpu.json`) and the service CPUs' clock frequency (`host.freq.tsv`), and repeats a repetition whose CCX was not otherwise idle. `opponents/open_schedule.py` runs a plan of repetitions; `opponents/open_summary.py` copies the small receipt files into `data/raw/opponents/` and prints the table and the evidence rows.

```sh
bash docs/paper/opponents/open-run.sh vosk-small 1 4 0          # round 1 on CPU 4, meter on CPU 0
bash docs/paper/opponents/open-run.sh vosk-small-w8 1 8-15 3    # eight workers on CPUs 8-15
python3 docs/paper/opponents/open_summary.py table docs/paper/data/raw/opponents
```
