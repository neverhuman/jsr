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

Export the verified rotation in the manuscript generator's JSONL format once
every configuration has three eligible repetitions. The output carries the same
configuration IDs, the digest of the staged audio, and the timing and transcript
proofs. Run `just paper` after adding the exported rows.

```sh
mkdir -p docs/paper/data/evidence
python3 python/ultra/tools/paper_cold_aggregate.py \
  docs/paper/evidence.json target/paper-evidence \
  docs/paper/data/evidence/xbabe3-cold-dev.jsonl --format manuscript-jsonl
```
