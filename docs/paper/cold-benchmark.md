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

## Opponents on other machines

The systems added for the wider comparison (owner ruling O16) run under the same contract on xbabe0 (Zen 2), xbabe6 (Xeon Platinum 8358, AVX-512 VNNI) and xbabe4 (Apple M3 Ultra) through `docs/paper/opponents/`. `run-opponent.sh ARM BUDGET ROUND` is one repetition; it has one `configure()` case per system, writes the same receipt files as `run-cold.sh` plus `budget.json`, `environment`, per-CPU `/proc/stat` before and after, and the clock of the service cores (`freq.tsv`), and it waits while `QUIET.lock` holds a quiet window on xbabe0. `plan.sh` runs a plan in phases so that each arm is alone on its L3 group and the host stays inside the agent's core budget (`plan-zen2.txt` is the xbabe0 plan). `collect.py` pulls the receipts, scores them with the pinned `ultra-score` as `paper_dev_audio.py score` does, prints every repetition and the median, and exports the small receipt files and the evidence rows.

- **Budgets.** `t1` is one pinned core and one thread. `t8` is eight pinned cores with eight threads inside one recogniser (`parallel: intra_op`). `w8` is eight pinned cores with eight single-thread recognisers side by side, each on its own core and its own duration-balanced shard of the panel (`parallel: workers`), metered as one process tree, so eight model loads are inside the cost. Engines that are single-threaded per stream (ours, the Zipformer, Moonshine) have no `t8`.
- **Views.** `workload.py` stages hard-linked views of the verified panel: `panel730` (the cost and WER workload), `panel73` (every tenth file, one utterance per speaker) and `panel10` for arms that cannot finish the panel, and `lag100` for traces. A row from a smaller view carries that view's digest and never joins the 730-utterance group.
- **Setup.** `setup-uv.sh`, `setup-python.sh` and `setup-rust.sh` install hash-verified, isolated toolchains and environments under the machine's opponents directory; nothing is installed system-wide. Every model file and binary is hashed into `config.sha256`.
- **macOS.** There is no CPU affinity and no procfs. The thread count is the only bound, and `bsd_time_measure.py` meters with `/usr/bin/time -l` and `getrusage`: CPU seconds are the kernel's counters for the waited-for tree, and peak RSS is the largest process (the sum over the eight workers in `w8`), not a sampled sum over the tree.
- **Our engine on macOS.** `ultra-cold-wav` does not build there: `ultra-bridge` and `ultra-stream` depend on the Linux-only `ultra-eval`. The Apple rows of our engine use the engine's own tool, `ultra-engine-eval --no-default-features --panel … --precision int8 --streams 1 --packet-ms 20 --qualifying`, on an npz view of the same audio (`engine_cache.py`); its peak RSS includes the panel's PCM, which the tool holds in memory. The same arm (`jsr-d2e-int8-enginetool`) runs on xbabe0 beside `ultra-cold-wav` so that the harness difference is measured. Until main compiles on aarch64 the tool is built with `aarch64-pqscan-compile-fix.patch`.
- **The Zipformer on macOS** runs through sherpa-onnx's Python binding with the Rust service's settings (`sherpa_zipformer.py`); on Linux that adapter also writes the partials trace.

## Open-source opponents (Vosk, Whisper.cpp streaming, onnx-asr Parakeet)

These arms use the same panel, meter and scorer. Each system has its own virtualenv and model directory under `$JSR_DATA/opponents/` (`vosk/`, `whispercpp-stream/`, `parakeet-onnx-asr/`); versions, sources and hashes are in the candidate manifests. [Vosk Large](../ultra/candidates/vosk-en-us-0.22.yaml) · [Vosk Small](../ultra/candidates/vosk-small-en-us-0.15.yaml) · [Whisper.cpp Streaming Base](../ultra/candidates/whispercpp-stream-base-en.yaml) · [Whisper.cpp Streaming Tiny](../ultra/candidates/whispercpp-stream-tiny-en.yaml) · [Parakeet TDT v3 with onnx-asr](../ultra/candidates/parakeet-tdt-0.6b-v3-onnx-asr-int8.yaml)

`run-cold.sh` knows the one-thread arms (`vosk-large`, `vosk-small`, `whispercpp-stream-base`, `whispercpp-stream-tiny`, `parakeet-v3-onnxasr-int8`) and the session anchors (`jsr-h22ctl-79800-ema-int8`, `jsr-d2e-16457-int8`, `r0-zipformer-int8`); their commands are defined once in `opponents/open-arms.sh`. The streaming runners feed 20 ms packets as fast as possible and write the partial results they read to `trace.jsonl` (one JSON object per result: `id`, `audio_ms`, `wall_ms`, `text`, `final`, plus `audio_s`, `chunk_wall_ms`, `cum_wall_ms` for the word-lag harness). "Whisper.cpp streaming" is the Picovoice benchmark's engine: consecutive 3 s chunks, each decoded on its own.

**Timing protocol on the shared host (v3).** The memory bus of the 3995WX is shared by all cores, so a memory-bound recogniser slows down when other cores are busy even if it is alone on its cache group (measured: up to 1.75 times for our engine, 30% for the Zipformer). One arm is therefore timed on the whole host at a time. The host keeps an ordered queue (`TIMING.order`), a lock directory (`TIMING.lock`, at most 30 minutes per hold) and a progress file (`TIMING.done`). `opponents/open_turn.py` takes the lock for an entry only when every earlier entry is done, starts each hold with one repetition of our engine as the session anchor, runs the entry's repetitions through `opponents/open-run.sh`, and appends `DONE N`. Systems at or below about 0.25 CPU-seconds per audio-second are timed on the full panel. Slower systems are timed on `panel73` (every tenth utterance) and take their WER from one full-panel pass on another host with the same binary and model hashes. The 73 shared hypotheses are compared with every timed repetition.

`open-run.sh` wraps `run-cold.sh` for a one-core full-panel repetition and repeats its steps for the other cases (a CPU list, or `panel73`; `<arm>-t8` = eight intra-op threads, `<arm>-w8` = eight single-thread recognisers side by side). It records per-CPU busy time (`host.cpu.json`), the service CPUs' clock frequency every 10 s (`host.freq.tsv`), the panel, its audio seconds and the model size. It sets aside a repetition that was not alone on its cache group and on the host. `opponents/open_summary.py` copies the small receipt files into `data/raw/opponents/` and prints the table, the evidence rows and the anchor rows.

```sh
OPEN_RUNS=$JSR_DATA/opponents/bench-open/runs-v3 \
  python3 docs/paper/opponents/open_turn.py docs/paper/opponents/open-plan-xbabe0.txt "$OPEN_RUNS/logs"
python3 docs/paper/opponents/open_summary.py table docs/paper/data/raw/opponents
python3 docs/paper/opponents/open_summary.py rows docs/paper/data/raw/opponents "<session>"
```
