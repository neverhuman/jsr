# Service CPU control with a separate observer

The new measurement envelope runs the inference service on core 52, the observer
on core 48 and the outer supervisor/fleet guard on core 49. On recorded H0 these
use separate service/observer L3 and SMT domains. The lead reviewed and committed
the instrument at `64c1b4dde99a1b2ec233bc94594fbdfac033e4ce`; the runtime owner
independently reviewed it and posted PASS at 01:25:30 UTC.

This is an exploratory DEV control while training and other work continue.
Guarded wall time supplies no throughput, latency or qualification evidence.
The original [six-row comparison](2026-11-20-trained-cpu-benchmark.md) retains its
original artifacts and co-located measurement conditions. Matching observer
placement is required for a new R0/R1 frontier.

## Completed R0 control

Both runs use the retained Zipformer INT8 bundle, eight streams, one inference
thread, 20-ms packets and unpaced complete-workload inference. The full panel is
the same 400 dev-clean examples, 2559.535 audio-seconds, 7715 Rust-normalized
reference words and six speaker clusters. Four examples exceed 20 seconds.

| Measurement | Full DEV400 | Smoke8 |
|---|---:|---:|
| Actual Final texts matching the historical control | 400/400 | 8/8 |
| Rust WER | 2.851588% | 0% |
| Reference words / speaker clusters | 7715 / 6 | 126 / 1 |
| Whole-service CPU seconds | 259.989309 | 7.601135 |
| Whole-service CPU-s/audio-s | 0.1015767743 | 0.1933639023 |
| Warm-service CPU-s/audio-s | 0.1000882227 | 0.0969507759 |
| Observer CPU seconds, outside service | 265.276168 | 10.409495 |
| Aggregate peak RSS bytes | 239263744 | 225751040 |
| Aggregate peak PSS bytes | 235634688 | 222131200 |
| Peak sampled processes / threads | 1 / 2 | 1 / 2 |
| Resource samples | 6806 | 337 |

Full-panel S/D/I are **184/14/22**. All accepted audio is accounted for and
processed; unfinished audio is zero. CPU-only verification passes. Final and
terminal committed transcript views match the historical outputs exactly.
Incremental Commit, automatic endpointing and latency qualification remain
unavailable. The tiny smoke panel establishes pipeline integrity, with no
additional independent statistical support.

The polling period is configured to 20 ms. Actual scans overrun it: the observer
consumes nearly a full core on this host. Its CPU is reported separately from
service cost. Service affinity is set and verified in the child before exec;
observer affinity is verified at bind and after drain. Later worker/thread mask
drift is outside this instrument's verification scope. Sampled memory/process
peaks can miss brief spikes.

The historical full-panel service reading is 0.1371899818 CPU-s/audio-s. Its date,
executable and observer/governance placement differ. These single observations
cannot isolate the causal effect of observer placement or establish a repeatable
speedup. Both rows remain explicitly unqualified.

## Build and reproduction

The release build is clean before and after compilation, with unchanged HEAD
and hashes of all tracked build sources. The build proof retains rustc, command,
environment, logs and the three executable hashes. Inference uses the retained
binaries, and references are opened by Rust only after inference finishes.

All paths below are relative to `/mnt/xbabe3-external/jsr/runs/bridge/`:

| Artifact | Identity |
|---|---|
| `m2-observer-build-20261121/build-provenance.json` | `cd4a41d711692ea7b3791aa96f7b6fc3fd56c8fd8ac5526f31715bc9ff26ed1b` |
| Retained `runtime/ultra-bridge` in that build | `56a61c9a3ed04ec9ee60fc31fe59c2b61230ed7b2ab5af15d5e7d603ac4010a0` |
| Retained `runtime/ultra-measure` | `98a8ca0ff49f1ec4d963c0ebc6147ea117b95325fdd030ba942a3595ffb1424c` |
| `m2-observer-control-binding-20261121.json` | `4f9a6ff9524f660b23ac8a7db03b82c40a8441f875ea0e5e68ee62e38835acd8` |
| `m2-observer-r0-review-v2-20261121.json` | `dcdf14a8b84f453ba820d29e1100c6e07ea59dfb71c5b57e11015a66efb95d2a` |
| `m2-observer-r0-dev400/final.jsonl` | `f36db48e80b43c3751ba53bff409e5df1bf02a8c1cb9d05e5eb83181945c324d` |
| `m2-observer-r0-dev400/scores.jsonl` | `88caf9c1cc0a977e9a903c8769f9cc63c1c5d4c46a059da813f9a4c710beb41e` |

The binding retains original and resolved job identities, the build proof and
guard/helper copies. Each run retains actual output, events, observations,
outcomes, measurement, receipts, Rust scores and checksums. The fleet guard uses
the existing exploratory lease and CPU target 85%, with GPU monitoring disabled.

Run [verify-service-control.py](verify-service-control.py) from the active checkout
with `ROOT BINDING.json RUN BASELINE NEW_OUTPUT.json`. It verifies artifact hashes,
source content against the recorded Git revision, exact actual-output joins,
effective job configuration, governance and retained Rust scores. It performs no
scoring. Reproduction uses the retained guard wrapper and binaries, a fresh output
directory, the resolved job configuration and the recorded core allocation.

## Follow-up service I/O implementation

The shared JSONL writer originally passed raw files to `serde_json::to_writer`,
which can issue one write per JSON fragment. The follow-up implementation
serializes one complete record plus newline, then calls `write_all`. The record
is immediately available after the call; there is no delayed event buffer.
Serialization, temporary memory, output writing and drain remain inside service
accounting. Short writes retry, while sink failures invalidate the operation.

Two focused tests cover Unicode/escaped line boundaries, complete-record write
counts, partial writes and failures. The mapped bridge suite, Clippy and full
`just required` pass, with audit score 85 and zero caps. A clean-source measured
comparison precedes any CPU benefit claim for this change.
