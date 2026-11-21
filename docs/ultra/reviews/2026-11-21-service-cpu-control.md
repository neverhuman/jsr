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
`just required` pass, with audit score 85 and zero caps, before the measurements
below.

The change is committed at `217962a4cb9aa4ac53c1c68f3cb859b897b795a3`.
Its clean before/after release build retains bridge SHA-256
`454eb0b4044815395c613d2c23a55cb06c48462b974ad5d1415c9d98430053a8`.
The measurement and standalone scorer binaries are byte-identical to the prior
control. Between these commits, the shared JSONL writer is the only product
behavior change; the accompanying tests and review files also change.

| R0 full DEV400 | Original writer | Complete-record writer |
|---|---:|---:|
| Whole-service CPU seconds | 259.989309 | 221.591181 |
| Whole-service CPU-s/audio-s | 0.1015767743 | 0.0865747806 |
| Warm-service CPU-s/audio-s | 0.1000882227 | 0.0857113425 |
| User / system CPU seconds | 244.256469 / 15.732840 | 220.211411 / 1.379770 |
| Aggregate peak RSS bytes | 239263744 | 238399488 |
| Aggregate peak PSS bytes | 235634688 | 234821632 |
| Peak processes / threads | 1 / 2 | 1 / 2 |
| Rust WER / exact original Final texts | 2.851588% / 400 | 2.851588% / 400 |

The first shared-host A/B screen records **14.77% less whole-service CPU** with
all 400 actual outputs unchanged. Both use identical model/audio, streams, thread
count, packetization, output policy, observer binary, masks and guard settings.
This is one sequential pair; frequency/cache conditions and concurrent host work
remain uncontrolled. Repeated paired confirmation is required for a repeatable
effect claim. Wall throughput, capacity and profile qualification remain absent.

Under the same artifact root, `m2-record-io-binding-20261121.json` has SHA-256
`ce3edfa26cc3faa9a764c142ee2ab4baf3ee0b9d25907c909ea9f1322a823d01`;
`m2-record-io-build-20261121/build-provenance.json` has
`da90e37539f4878dc129d4396e977a2380dd2a8749139eb50855a06c2d9af2b4`.
The full R0 review `m2-record-io-r0-review-20261121.json` has
`54193ab52a5374e20352a95b4ad9b49602d66746e6823aa8ad6eb47d7192252c`.

The native D2 smoke also reproduces all eight retained FP32 texts. Its first
verifier attempt refused a job comparison because native jobs may omit unused
Python paths, which Rust defaults to empty paths. The verifier now applies those
exact missing-key defaults for native jobs and preserves any supplied value;
all other job fields remain checked. The failed attempt published no review.
The corrected check passes, as does the full D2 same-workload measurement.

## Trained D2 FP32 through the current service

The unselected first D2 step-5000 FP32 snapshot runs through the same retained
`217962a` bridge binary, observer and workload configuration as the new R0 row.
All 400 actual Final texts match the older native and NumPy FP32 runs exactly,
SHA-256 `464c11957da7c83e7aac8e8e58644706c5db9645bc2b6ab0c90cf686562b9530`.
Rust scores remain **21.879456% WER**, S/D/I **1184/375/129**, on 7715 words and
six speaker clusters. Forced advances and unfinished audio are zero.

| Current complete-workload observation | R0 native INT8 | D2 step-5000 native ORT FP32 |
|---|---:|---:|
| Whole-service CPU seconds | 221.591181 | 497.767787 |
| Whole-service CPU-s/audio-s | 0.0865747806 | 0.1944758665 |
| Warm-service CPU-s/audio-s | 0.0857113425 | 0.1939015434 |
| Observer CPU seconds, outside service | 224.758425 | 497.215244 |
| Aggregate peak RSS bytes | 238399488 | 777990144 |
| Aggregate peak PSS bytes | 234821632 | 763746304 |
| Peak processes / threads | 1 / 2 | 1 / 5 |
| Rust WER | 2.851588% | 21.879456% |

The prior D2 native reading is 0.1686292112 CPU-s/audio-s. The current observation
is 15.33% higher. Runtime code, observer placement, I/O implementation and host
conditions differ, so these rows cannot isolate the cause. The data establishes
no R1 CPU improvement. The current D2 artifact also remains substantially less
accurate than R0. Neither artifact has a qualified profile or a confirmed
capacity/latency result from these runs.

The D2 review `m2-record-io-r1-d2-review-20261121.json` has SHA-256
`f6934e0954248cad3411c49306a19b8b1c75015b95831dff2bf57946f1da5747`.
Its verifier checks actual Final-to-outcome-to-Rust-score joins, complete accepted
and processed audio, identical reference populations, executable/build and
effective job identities, placement and guard records. The selected D2 checkpoint,
full DEV selection, qualifying window, sustained runs and next-day matched
reproduction remain separate work.
