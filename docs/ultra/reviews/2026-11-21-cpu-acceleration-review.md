# CPU acceleration: independent profile and review record

The owner requests an aggressive CPU port with Rust preferred. The lead has
activated `ultra-engine` for fused Rust inference and retained the ORT path as an
independently measured control. No acceleration result is established yet.
Ownership and implementation stages are in [ENGINE_PLAN.md](../ENGINE_PLAN.md).

## Hardware and reference

H0 is an AMD Threadripper PRO 3995WX: 64 physical cores, two hardware threads per
core, AVX2/FMA and 16 separate 16-MiB L3 cache domains. It has no AVX-512 or VNNI.
Core 52 uses L3 domain 13; cores 48–51 use domain 12. These are measured host
properties from `lscpu`, not assumed topology.

The current trained-model control is D2's unselected step-5000 FP32 export.
The native service accounts for all 400 DEV examples and scores 21.879456% WER
at 0.168629 CPU-s/audio-s, with the observer co-located on its core. These are
exploratory readings, not a single-stream engine baseline or a qualifying speed
claim. See [the training/inference report](2026-11-20-trained-cpu-benchmark.md)
for identities, resource scope, comparison rows and limitations.

The current Rust implementation already uses caller-backed ORT input/output
tensors. There is no extra ORT-output-to-Vec copy at that API boundary. It still
gathers/scatters independent stream caches, validates their outputs and constructs
tensor handles/name arrays for repeated session calls. Decoder context is fed
again on every greedy round. These are source findings, not measured dominant
fractions. A new language or storage format alone does not establish acceleration.
[Safetensors](https://huggingface.co/docs/safetensors/index) supplies safe tensor
storage and zero-copy loading; execution kernels remain a separate concern.

## Diagnostic protocol

`profile-r1-cpu.py` loads only the retained NumPy consumer, verified bundle,
audio-only manifest and PCM. It receives no labels, timing references or expected
outputs. It measures process CPU around encoder, decoder, joiner and frontend
work, retaining the remaining driver/packing CPU separately. The nested greedy
timer is explicitly non-additive. Load/verification cost is recorded separately.

The initial panel is the same eight source examples as the retained smoke checks:
39.31 audio-seconds, eight streams, one inference thread, 20-ms packets and core
52. Two jobs run sequentially under the existing exploratory lease and fleet
guard. The first uses phase timers without operator tracing; the second enables
[ORT operator profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
Operator traces contain instrumented wall durations. They are not process CPU
fractions, latency evidence or uninstrumented speed measurements. Guard pauses
and shared-host interference can affect them. This consumer diagnostic excludes
the complete service, IPC and measurement observer.

Both runs retain source, configuration, actual output and hashes. Output is
compared only after inference. Decoder context-pair repetition counts estimate
reuse opportunities; they do not measure a cache speedup. Encoder state input
payload totals describe tensor volume, not measured memory traffic or bandwidth.
Hardware performance counters are unavailable (`perf_event_paranoid=4`); no host
security setting was changed.

## First measured profile

Both diagnostic runs complete and reproduce all eight retained D2 FP32 texts,
SHA-256 `496d4c18b6c4f2d0f90f2ed970bca0463bbc428f17c710b3446b4a26b14127d0`.
The source hash is
`a8145d0a954789e0857e56f8e45fd96c09937aeffac66b25a9aa022556b6c03a`.
The unprofiled consumer takes 4.597472334 process CPU-seconds, excluding load:

| Component | Process CPU seconds | Fraction |
|---|---:|---:|
| Encoder session calls | 4.006329461 | 87.1420% |
| Decoder session calls | 0.020429076 | 0.4444% |
| Joiner session calls | 0.081688565 | 1.7768% |
| NumPy frontend | 0.137025952 | 2.9805% |
| Other Python driver and packing | 0.351999280 | 7.6564% |

There are 50 encoder calls and 278 calls each to decoder and joiner. The decoder
receives 1128 context rows containing 137 unique context pairs. The encoder's
input state payload sums to 1,652,246,472 bytes. Neither count establishes the
benefit of caching or zero-copy execution. These are reference-Python fractions;
native phase counters are needed before attributing its CPU in the same way.

The instrumented run takes 6.824745233 process CPU-seconds. Its much larger cost
demonstrates profiling perturbation; these single runs do not estimate a stable
overhead ratio. Encoder node wall durations sum to 7,912,398 microseconds:

| Encoder operation | Instrumented node wall microseconds | Calls |
|---|---:|---:|
| MatMul | 3,804,923 | 8100 |
| Concat | 1,228,684 | 6000 |
| Slice | 610,882 | 8000 |
| Split | 608,011 | 1750 |
| Conv | 598,166 | 900 |
| Transpose | 314,161 | 4050 |

These wall durations must not be called CPU fractions or used as speed evidence.
Together with the unprofiled phase result, they prioritize encoder matrix work
and graph data movement. Removing decoder work alone has little room in this
reference path. Native improvements must still be measured independently.

Artifacts are under `/mnt/xbabe3-external/jsr/runs/bridge/`:

| Profile artifact | SHA-256 |
|---|---|
| `d2-s5000-cpu-phases-8/profile.json` | `9474e300282006656956dd86b8f17f10ef0e6a5cf058cd84d9ceeeed2b4407c9` |
| `d2-s5000-cpu-operators-8/profile.json` | `cfc46593b79970cd7b2a20136bb9bc02d6790e5fd3b3e4830706517d8814f54d` |

Each profile binds its workload, bundle, source, outputs and raw trace hashes.
Its adjacent `.job.json` and `.governance/` preserve configuration and guard
records. The incorrect-consumer-hash fixture rejects before loading the model
and publishes no profile. These diagnostics make no quality, capacity or profile
qualification decision.

## Review conditions for the accelerated engine

1. FP32 must reproduce the learned reference, including frontend, absolute RoPE
   positions, attention cache lengths, convolution state, partial chunks and
   greedy symbol caps. Validate layer outputs and all 400 actual transcripts;
   retain every mismatch. Different positions/cache occupancies must coexist in
   the same batch.
2. INT8 needs its own quality evidence and batch-invariance checks at 1, 3, 5, 16
   and 32 streams. Per-row scales must depend only on that row. Whole graph and
   effective consumer batching are separate claims.
3. AVX2 `vpmaddubsw` saturates adjacent U8×S8 products at signed 16-bit range.
   Full-range operands need an exact widened method or proven safe bounds such
   as A7/W7. Include extreme and cancellation cases. ORT's
   [quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
   documents this non-VNNI issue and reduced-range mitigation.
4. Fusing LayerNorm after a residual must preserve its full-channel reduction.
   It cannot be computed independently per output tile. Floating-point error
   gates must handle values near zero and cumulative error through 16 blocks.
5. Validate safetensors offsets, shapes, byte lengths, dtypes and alignment before
   SIMD or typed memory views. Bound allocations and preserve terminal failures
   and processing truth. Count retained arenas and stream state in memory.
6. Measure packed GEMM on actual layer shapes against the optimized control.
   Rust remains preferred; a tuned C ABI kernel is a permitted comparator. A
   custom kernel must earn its place with measured end-to-end behavior.
7. Treat rooflines, 0.05 CPU-s/audio-s and density goals as hypotheses. Compare
   identical trained weights, precision, workload and output policy. Put the
   observer on a different CCX and record its sampling period/CPU cost. Engine
   timing and whole-service timing remain distinct; qualification still needs
   the accepted latency, memory, backlog and sustained-operation evidence.

Ternary/T-LUT2 requires appropriate trained weights and a later experiment
identity. An untrained low-bit model cannot establish a trained-model CPU win.

## Subsequent implementation and quantization review

The first pure Rust engine source is `5221b05`. Its independent
[E0/E1 review](2026-11-21-engine-e1-review.md) identifies a safe-SIMD size-overflow
hole and loader, lifecycle, provenance and evaluator gaps. These findings require
fixes before engine acceptance; there is no independently accepted engine speedup.

The ORT v1.1 consumer update `097352b` is merged through `6b8bb6c`. Independent
execution passes four bundle-contract tests and the two eight-example native/
Python token/text fixtures. The latter runs stream settings up to 32 with only
eight source examples, so it does not establish 32 active-stream coverage.
Effective observer pinning and the remaining measurement fixes are assigned to
the runtime owner.

Independent Rust rescoring of three completed fixed-capture calibration variants
joins all 400 IDs, reference strings and FP32 texts to the retained D1 native
panel. The original sweep's execution provenance remains separate from this
retained-output check:

| Variant | Rust WER (%) | Change from FP32 (points) | Exact texts / 400 |
|---|---:|---:|---:|
| FP32 control | 29.734284 | — | 400 |
| Static MinMax encoder and joiner | 30.304601 | +0.570318 | 178 |
| Percentile 99.99 encoder, FP32 joiner | 29.876863 | +0.142579 | 286 |
| Static MinMax encoder, FP32 joiner | 29.786131 | +0.051847 | 272 |

The last variant passes the preset **point** margin of +0.10. Its six-speaker
descriptive one-sided 95% bootstrap bounds are −0.120228 / +0.269251 points
(10,000 replicates, seed 20261120). This does not establish formal noninferiority
or transcript equivalence. Canonical export provenance, full native parity,
batch/state checks and CPU cost remain to be verified.
The immutable copies, Rust score records and reports are under
`runs/bridge/a2-fixed-sweep-review-20261121` on the data drive. Their
`review-provenance.json` has SHA-256
`07a3450e5742a15cd17cf2cdc122cb5ce029c84eda5c707dcac9d97f2968ceb5`.
The MinMax encoder-only Rust report has SHA-256
`316a9a030f2123cbe64069d6a9fcc947179e36b423cd1f840cc198b477bca9af`.
