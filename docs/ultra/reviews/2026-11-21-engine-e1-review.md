# Independent E0/E1 engine review: 5221b05

The first pure Rust engine implementation is available at `5221b05` on
`lane/engine`. This review reads that immutable source from the active checkout;
it does not enter the writer's worktree. The engine is not accepted or measured
by this review. Its shape matches the owner's CPU acceleration request: packed
FP32 GEMM, AVX2/FMA dispatch, streaming cache rings and a safetensors converter.

## P1: overflow at the safe SIMD boundary

`PackedF32` exposes its fields publicly, and `gemm_f32` uses unchecked products
in length assertions. A safe caller can construct `n=16`, `k=1<<63`, empty
`data`, 16 zero bias values, then supply empty input, `rows=2` and 32 output
values. In a release build, `rows*k` and `panels*k*16` wrap to zero. The input
and panel slices can both be empty while the tile's length assertions pass.
Its first unchecked SIMD weight load then reads outside the empty slice.

This is a source-derived counterexample; the unsafe fixture was not executed.
Use checked multiplication/addition before forming pointers, reject invalid
dimensions, and keep packed storage behind validated construction. Add a
release-mode regression that proves rejection before any SIMD load. Normal-shape
scalar/SIMD parity does not prove this boundary. API acceptance requires this fix.

## Loader and artifact acceptance

- `Model::load` compares checkpoint CMVN mean with `arch.json`, but not inverse
  standard deviation. A finite positive replacement `istd` changes the frontend
  while the verified tensor file remains unchanged. Compare both tensors.
- File validation rejects slash-containing paths but permits escaping symlinks.
  Canonicalize and require containment. Verify every declared payload, including
  BPE; the current loader checks only the files it opens.
- Validate positive, bounded architecture dimensions and checked allocation
  products before division, packing or cache allocation. Validate the declared
  frontend and decoder rules instead of ignoring unsupported fields.
- The loader decodes F32 bytes into owned aligned vectors. This avoids unaligned
  typed casts, but is not mmap or zero-copy loading. Report the actual memory path.
- Converter `git_commit()` records checkout HEAD, not executed source content.
  Reconvert from an immutable clean revision or bind exact converter and imported
  trainer source bytes before/after execution. Preserve original launch limits.
- `convert` permits a nonempty directory without `arch.json` and can overwrite
  partial payloads. Require a fresh directory and exclusive artifact publication.

## Runtime and numerical acceptance

- `decode_tapped` discards feature prefixes before encoding; encoder state then
  advances before greedy decoding can fail. Add terminal failure or rollback
  semantics so retries cannot silently lose required work. Validate closure and
  reject audio accepted after finish before backend integration.
- Four fresh vectors in `decode_tapped`, plus the decoder offset vector, conflict
  with the no-allocation-after-warm-up comment. Remove them or correct the claim
  and measure allocations. Reused tensor arenas alone do not prove that property.
- `Precision::Int8` can pass construction after `quantize_int8`, then reaches the
  current `unreachable!` stub. Return `Unsupported` until E2 is implemented.
- Layer tests need explicit finite checks on both sides: `f64::max` folds can
  ignore NaNs. The full test must require at least 50 actual reference examples,
  rather than silently checking a smaller count. Retain absolute and normwise
  relative errors and every mismatch.
- Test the exponential approximation at its stated range endpoints and with
  NaN/Inf. At `EXP_LO`, exponent construction flushes a nonzero subnormal result
  to zero, so the stated relative error needs an underflow exception. The positive
  endpoint rounds to exponent 127 as intended; an initial review concern about
  exponent 128 was withdrawn after checking F32 arithmetic. SIMD min/max can also
  mask NaN differently from the scalar path. A learned-panel parity pass does not
  prove a global approximation error bound.

ISA dispatch checks both AVX2 and FMA. Normal-size SIMD loads/stores use explicit
slice bounds and unaligned instructions, and LayerNorm reduces whole rows. These
are useful implementation properties, subject to the overflow fix above.

## Evaluation admission

The first evaluator loads transcripts and reference tokens before inference.
Move reference access after retained inference outputs and timing stop, or use
a separate Rust scoring stage. Join reference tokens by actual IDs, audio and
checkpoint identity; a positional vector comparison is insufficient. Retain
S/D/I, source cluster identities, executable/source hashes and complete outcomes.

Reject empty workloads and unavailable/nonfinite CPU clocks before publishing
quality or cost. Thread scheduler CPU is an engine diagnostic, excluding cold
load and any work on other threads. It cannot replace whole-service accounting.
The evaluator's source-commit environment field also needs a verified producer.

The existing [six-row benchmark](2026-11-20-trained-cpu-benchmark.md) and
[reference profiles](2026-11-21-cpu-acceleration-review.md) remain the controls.
No new engine throughput, parity, memory, latency or qualification claim follows
from this source review. Findings have been handed to the engine owner.

## Independent retained-output check

Separate completed files under `runs/engine` contain 400-example D1 evaluations.
Exact copies are retained under
`runs/bridge/engine-retained-output-review-20261121` on the data drive. Every
actual ID joins the frozen D1 panel; the actual hypothesis strings are converted
to Rust score records and rescored with the retained Rust executable:

| Retained run | Rust WER (%) | Change from FP32 (points) | Exact texts / 400 |
|---|---:|---:|---:|
| FP32, one stream | 29.734284 | 0 | 400 |
| FP32, 16 streams | 29.734284 | 0 | 400 |
| INT8, 16 streams | 29.941672 | +0.207388 | 328 |

Both FP32 score files equal the previously verified native D1 file, SHA-256
`d1d541260bb50400cad87eb02d12c64c1d929aedd4a3e6c9f9eea697a6001b80`.
The initial INT8 variant fails the preset +0.10 point margin. Its Rust report
has SHA-256 `be0af36229ac7f79d3ad1fcc3b9bbb565b81ab58536c86f910ab4daeacf3f4d4`.
The content-review provenance has SHA-256
`7301820fbe7845de2665130d416b7262abdb014f0f5084138ed33b8ac5ec669b`.

This validates retained output content, not the original execution or its source.
The files report `git_commit: unknown`; the INT8 result cannot come from the
stub in `5221b05`. Exact E2 source and binary provenance remain necessary.
The files' engine CPU readings (0.180145 FP32/one stream, 0.118374 FP32/16 streams,
0.058694 INT8/16 streams) are preserved as source-reported exploratory diagnostics
with `performance_evidence_accepted: false`. They exclude service/cold-load scope
and do not establish a speedup. A fresh independent source-bound run is required.

## Follow-up source review of d596935

The owner rebased the engine onto `64c1b4d` and delivered `d596935`. Private
packed types plus checked construction close the original forged-weight P1
counterexample. The source also adds terminal stream failures, path containment,
positive architecture bounds, feature/decoder checks, finite layer-parity checks,
post-inference reference access, ID-based token joins and executable hashes.
The evaluator now records `int8_fp32_kinds`, closing the ablation-policy identity
omission. These are source findings; independent execution on a clean retained
build remains separate.

Multi-chunk attention retains each sub-chunk's trained attention window. Its
ring reserves space for the old left context plus all ready sub-chunks, then
selects keys through the current sub-chunk's end. Source review finds no exposure
of a later sub-chunk through that window. The scheduling grid and maximum ready
chunks must still be bound to each evaluated operating point.

Several earlier findings remain open in this delivered source despite the
owner's DONE text: CMVN comparison covers `mean` only; payload verification
selects weights, tokens and filterbank rather than every declared payload/BPE.
The quantizer retains NaN-blind ranges and lacks explicit finite-inverse handling
for tiny ranges; public dimensions/accumulator bounds still need validation.
Trained S dimensions fit the integer accumulator, which does not validate every
publicly constructible matrix. With now-private weights bounded to [-127,127],
activation -128 alone is compatible with the abs/sign kernel: adjacent products
are bounded by 32512. An extended operand/accumulator proof can cover it; rejecting
-128 is not inherently required.

Two concrete density-harness issues prevent accepting its capacity claims:

- After EOF is accepted, every pending chunk receives the utterance-end capture
  deadline. Earlier backlogged chunks therefore receive understated lateness.
  Each chunk needs its own required-sample deadline, capped at EOF for padding.
- Duration cutoff drops active work while CPU/audio divides by accepted audio.
  Accepted, processed and unfinished audio, cutoff outcomes and applicable drain
  must be recorded before a passing capacity interpretation is possible.

The owner's newer centered INT8 result is a distinct candidate from the failed
initial INT8 row above. Its quality and CPU readings await an independent retained
output/build review. The reported 24/core and 96/CCX remain scheduler experiments;
word latency, finalization, whole-service accounting and same-harness ORT evidence
are separate gates. The lead authorized experimental integration while assigning
the remaining findings to the engine owner; no deployment qualification follows.
