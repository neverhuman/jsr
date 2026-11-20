# Trained CPU benchmark: M1 DEV evidence

This implements the first research milestone of the owner-supplied training-to-CPU
benchmark plan. It is an exploratory DEV comparison. It does not qualify a release,
establish native R1 parity, or replace the selected D2 comparison.
On the matched panel, D1's 29.83% WER and 0.14086 CPU-s/audio-s offer no observed
advantage over Zipformer's 2.85% and 0.13719. Nemotron reaches 2.16% with
0.47213 CPU-s/audio-s through the completed-file control. These are retained
artifact results with six speaker clusters and shared-host measurement limits.

## Reproducible chain

The new `ultra-bridge` workspace member prepares an audio-only workload, runs R1's
delivered NumPy/ONNX Runtime consumer or the native R0 Zipformer control, measures
the complete inference service with Rust `ultra-measure`, retains actual outputs,
and scores them with Rust. A separate Nemotron completed-file control uses the
same prepared PCM, whole-service observer, strict output join and scorer.
The reviewed bridge implementation is committed at `ed7f1df` on `main`; the lead
verified its push. Runs retain the executable and source actually used, including
earlier pre-commit snapshots, rather than substituting this later commit as their
build identity.

The worker receives opaque IDs, PCM and inference settings. Reference transcripts,
speaker IDs and timing references are opened by the scoring stage after inference
exits. Every expected instance needs a matching successful outcome, PCM hash and
complete drain. Missing output cannot become an empty transcript. Run artifacts
include the actual executable and worker/consumer source, effective configuration,
CPU-backend observations, event log, process-tree measurements and checksums.

Commands and job schemas are in [the bridge README](../../../crates/ultra-bridge/README.md).
The protocol identity is `M1-DEV-20261120`: first 400 dev-clean examples, eight
streams, one inference thread, 20 ms packets, core 52 on H0, with the recorded
exploratory lease. Fleet guards apply to the actual process groups. D2 continues.
Wall throughput and batch wall timings from these guarded runs are unavailable
as timing evidence. CPU and memory readings describe this shared-host experiment.
M1 admission is bound to the actual clean/other DEV cache hashes in `DATA.lock`,
with arbitrary-cache and cross-split rejection before NPZ parsing. The frozen
panel manifest, not a caller-supplied dataset label, identifies the measured audio.
The exact panel contains **2559.535 audio-seconds**, **400 utterances**, **7715
reference words** and **six speaker clusters**. Its audio-only workload manifest
hash is `fb8241b9e609820a27fa36ad6ae85631bcbf3f99af1913acc097d7ba18f3301e`.
Four panel utterances exceed D1's 20-second training inclusion limit; none is
removed from inference or scoring. Durations span 1.56 to 24.03 seconds.

## D1 training account

R1-D1 is a 58,657,858-parameter streaming Conformer transducer with auxiliary CTC.
The retained checkpoint is step 7800 from
`D1d_S_preln_lr1e-3_b300_sa4k`. The recorded recipe uses train-clean-100, seed 1234,
AdamW learning rate 0.001, 1000-step warm-up, 300-second padded batch budget,
maximum batch 64, gradient clip 5, and SpecAugment beginning at step 4000. Training
uses BF16 autocast on one GPU. Utterances longer than 20 seconds are excluded.
This checkpoint was stopped for the D2 warm start rather than selected as a
converged model through full clean/other DEV evaluation.

The 156 logged training intervals sum to **577.775 audio-hours** and **1.793 hours**
of interval wall time. Start through the last recorded DEV evaluation spans
**1.827 hours**, yielding approximately **316 audio-hours per elapsed hour**.
These are recorded run durations, not GPU kernel utilization measurements. The
elapsed figure includes the recorded evaluation/checkpoint overhead within that
span. Earlier attempts are summarized below; preprocessing, launch gaps and
original source-at-launch evidence remain outside the complete cost account.
Unique excluded utterances and effective
retained training hours have not been reconstructed from the cumulative counters.
The training-log hash is
`347ecd07e69f6b40c665bc1b0afa489cefb3b941c90333083200cdc1afc0e533`.

Earlier retained attempts are included in the account below. Each start record
declares `device: cuda` and BF16 autocast. Hours are start-to-last-record spans
from each immutable log, rather than measured accelerator utilization. Logged
audio can omit an unlogged partial interval before an early stop.

| Run | Last logged/evaluated step | Recorded elapsed h | Logged audio h | Recorded outcome |
|---|---:|---:|---:|---|
| D0_overfit16 | 1500 | 0.288 | 94.112 | 16-example implementation exercise; 9.462% on its overfit evaluation, no generalization claim |
| D1_S_tc100_gpu | 1498 | 0.908 | 340.684 | All five progress DEV evaluations at 100% WER; stopped |
| D1b_S_preln_tc100_gpu | 593 | 0.365 | 133.284 | Both progress DEV evaluations at 100% WER; stopped |
| D1c_S_preln_lr1e-3_b300 | 2489 | 0.646 | 181.611 | All three progress DEV evaluations at 100% WER; stopped |
| D1d_S_preln_lr1e-3_b300_sa4k | 7800 | 1.827 | 577.775 | Retained D1 checkpoint; progressing, not converged |

These five recorded spans total **4.034 hours**, including **1.919 hours** across
the three collapsed D1 attempts, **1.827 hours** for retained D1, and **0.288 hours**
for D0. This is a disclosed lower-bound development account before D2/MR,
additional diagnostics, preprocessing and unrecorded gaps. It must not be
reported as a complete project GPU-hour bill. Their source logs remain under
`/mnt/xbabe3-external/jsr/runs/train/<run>/train.jsonl`.
Recipe changes across these attempts include normalization, learning rate,
batch budget, warm-up and augmentation onset. They do not isolate the cause of
learning improvement as a controlled architecture experiment.

The trainer's progress evaluations use its normalization and 7678 reference words;
the Rust board uses 7715 reference words. Do not interpret their small WER
difference as an export parity measurement.

| Step | Elapsed hours | Logged audio-hours through preceding training log | Trainer DEV-clean WER (%) |
|---:|---:|---:|---:|
| 1357 | 0.531 | 100.086 | 84.670 |
| 2000 | 0.660 | 148.236 | 60.485 |
| 2712 | 0.804 | 200.125 | 50.612 |
| 4000 | 1.061 | 296.262 | 40.883 |
| 4070 | 1.076 | 299.975 | 42.785 |
| 5426 | 1.347 | 399.989 | 37.301 |
| 6000 | 1.463 | 444.672 | 33.876 |
| 6781 | 1.620 | 500.053 | 33.303 |
| 7800 | 1.827 | 577.775 | 29.812 |

Source: the run's `config.json`, `R1-D1.json` and `train.jsonl`; the latter contains
one start record, 156 training records and nine DEV evaluations. These are
training-progress summaries, not independent confirmation rows.
Audio exposure is a lower bound through the most recent 50-step training log;
evaluations between those log boundaries have a remaining partial interval.

## Verified export

The fresh bundle is
`/mnt/xbabe3-external/jsr/bundles/r1-d1-reviewed-20261120-v2`.
It was regenerated through the existing trainer export API after verifying every
trainer source file against immutable commit
`99a033865f133c465403fd50411272de7d28f8ad`, then verifying unchanged source and all
payload hashes again after export. The sidecar explicitly scopes its proof to
export; it does not reconstruct the original training launch.

| Artifact | SHA-256 |
|---|---|
| Source checkpoint | `b3344342ae2d60d095bfbd86ba28fc4f1571f0287b65fba2d2e4b89166fdae38` |
| Bundle manifest | `311ef3db230b2e5d4e18542266c7b74d2f122a37aa47959b3874194ae6ac5481` |
| Export-source sidecar | `b7fb7fbaddd736e12cb4dc5ddb80810efe88b77ec5bc320231c22d8839ccfefb` |

The recorded environment is Torch 2.7.0, NumPy 1.26.4, ONNX 1.23.2 and ONNX Runtime
1.32.0. Mixed INT8 means dynamic INT8 encoder/joiner linear operations with a
floating-point decoder and remaining floating-point operations. The encoder
already emits the 320-dimensional projection. Cache batching uses the manifest's
declared stream axis; MR state layouts are not inherited from dense layouts.
The selected encoder INT8, decoder FP32 and joiner INT8 files total **64,359,992
bytes**. The dense encoder's declared cache tensors occupy **6,635,528 bytes per
stream**. The latter is layout arithmetic, not a measurement of incremental
resident stream-state memory, which remains unavailable. R0's manifest records
72,649,734 packed weight bytes; its trained recipe/exposure differs from D1.

The original bundle's aggregate proof reports 400/400 FP32 agreement with trainer
evaluation and only 210/400 INT8 transcript agreement with FP32. It contains two
summary rows, not per-example trainer parity evidence. The original trainer
comparison still needs its individual outputs. FP32 parity does not establish
INT8 equivalence.

The runtime owner later supplied actual 400-row native/reference parity files.
Independent inspection verifies **400/400 FP32 text and token-sequence matches**;
Rust rescoring gives **29.734284%** for both (S1558/D601/I135). Dynamic INT8 has
**355/400 matches**, reference WER **29.708360%** versus native **29.552819%**:
the absolute **0.155541-point** difference and transcript agreement both miss
the recorded handoff conditions. All IDs, indices and speakers join the frozen
DEV records. The raw proof hashes are
`e0557fb420becd7ca8725ad4e3fe17f9910a41306400c785902a06183ea9840b` (FP32) and
`8cb4cd9a91661266c753a7061b49d819798c542c4d7f428d24a2170aeee2d11d` (INT8).
The implementation is merged at `abed965`. A later sidecar binds retained
binaries, the loaded ORT library and current source hashes, but the old proof
JSONs omit fields that the current driver always writes. That sidecar therefore
does not establish the exact source of the earlier execution. Original execution
provenance and an immutable independent 400-example rerun remain open. A separate
independent native run verifies 16/16 FP32 and 15/16 INT8 text/token matches on its
small fixture; it does not replace the 400-example handoff. The pure-engine CPU
figures are excluded from the whole-service board.

The saved reference outputs also confirm **210/400** FP32/INT8 text and token
matches at sixteen streams. Their aggregate WER delta is only **−0.025924 points**
(separate one-sided 95% descriptive bounds −0.315499 / +0.207708). A small WER
change does not establish transcript equivalence. The rescoring files and reports
are retained under `/mnt/xbabe3-external/jsr/runs/r1_rust_parity/review_*`.

Independent synthetic inspection also establishes that the **current dynamic
INT8 encoder is batch-composition-sensitive**. With NumPy seed 20261120, identical
row 0 features of shape 19×80 and zero caches/absolute positions, encode it alone
and beside seven other synthetic rows. The FP32 projected output is identical in
this fixture; INT8 differs by maximum **0.2657125**, mean absolute **0.0445544**.
This measures encoder output, not WER or a timing ratio. It does not by itself
prove the cause of the native owner's observed transcript mismatches.

Reproduce with `.venv/bin/python docs/ultra/reviews/probe-r1-batching.py BUNDLE`.
The retained `r1-export-batch-diagnostic.json` under the bridge run root binds
manifest/script hashes, versions, seed and state initialization. Batch grouping
is consequently part of the present dynamic artifact's operating point. Today's
fixed eight-stream result remains unchanged; it cannot establish invariance at
other densities or packet schedules.

The lead assigned a statically calibrated export for a later freeze artifact.
It must have a separate identity and TRAIN-only calibration provenance, actual
precision details, batch/state/partial-chunk checks and independent quality proof.
The first v1.1 MinMax artifact is batch-invariant in the trainer's reported
2/8/16-row checks, but fails its quality gate: 30.477% versus 29.812% trainer WER
(+0.67 points), with 187/400 identical transcripts. Encoder-only and joiner-only
static INT8 variants also miss +0.10 points. These failed attempts remain
recorded; calibration sweeps and a floating-point joiner are owner follow-ups.
Review additionally requests exact consumed calibration-row provenance, bounded
sampling during collection, actual joiner input/context pairs and distinct file
versus selection hashes. No static artifact is accepted by this report.
It has not replaced the measured dynamic bundle. Native plumbing
remains experimental, and the verified 88.75% INT8 agreement is below the recorded
handoff target. No output-preserving equivalence is asserted.

## Measurement and claim limits

The completed primary pilot rows use the same 2559.535 seconds of prepared audio:

| System and path | Rust WER (%) | S / D / I | Whole-service CPU-s/audio-s | Warm CPU-s/audio-s | Aggregate peak RSS bytes | Sampled PSS bytes | Peak threads |
|---|---:|---|---:|---:|---:|---:|---:|
| R0 Zipformer INT8, native sherpa 2.0.2 / ORT 1.28.2 | 2.8516 | 184 / 14 / 22 | 0.137190 | 0.136190 | 238,551,040 | 234,956,800 | 1 |
| R1-D1 mixed INT8, NumPy / ORT 1.32.0 bridge | 29.8250 | 1572 / 591 / 138 | 0.140862 | 0.140561 | 592,748,544 | 575,327,232 | 7 |
| Nemotron 560 ms Q8_0, completed-file CPU control | 2.1646 | 137 / 15 / 15 | 0.472127 | unavailable | 1,283,899,392 | 1,277,926,400 | 17 |

All three rows account for all 400 inputs, 7715 reference words and six speaker
clusters, with zero unfinished samples. R0/Nemotron decoder cap counts are unavailable;
D1 records one forced advance. The runtime paths and ORT versions differ, so this
does not isolate an architecture effect. CPU readings are single exploratory
observations, not a statistically established cost ratio. D1 uses about 2.7% more
CPU and substantially more memory in this pilot while making many more errors.
It provides no observed quality/cost advantage over the deployed R0 artifact.

Rust's paired speaker bootstrap (10,000 replicates, seed 20261120, alpha 0.05)
reports **D1 minus R0 = +26.9734 WER points**, **+945.9% relative WER**, with separate
one-sided 95% lower/upper bounds **+23.6615 / +31.0063 points**. These are the fifth
and ninety-fifth percentile bounds, not a two-sided 95% interval. This bootstrap
has only six independent speaker clusters, making its bounds fragile and
descriptive only. The primary
formal selected-D2-versus-R0 decision remains **INCONCLUSIVE**: D2 selection and
the qualifying experiment have not occurred. Other system rows are descriptive.
The paired machine-readable report is
`/mnt/xbabe3-external/jsr/runs/bridge/m1-r0-r1-paired-20261120.json`.

Nemotron's row retains `unknown-overlap` and internal-evaluation rights scope.
The measured GGUF is 699,872,960 bytes, SHA-256
`d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d`;
the actual CLI binary hash is
`41b19d48dcdc53af10baec48d543033f534e4cc5ad09b05227def417adfbe435`.
Loaded CPU-library evidence is positive; the wrapper does not infer CPU-only
operation merely from an unavailable GPU check. Its whole-service CPU is
1208.424416 seconds, including WAV creation, verification, load, CLI execution,
collection and drain. The retained directory is `m1-corrected-n560-dev400`, and
its `scores.jsonl` hash is
`cc78352ca43f0b296390b615fe82571236335882927d91f163b31499f23f9e8f`.
This descriptive row receives no live latency or automatic endpointing label.

The fresh D1 mixed-INT8 row completes all 400 inputs with **29.8250% Rust WER**:
1572 substitutions, 591 deletions and 138 insertions over 7715 reference words.
Its whole-service cost is **0.140862 CPU-s/audio-s**; the post-load snapshot split
is **0.140561**, with 0.77 sampled CPU-seconds attributed before that snapshot.
Aggregate peak RSS is **592,748,544 bytes**, sampled PSS **575,327,232 bytes**, and
observed peak thread count **7**. One forced decoder advance is recorded. These
are exploratory readings from the retained pre-commit runtime snapshot, with
`valid: true`, `evidence_valid: false`, `qualifying: false`.

Artifacts are under
`/mnt/xbabe3-external/jsr/runs/bridge/m1-corrected-r1-dev400`.
Its `scores.jsonl` hash is
`c6b7ac1e1abbe134a5c706debcf385caf847477572856aa5c1a1d009af680fe1`.
The prior prototype and this fresh verified export emit identical text for all
400 input IDs. Their raw output-file hashes differ because the prototype used
completion order and the fresh output uses source order. All eight export payload
hashes are identical across the old and verified bundles; the fresh manifest
repairs source provenance. This is a repeat of bridge output, not independent
trainer/native parity or next-day CPU reproduction.

Two fresh eight-example A/A runs (`m1-corrected-r1-aa8` and `-repeat`) match the
bundle, workload, executable, worker, consumer, observer and inference settings.
Their source-sorted Final and terminal committed-text files are byte-identical,
SHA-256 `5eff4c9c22b26d3502705e571d6d3179ede15de6ed4ecda5f6544c281a751077`.
Both Rust scores are 26.984127% over 126 words. This is same-day transcript
repetition; their short-run CPU readings do not establish the frozen second-day
5% median-cost tolerance.

The separate paced eight-example run accounts for all input and scores its actual
output: 26.190476% WER (S23/D9/I1 over 126 words). All **1968** observations occur
after the accepted samples' capture deadlines; minimum observed slack is
0.000230451 seconds. This necessary observation check is not a packet-release
trace or latency qualification. Dynamic INT8 batching can change the output;
the paced score does not reuse the unpaced transcript. Guarded wall metrics,
CQ/FC and backlog qualification remain unavailable.

The six-run machine-readable frontier is
`/mnt/xbabe3-external/jsr/runs/bridge/m1-evidence-summary-20261120.json`, SHA-256
`2f14a1ff133cb6dcddef39e6e4f52a4f688e755faf9a4ed185c6f3c01a83edeb`.
It binds run manifests, Rust score artifacts and unavailable measurements.
Recreate it with
`.venv/bin/python docs/ultra/reviews/verify-m1-evidence.py RUN_ROOT NEW_OUTPUT.json`.
The verifier checks all retained inference/scoring checksums, exact panel joins,
complete outcomes, A/A identities and paced observation deadlines. It performs
no independent text normalization or scoring and refuses an existing output file.

Earlier instrument attempts remain retained rather than disappearing from the
account:

| Attempt | Disposition |
|---|---|
| `m1-smoke-r1-int8` | Invalid receipt: model-batch timing fields absent; retained as a failed instrument run |
| `m1-smoke-r1-int8-v2`, `m1-dev400-r1-int8` | Superseded prototypes: packet/IPC timings were labelled as model-batch timings; old export provenance |
| `m1-smoke-r0-int8`, `m1-dev400-r0-int8` | Superseded prototypes: unavailable decoder caps were incorrectly represented as zero |
| `r1-d1-reviewed-20261120` export directory | Failed preparation: missing ONNX dependency; never admitted as a complete bundle |

Corrected runs retain unavailable counters as null, preserve strict receipt
rejection and suppress guarded timing evidence. None of the superseded receipts
supports a performance/profile claim. Their actual text can serve only the
explicitly labelled diagnostic repeat check above. The corrected three-system
board uses fresh runs, not the most favorable prototype timing.

The complete service scope includes load, model verification, audio I/O, features,
IPC, decoding, formatting, collection and drain. Reference preparation, the
measurement observer and scoring are outside that scope consistently for all
systems. R0/R1 warm CPU subtracts a post-load process-tree snapshot with explicit
clock-tick resolution. Nemotron does not expose an equivalent warm CPU split.
The 39.31-second R0 smoke cost **0.215940 CPU-s/audio-s** including cold load,
versus **0.132246** after its 3.29-CPU-second load snapshot. The full panel's warm
reading is **0.136190**. This demonstrates the cold-load tax for short workloads
in this bridge. It does not fully reconcile earlier 0.079-versus-0.20 Zipformer
screens with different paths/settings; neither older figure substitutes for the
current matched service row.
RSS is the observer's conservative aggregate of process high-water marks; PSS and
thread peaks are sampled. Peak memory and cold cost are therefore tied to this
workload and polling scope.

The bridge credits processed samples only after verified EOF drain because its
reference consumer lacks an authoritative streaming watermark. Intermediate
backlog observations are conservative upper bounds. All primary inputs must
complete; a completed subset never substitutes for the full quality panel.
R0 pads 800 ms at controlled EOF. R1 follows its bundle's partial-chunk drain.
Nemotron is a completed-file control despite its internal 560 ms context setting.

`summary.valid` means measurement/output integrity. `evidence_valid` separately
reports the unchanged strict receipt gate. Uncommitted runtime source, guarded
timing and unavailable native decoder cap counts prevent a qualifying receipt.
The hardened bridge always writes `completion_status: exploratory`; it cannot
issue a qualified receipt. Raw event hashes are exact-log integrity hashes; A/A
transcript equivalence compares the source-sorted output views. R0/R1 commit
their terminal text through Final and do not emit incremental Commit events.
CQ, stable-word delay, FC, automatic endpointing, profile capacity, backlog
qualification and real one-hour sustained operation remain **unavailable** here.

The board has only six independent speaker clusters. DEV use, limited cluster
support and shared-host readings prevent a confirmatory quality/cost claim. The
frozen +0.10 WER-point margin and 5% second-day CPU tolerance are preserved for
later confirmation, not decided from this pilot. No public-test or sealed OOD
references were opened by this work.

## Verification and remaining handoffs

Independent synthetic FC diagnostics exercise joint forced-segment ownership,
duplicate delivery, gaps/unmatched denominators, premature closures and malformed
closures. All four pass. They prove arithmetic/lifecycle behavior, not latency.
Run them with `cargo test -p ultra-bridge --test finalization`.
Worker regressions cover crash, timeout, wrong/stale/missing output, nonzero exit,
oversized output and actual grandchild cleanup. Bundle regressions reject corrupt
native payloads and incomplete manifests before native library loading.

On `ed7f1df`, the retained 17 qualification diagnostics also pass: duplicate and
tampered delivery, UTF-8 boundaries, atomic failure, incomplete/unflushed Final,
impossible processing, duplicate scoring snapshots and malformed receipts. Full
`just required` passes with `RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=6`, including
104 Python tests with one skip, contracts, security and Jankurai **85 / zero caps**.
The subsequent C1/runtime integration gates report 105 Python tests with one skip.
Parallel CPU-burner timing assertions failed on the restricted shared build cores
in an earlier attempt; serial test scheduling passed without changing thresholds.

The runtime owner retains native R1 and common-suite implementation. Independent
execution on merged `6dc7f0d` verifies frontend parity against the actual
BundleRecognizer on five edge waveforms and 50 stratified DEV utterances:
31,772 frames, maximum absolute difference **7.153e-7**, below 1e-4. The native
sherpa ABI/decode tests, checked result/free tests and coordinator fixture pass.
The missing-result-to-empty path now returns an error. B1's later default native
parity test also passes with the small-fixture results stated above. Symlink
containment, checked state-size arithmetic, cache-value validation, terminal
inference failures and complete retained-state accounting remain source-review
follow-ups; passing parity tests do not close those gates. The lead retains D2
selection, data/timing audit, freeze, coordinated qualifying window, independent
reproduction and release evidence.

Independent C1 synthetic review found acceptance of NaN/infinite reference times.
The lead fixed it in `cf822ff`; all nine timing tests and six independent
nonfinite/negative-duration probes now pass. This validates data rejection, not
the accuracy of the alignments or the still-open listening audit. Rust timed-word
normalization and explicitly scoped populations are required before CQ/FC claims.
The explicit corpus integration diagnostic
`cargo test -p ultra-bridge --test timing_join -- --ignored --nocapture` passes on
the pinned DEV400 score records and MFA dev-clean file: **7678 raw timed words
map to all 7715 Rust reference words, with no dropped input**. It verifies exact
normalized word identity, finite ordered intervals and audio-end bounds for all
400 inputs. It is ignored by default because it needs the retained data-drive
artifacts. This confirms the available join, not alignment accuracy or latency.

The clock audit prompted `6dc7f0d`: coordinator Gap, hard-cap and cancellation
events now use an injectable monotonic clock, and a hard cap without a real
backend Final closes as incomplete rather than inventing flush evidence. The
independent injected-clock fixture passes. Production common-suite callers still
need to share the same clock instance between backend and coordinator; default
factories create separate origins. Controlled EOF remains distinct from automatic
endpointing and successful forced segmentation. M1 timestamps/drain checks are
independent of that coordinator, so its quality/cost rows retain their scope.

The accepted plan's remaining requirements include D2's fixed clean/other DEV
selection, native same-precision parity, reference timing audits, common-suite
fault tests, five paired timing blocks, both real one-hour runs and next-day
reproduction. Training MR on other hosts is now an owner-authorized background
activity in the coordination log; it remains outside the initial freeze and
needs its matched step-7800 control and separate experiment identity.
The recorded data ruling blocks TED-LIUM 3 on rights; it was not fetched or
silently replaced. The MFA listening audit remains open. These dependencies do
not supply missing domain-shift or latency evidence to this DEV pilot.
