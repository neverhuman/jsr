# Trained CPU benchmark: D1 and first D2 snapshot on DEV

This delivers the M1 research comparison and an initial native FP32 service
extension, including D2's first retained step-5000 snapshot. It is an exploratory
DEV comparison. Release qualification and the
selected D2 comparison remain open.
On the matched panel, D2 step 5000 reaches 21.88% WER at 0.16863 CPU-s/audio-s
through native FP32 inference, improving on D1 native FP32's 29.73%. Both remain
well behind Zipformer's 2.85% and 0.13719. D1's mixed-INT8 bridge result is
29.83% at 0.14086. Nemotron reaches 2.16% with
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
All current M1/M2 CPU rows carry an additional caveat: **observer co-located on
the service core**. During the D2 follow-up, a read-only process check found the
observer and service both restricted to core 52; after 11:02 wall time they had
consumed 4:19 and 4:15 CPU time respectively. Observer CPU is excluded from service
cost consistently, but this placement competes for execution and cache resources
and may affect service CPU cost. The size and direction of that effect have not
been measured. The lead's freeze rule now requires a different CCX for the
observer, a fixed sampling period and separate observer CPU accounting. A matched
R0 rerun with that placement is required before freeze; existing rows are retained
with this limitation.
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
The unique duration-filter population is reconstructed below; cumulative training
drop counters are not used as unique exclusion counts.
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

An independent audit verifies all 14 train-clean-100 parquet shard hashes against
the pinned DATA.lock manifest and inspects every audio header. The population is
**28,539 utterances / 100.590880 raw audio-hours**. The recorded strict `>20 s`
filter excludes exactly **one unique utterance**, `7859-102521-0017` (392,400
samples, 24.525 seconds), leaving **28,538 / 100.584067 hours** eligible. Twenty
ordinary examples and the excluded example were decoded to verify header frame
counts against the decoder used by training. These are raw audio durations;
logged training exposure uses feature-frame durations. Eligibility does not prove
that an interrupted run consumed every eligible example or reconstruct launch
source provenance. The audit source is `training-exposure.py` in this review
directory, invoked with the retained run config and a new output path. Its result
`/mnt/xbabe3-external/jsr/runs/bridge/d1-training-eligibility-20261120.json` has
SHA-256 `3d1fb969db3b309120c51b78db240c8b85aece7e21af62606cdb2dae03ce7a94`.

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

D2 continues from D1 step 7800 with a fresh 102,000-step schedule, 600-second
padded batch budget, maximum batch 128 and all three TRAIN splits. Its retained
trainer DEV-clean progress evaluations include **21.907% at 5000** and
**21.894% at 6854**, both on
400 examples / 7678 words with zero forced advances. These establish further
learning, without full clean/other selection or a final frozen D2 candidate.
The first retained step-5000 snapshot now has its own research export, described
below. The D2 data-eligibility and full cost accounts remain
separate from D1's completed pilot. The completed D2 eligibility audit verifies
all 126 TRAIN shard hashes and the complete per-split file sets against DATA.lock:

| TRAIN split | Unique utterances | Unique excluded >20 s | Eligible utterances | Eligible raw audio-hours |
|---|---:|---:|---:|---:|
| clean-100 | 28,539 | 1 | 28,538 | 100.584067 |
| clean-360 | 104,014 | 17 | 103,997 | 363.498553 |
| other-500 | 148,688 | 36 | 148,652 | 496.631291 |
| Total | 281,241 | 54 | 281,187 | 960.713911 |

The nominal 960-hour corpus contains 961.054398 raw audio-hours; the filter
excludes 0.3404875 hours. All 54 excluded examples and 60 ordinary examples were
decoded to check their header lengths. The independently rerun clean-100 row is
identical to D1's audit. The result
`/mnt/xbabe3-external/jsr/runs/bridge/d2-training-eligibility-20261120.json` has
SHA-256 `4c5415d147599f2abb9dfa9a2ecf7baa3bde1cd83b43b5ed1fff22d1facc1559`.
Both audits bind the original script SHA-256 `0329fa7f...`, retained exactly as
`training-exposure-0329fa7f.py` under the bridge run root. The current source
also refuses incomplete shard sets; that check was independently applied to the
retained audit. These are eligible populations on the pinned data, rather than
counts of examples actually consumed or evidence of hashes captured at launch.

The subsequent trainer progress evaluation at **step 10,000 is 17.309% DEV-clean
and 31.719% DEV-other WER**, 400 examples each. These remain trainer-normalized
progress figures. Independent reading of `D2_S_960h/dev_curve.jsonl` confirms both
records. A formatter error produced an empty coordination summary; the trainer
replaced that shell reporter with a locked Python reporter. The empty summary is
not evidence, and the actual records are retained.

The first-snapshot cost account retains the exact log prefix through step 5000's
progress evaluation, rather than hashing a later, still-growing training log.
Its 100 training intervals record **700.183378 audio-hours**, **1.774422 interval
wall hours**, and **1.775299 hours from start through progress evaluation**
(394.403 audio-hours per elapsed hour). The run starts from D1 step 7800's model
with a fresh schedule; these figures are additional to D1's training account.
They are GPU-attached elapsed lower bounds, not accelerator utilization or a
complete training bill. Original training source at launch remains unreconstructed.
The files `d2-s5000-training-prefix.jsonl` and `d2-s5000-training-account.json`
under the bridge run root have SHA-256
`1d6a755ba0d202089cc464f5a0b48a4241daa733dd49bce97f6a63a325d23763`
and `c4ed069bfe1756e730f6c89acccba78209b55d2242ddfcf6271898ac6ebb5350`.

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
summary rows. The later v1.1 delivery adds `dev400_fp32.jsonl`, SHA-256
`8d63048f946837b47fe29e851e13f62f3a7c291dd4c052db97996998046bb5c1`,
with individual trainer and bundle text for all 400 IDs: every pair matches.
Its FP32 payloads equal the reviewed bundle's. Inspection of the delivered
`--compare-checkpoint` implementation confirms it loads ASRModel and performs
trainer greedy evaluation to populate `trainer_hyp`. This verifies the retained
content and code path; original execution provenance is a separate gate.
FP32 parity does not establish INT8 equivalence.

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
provenance remains open for those older files. A separate independent native run
verifies 16/16 FP32 and 15/16 INT8 text/token matches on its small fixture. The
later immutable FP32 service run below verifies all 400 texts independently;
native INT8 acceptance remains open. The older pure-engine CPU figures are
excluded from the whole-service board.

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
Independent Rust rescoring of the retained per-example MinMax outputs confirms
the failure: **29.734284% FP32 versus 30.408296% INT8**, delta **+0.674012 points**,
with 187/400 identical texts. All 400 IDs and reference strings join the frozen
panel. The six-cluster descriptive one-sided 95% bounds are +0.192941 / +1.040892
points (10,000 replicates, seed 20261120); this is no new formal comparison.
The retained `a2-minmax-rust-report.json` under the bridge run root has SHA-256
`407827e032ec116eae9603caf14b39fcdae8248517e448b0940c25d814457de9`.
Source review of `e7c6efa` confirms bounded sampling during collection, actual
joiner input/context pairs, consumed-row metadata and distinct file versus
selection hashes. Exact source/data/payload provenance and new quality results
remain to be accepted. Follow-up review also requests immutable sweep directories
and correct v1.1 consumer handling of a floating-point joiner or per-row dynamic
quantization with `max_batch: 1`. Direct graph batching and effective consumer
invariance are different claims. No static artifact is accepted by this report.
It has not replaced the measured dynamic bundle. Native plumbing
remains experimental, and the verified 88.75% INT8 agreement is below the recorded
handoff target. No output-preserving equivalence is asserted.

The **first retained D2 step-5000 snapshot** has a separate FP32 research bundle,
`/mnt/xbabe3-external/jsr/bundles/r1-d2-s5000-fp32-research-20261120`. It is not the
selected D2 candidate. Existing A2 export orchestration verifies all trainer source
bytes against immutable `24c1b4b` before import and checks unchanged bytes afterward.
Every payload hash and the completed checkpoint hash are independently verified.

| D2 step-5000 artifact | SHA-256 |
|---|---|
| Source model checkpoint | `1ca4e893eff1c3b9af74e4d6e4823426113dbe2baa0fbd3a47a5ded3806523c7` |
| Bundle manifest | `3359fb0f7e221b527fbf6949c90db40a25947097911cb2fd768072991fd8cad0` |
| Export-source sidecar | `579f703966778ca4cdc70d6c3b328f9925c655c409ada844125342d5040d29ad` |

The reviewed older export helper also writes dynamic INT8 payloads; only FP32 is
loaded for this follow-up. Those inactive files have no new quality or acceptance
claim. This preserves one reviewed export path and gives the measured bundle a
distinct identity. The initial native/NumPy FP32 smoke comparison matches all
eight texts and scores 18.253968% over 126 reference words from one speaker. It
does not establish full-panel parity or a performance advantage.

## Measurement and claim limits

The completed primary pilot rows use the same 2559.535 seconds of prepared audio:

| System and path | Rust WER (%) | S / D / I | Whole-service CPU-s/audio-s | Warm CPU-s/audio-s | Aggregate peak RSS bytes | Sampled PSS bytes | Peak threads |
|---|---:|---|---:|---:|---:|---:|---:|
| R0 Zipformer INT8, native sherpa 2.0.2 / ORT 1.28.2 | 2.8516 | 184 / 14 / 22 | 0.137190 | 0.136190 | 238,551,040 | 234,956,800 | 1 |
| R1-D1 mixed INT8, NumPy / ORT 1.32.0 bridge | 29.8250 | 1572 / 591 / 138 | 0.140862 | 0.140561 | 592,748,544 | 575,327,232 | 7 |
| R1-D1 FP32, native Rust / ORT 1.32.0 service | 29.7343 | 1558 / 601 / 135 | 0.193986 | 0.193365 | 761,176,064 | 757,180,416 | 4 |
| R1-D2 step 5000 FP32, NumPy / ORT 1.32.0 bridge, unselected | 21.8795 | 1184 / 375 / 129 | 0.192121 | 0.191707 | 843,157,504 | 819,618,816 | 5 |
| R1-D2 step 5000 FP32, native Rust / ORT 1.32.0, unselected | 21.8795 | 1184 / 375 / 129 | 0.168629 | 0.168113 | 750,559,232 | 746,688,512 | 4 |
| Nemotron 560 ms Q8_0, completed-file CPU control | 2.1646 | 137 / 15 / 15 | 0.472127 | unavailable | 1,283,899,392 | 1,277,926,400 | 17 |

All six rows account for all 400 inputs, 7715 reference words and six speaker
clusters, with zero unfinished samples. R0/Nemotron decoder cap counts are unavailable;
D1 records one forced advance and D2 step 5000 records zero. The runtime paths and ORT versions differ, so this
does not isolate an architecture effect. CPU readings are single exploratory
observations, not a statistically established cost ratio. D1's mixed-INT8 bridge uses about 2.7% more
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
seven of eight paced texts match the unpaced run, and its score uses its actual
transcript. Guarded wall metrics,
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

The native FP32 extension is committed at
`24c1b4b72c838edf4a3805f4331fe532c57b9c58`. It uses the same audio-only job,
whole-service observer, Rust scheduler, output ledger and scorer as M1, calling
`OrtTransducerBackend` directly. Compatible encoder chunks are batched; processing
credit stays at zero until verified EOF drain. The wrapper records actual cap
counts, requires complete PCM digests and becomes permanently failed after any
protocol or inference error. A scoped deadline thread terminates a hung native
service with exit 124, causing an invalid run. Its resource cost is measured.
This raw-engine research wrapper does not qualify the deployment coordinator's
backpressure, hard-cap or endpointing behavior.

The release build verified clean, unchanged inference source before and after
compilation. Executable SHA-256 is
`77ce5d05beea1af73d25a4ed33b5a1a7b8efb242ae32cabadd3f06ca56feb22b`.
Build/compiler/lock provenance is retained in `m2-build-provenance-20261120.json`
(SHA-256 `7f72a4639bf7d133c38398d1ec85954ced8b7a9e9a1df26929af603ca51341d5`).
The actual loaded ORT library is version 1.32.0, SHA-256
`6c1585d1bbf45b0712ea4a21b705c5fcb4ba04ff2e30f1fbf81a49faf658638f`;
no Python runtime is used. Selected FP32 encoder/decoder/joiner files total
**233,405,026 bytes**. Runtime artifacts live under `m2-native-fp32-dev400`.

All **400/400 native FP32 transcripts match the retained FP32 bundle/trainer
transcripts**, with zero WER-point difference. The comparison preserves the
separate limitation on the reference execution's original provenance; it does not
repair earlier proof JSONs. Its comparison file is
`m2-native-fp32-reference-comparison.json` (SHA-256
`a0e32caf7d8db60d385a3d1c000d6c1e965deb96e97dbcf78583ab0a912797c4`).
Native `scores.jsonl` SHA-256 is
`d1d541260bb50400cad87eb02d12c64c1d929aedd4a3e6c9f9eea697a6001b80`.
The service used **496.515207 CPU-seconds**, including a 1.59-CPU-second post-load
snapshot split, and recorded one forced advance. The FP32 native and mixed-INT8
bridge rows differ in precision and runtime; their cost difference does not
isolate Rust-versus-Python overhead. Neither D1 point shows an advantage over R0.
Cold load here means a new service process; host filesystem caches were not flushed.

The native-versus-R0 paired report uses 10,000 replicates and the same seed:
**+26.8827 WER points**, with separate descriptive one-sided 95% bounds
**+23.4005 / +31.1733**. Six speaker clusters still give fragile bounds. This adds
no formal candidate-versus-reference claim; selected D2 confirmation remains open.
The paired file is `m2-r0-native-fp32-paired-20261120.json`.

Native eight-example A/A and repeat, 10/37 ms packetization and 20 ms paced checks
all produce identical Final/terminal committed text, SHA-256
`48c79da540e25bc2876eefc4efc076ad68e8ce00df019bda9a398b6498b59bfd`.
Each Rust score is 26.190476% over the same 126 words and one speaker; repetitions
add no independent support. All 1968 paced observations follow capture deadlines,
with minimum slack 0.000081228 seconds. This is an observation check, not a
packet-release trace, latency claim or next-day reproduction.

The checked four-row frontier plus native repetition evidence is
`m2-evidence-summary-20261120.json` (SHA-256
`42bdd7218097d5ad630e49cd3b46bdf9f33943ab1d9d6a7f4d5fc906d0038a37`).
Recreate it with `.venv/bin/python docs/ultra/reviews/verify-m2-evidence.py RUN_ROOT NEW_OUTPUT.json`.
It verifies artifact checksums, exact reference/audio joins, scorer normalization,
build identity and native repetition/capture observations. CQ/FC, capacity,
sustained operation and second-day reproduction remain explicitly unavailable.

The **D2 step-5000 native follow-up**, completed November 21, uses that same
retained executable, ORT library, audio panel and 8-stream/1-thread/20-ms job.
All 400 inputs drain successfully, with zero forced advances and **21.879456%
Rust WER** (S1184/D375/I129 over 7715 words). Whole-service CPU is
**431.612368 seconds**, including 1.32 CPU-seconds before the post-load snapshot.
Its directory is `m2-d2-s5000-native-dev400`; `scores.jsonl` SHA-256 is
`9522df422b9a87ea3377a48e6a1da37d1c33c53cda8945aa3e6bd4fa9c6808a9`.
This improves on D1's native FP32 error rate while still showing no observed
quality/cost advantage over R0. No architecture or confirmed speed claim follows
from these single shared-host observations, and D2 training continues.

The descriptive R0-versus-D2-snapshot paired report uses the same 10,000
speaker-cluster bootstrap replicates and seed. D2 minus R0 is **+19.027868 WER
points**, **+667.3% relative**, with separate one-sided 95% bounds
**+16.961618 / +21.745201 points**. Six clusters remain fragile support.
The file `m2-r0-d2-s5000-paired-20261120.json` has SHA-256
`9ab0956d3c1c8d1ea7ee66aedbfd2f3861d67fdb1a54874001704921be3c1a72`.
This is a first-snapshot research row, not the selected-D2 formal decision.

The full D2 FP32 bridge reference also completes all 400 inputs. **400/400
transcripts match native FP32**, with zero WER-point difference and byte-identical
source-sorted Final files, SHA-256
`464c11957da7c83e7aac8e8e58644706c5db9645bc2b6ab0c90cf686562b9530`.
Both actual Rust score files have the same hash quoted above. This establishes
native-versus-exported-consumer transcript equivalence on this panel; full
D2 checkpoint-versus-export per-example parity remains a separate open check.
The bridge uses **491.740227 CPU-seconds**, with a 1.06-CPU-second load snapshot
split, zero forced advances and five observed peak threads. Its directory is
`m2-d2-s5000-bridge-dev400`. At matched FP32 precision, native's observed CPU cost
is about **12.2% lower** in these two runs. Shared-host interference, co-located
observers and a single observation per path prevent an established speedup claim.
The bridge loads Python ORT's retained CPU pybind library; native loads the
separately hashed C API library, both version 1.32.0.

The follow-up's retained executable is built from `24c1b4b`, as bound by its
SHA-256 and the unchanged build record. The current runner's `plan.source_commit`
field reads the launch checkout's HEAD, which can be later than a retained
executable's build. It must not substitute for build provenance. The D2 native
run launched at clean `24c1b4b`; the subsequent bridge reference run launches
after documentation commits. The follow-up verifier distinguishes launch checkout
from compiled source, preserving the existing manifests rather than rewriting
their metadata. Future manifests need separate fields for these identities.

The new six-row frontier is `d2-s5000-evidence-summary-20261121.json`, SHA-256
`388f48bc65b84526dafe8532fae6ef329791e709d37840e89b59c979c03bf995`.
Recreate it with `.venv/bin/python docs/ultra/reviews/verify-d2-evidence.py RUN_ROOT NEW_OUTPUT.json`.
It verifies every inference/scoring checksum, actual Final-to-outcome-to-score
hypothesis joins, exact audio/reference populations, source checkpoint/export
payloads, immutable export source bytes, native build identity, training-prefix
totals and actual full-panel native/NumPy parity. It retains every mismatch if
one occurs and distinguishes the minimum handoff gates from exact equivalence.
The paired scorer executable is retained with
`m2-r0-d2-s5000-paired-provenance.json`; its original standalone build source was
not captured. Rust normalization and input/output hashes remain bound. The prior
M1/M2 frontier files are unchanged. All qualification-dependent fields remain null.

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
Peak process count is unavailable in the current receipt; it is not inferred
from the configured process tree or the peak thread count.

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
The M2 code committed as `24c1b4b` passes the full gate on merged `29b1b2c` plus
those exact changes: native batch/partial-chunk and deadline tests, hardened
native parity/failure tests, 105 Python tests/one skip, contracts, security and
audit 85/zero caps. The release build then verifies unchanged clean source.
Parallel CPU-burner timing assertions failed on the restricted shared build cores
in an earlier attempt; serial test scheduling passed without changing thresholds.

The runtime owner retains native R1 and common-suite implementation. Independent
execution on merged `6dc7f0d` verifies frontend parity against the actual
BundleRecognizer on five edge waveforms and 50 stratified DEV utterances:
31,772 frames, maximum absolute difference **7.153e-7**, below 1e-4. The native
sherpa ABI/decode tests, checked result/free tests and coordinator fixture pass.
The missing-result-to-empty path now returns an error. B1's later default native
parity test also passes with the small-fixture results stated above. The subsequent
`dd5078c` delivery adds canonical bundle containment, checked state-size arithmetic,
unique create-new parity outputs and explicit original acceptance booleans.
Merged `29b1b2c` adds finite cache checks, bounded integer cache lengths, deferred
cache/position updates after successful greedy decoding, and terminal handling of
native/coordinator failures. Its injected late-failure regression and the updated
native parity test execute successfully in the independent full gate. The state
counter at that revision includes encoder caches and major workspace allocations;
the later `097352b` delivery adds the excluded frontend/shape-vector allocations.
Aggregate measured service memory remains the authority. The lead retains D2
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
independent injected-clock fixture passes. The `daf50bb` integration connects
production runner/suite callers to a shared backend/coordinator clock. Default
factories still require callers to select the shared-clock API explicitly.
Controlled EOF remains distinct from automatic
endpointing and successful forced segmentation. M1 timestamps/drain checks are
independent of that coordinator, so its quality/cost rows retain their scope.

The newly merged `ultra bench-suite` and native Nemotron adapter remain
experimental. `29b1b2c` implements reference-blind latency input scheduling,
post-inference timing joins, consistent precision across subjobs, explicit
engine/service cost scopes, refusal of unverified qualification, positive
CPU-library evidence, retained actual artifacts and file-backed child-output
capture. Independent integration tests pass. Remaining WP5 scope includes
complete immutable per-job manifests, create-new suite artifacts, explicit
controlled-EOF labels and correctly scoped timing populations. At that revision,
GGUF hashing allocates the whole file before its resource window. The later
`097352b` delivery adds 64-KiB streaming hashes, preparation/child CPU scope,
per-job manifests, create-new artifacts, controlled-EOF/population labels and
separate build/launch source fields. These mechanisms are independently reviewed
in source; the lead retains integration gates. The new v1.1 test has eight source
examples, so `streams=16/32` settings do not exercise that many active instances.
Follow-up requests cover actual 16/32-stream fixtures, serialized encoder-call
counting, effective observer placement and fail-closed topology validation.
Declared disjoint CPU sets alone do not demonstrate an observer was pinned.
Common-suite engine cost rows
do not replace the measured bridge service rows or establish qualification.

The accepted plan's remaining requirements include D2's fixed clean/other DEV
selection, native INT8 same-precision parity, reference timing audits, common-suite
fault tests, five paired timing blocks, both real one-hour runs and next-day
reproduction. Training MR on other hosts is now an owner-authorized background
activity in the coordination log; it remains outside the initial freeze and
needs its matched step-7800 control and separate experiment identity.
The recorded data ruling blocks TED-LIUM 3 on rights; it was not fetched or
silently replaced. The MFA listening audit remains open. These dependencies do
not supply missing domain-shift or latency evidence to this DEV pilot.

The subsequent owner instruction activates an aggressive CPU port with Rust
preferred. The lead's [engine plan](../ENGINE_PLAN.md) assigns a fused Rust engine
and a separately improved ORT control. The independent
[acceleration review](2026-11-21-cpu-acceleration-review.md) records reference-blind
profiles, actual ISA/topology and acceptance constraints. The first diagnostic
attributes 87.14% of reference-consumer process CPU to the encoder; these are
small-panel diagnostic fractions, not native or whole-service measurements.
No accelerated engine speedup or deployment qualification is established yet.
