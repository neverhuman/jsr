# H54: v3 release qualification

**Status: COMPLETE for v3.0.0 (paper-lane, 2026-11-23 15:45Z; exploratory, DEV only).** H54 registers no gates of its own. It collects the receipts behind the v3.0.0 release decision, measured against gates that other lanes registered before the data existed:
- release gates E1-E4: eval-lane, 09:02:20Z;
- S2 gates (a) and (b) and the release-candidate rule: eval-lane, 10:04:38Z;
- gate (b) Amendment 1: eval-lane, 11:23:12Z;
- warm-AGC criteria: claude-lead, 09:34:43Z, amended 11:48:10Z;
- release-decoder settings: claude-lead, 11:12:38Z.

Every number is **exploratory, LibriSpeech DEV only**. No test-set number appears here. LibriSpeech test-clean / test-other were read several times before the freeze, per the disclosure in [`FREEZE.md`](../../docs/ultra/FREEZE.md) §4. The release notes are [`docs/ultra/releases/v3.0.0.md`](../../docs/ultra/releases/v3.0.0.md).

## Question
Does one frozen artifact (weights + decoder + scorer) pass every registered v3 release gate? And what exactly ships as v3.0.0, in what default state, with which limits?

## Setup
- **Artifacts:**
  - **A:** B0 (A1 step 49,000 EMA, INT8 `4701787c…`) + the release decoder. This is the fallback.
  - **B:** the S2 release candidate (RC1, `H53_RC1_s7919`) + the release decoder. This is the preferred artifact.
- **Release decoder:** `ultra-engine-commit`, binary sha256 `9a142e8e…` (main bb1071a).
  - **Stream:** greedy, chunk 4.
  - **Commit pass:** chunk 16, beam 16, M5 / LODR 0.80 / 0.60, tail pad 320 ms, 40 s cap. M5.ulm `72a621c2…`.
  - **Scorer:** `ultra-score` `d784f8ca…`.
- **Paired statistics:** against B0 INT8's 8,773 edits. Paired speaker-cluster bootstrap, 10,000 replicates, seed 20261122. Brackets are one-sided 95% bounds unless a row says otherwise.
- **Hosts (role labels):**
  - **WER:** full dev on x2 (runtime-lane) and x0 (bench-open), and on H0 and the 24 GB cards (eval-lane).
  - **Timing and energy:** quiet holds on x0 (Zen 2, one core, the owner's timers paused), with energy's 1 Hz log.
- **The release decision** (eval-lane's plan, 09:02:20Z):
  - a row is eligible if it passes E1 and E2;
  - the bundle is judged against B0;
  - effect sizes are never added across rows.

## Results: qualification receipts
Paths are on the data drive unless they start with `research/` or `docs/`. Full sha256 values are in [`MANIFEST`](MANIFEST).

bench-open's x0 receipts are also committed in the repo, byte-identical to the files cited below, under [`docs/paper/data/raw/opponents/bench-open-release/`](../../docs/paper/data/raw/opponents/bench-open-release/README.md) (PR #109). That covers the 14g / 14i / 14k / 14m runs, their joins (`joins/14m.jsonl`, `joins/14i_q_cpu_summary.json`, `joins/14k.jsonl`, `joins/14fg_power_holds.jsonl`) and the release-eval rows.

| # | Check | Registered rule | Artifact | Result | Verdict | Receipt (sha256 prefix) | Posted |
|---:|---|---|---|---|---|---|---|
| 1 | **E1, WER** | ≤ 7,895 edits; pooled upper < 0; dev-clean and dev-other uppers ≤ +0.20 pp | A (binary 9a142e8e, 40 s cap) | **Final text, utterance-segmented: 6,455 edits = 6.087%** (3.236 / 9.121). The live stream is B0 INT8 text for text: 8,773 edits = 8.272%. Pooled −2.186 [−2.435, −1.943]; dev-clean −1.225 [−1.427, −1.043]; dev-other −3.209 [−3.702, −2.747] | **PASS** | `evals/eval-lane/h41/verdicts/release-row-B0-CP-LM-pad320-bb1071a.json` (0c11b857). Input `runtime-lane/release_bb1071a_pad320_fulldev/…committed.score-input.jsonl` (65fb4ddd). Run receipt `…/release_bb1071a_pad320_fulldev.receipt.json` (d860ea86) | eval-lane 12:08:17Z |
| 2 | Second-host reproduction | same text from a second host | A on #106's binary (fb8a95e5, 30 s cap) | 6,471 edits on x0, equal edit for edit to runtime-lane's row on that binary. The 40 s cap (row 1) changed 10 long utterances, for 16 fewer edits | reproduced | `opponents/bench-open/release-eval/rows/B0-A1-49000/ROW.json` (137376ab) | bench-open 11:49:14Z |
| 3 | INT8 parity of the LM final pass | native minus PyTorch within noise | #106 binary, C half, pad 0 | 5.981% against PyTorch's 5.975%: +0.0056 pp [−0.043, +0.057]. With the LM off, the commit path is byte-identical to #76 on 100 / 100 | parity | `runtime-lane/pr106_c_half_parity/pr106_c_half_parity.json` (2125f120) | runtime-lane-h45 11:24:00Z |
| 4 | **E2, latency** | protected paced first-correct-word p95 ≤ +20 ms over B0, median not higher | A | The provisional stream equals B0 INT8 text for text (5,567 / 5,567), so E2 is B0's stream by construction | **PASS (by construction)** | row 1's receipt | eval-lane 12:08:17Z; claude-lead 11:33:32Z |
| 5 | E2 for streaming L4 (14k) | same | B0 + streaming L4 | p95 **+40.1 ms** [−159.8, +320.4]; median −0.0 ms | **FAIL**, so L4 is not a release option | `evals/eval-lane/h43/protected-e2/14k-pooled-l4-vs-b0.paced.json` (c9e0d524) | eval-lane 11:33:00Z; claude-lead 11:33:32Z |
| 6 | **Timing, memory, energy (14m)** | quiet x0, 1 core, 730-utterance panel, 3 rounds | A | Medians of 3 clean rounds: **0.0702 CPU-s per audio-s** (stream 0.0339 + final 0.0360) = **14.3×** (rounds within ±0.15%); **686 MiB**; final text p50 / p95 / max **210 / 589 / 1,127 ms**; core / package **0.72 / 1.79 W** per stream. The rep run after the 12:31Z CI lapse was repeated | measured | `opponents/bench-open/watts-join/14m.jsonl` (90a58ddd; all 9 reps with input hashes). Per-run dirs on H0 `/dev/shm/bench-open/xbabe0/runs-v3/1c-assay/cold/round-N/commit-*/` | bench-open 14:01:25Z |
| 7 | Long-session cost | flat over long sessions | A's binary; `CommitPassSession` at bb1071a | 14m: 0.0696-0.0701 CPU-s per audio-s in each 10-minute window of an 87.7-minute session (730 utterances, a full reset after each); slope +0.1% per hour. runtime-lane: a 35-minute continuous session with no full reset (270 endpoints). RSS flat at 919 MB; steady state 0.074-0.075 CPU-s per audio-s; the committed text reached 27 KB. On a quiet core the engine scales linearly (3.000× the audio took 2.993× the CPU) | **flat** | the 14m runs' `commit.jsonl` (hashes in `14m.jsonl`); `runtime-lane/release_bb1071a_35min_exposure/exposure_35min_summary.json` (151f741f; its SHA256SUMS checks 8 / 8) | bench-open 13:35:58Z; runtime-lane 14:10Z; claude-lead 14:11:28Z |
| 8 | S2 gate (a), S2a-BP1 | stop if chunk 4 is > +0.20 pp worse than B0 | S2a step 5,500, EMA | 7.995% against 8.302: −0.307 [−0.402, −0.215] | CONTINUE | `evals/eval-lane/h41/verdicts/s2-gate-a-BP1.json` (fa856783) | eval-lane 11:14:44Z |
| 9 | S2 gate (a), S2a-BP2 | same | S2a step 10,500, EMA | 7.997%: −0.306 [−0.388, −0.223] | CONTINUE (plateau) | `…/s2-gate-a-BP2.json` (deb7b6df) | eval-lane 12:21:34Z |
| 10 | S2 gate (a), S2b-BP1 | same | S2b step 5,500, EMA (source 62cf5017…) | 7.584%: −0.719 [−0.839, −0.601] | CONTINUE | `…/s2-gate-a-S2b-BP1.json` (c31d9fff) | eval-lane 13:16:04Z |
| 10b | S2 gate (a), S2b-BP2 | same | S2b step 10,500, EMA (source 08670104…) | 7.490%: −0.813 [−0.929, −0.702] | CONTINUE | `…/s2-gate-a-S2b-BP2.json` (d67e6bdf) | eval-lane 14:04:35Z |
| 11 | **S2 gate (b), stage 1** (Amendment 1) | intersection-union test at one-sided 97.5%: c4 upper < 0, c8 upper ≤ 0.094, c16 upper ≤ 0.20 | S2b-BP1 stable EMA | c4 −0.719 [upper −0.583]; c8 −0.775 [−0.649]; c16 −0.707 [−0.583] | **PASS**: S2 is the release model, RC1 goes ahead | `…/gate-b-S2b-BP1.json` (1f8c6b42) | eval-lane 13:24:51Z; claude-lead 13:25:12Z |
| 12 | Branch point with the release decoder (reference) | none (context) | S2-BP1 EMA INT8 (0f35bfb1) | 5.898% (6,255 edits): −0.189 [−0.282, −0.096] against row 1 | reference | `opponents/bench-open/release-eval/rows/H53_S2_BP1-ema/ROW.json` (00bdc4bb), `paired_vs_B0_release_bb1071a.json` (44240ccb) | bench-open 12:26:24Z |
| 12b | Trajectory, release decoder (reference) | none (context) | S2a-BP2 EMA INT8 (4215fe4e) | 5.883% (6,239 edits): −0.204 [−0.312, −0.100] against row 1. Live stream 7.988% | reference | `runtime-lane/release_bb1071a_s2a_bp2_fulldev/summary.json` (86040131; its SHA256SUMS checks 9 / 9) | runtime-lane-h45 14:32:54Z |
| 12c | Trajectory, release decoder (reference) | none (context) | S2b-BP1 EMA INT8 (3ddc1419) | **5.492%** (5,824 edits): −0.595 [−0.733, −0.464] against row 1. Live stream 7.579% (8,038). The lead's trajectory line is B0 6.087 → S2-BP1 5.898 → S2b-BP1 5.492, full dev, same binary | reference (an uncooled branch point, not a release row) | `runtime-lane/release_bb1071a_s2b_bp1_fulldev/summary.json` (1e444d13; its SHA256SUMS checks 10 / 10) | runtime-lane-h45 14:26:35Z; claude-lead 14:27:33Z |
| 13 | **Candidate rule + A6 look** | EMA unless raw is ≥ 0.05 pp better on S. Then on the confirmation half: upper < 0. On full-dev INT8: pooled upper < 0, splits ≤ +0.20. Latency: median not higher, paired p95 ≤ +20 ms, guards within 0.50 pp. CPU ≤ 1.05×. INT8 minus PyTorch within ±0.05 pp. c16 upper ≤ +0.20 | B (RC1) | **Candidate = RC1's EMA**: raw − EMA on S = +0.027 pp (raw needed ≤ −0.05). **WER items of the A6 look:**<br>• confirmation half, chunk 4 (PyTorch): −0.889 [−1.086, −0.702], p 0.0001, which survives Holm;<br>• native INT8, chunk 4, full dev: **7.447%** against 8.272%, −0.825 [−0.962, −0.694]; dev-clean −0.534 [upper −0.425], dev-other −1.135 [upper −0.890];<br>• INT8 − PyTorch: −0.003 pp;<br>• chunk 16, full dev: −0.771 [−0.875, −0.667].<br>PyTorch EMA on full dev: 7.450 / 7.110 / 6.931 at chunks 4 / 8 / 16. **Latency items** (paced full dev, design end clock, paired against B0):<br>• first correct word median 360 → 190 ms (−170 [−170, −160]);<br>• p95 480 → 420 ms (−60 [−70, −50]);<br>• first-word coverage +0.90 pp [+0.47, +1.35];<br>• first-word error −0.90 pp;<br>• correct-by-300 ms stable +39.3 pp [+38.1, +40.4];<br>• first token mean 312 → 166 ms.<br>**CPU item:** 0.0705 against B0's 0.0702 CPU-s per audio-s on the same binary = 1.004× (limit 1.05×; 14p and 14m) | **A6 PASS** on every registered item (weights: EMA) | `evals/eval-lane/h41/verdicts/rc-RC1-*.json`: weights-rule fc4376cc, A6-conf-c4 6012ad56, A6-int8 dbe4ed3e, A6-c16 21388c07, fulldev-c4 5c3d5dc3, fulldev-c8 597e922a. Latency: `evals/eval-lane/h43/latency-fulldev/RC1-c4-vs-B0-c4.json` (a88f107d), `RC1-c4.json` (b62ce43e). Export `evals/eval-lane/export/H53_RC1-ema/int8/` (weights.int8 edac19d8) | eval-lane 15:02:09Z, 15:10:13Z, 15:38:39Z |
| 14 | **E1 on the candidate** (final text, utterance-segmented; the live stream row beside it) | as row 1 | B: RC1 EMA INT8 (edac19d8) + binary 9a142e8e | **Final text, utterance-segmented: 5,808 edits = 5.476%** (2.922 / 8.197). Against B0 INT8: −2.796 [−3.119, −2.479]; dev-clean −1.540 [−1.787, −1.315]; dev-other −4.133 [−4.766, −3.536]. **Against row 1 on the same binary: −0.610 [−0.737, −0.489].** The live stream is 7,898 edits = 7.447% (3.928 / 11.195), equal to RC1's native INT8 decode on 5,567 / 5,567. paper-lane got its splits by scoring the provisional file with the pinned scorer, and the pooled value matches eval-lane's A6 row | **PASS** | `evals/eval-lane/h41/verdicts/release-row-RC1-release-decoder.json` (55e75640); `…/release-row-RC1-vs-B0-release-row.json` (6ed83245). Inputs: `evals/eval-lane/rc/RC1/release-decoder/release-decoder.committed.score-input.jsonl` (e293e4fd), `…provisional.score-input.jsonl` (434803dd) | eval-lane 15:12:19Z |
| 15 | **Candidate timing (14p)** | as row 6 | B: RC1 EMA INT8 (edac19d8) + binary 9a142e8e | Medians of 3 clean rounds: **0.0705 CPU-s per audio-s** (stream 0.0342 + final 0.0360) = **14.3×**; **686 MiB**; final text p50 / p95 / max **211 / 586 / 1,126 ms**; core / package **0.72 / 1.64 W** per stream; 42.5% of segments revised. B0 is the same within 0.5% (row 6) | measured | `opponents/bench-open/watts-join/14p.jsonl` (7f7eec35; per-rep `commit.jsonl` 918297cd / c0e751f3 / 76d7f2ea) | bench-open 15:40:13Z |
| 16 | Sentinel-0 CPU (14i): Q2, Q6 | Q2 gated / ungated ≤ 0.02 on non-speech; Q6 ≤ 1.05 on speech | #77 session 5ca3583, replay 500 / idle 1,500 | Q2 0.0036; Q6 1.003. On the 1 h, 5%-duty cell: 0.029 core W per stream against 0.50 W for C1 (−94%); package about 0.09 against 1.59 W | **PASS** (Q2, Q6) | `opponents/bench-open/watts-join/14i_q_cpu_summary.json` (6f938e7e) | bench-open 12:16:50Z |
| 17 | Sentinel-0 Q4b, tier B torture | first-word recall, gated − C1: point < −0.5 pp fails | same | **−0.81 pp** [−1.40, −0.22] on 13,400 items; N10 noise −2.26 pp | **FAIL**, so the gate is opt-in | [`research/H47_always_on_sleep_gate/results/Q4b_tierB_torture.json`](../H47_always_on_sleep_gate/results/Q4b_tierB_torture.json) (e5e4b825) | h47-sleep 10:34:40Z; claude-lead 10:35:26Z |
| 18 | Sentinel-0 Q5, added false words | Garwood one-sided 95% upper ≤ 0.10 per hour | same | 0 words in 30 h of non-speech; one added word on the stream's gaps gives an upper bound of 0.1008 per hour | NOT PASSED (by 0.0008 per hour) | [`research/H47_always_on_sleep_gate/results/Q_5ca3583_r500_i1500_Q5_30h.json`](../H47_always_on_sleep_gate/results/Q_5ca3583_r500_i1500_Q5_30h.json) (bb1e0f7a) | h47-sleep; claude-lead 09:39:33Z |
| 19 | Sentinel-0 Q5b (≥ 50 h fresh) | as Q5 | same | no result posted. It was running on the Mac when h47-sleep went unstaffed (10:43Z) | open | none | none |
| 20 | Warm AGC C1 + C2 (fresh 730-utterance panel) | C1: ≥ 80% recovery of WER and first-word loss at −20 / −30 dB. C2: 0 dB WER upper ≤ +0.20 pp, first-word error ≤ +0.5 pp | binary 02e1fa4c (#108), warm manifest mode | Recovery 121% / 101% (WER), 102% (first word at −30 dB); 0 dB +0.034 pp [upper +0.120]; AGC off is identical on 2,920 / 2,920 | **PASS** (warm). Cold per clip, reported only: C1 FAIL (12-13%), C2 pass | `evals/eval-lane/h43/gain-panel-fresh/v2-02e1fa4cd1f7/primary/criteria_1_2.json` (ee627a19); `…/secondary/criteria_1_2.json` (e598d99a) | eval-lane 12:59:42Z |
| 21 | Warm AGC C3, the 27.56 h stream | \|on − off\| ≤ 0.087 pp, no new duplicates | same, 164 pieces of about 10 min, warm within each (design registered at 13:04:13Z) | 11.559% on against 11.112% off: **+0.447 pp**; inserted repeats 10 against 8. 121 of 164 pieces worse | **FAIL**, so warm AGC is opt-in | `evals/eval-lane/h43/agc-c3/pieces-02e1fa4cd1f7/criterion3.json` (e65699cf) | eval-lane 14:05:21Z; claude-lead 14:05:37Z |
| 22 | Warm AGC C4, CPU | ≤ 1% added engine CPU, quiet x0 | same | not run: dropped as moot after C3 failed | n/a | none | claude-lead 14:05:37Z |
| 22b | Chunk-coalesced wakes (#105), 14o | per-packet stays the default until a paired quiet-host receipt exists (claude-lead 11:11:19Z) | `ultra` 296f469c, one paced B0 stream, 300 s speech + 300 s pink | Round 1: CPU −0.6% on speech (0.0622 → 0.0618), −0.8% on pink; core watts the same within noise. Not design-clock identical: the 30 s cap closure moves by one revision, and the segmentation and final text differ | **opt-in, experimental** | x0 run dirs `round-N/wake-ab/`; *pending* a data-drive join (`join_14o.py` exists). Rounds 2-3 are for the record | bench-open 14:46:46Z; claude-lead 14:47:10Z |
| 23 | Level sensitivity (B0) | the gain-stress trigger: > 2× spread | B0 INT8, 160 ms, 730-utterance panel | −20 dB **+1.14 pp** [+0.87, +1.41]; −30 dB **+3.05 pp** [+2.58, +3.54]; first-word error 8.49 / 11.51 / 16.44% | trigger fired, so it is listed as a limit | `evals/eval-lane/h43/gain-panel/summary.json` (d19b35fd) | eval-lane 09:17:59Z |
| 24 | Paced live energy (14f) | none (context) | B0 greedy, `ultra stream --pace 1.0` | core 0.545 W (speech) / 0.500 W (pink −30 dBFS) per stream; package 2.04 / 1.97 W; 0.063 CPU-s per audio-s | reference | `opponents/bench-open/watts-join/power_holds_14f.summary` (2a83180f) | bench-open 09:09:50Z |
| 24b | Pseudo-label tail truncation (found after qualification) | none (a defect found after the gates) | R0 labeller v1.1 (0.3 s pad, full chunks only) against v1.2 (0.8 s pad + sherpa's chunk rule) | DEV S, R0 greedy: 6.476% (v1.1) against 5.680% (v1.2), +0.8 pp; last reference word missed on 582 against 170 utterances. On a 1,600-row Libri-Light tier-A sample, 2.44% of rows gain one final word with v1.2. v3.0.0 (small + medium) was trained on v1.1 labels | **limit, disclosed** (the fix is v3.1, H58) | label conventions in the H52 record (`CONVENTIONS.json`) and beside the raw shards (`LABEL_CONVENTIONS.json`); no data-drive receipt for the S check was posted | h52-librilight 16:35:50Z; claude-lead 16:36:19Z and its message to paper-lane |
| 25 | Libri-Light data | overlap verdicts logged before use; sealed manifests | small and medium, v1.1 R0 labels | small: 99,109 utterances, 378.65 h. Medium: 888,265 utterances, 3,395.81 h. Overlap verdicts against all four dev and test sets are in `docs/ultra/LOOKS.jsonl` | sealed | small `SHA256SUMS` 8f371a65; medium `SHA256SUMS` 7e87f000; H52 README and MANIFEST | h52-librilight 11:44:38Z |

## Not done for v3.0.0
- **A live endpointer for the commit pass.** Every final-text WER is utterance-segmented, with the LibriSpeech segment boundaries as endpoints. The release notes report the live stream WER beside it. Integration is the first v3.1 item (claude-lead 14:09:53Z).

These are items from the v3 plan's Phase 9 ([`H43 PLAN`](../H43_v3_baseline_and_measurement/PLAN.md)) that have no receipt yet:
- **Whole-device watts by duty level** on the owner's metered target device. The device is still pending from the owner.
- **The gross 5%-duty energy endpoint (E3)** from protected paired holds.
- **A 24 h paced soak** of the release artifact.
- **The overnight gated matrix:** B0 / new weights, each with and without the sleep gate, with the compact LM and with the PQ kernels.
- **Competitor rows beside the release artifact.** The paper's table draws ULTRA as A1 step 30,000 on a greedy stream. The release artifact has no row there yet.
- **The opt-in features composed with the commit pass in one binary.** Each feature has its own entry point (see the release notes).

## What went wrong or surprised us
- **The streaming LM looked like the release path until E2.** It costs only 2% more CPU, but its p95 first-correct word fails the +20 ms rule. That p95 is fragile at n ≈ 73. Keeping the LM out of the stream and in the endpoint commit pass sidestepped the question and gave a larger WER gain.
- **A cold AGC is not a warm AGC, and a speech panel is not a stream.** #104's first binary failed C1 because it ran one process per clip. The fixed binary (#108) passed C1 and C2 in warm manifest mode, then failed C3 on the 27.56 h stream (+0.447 pp). The likely cause is that its level estimate tracks the noise gaps between utterances. The first word after power-on is still uncorrected.
- **More LibriSpeech alone made B0 worse** (A1C +0.08 to +0.19 pp). S2's gain over B0 is measured only as a whole row: the new data, dp 0.005, the extra training and the cooldown together. dp's own WER effect was never isolated, because the factorial was cut.
- **A long-session scare turned out to be load, not code.**
  - At 13:04Z, two timings of a never-reset stream (25.9× real time at 600 s, 15.3× at 1,800 s) were fitted as super-linear cost.
  - On a quiet core the engine scales linearly: 3.000× the audio took 2.993× the CPU. The two points had been taken under different H0 load.
  - eval-lane withdrew the finding at 14:11Z. The C3 result it came with stands.
- **The pseudo-labels lost some final words.** After qualification, h52-librilight found that the v1.1 R0 labeller dropped the final partial chunk. On DEV S that costs R0 0.8 pp; 2.4% of Libri-Light rows are affected. v3.0.0 was trained on those labels, so the release notes list it as a limit, and the relabelling is v3.1 (H58).
- **Operations:** the D0 / D1 DiLoCo proof failed twice on a transport defect, and its runs were voided. An x0 runner crash let the owner's CI jobs run for 88 s (12:31Z). The one 14m rep whose idle floor followed them was set aside and repeated.

## Decision / consequence
- **v3.0.0's decoder** is the greedy chunk-4 stream with the chunk-16 commit pass (beam 16 + M5 / LODR). Streaming L4 stays a documented row only.
- **How the commit pass ships:** as the `CommitPassSession` API and the `ultra-engine-commit` eval CLI, with no live endpointer. The headline WER is labelled "final text, utterance-segmented", and the live stream WER is reported beside it for each artifact.
- **v3.0.0's model** is S2's RC1 EMA: row 13 passed on every item.
- **The Sentinel-0 gate** ships opt-in. **Chunk-coalesced wakes** ship opt-in and experimental (row 22b).
- **Warm AGC** ships opt-in: C3 failed (row 21).
- **The tag** waits for the notes PR to merge. claude-lead and v3-integrator tag it; paper-lane does not.

## Pointers
- Release notes: [`docs/ultra/releases/v3.0.0.md`](../../docs/ultra/releases/v3.0.0.md).
- Receipts with full sha256: [`MANIFEST`](MANIFEST).
- **Training:** [H53](../H53_v3_trunk_and_capacity/README.md) (Amendments 5-7, 6.1 and gate (b) Amendment 1; the data-drive copy of AMENDMENTS.md carries the later ones).
- **Data:** [H52](../H52_libri_light_semi_supervised/README.md).
- **Decoder:** [H45](../H45_zero_training_wins/README.md), [H46](../H46_lm_and_text/README.md), [H42](../H42_ngram_lm/README.md).
- **Always-on:** [H47](../H47_always_on_sleep_gate/README.md). **Baseline and measurement:** [H43](../H43_v3_baseline_and_measurement/README.md).
- Scoreboard feed: `/mnt/xbabe3-external/jsr/v3-scoreboard/scoreboard.json`. Its curated rows predate the native release-decoder row, so for that row use row 1 above.
- Coordination: `AGENT_CHAT.md` (local). The posts are cited by role and UTC time.
