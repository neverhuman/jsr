# R10 v4.2.1 release energy campaign: pre-registration

Written 2026-11-27 12:12 PM MST, before any R10 measurement run. Lane: release/v4.2.1.

## Question
Single-core CPU-seconds, idle-subtracted core energy per audio-second and peak RSS of the isosyll v4.2.1 live mode
(v4.2.0 live path plus the research H318 endpoint rerank: the frozen H261 content-word reranker with the
fitLive_s375 weights, run only when the endpoint beam's top1-top2 score margin is below 3.75, about 59% of
endpoints), beside the v4.2.0 live mode and the R7/R8 anchors; and the added endpoint delay.

## Method (R8/R9 unchanged)
- Tool, host, core, idle windows, thread limits, locks and rerun rule exactly as R8/R9 (`runs/r8/PREREG.md`,
  R7 method: energy_bench.py run, xbabe0, pinned CPU 16 with SMT sibling idle, observer CPU 31, 60 s idle before
  and after, one repetition, QUIET.lock per arm, TIMING.lock only when no CI VM runs, one in-place rerun of a
  core-unclean arm). The campaign holds the shared xbabe0 meter lease (`/dev/shm/jsr-h172-meter.lease`).
- Arms, all on the full 730-utterance panel, three counterbalanced blocks (orders fixed in campaign.sh):
  anchor730-live (RC2 live +LM, R7 binary), anchor730-zip (Zipformer INT8, R7 binary),
  v42-live-730 (published isosyll v4.2.0 x86_64-linux binary, `transcribe --json --mode live`, v4.2.0 model),
  v421-live-730 (isosyll v4.2.1 glibc build of release/v4.2.1 ca4c230b, same command, v4.2.1 model = v4.2.0 model
  plus live-content-fit.json, fitLive_s375 sha 5a36e403).
- Reported per arm: CPU-s/audio-s = median over the three blocks; J/audio-s and RSS = core-clean blocks only.
  Paired ratio v421/v42 = median over blocks of the within-block ratio.
- Endpoint delay: `isosyll transcribe` reports per-file wall_s; one process decodes the panel in order on the pinned
  core, so the per-utterance paired difference v421 - v42 (same block, same file, first file dropped because it
  carries the lazy model load) is the added endpoint work. Reported: median and p95 of the added delay over the
  729 files, pooled over the core-clean blocks, and p95 of per-file wall_s for each arm.

## Validity rule
The block-1 anchors must reproduce R6 within +/-5 percent: RC2 live +LM 0.328 J/audio-s (0.3117 to 0.3444) and
Zipformer INT8 0.573 J/audio-s (0.5444 to 0.6017). If either misses, R10 is VOID: report the numbers as they are,
do not tune, and do not colour the hero figure from R10.

## Budget statement (fixed before measuring)
Research H314 (codex, paced, ARM xbabe4, skip 3.75) measured +2.97% / +2.92% process CPU and endpoint p95
+1.63 / +2.61 ms. Jepson chose skip 3.75 ("the timed 59% setting"). R10 reports the v421/v42 ratio and the delay as
measured, with no retuning.

## What this does not claim
CPU-reported core registers, not wall power; one core of an otherwise idle 64-core part (boost on). Batch decode of
recorded files, not a paced microphone; the added delay is compute on one core, not network or audio buffering.
The timed v4.2.1 binary is built from ca4c230b; the release binary differs only by later docs commits (same code).
