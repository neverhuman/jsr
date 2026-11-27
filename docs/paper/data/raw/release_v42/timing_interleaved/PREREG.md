# R9 v4.2.0 release energy campaign: pre-registration

Written 2026-11-27 (MST), before any R9 measurement run. Lane: release/v4.2.0.

## Question
Single-core CPU-seconds and idle-subtracted core energy per audio-second of the isosyll v4.2.0 final mode
(F3 soup c32 B16, TRAIN-30% confidence-gated B48 re-search, exact H261 content-word t2 rerank), beside the
v4.1.0 final mode and the R7/R8 anchors, and the cost of the confidence gate alone.

## Method (R8 unchanged)
- Tool, host, core, idle windows, thread limits, locks and rerun rule exactly as R8 (`runs/r8/PREREG.md`, R7 method:
  energy_bench.py run, xbabe0, pinned CPU 16 with SMT sibling idle, observer CPU 31, 60 s idle before and after, one
  repetition, QUIET.lock per arm, TIMING.lock only when no CI VM runs, one in-place rerun of a core-unclean arm).
  The whole campaign holds the shared xbabe0 meter lease (`/dev/shm/jsr-h172-meter.lease`).
- Arms, all on the full 730-utterance panel, three counterbalanced blocks (orders fixed in campaign.sh):
  anchor730-live (RC2 live +LM, R7 binary), anchor730-zip (Zipformer INT8, R7 binary),
  v41-final-730 (published isosyll v4.1.0 x86_64-linux binary, `transcribe --json --mode final`, v4.1.0 model),
  v42-final-730 (isosyll v4.2.0 glibc build of release/v4.2.0, same command, v4.2.0 model),
  v42-nogate-730 (timing control: the same source with the confidence gate off; never shipped).
- Reported per arm: median over the three blocks of CPU-s/audio-s and J/audio-s (core-clean repetitions only).
  Paired ratios: median over blocks of the per-block ratio v42/v41 and v42/v42-nogate (the gate's own cost).

## Validity rule
The block-1 anchors must reproduce R6 within +/-5 percent: RC2 live +LM 0.328 J/audio-s (0.3117 to 0.3444) and
Zipformer INT8 0.573 J/audio-s (0.5444 to 0.6017). If either misses, R9 is VOID: report the numbers as they are,
do not tune, and do not colour the hero figure from R9.

## Budget statement (fixed before measuring)
The research budget for the gate was +5% process CPU over the ungated F3 B16 final pass. Jepson locked B48 at the
TRAIN-30% gate knowing research measured +5.8% / +5.9% on xbabe4. R9's v42/v42-nogate ratio is reported as measured,
over or under +5%, with no retuning.

## What this does not claim
CPU-reported core registers, not wall power; one core of an otherwise idle 64-core part (boost on).
