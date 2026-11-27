# R8 v4.1.0 release energy campaign: pre-registration

Written 2026-11-27 (MST), before any R8 measurement run. Lane: release/v4.1.0.

## Question
Idle-subtracted core energy (J per audio-second) and single-core speed of the v4.1.0 model (H247 step 450 INT8,
same architecture and size as RC2) in the released live +LM and final decoder settings, measured beside RC2 and
Zipformer INT8 in the same blocks on the same core.

## Method (R7 unchanged)
- Tool, host, core, idle windows, thread limits, locks and rerun rule exactly as `runs/r7/PREREG.md` (energy_bench.py
  run, xbabe0, pinned CPU 16 with SMT sibling idle, observer CPU 31, 60 s idle before and after, one repetition,
  QUIET.lock per arm, TIMING.lock only when no CI VM runs, one in-place rerun of a core-unclean arm).
- The whole campaign holds the shared xbabe0 meter lease (`/dev/shm/jsr-h172-meter.lease`), so it never overlaps
  another metered run on that host.
- Binaries: the frozen 3bd85e9e `ultra-cold-wav` and `ultra-engine-commit` of R6/R7. `r8-final` changes only the INT8
  model directory (M5 LM and flags are R7's). `r8-live-lm` runs the v4.1.0 live defaults: H247 weights, beam 12,
  beam margin 4.0, blank penalty 0.24, compact M5 LM (`M5.v2.ulm`, sha256 58ef726b...) at weight 0.70 / LODR 0.50.
- Arms (panel73, 3 counterbalanced blocks: forward, reversed, rotated by two): rc2-live-lm, v41-live-lm, r0-int8-r6
  (Zipformer INT8), rc2-final, v41-final. Every block also runs RC2 live +LM, Zipformer INT8, v4.1.0 live +LM and
  v4.1.0 final on the full 730-utterance panel (block 1's RC2 and Zipformer full-panel runs are the validity anchors).
- Reported energy per arm: median of the 3 panel73 block values (the matched colour). v4.1.0 speed and CPU-s per
  audio-second: the 3 full-panel repetitions (median), the same form as the R6 rows of RC2. Paired speed ratios
  (v4.1.0 live vs Zipformer, v4.1.0 live vs RC2 live): median over blocks of the per-block ratio.

## Validity rule
The block-1 full-panel anchors must reproduce R6 within +/-5 percent: Isosyll RC2 live +LM 0.328 J/audio-s
(0.3117 to 0.3444) and Zipformer INT8 0.573 J/audio-s (0.5444 to 0.6017). If either misses, R8 is VOID: report the
numbers as they are, do not tune, and do not colour the hero figure from R8.

## What this does not claim
CPU-reported core registers, not wall power; one core of an otherwise idle 64-core part (boost on).
