# H75: error anatomy of the shipped v4.0.0 live and final text on DEV S

**Status:** EXPLORATORY, DEV selection half S only (2,782 utterances, 36 speakers, 52,196 reference
words). C half, TEST and HOLDOUT untouched. claude lane, WER push 2026-11-26.
(Claimed first as "H67"; renamed because codex's H67 claim was earlier.)

## Question
Where are the remaining errors of the system that actually ships (v4.0.0 = RC2 INT8 weights with the
live LM and the M5 commit pass), and which share can language correction reach versus acoustics?

## Setup
- Signed `isosyll` v4.0.0 x86-64 glibc binary and model archive from
  `/mnt/xbabe3-external/jsr/releases/v4.0.0-dist/`, `transcribe --mode final|live --json --threads 1`,
  24 shards per mode on xbabe3 (nice 10; wall times are not timing evidence, the host was loaded).
- S-half WAVs exported by `tools/export_s_half_wav.py` through `wer_diag.load_selection_dev`
  (C-half guard) to `/mnt/xbabe3-external/jsr/wer-push/shared/s_half/`.
- Official counts: pinned `ultra-score` (sha256 d784f8ca…, `ultra-en-wer-norm/v1`).
- Anatomy: `tools/anatomy.py` (local Levenshtein over `ultra.wer.normalise`; ~1% more edits than the
  pinned normaliser, read as shapes). Word frequency classes come from train-960 transcript counts.

## Results
| row | pooled | dev-clean | dev-other | edits |
|---|---:|---:|---:|---:|
| RC2 greedy c4 (H61 reference) | 6.9469% | 3.4959% | 10.8289% | 3,626 |
| live: c4 beam 4 + live LM 0.70 / LODR 0.50 | **6.26%** | 3.14% | 9.77% | 3,267 |
| final: commit c16 beam 16 + M5 0.80 / LODR 0.60 | **5.44%** | 2.68% | 8.54% | 2,838 |

final − live: −0.845 pp macro, one-sided 95% [−1.056, −0.663].

- **Speaker concentration.** Five dev-other speakers (1650 22.8%, 1585 17.1%, 7697 13.2%, 5543 12.5%,
  2506 10.7% final WER) hold 15% of the words and 42% of all edits, in greedy, live and final alike.
  The other 31 speakers are at 3.79% final and 4.36% live.
- **Rare words.** Words with 0 train-960 occurrences err 78% of the time, 1-9 occurrences 31%,
  10-99 occurrences 9.8%, against 3.4% for words seen 1000+ times. These 8.6% of reference words carry
  30% of final edits (854 / 2,864). Most are names with a consistent wrong spelling
  (ALEKSANDROVNA→ALEXANDROVNA ×19, BOZZLE→BOSWELL ×11, SERAFIMA→FEMALE / SERAPHIMA).
- **Function words.** THE↔A 73, IN↔AND 53, AN↔AND, IS↔AS and THE / A insertions ≈ 8% of final edits.
- **Position and length** as in H60: word 1 7.6 errors per 100 against 4.4 from word 16; 1-9-word
  utterances 7.5% against 4.5% for 35+ words.

## Decision / consequence
- Getting to 2-3% pooled is mostly a hard-speaker problem (acoustics, robustness, adaptation); the
  31 easier speakers are already near 3.8% final.
- Language correction (H68) targets the function-word and context share; a name cache can only help
  names the model sometimes gets right.
- Every arm tonight reports top-5-speaker WER and rest WER separately.

## Pointers
`/mnt/xbabe3-external/jsr/wer-push/H75/`: `{live,final}.score-input.jsonl`, `live_vs_final.{json,md}`,
`greedy_vs_live.{json,md}`, `anatomy.json`, `runs/` (raw isosyll JSONL).
