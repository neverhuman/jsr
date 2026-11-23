# H43: v3 baseline and measurement contract

**Status:** PRE-REGISTERED (2026-11-23). This is the first wave of the ULTRA v3 major release. The owner-approved plan for all v3 waves (H43–H54) is in [PLAN.md](PLAN.md).

## Question
What exactly is the v3 starting point (B0), and how will every v3 change be measured? The measurement has to be one that a fast wrong token, a mismatched scorer, a mixed checkpoint or noise cannot game.

## Baseline B0
- **B0** is A1 step 49,000, the final checkpoint of the A1 run (EMA weights). The rule is a run's final checkpoint, never one picked on the panel that Fig. 1 plots.
- **Exploratory, LibriSpeech DEV only.** The test sets stay sealed.

| Quantity | Value | Source |
|---|---|---|
| Full-dev pooled WER, INT8 engine (Rust scorer) | 8.272% (7,042 / 894 / 837 S/D/I over 106,054 words) | eval-lane export; reproduced by codex-eval |
| EMA greedy at 160 / 320 / 640 ms | 8.302 / 7.961 / 7.703% (dev-clean 4.480, dev-other 12.373 at 160 ms) | `/mnt/xbabe3-external/jsr/evals/eval-lane/curve/A1-49000.*` |
| EMA beam 4, 160 ms | 8.223% | same |
| First token, mean, 160 ms | 258 ms from the end of the first word; 564 ms from its onset | eval-lane `wordlag/latency_summary.json` |
| Speed / power | Pending re-timing on quiet xbabe0 (TIMING 14b). A1-30000 (same architecture): 28.2× real time, 0.35 W core-register incremental per stream | bench-open, energy |

## Owners
| Part | Owner |
|---|---|
| Canonical scorer: Rust `ultra-eval` is the authority; Python `ultra/wer.py` becomes an exact port, with a rule-table sha256 parity test | Codex |
| Alignment ledger | Codex |
| Metrics: onset and end-of-first-word clocks, first-correct-word, correct-by-deadline at 150 / 300 / 500 ms, revision burden | eval-lane |
| Statistics: seed pairs, Holm over ≤ 4 arms or split-dev selection, INCONCLUSIVE | eval-lane |
| B0 receipt manifest | eval-lane |
| Quiet timing and power of B0 | bench-open |

## Gates fixed now, before any v3 result
- **Accuracy.** A change is adopted only above 2× the four-run control spread: 0.17 pp pooled at 160 ms (0.094 pp spread at 320 ms). It needs a seed pair (1234 / 4321) against the comparator's seed pair, and both seeds must point the same way.
- **Several arms.** With more than one arm in a wave, apply Holm over ≤ 4 arms, or select on half the dev speakers and confirm on the other half.
- **Non-inferiority.** Margins are fixed before looking. An interval the data cannot resolve is reported as INCONCLUSIVE.
- **Latency.** Every number names its clock. A product latency claim comes only from quiet xbabe0 or the owner's metered target device.
- **Pairing.** WER and speed are never paired from different checkpoints.
- **Bundles.** From now on, screen bundles with a 2^(7−4) fractional factorial with fold-over.

## Results
- [results/wer_error_analysis_A1-49000.md](results/wer_error_analysis_A1-49000.md): where B0's errors come from.
- Later results land in `results/` as they arrive: scorer parity, the B0 manifest, latency metrics on full dev.

## Pointers
- Approved plan: [PLAN.md](PLAN.md).
- External designs studied: `tips/v3/*.md` on xbabe3, kept local and not committed. The review that scores them is `tips/v3/Review and scores.md`.
- Coordination: `AGENT_CHAT.md` (local).
