# v4.1.0 R8 quiet-host paired timing and energy (full panel)

Copied byte-exact from the benchmark host's `opponents/energy/runs/r8/` (energy_bench.py receipts). Design and
validity rule: `PREREG.md` (written before any R8 run). Driver: `campaign_sh.txt` (`campaign.sh` at the origin, renamed so linters skip a receipt); arm commands: `run-energy_sh.txt` (the origin `run-energy.sh`; same arms as docs/paper/run-energy.sh)
(`r7-live-lm`, `r7-zipformer`, `r8-live-lm`, `r8-final`); the driver's own log: `campaign_log.txt` (`campaign.log` at the origin).

- **Arms (full 730-utterance DEV panel, one pinned core, 60 s idle windows, frozen 3bd85e9e binaries):**
  `anchor730-live` (v3.1.0 RC2 live +LM, the v4.0.0 live default), `anchor730-zip` (Zipformer INT8),
  `v41-live-lm-730` (v4.1.0 live defaults: H247 step-450 INT8, beam 12, margin 4.0, blank penalty 0.24, compact M5 LM)
  and `v41-final-730` (v4.1.0 final: commit pass, M5 LM, beam 16), in three counterbalanced blocks.
- **Change after pre-registration (before any R8 measurement):** the five panel73 arms were dropped for time
  (logged at the top of `campaign_log.txt`); the full-panel arms are unchanged.
- **Validity:** block-1 anchors within 5% of R6: v3.1.0 live +LM 0.3247 J/audio-s (R6 0.328, -1.0%) and
  Zipformer INT8 0.5771 (R6 0.573, +0.7%), both core-clean, so R8 is VALID.
- **Unclean blocks:** an arm whose core repetition was unclean was kept as `<arm>.unclean.json` and rerun once.
  In block 3 both live arms (`anchor730-live`, `v41-live-lm-730`) stayed core-unclean after the rerun
  (`sibling_busy`: the measured core's SMT sibling was busy); they are kept and left out of the energy and RSS medians.
  `tools/paper_data/v41_rows.py`: CPU and wall time are medians of all three blocks (an unclean block can only be the
  maximum, so each median is a clean block's value); energy, RSS and paired ratios use core-clean blocks only.
- Per-utterance transcripts and per-second series are not copied. The block-1 transcripts of both v4.1.0 arms are
  byte-identical (730/730) to the `isosyll` v4.1.0 transcripts scored in `raw/versus/v410`.
- Host names are role labels; no address appears.
