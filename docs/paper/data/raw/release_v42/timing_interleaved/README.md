# v4.2.0 R9 quiet-host paired timing and energy (full panel)

Copied byte-exact from the benchmark host's `opponents/energy/runs/r9/` (energy_bench.py receipts). Design and
validity rule: `PREREG.md` (written before any R9 run). Driver: `campaign_sh.txt` (`campaign.sh` at the origin, renamed so linters skip a receipt); arm commands: `run-energy_sh.txt` (the origin `run-energy.sh`); the driver's own log: `campaign_log.txt` (`campaign.log` at the origin).

- **Arms (full 730-utterance DEV panel, one pinned core, 60 s idle windows):** `anchor730-live` (v3.1.0 RC2 live +LM,
  the v4.0.0 live default) and `anchor730-zip` (Zipformer INT8), the R8 anchors; `v41-final-730` (the published
  `isosyll` v4.1.0, final mode); `v42-final-730` (the v4.2.0 release-candidate build d0d03dc3, final mode: F3 soup,
  c32 beam 16, gated beam 48, content-word t2 rerank); `v42-nogate-730` (the same build with the confidence gate off),
  in three counterbalanced blocks.
- **Timed build vs the published binary:** the published v4.2.0 binary (260de332) differs from the timed d0d03dc3 build
  only in how the reranker's rotary-position tables are computed once at load (f64 rounded to f32 instead of f32, for
  cross-platform identical words). The decode's work per utterance is unchanged. The block-1 `v42-final-730`
  transcripts are byte-identical (730/730) to the pre-fix transcripts and differ from the published binary's on 12/730.
  The block-1 `v41-final-730` transcripts are byte-identical (730/730) to the v4.1.0 transcripts in `raw/versus/v410`.
- **Validity:** block-1 anchors within 5% of R6 and core-clean: v3.1.0 live +LM 0.3336 J/audio-s (R6 0.328, +1.7%)
  and Zipformer INT8 0.5808 (R6 0.573, +1.4%), so R9 is VALID. The host's four idle CI runner VMs stayed up
  (the driver could not take the exclusive timing lock), so several repetitions carry `foreign_load_changed` or idle-drift flags.
- **Unclean blocks:** an arm whose core repetition was unclean was kept as `<arm>.unclean.json` and rerun once.
  Still core-unclean after the rerun: `v42-final-730` and `v41-final-730` in block 1, `anchor730-live` in block 3. They are
  kept and left out of the energy and RSS medians and the paired ratios (`tools/paper_data/v42_rows.py`); CPU and wall
  time are medians of all three blocks.
- Per-utterance transcripts and per-second series are not copied.
- Host names are role labels; no address appears.
