# v4.2.1 R10 quiet-host paired timing and energy (full panel)

Copied byte-exact from the benchmark host's `opponents/energy/runs/r10/` (energy_bench.py receipts). Design and
validity rule: `PREREG.md` (written before any R10 run). Driver: `campaign_sh.txt` (`campaign.sh` at the origin, renamed so linters skip a receipt); arm commands: `run-energy_sh.txt` (the origin `run-energy.sh`, which adds the `r10-isosyll` arm: `r9-isosyll` with the mode as an argument); the driver's own log: `campaign_log.txt` (`campaign.log` at the origin).

- **Arms (full 730-utterance DEV panel, one pinned core, 60 s idle windows):** `anchor730-live` (v3.1.0 RC2 live +LM,
  the v4.0.0 live default) and `anchor730-zip` (Zipformer INT8), the R8 anchors; `v42-live-730` (the published
  `isosyll` v4.2.0 x86_64-linux binary, `transcribe --json --mode live`, the v4.2.0 model); `v421-live-730` (the
  `isosyll` v4.2.1 x86_64-linux release binary, ca4c230b, same command, the v4.2.1 model), in three counterbalanced blocks.
- **Per-file outputs:** `block*/v42-live-730.out.jsonl` and `block*/v421-live-730.out.jsonl` are the two isosyll arms'
  own stdout (`runs/out/r10/b<N>-r10-<arm>/rep1-w0.jsonl`: text and per-file `wall_s`). The added utterance-end delay is
  the per-file paired difference (first file dropped: it carries the lazy model load); `tools/paper_data/v421_rows.py`.
  The v4.2.1 transcripts in every block are the scored `raw/versus/v421/v421-live.panel730` transcripts.
- **Validity:** block-1 anchors within 5% of R6 and core-clean: v3.1.0 live +LM 0.3267 J/audio-s (R6 0.328, −0.4%)
  and Zipformer INT8 0.5815 (R6 0.573, +1.5%), so R10 is VALID. The host's four idle CI runner VMs stayed up
  (the driver could not take the exclusive timing lock), so repetitions carry `foreign_load_changed` or idle-drift flags.
- **Unclean blocks:** the block-1 live anchor was core-unclean (`sibling_busy`) and kept as `anchor730-live.unclean.json`;
  its one rerun was clean. Every arm is core-clean in all three blocks.
- Per-second series are not copied.
- Host names are role labels; no address appears.
