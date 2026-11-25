# RC2 (v3.1.0) R6 quiet-host interleaved timing, subset

Copied byte-exact from `release/v3.1.0/timing_interleaved/` (210 files, SHA256SUMS verified at the origin). Kept here:
the audited `summary_report.json`, the per-block per-arm energy-bench receipts (`block<N>/<arm>.json`, which hold
cpu, wall, peak RSS and core joules), and the origin `SHA256SUMS` and `BIN_SHA256SUMS`. Not copied: the
`hold_audit.log` and `timing_quiet_excerpt.log` of each block (a log is never evidence here), per-utterance transcripts (`logs/`), per-second power series (`series/`), the 730-row
audio manifest and the `14s/` paced load receipts. `SHA256SUMS` therefore lists files absent here; the entries for
the kept files match this folder.

Design: 3 counterbalanced blocks x 5 arms (live c4 greedy, live c4+LM, live c8+LM, final M5, Zipformer INT8 anchor)
on one pinned core of the quiet host, ultra commit 3bd85e9e. Host names are role labels; no address appears.
`tools/paper_data/v31_rows.py` derives `data/evidence/xbabe0-v31-release.jsonl` from these files.
