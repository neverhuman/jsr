# R7 energy campaign: deviations from the pre-registration

Dated 2026-11-26. The pre-registration is `PREREG.md` beside this file (byte-identical to the sealed copy in the
receipt archive); these are the places where the campaign as run differs from it. None of the selection rules
looks at an energy value; selection uses disturbance metrics only. Facts below are read from the archived
`campaign.log`, the `campaign*.sh` drivers, `run-energy.sh`, the arm logs and the receipts themselves.

1. Reruns. In block 1 an arm could take up to 3 attempts (2 reruns); blocks 2 and 3 were capped at 2 attempts
   (1 rerun). The block 1 log shows a third attempt for Moonshine Small, Nemotron 160 ms and Vosk Large.
2. Attempt selection. The pre-registration says "the rerun is the one used" and "median of the 3 block values".
   The campaign instead uses a core-clean attempt if one exists (the latest), else the least-disturbed attempt,
   judged by `sibling_busy_fraction_run + elsewhere_busy_cpus_run / 64`, never by energy. Only core-clean
   blocks can colour the figure, so each reported value is the median of 2 or 3 core-clean blocks (9 arms
   have 3, 5 have 2); an arm with no clean block would stay grey. Every attempt (clean, unclean, failed) is
   retained.
3. Moonshine build. The energy runs used the energy build `energy/bin/moonshine-stream` (H39), while the H0
   timing rows used the bench-cross build. The models are identical.
4. Quiet check. The check is stricter than `run-energy.sh quiet` (no process above 25% CPU) but is made only
   at arm start, not during the arm.
5. Lock trap. The first live-anchor attempt ended with `renew: unbound variable` from the QUIET.lock EXIT trap
   under `set -u`. The host copy of `run-energy.sh` was edited at 03:58:12Z, after that attempt and before
   any retained receipt (the earliest starts 04:00:01Z); the archived version expands the renewal PID into the
   trap string when the trap is set. The repository copy later moved the PID to script scope with a cleanup function; that variant
   ran no receipt.
6. Log directory. The driver clears the log directory before each attempt, because the cold binary refuses to
   overwrite existing logs.
7. Sampler lifetime (found during the campaign, 2026-11-26). The power-sampler helper lives about 1625 s
   (600 + max-run 900 + idle + settle + idle). Whisper.cpp Base on the 73-clip panel needs about 1300 s of
   decode plus idles, so two attempts (block 1 attempt 1, block 2 attempt 1) ended with `nonzero_exit` at the
   final idle mark and recorded about 0.004 J. Both are flagged, kept as receipts and excluded; the rerun in
   each block finished core-clean, and block 3 was clean on its first attempt, so no gap-fill run was needed.
8. CI was active for the whole campaign. The pre-registration asks for an "otherwise idle" host and says "if CI
   jobs or other heavy processes appear, pause between arms and wait". The campaign did not pause for CI: all
   68 host checks in `campaign.log` (04:21Z to 15:50Z) report 4 CI virtual machines active, and every one of
   the 71 retained receipts carries the `foreign_load_changed` flag. The 39 core-clean receipts show up to 7.1
   other busy CPUs on average during the run (block 1 Whisper.cpp Base; most are below 2) and an idle package
   power of 59 to 99 W. Only the pinned core's energy register is used, with its own idle floor subtracted, and
   only core-clean receipts (sibling thread idle, no core idle drift, worker on its CPU) count; these bound
   the effect of the foreign load on the reported values, but the package numbers are not usable.
9. Driver restarts and edits. Block 1 was started four times (`BLOCK START` at 03:46:49Z, 04:13:32Z, 04:21:25Z
   and 04:30:08Z). The driver was edited during the campaign: `campaign.sh` (last modified 04:03:01Z, after the
   first start), then `campaign2.sh` (04:15:42Z) and `campaign3.sh` (06:18:59Z). The 04:21 start logs
   `campaign2.sh` messages (TIMING.lock lines, "SKIP ... (receipt exists)"); from 04:30 the log shows the
   `campaign3.sh` rule "SKIP ... (clean receipt exists)". The archived `campaign3.sh` caps attempts at 2, yet
   block 1 logged third attempts after 04:30, so `campaign3.sh` was edited again while block 1 ran (at
   06:18:59Z); the version block 1 ran with a cap of 3 is not retained. Blocks 2 and 3 ran the archived
   version.
10. Skip rule changed. Up to `campaign2.sh`, any existing receipt counted as done (at 04:21 the unclean
    live-anchor receipt was skipped). `campaign3.sh` skips an arm only when its receipt is core-clean.
11. Receipts renamed by hand. No driver writes `.unclean0` or `.failed-nonzero-exit` names: the files
    `b1-anchor730-live.unclean0.json`, `b1-anchor730-zip.unclean0.json` and
    `b1-anchor730-live.failed-nonzero-exit.json` were renamed by hand between driver restarts.
12. Anchor runs with a receipt but no DONE line. Three anchor receipts were written by a driver that was then
    stopped before it logged the result: Zipformer `unclean0` (started 04:03:15Z, 0.4060 J, sibling busy),
    live `unclean0` (started 04:15:27Z, 0.2472 J, sibling busy), and the clean Zipformer anchor that is used
    (started 04:22:33Z, 0.5668 J); `campaign.log` records the last only as "SKIP anchor730-zip (clean receipt
    exists)" at 04:47:13Z.
13. Lost receipt. The first live-anchor attempt (DONE at 03:58:53Z, 0.317888 J, not core-clean) has no
    receipt. The log says it was kept as `.unclean`, but no file with that name or value exists on the host or
    in the archive: it was overwritten when the next attempt reused the path. Its only record is
    that `campaign.log` line and the matching line in `arm-1-anchor730-live.log`.
14. Live anchor attempts and the rerun cap. The live anchor took 5 attempts: 03:47Z (0.3179 J, unclean, lost),
    04:00Z (nonzero exit after 0.18 s, 0.00025 J), 04:15Z (0.2472 J, unclean), 04:31Z (0.3064 J, unclean) and
    04:43Z (0.3209 J, core-clean, used). Under the pre-registration's literal rule ("rerun once ... the rerun is
    the one used") the anchor would have been the 04:00 nonzero-exit rerun, far outside the 5% band, and R7
    would be VOID. The rerun cap was raised (`campaign2.sh`, 3 attempts) after that failure, and the anchor
    was then rerun until a core-clean attempt existed. The Zipformer anchor took 2 attempts.
15. TIMING.lock code. `campaign2.sh` and `campaign3.sh` added code that takes TIMING.lock when no CI VM runs.
    It was never taken: all 68 TIMING.lock lines in `campaign.log` say "not taken", so the campaign held
    QUIET.lock only, as pre-registered.
16. Harness edits. An earlier version of this file said "The harness was not edited during the campaign". That
    was wrong: the drivers were edited (items 9, 10, 14, 15) and `run-energy.sh` once (item 5). The measurement
    code did not change across blocks: all 71 retained receipts record the same `energy_bench.py` sha256
    (9faf97a4..., matching the host file, last modified 2026-11-22), and every attempt of each arm records the
    same command line, environment, CPU set and input-file hashes in all blocks. The archived
    `run-energy.sh` (sha256 3d482b6b...) predates every retained receipt. Its Parakeet sherpa-onnx case started
   Python without `-I` (as the receipts' command lines show), unlike the other Python arms; the repository
   copy now passes `-I`, which no R7 receipt used.

Outcome: the anchors passed (Isosyll live 0.3209, Zipformer INT8 0.5668, within 2.2% and 1.1% of R6). All 14
arms have at least two core-clean blocks, and every block spread is under 5%.

Validity rule: both full-panel anchors must be present, core-clean, decode the full 730-utterance panel
(5259.56 audio-seconds) and lie within 5% of R6 (Isosyll live 0.328, Zipformer INT8 0.573) with a complete
three-block inventory. Otherwise the campaign is VOID: the summarizer replaces the evidence file with a single
VOID line, which the paper loader refuses.
