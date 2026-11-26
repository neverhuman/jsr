# R7 energy campaign: deviations from the pre-registration

Dated 2026-11-26. The pre-registration (PREREG) is not kept in this repository; these are the places where the
campaign as run differs from it. None of them looks at an energy value; selection uses disturbance metrics only.

1. Reruns. Block 1 allowed up to 3 reruns per arm; blocks 2 and 3 were capped at 2.
2. Attempt selection. The pre-registration says "the rerun is the one used". The campaign instead uses a
   core-clean attempt if one exists (the latest), else the least-disturbed attempt, judged by
   `sibling_busy_fraction_run + elsewhere_busy_cpus_run / 64`, never by energy. Only core-clean blocks can
   colour the figure; an arm with no clean block stays grey. Every attempt (clean, unclean, failed) is retained.
3. Moonshine build. The energy runs used the energy build `energy/bin/moonshine-stream` (H39), while the H0
   timing rows used the bench-cross build. The models are identical.
4. Quiet check. The check is stricter than `run-energy.sh quiet` (no process above 25% CPU) but is made only
   at arm start, not during the arm.
5. Lock trap. The QUIET.lock EXIT trap was fixed under `set -u` (the renewal PID is now script-scope and the
   trap calls a cleanup function).
6. Log directory. The driver clears the log directory before each attempt, because the cold binary refuses to
   overwrite existing logs.
7. Sampler lifetime (found during the campaign, 2026-11-26). The power-sampler helper lives about 1625 s
   (600 + max-run 900 + idle + settle + idle). Whisper.cpp Base on the 73-clip panel needs about 1300 s of
   decode plus idles, so two attempts (block 1 attempt 1, block 2 attempt 1) ended with `nonzero_exit` at the
   final idle mark and recorded about 0.004 J. Both are flagged, kept as receipts and excluded; the rerun in
   each block finished core-clean, and block 3 was clean on its first attempt, so no gap-fill run was needed.
   The harness was not edited during the campaign.

Outcome: the anchors passed (Isosyll live 0.3209, Zipformer INT8 0.5668, within 2.2% and 1.1% of R6). All 14
arms have at least two core-clean blocks, and every block spread is under 5%.

Validity rule unchanged: both full-panel anchors must be present, core-clean and within 5% of R6 (Isosyll live
0.328, Zipformer INT8 0.573) with a complete three-block inventory, otherwise the campaign is VOID and no
evidence file is written.
