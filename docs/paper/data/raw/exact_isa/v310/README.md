# RC2 exact-ISA witness (transcript identity only; no timing, no WER decision)

`rc2-compare.json` is the output of `rc2_compare.py`. It compares each witness run with the confirmatory look's Zen 2 (Threadripper PRO 3995WX)
transcripts in `../../versus/v310/*.panel730.score-input.jsonl`. Every witness run uses binaries built at engine commit 3bd85e9
on its own host, and the frozen RC2 bundle (INT8 prepared sha256 94a9879f…).

- **Xeon Platinum 8358:** 13 runs. Dispatch default (= avx2fma), avx2fma and avx512vnni. Each covers c4 greedy at 1 and 16 streams, c4 + LM b4 and c8 + LM b4, plus the M5 final commit.
- **Apple M3 Ultra (NEON):** 5 runs. c4 greedy at 1 and 16 streams, c4 + LM, c8 + LM, final M5.
- **Forced scalar (M3):** an optional control. Not part of the release witness; listed with no comparison while it runs.

Codex independently audited the Xeon runs (`rc2-x6-transcript-audit.json` 041f9169…) and the first three NEON runs (5b533a44…).
