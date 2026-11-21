# H7: Multirate encoder (S_MR) vs dense S

**Status:** EXPLORATORY.

## Question
Does the multirate encoder keep quality while cutting compute? In S_MR, the middle 10 blocks run at 12.5 Hz and the outer blocks at 25 Hz (spec v1.0-B §17.2–17.4).

## Setup
- **Model:** preset `S_MR`, built in d340276 with a streaming path in cea516f. The engine supports it (aed94af).
- **D1MR:** S_MR on train-clean-100, run to 40k steps on the xbabe1 4090.

## Results
- **Quality:** the D1MR final (40k) scores **18.0% full dev-clean** and 36.5% dev-other. Dense D1d stopped at **29.8%** (7.8k steps). At the matched step (~7.8k), S_MR was about 5 points worse; it overtook dense by about 10.8k.
- **Engine INT8** (dev tool, exploratory):

| Host | Dense | S_MR |
|---|---|---|
| M3 Ultra | 60–64 streams/core | 84 streams/core |
| Zen 2 | 96/CCX | 136/CCX |

- **CPU per stream:** about 25% lower for S_MR.
- **Single stream:** S_MR is not much faster; it is memory-bandwidth bound, with similar weight bytes.

## Consequence
- S_MR is the density design. Two 960 h runs followed: D2MR and D3MR (H9).
- The density numbers above are **dev-tool numbers**. The claim needs the single-harness rerun (H10).

## Pointers
- Commits: d340276, cea516f, aed94af, 2a966c0, 87a4a51.
- Run dirs: `/mnt/amphora-backup/jsr-data/runs/train/D1MR_S_MR_tc100` (xbabe1); copy on H0 at `/mnt/xbabe3-external/jsr/runs/train/D1MR_S_MR_tc100_copy`.
