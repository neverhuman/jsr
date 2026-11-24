# Training dynamics of the release lineage (full LibriSpeech DEV, `ultra-score` d784f8ca)

- `stage1/step<N>.c{4,16}.score.json`: stage-1 EMA checkpoints (LibriSpeech 960 h only), PyTorch streaming decoder, from
  eval-lane `curve/` (`<run>-<N>.ema.c<C>.ms8.score.json`). c16 at 35k/40k/45k was never scored.
- `stage1/exposure.csv`: cumulative `audio_s` of the stage-1 `train.jsonl` (980 train rows, steps 50..49,000 with no
  duplicates; sha256 89d75fbda678d7628e2cce43e5cd1161385b918569633de69c831e8cabf02e3c) at each scored step, in hours.
- `stage2/`: INT8 engine with the release decoder (bb1071a) on full DEV: the stage-1 end row, Libri-Light small
  branch points at 10k and 20k audio-h, Libri-Light medium at +10k, and the cooldown checkpoint that shipped as
  v3.0.0. Receipts are those listed in `research/H54_v3_release_qualification/MANIFEST`; branch-point exposures are
  the nominal definitions of `docs/ultra/releases/v3.0.0.md` (Training).
- Descriptive only: no model selection was made from these plots after the fact.
