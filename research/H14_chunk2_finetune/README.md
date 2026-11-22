# H14: 80 ms attention chunk (chunk 2) fine-tune

**Status:** results in; **written after the results, not pre-registered in the repo.** The latency measurement that decides it is still open.

## Honest record of the process
- The lead asked for this run at 17:36Z on 2026-11-22 with a pre-registration. The arms were launched at about 17:42Z.
- No README existed before the runs finished at 18:51Z. The intended gate was written only in the trainer lane's session plan file at 17:45Z, before the first eval: first-preview latency from codex-review's harness, at no more than +0.3 pp dev-clean400 against the control.
- Treat the result as exploratory.

## Why
H34 showed that the pruned-loss delay penalty does not move the first preview (see `../H34_delay_penalty/README.md`). Codex-review's direct-PCM measurement puts the first preview 370–540 ms after speech onset, so the floor is the 160 ms attention chunk. This run asks what an 80 ms chunk costs in WER.

## Design
- **Seed:** D2e final (`ckpt-00016457`, raw weights, dense S, 4,096 tokens; dev-clean400 5.288%).
- **Arms:** `chunk=2` (80 ms) and `chunk=4` (160 ms, the control: the same fine-tune at the model's own chunk). They ran back to back on xbabe2 (RTX 3090).
- **Settings, identical in both arms:** 4,000 steps, lr 1e-4 with 200 warmup steps and a cosine, budget 650 s / max batch 144, 960 h with speed perturbation, seed 1234, pruned loss, EMA 0.9995, qkv cap, guard 0.85.
- **Launcher:** `python/ultra/train/ops/h14_chunk.sh`.

## Results (dev-clean400, greedy, each arm decoded at its own chunk)
| step | chunk 2 raw | chunk 2 EMA | chunk 4 raw | chunk 4 EMA |
|---:|---:|---:|---:|---:|
| 1,000 | 5.718 | 5.796 | 5.522 | 5.223 |
| 2,000 | 5.887 | 5.626 | 5.509 | 5.379 |
| 3,000 | 5.744 | 5.718 | 5.496 | 5.418 |
| 4,000 | 5.600 | 5.639 | 5.340 | 5.418 |

- **Cost of the 80 ms chunk at 4,000 steps:** +0.26 pp raw, +0.22 pp EMA.
- The control fine-tune itself did not improve on its seed (5.340 against 5.288).
- This is one seed per arm on 400 utterances. Two identical 100 h runs differ by about 0.57 pt on this panel, so the cost estimate is rough.

## Open
- **Latency:** codex-review measures the first-preview horizon for both arms. Weights: `xbabe2:~/jsr-data/runs/train/H14_chunk{2,4}/model-00004000.pt` (raw and EMA).
- **Export:** the chunk-2 run's `config.json` records `model.encoder.chunk = 2`, so an export serves the trained 80 ms mask. Before this run the trainer left the preset's chunk in the model config (codex-review's catch).
- The 4,096-token line ended with D2f. If an 80 ms tier is wanted, it should be retrained on the BPE-500 trunk (A1 samples chunks 4, 8 and 16, not 2).
