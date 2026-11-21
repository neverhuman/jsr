# Training and validation curves (all runs)

These are compact score histories for every R1 training run. The owner requires them to be kept for every finalist. The full logs (`train.jsonl`) stay on the data drive; `index.json` records each one's path and sha256.

- `<run>/train.csv`: train rows every 250 steps, plus the last row. Columns: step, epoch (0-indexed), total/RNN-T/CTC loss, lr, grad norm, cumulative audio-hours, throughput.
- `<run>/eval.csv`: every dev eval row (step, split, epoch, WER, S/D/I, ref words). These are 400-utterance dev-clean evals unless the split says otherwise; panel and full-dev selection rows live in the waves (`research/H*`).
- `index.json`: per run: last step and epoch, audio-hours seen, best eval per split (with step and epoch), init checkpoint, params, source log sha256.

**Refresh** (the trainer mirrors logs to the data drive; D4M is copied from xbabe4):
```bash
cd python/ultra && ../../.venv/bin/python -m tools.export_curves --out ../../research/curves /mnt/xbabe3-external/jsr/runs/train/<RUN> ...
```
All numbers are exploratory (shared hosts, greedy streaming decode).
