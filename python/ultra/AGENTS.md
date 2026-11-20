# python/ultra: agent guidance

This tree is the dated Python exception for training and data preparation (`EXCEPTION.md`, `docs/exceptions/2026-11-20-python-ultra-training.md`). It never owns product truth. Streaming runtime, scoring authority, receipts and release gates are Rust (`crates/`).

| Path | What | Proof |
|---|---|---|
| `ultra/libri.py` | LibriSpeech parquet reader and decoded caches on the data drive | `just pytest` |
| `ultra/sp_a.py`, `ultra/sp_a_fleet.py` | SP-A stream-density harness: fixed capture deadlines, grid decode, fleet validation | `python/ultra/tests/test_sp_a_*.py` |
| `ultra/wer.py` | provisional WER (re-scored by `ultra-score` before any claim) | `test_wer.py` |
| `train/` (lane/trainer) | streaming transducer trainer | `test_train_*.py` |
| `waves/<id>/` | one folder per wave: `PREREG.md` first, then README with the verdict on line 1 | — |

**Rules:**
- **Data:** data, checkpoints and run outputs live under `/mnt/xbabe3-external/jsr`, never in git.
- **Timing:** timing runs follow the core partition and lease in `AGENT_CHAT.md`. Per-block runs are exploratory; qualifying runs need `BENCH_LEASE.host`.
- **Pre-registration:** pre-register before running. Amendments are dated and recorded before any confirmation read.
- **Environment:** the venv reuses the system torch (numpy < 2). Do not upgrade numpy in `.venv`.
