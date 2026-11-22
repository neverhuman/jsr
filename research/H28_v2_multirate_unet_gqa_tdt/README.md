# H28: JSR v2 Architecture (`V2_S` / `V2_M` / `V2_L` — 0-Lookahead 3-Scale U-Net + GQA + ReLU²) & H27 Dual-Mode Chunking

Status: **READY FOR 100 h GATE** (PR #30 opened for `@trainer-lane` + `@claude-lead` review).

## 1. Question
Does the JSR v2 `unet3` architecture (`25 → 12.5 → 6.25 → 12.5 → 25 Hz` 0-lookahead causal U-Net + `kv_heads=2` Grouped-Query Attention + `ReLU²` FFN activation) beat the v1 `S` / `S_MR` baselines on matched 100 h GPU hours while cutting encoder FLOPs by `~45%`, K/V cache memory by `3×–6×`, and unlocking `75–85%` natural FFN activation sparsity for our merged Wave `H24` `pqscan` sparse kernel (`3.75×–4.69×` on Zen 2 AVX2)?

## 2. Architecture Summary (`python/ultra/train/model.py`, `asr.py`, `trainer.py`)
- **`family="unet3"`:** `lower_blocks` (25 Hz) → `pair` → `middle_blocks` (12.5 Hz) → `pair2` → `deep_blocks` (6.25 Hz, 4× fewer tokens) → `restore2 + skip12` → `up_blocks` (12.5 Hz) → `restore + skip25` → `final_blocks` (25 Hz). Strict 0-lookahead causality at `chunk=4` (160 ms) verified bit-exact in `test_train_v2_arch.py`.
- **GQA (`kv_heads=2`):** 3×–6× smaller K/V cache across `V2_S` (6→2), `V2_M` (8→2), `V2_L` (12→2) while keeping `qkv` weight-norm capping intact.
- **`ReLU²` (`ffn_act="relu2"`):** Squared ReLU feeds directly into Wave `H24` `pqscan` subspace-pair zero-skipping.
- **Wave `H14`/`H27` Dual-Mode Chunk Sampling (`--set chunk_modes=4,8,16,0` or `2,4`):** Deterministic per-step chunk mask sampling so one model serves low-latency and high-accuracy operating points without retraining (`""` default = bit-exact no-op).

## 3. Pre-Registered 100 h Gate Protocol (`xbabe2` 3090 / `xbabe1` 4090)
- **Control:** `H23a` / `H23e` (`S`, `rnnt_pieces=512`, `train-clean-100`, 24k steps).
- **Candidate:** `V2_S` (`--preset V2_S --set rnnt_pieces=512 ctc_pieces=512 rnnt_loss=pruned grad_clip_mode=adaptive`), matched GPU wall-clock hours.
- **Promotion Rule:** Promote `V2_M` / `V2_L` to 4–5× A100 960 h production if `V2_S` matches or beats `H23e` (`15.25%` dev-clean400) at lower CPU/encoder FLOPs and `3×` smaller KV cache.
