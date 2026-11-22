# H28: JSR v2 Architecture (`V2_S` / `V2_M` / `V2_L` — 0-Lookahead 3-Scale U-Net + GQA + ReLU²) & H27 Dual-Mode Chunking

Status: **READY FOR 100 h GATE** (PR #30 opened for `@trainer-lane` + `@claude-lead` review).

## 1. Question
Does the JSR v2 `unet3` architecture (`25 → 12.5 → 6.25 → 12.5 → 25 Hz` 0-lookahead causal U-Net + `kv_heads=2` Grouped-Query Attention + `ReLU²` FFN activation) beat the v1 `S` / `S_MR` baselines on matched 100 h GPU hours while cutting encoder FLOPs by `~45%`, K/V cache memory by `3×–6×`, and unlocking `75–85%` natural FFN activation sparsity for our merged Wave `H24` `pqscan` sparse kernel (`3.75×–4.69×` on Zen 2 AVX2)?

## 2. Architecture Summary (`python/ultra/train/model.py`, `transducer.py`, `asr.py`, `data.py`, `trainer.py`)
- **`family="unet3"`:** `lower_blocks` (25 Hz) → **Self-Conditioned `InterCTC`** (`inter_ctc=True`, supervises 25 Hz plosives before downsampling and feeds `inter_proj(softmax(inter_logits))` back into the stream) → `pair` → `middle_blocks` (12.5 Hz) → `pair2` → `deep_blocks` (6.25 Hz, 4× fewer tokens) → `restore2 + skip12` → `up_blocks` (12.5 Hz) → `restore + skip25` → `final_blocks` (25 Hz). Strict 0-lookahead causality at `chunk=4` (160 ms) verified bit-exact in `test_train_v2_arch.py`.
- **Causal Depthwise-Separable 2D Stem (`stem_type="dwsep"`):** `1→32` 3×3 s=2 → `32→32` DW 3×3 s=2 → `32→128` PW 1×1 (`5.2×` fewer stem MACs, identical causality contract).
- **GQA (`kv_heads=2`) + Per-Stage Left Context (`stage_left=(32, 64, 64)`):** 3×–6× smaller K/V cache across `V2_S/M/L`, with `1.28 s` local context at 25 Hz, `5.12 s` at 12.5 Hz, and `10.24 s` at 6.25 Hz.
- **Stability & Quantization Hygiene:** Pre-RoPE `QK-RMSNorm` (`qk_rmsnorm=True`), `norm_affine=False` (eliminates LN.g↔W scale freedom), export-folded residual `LayerScale` (`layer_scale_init=0.1`), and Zipformer-style `Activation Balancer` (`--set balancer_weight=0.001`).
- **`ReLU²` (`ffn_act="relu2"`) + 4-Token Predictor (`pred_ctx=4`):** Squared ReLU feeds Wave `H24` `pqscan` zero-skipping; 4-token stateless predictor (`context_len=4`) fixes `BPE-500` short-context under-conditioning at `<0.2%` CPU.
- **Training Generalization (All Default-Off Bit-Exact):** Fixed `H26` CR-CTC View-2 supervised CTC, O2-clean mathematical `synth_aug=true` (25% synthetic RIR + 25% pink/white spectral noise inside data workers, zero external files), smooth SpecAug ramp (`specaug_ramp_start=500`), and Wave `H14`/`H27` dual-mode `--set chunk_modes=4,8,16,0`.

## 3. Pre-Registered 100 h Gate Protocol (`xbabe2` 3090 / `xbabe1` 4090)
- **Control:** `H23a` / `H23e` (`S`, `rnnt_pieces=512`, `train-clean-100`, 24k steps — `15.251%`).
- **Candidate:**
  ```bash
  --preset V2_S --splits train-clean-100 --set rnnt_pieces=500 ctc_pieces=500 \
    tokenizer=$D/models/zipformer-en-2023-06-26/bpe.model rnnt_loss=pruned \
    grad_clip_mode=adaptive cr_ctc_weight=0.2 inter_ctc_weight=0.2 \
    balancer_weight=0.001 synth_aug=true specaug_ramp_start=500 \
    chunk_modes=4,8,16,0 ema_decay=0.9995
  ```
- **Promotion Rule:** Promote `V2_M` / `V2_L` to 5–6× A100 960 h production if `V2_S` matches or beats `H23e` (`15.25%` dev-clean400) at lower CPU/encoder FLOPs and `3×` smaller KV cache.
