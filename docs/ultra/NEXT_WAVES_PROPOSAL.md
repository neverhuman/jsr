# Next research waves (H12+): ideas from the six 2026-11-21 specs, ranked against our measurements

**Sources:**
- the six owner specs in `tips/`;
- two fresh-context readers (accuracy specs and speed specs);
- trainer-lane's and engine-lane's skims (AGENT_CHAT 22:08Z).

Ideas are ranked by expected value for us at our current state:
- dense R1 at 8.52% dev-clean against R0's 2.85%;
- the engine is compute-bound at batch (about 62 GMAC/s per core on Zen 2), and batching barely amortises (density × single-stream ≈ 1);
- training runs 1–3 days on 3090/4090/M3.

**Ground rule (the specs agree):** never pair a WER from one checkpoint with a speed from another. The 0.041 CPU-s/s figure is from the D2 step-5k checkpoint.

## A. Accuracy (the main gap)

| # | Idea (spec §) | Why us | Expected | Cost / Rust | First experiment → pass |
|---|---|---|---|---|---|
| **A1** | **MR pad-before-pairing bug fix** (CPU-ASR §4.1). Confirmed in `train/model.py` EncoderMR. | Every S_MR run trained with a train/infer mismatch at odd-length tails. | small, but it's a correctness fix | 1 h; Rust: confirm the per-stream engine is unaffected | Junk-padding invariance test; D2MR-final batch-1 vs batched WER. **Running (trainer, H12).** |
| **A2** | **Error-source decomposition before spending GPU-days** (CPU-ASR §10.1). | Decides the next waves: streaming vs full context, greedy vs beam-4, beam vs N-best oracle, per-length and final-word, FP32 vs INT8. | — (a decision tool) | about 1 day, CPU | A table splitting the gap into undertraining, streaming penalty and search. **Running (trainer, H12).** |
| **A3** | **Free checkpoint averaging of D2** (X1 §6.1) | Already have the snapshots. | 3–8% relative | free | Last 5–8 snapshots averaged; ≥ 3% relative on the 730 panel. Folded into B2/C1 selection. |
| **A4** | **Pruned RNN-T loss** (Frontier §9.2, CPU-ASR §10.7, X1 §6.1). The full 4097-way lattice costs about as much as the encoder. | We are **undertrained** (about 7–14 epochs vs about 30 for R0's recipe). A 2× or better step throughput means more epochs per day. | indirect, likely the largest recipe lever | about 1 day of integration (k2/torchaudio, wheel risk); no Rust change | Loss and gradient match the exact loss; ≥ 2× steps per GPU-hour; WER within noise at equal exposure. |
| **A5** | **Smaller shared tokenizer (500–1024 pieces, same as the CTC head)** (HYPER A5.1, CPU-ASR §13.3, V2 §4.1) | 4096 pieces is sparse for 960 h; it shrinks the joiner about 8×; it **enables R0 distillation** if tokens match R0's 500. | unverified, plausible | retrain; Rust: new bundle only (verify no vocab-size assumption) | Paired 100 h runs, 500 vs 4096; ≥ 5% relative. Combine with A4 into one "recipe parity" run. |
| **A6** | **Chunk-size sampling + full-context self-distillation / dual-mode RNN-T consistency** (CPU-ASR §10.5, X1 §6.2–6.3, X2 A3) | No inference change; attacks the streaming penalty (sized by A2). Use the RNN-T consistency loss; the paper reports the CTC variant hurt. | 5–10% relative (hype says 55–70%) | about 1.6–1.8× train compute | Warm start from D2; ≥ 5% relative on dev-other at equal steps, no worse emission delay. |
| **A7** | **QK-RMSNorm (+ ScaledAdam, optionally)** (HYPER T1, V2 §4.2) | Principled fix for the qkv runaway (H9); lets lr go higher. | stability; enables A4–A6 at higher lr | small Rust change (per-head norm on q, k) | 100 h at lr 1.2e-3, no divergence, WER ≤ baseline. Treat Muon/ScaledAdam "2×" claims as unproven. |
| **A8** | **Distil from R0 through its distributions, tokenizer matched** (CPU-ASR §10.4 Route B, Frontier §9.3) | A 2.85% teacher signal with no new data and no graph change. | **the most plausible large gain** (15–30% relative, unverified) | teacher inference on 960 h once; **needs owner sign-off (R0 as a teacher)** and A5 | ≥ 10% relative over a no-teacher control at equal exposure. |
| A9 | CR-CTC with the transducer (X1/X2 A2) | Published about 6% relative. | modest | 2 forward passes; no Rust change | After parity. |
| A10 | Public causal-Zipformer recipe as a control run (X2 A1) | Honest attribution of recipe vs architecture. | — | one GPU for days | Optional; decide after A2. |

## B. Speed (engine; bit-identical first, then new numerics behind gates)

| # | Idea (spec §) | Why us | Expected | Engine work | First experiment → pass |
|---|---|---|---|---|---|
| **B1** | **Cut the non-GEMM floor, bit-identical:** RoPE cos/sin table once per stream-step (today f64 sin/cos per attend call, `encoder/block.rs:198`); fuse residual + norm + row min/max (X2 S2, X1 Table 4, engine-lane) | On the Zen 2 panel profile, **about 30% of time is outside the GEMM** (gemm 0.028 of 0.0405). | 10–15% total, zero accuracy risk | moderate; continuity fixtures are the oracle | Per-phase profile before and after; fixtures unchanged. |
| **B2** | **Unsigned-offset 7-bit kernel** (X1 §4.2 K1/K2, V2 §2.1): drop `vpabsb` and 2× `vpsignb` (3 of 10 ops per 64 MACs); integer correction via a stored column sum | Instruction cut on a compute-bound kernel. int32 sums are identical to the existing act7 path. | engine-lane estimates 1.3–1.6× on the GEMM; **but our act7 measurement gave only +3–10%**, so bench first | new tile + scalar reference + prepared layout v2 + NEON twin | `kernel_bench` on a quiet core, K0 vs act7 vs offset tile; ≥ 1.2× and bit-exact. The act7 quality gate already passed (+0.013). |
| B3 | **K/V "state diet"** (int8/shared K/V, shorter left context) (X1 S4, X2 S6) | fp32 K/V is 6.7 MB per stream, a density limit above about 8 streams per worker. | density only | moderate | First sweep left context 128/64/32 at fixed GEMM work. |
| B4 | **Single-plane ternary LUT with 7-bit activations** (X1 S3) | Removes the 8-row cap that made R4T density lose to int8. | unknown on Zen 2 | kernel + an R4T gate at A7 | `tlut_bench` on H0 at 4/16/64 rows; ≥ 1.7× over int8 at ≥ 16 rows, exact; only then QAT. |
| B5 | Narrower FFN (4× → 2×) with distillation (CPU-ASR §7.1) | FFN is 70% of MACs; MACs pay when compute-bound. | about 25–35% CPU | none (new checkpoint) | After A8 exists. |
| B6 | Bounded beam (+ a longer-context predictor) (CPU-ASR §11) | Our predictor sees 2 tokens. | 3–8% relative WER | Rust decoder change | Size it with A2 in Python first. |
| B7 | **R0-E:** R0's weights in ultra-engine (X2 A0) | An equal-accuracy speed headline without waiting for training. | big if it works | **weeks** (Zipformer ops: SwooshL/R, BiasNorm, multi-rate bypass); **owner decision** | Later. |

## C. Claims our measurements contradict (do not plan on these)
- **"Zen 2 is bound by one multiply pipe, so act7/unsigned gives 1.33–1.6×":** act7 measured +3–10%. Bench before believing (B2).
- **"About 40% of single-stream cost is weight fetch":** batching amortises ≤ 8% on Zen 2 and about 0 on M3. We are compute-bound.
- **"The kernel runs at about 30% of roofline (27–36 GMAC/s)":** measured 62 GMAC/s per core.
- **"W4A8 / ternary fits in L3 and wins at batch":** R4T fit in L3 and still lost at batch (compute-bound).
- **"No speed perturbation / no qkv safeguards":** both exist (data.py; qkv cap 4d4fc03).
- **"u8 activations at [0, 254] are exact on AVX2":** `vpmaddubsw` saturates (2·254·127 > i16 max). Only exact with 7-bit.
- **AVX-512 VBMI / AMX / VNNI headline numbers:** our hosts have none of them (a VNNI host would be an owner hardware decision).
- **HYPER/V2 WER targets (about 1.8–2.4% on LibriSpeech only) and 8–12k streams per server:** projections without evidence.
- **Apple AMX via Accelerate:** breaks pure-Rust and bit-exactness.
- **640 ms commit:** risks A1 p99 ≤ 800 ms; cap at 320 ms unless measured.

## D. Proposed wave order
- **H12 (now, CPU):** A1 + A2 + A3. *Running.*
- **H13 (next free GPU):** "recipe parity" run = A4 pruned loss + A5 512-piece tokenizer + EMA, dense S, on 960 h, compared against D2 at equal exposure. A7 QK-norm is added if A2/A4 show lr-limited training.
- **H14:** A6 dual-mode / chunk sampling, warm from the best checkpoint (no inference change).
- **H15 (engine, parallel, CPU):**
  - B1 bit-identical non-GEMM cuts;
  - B2 bench, then the kernel if ≥ 1.2×;
  - B3 left-context sweep.
- **H16 (needs owner sign-off):** A8 distillation from R0 (after A5 matches the tokenizer).

## E. Owner decisions requested
1. **R0 as a teacher for distillation (A8).** R0 was trained only on LibriSpeech, so no new data enters. It is still an external model's outputs.
2. **R0-E** (B7): running R0's published weights in our engine for a same-accuracy speed row.
3. **Hardware:** an AVX-512 VNNI host would give a measured 1.9× engine speedup (X1/X2) but adds a new hardware row.
4. **Data beyond LibriSpeech** (noise/RIR augmentation; larger corpora): still open from before.
