# H51-PQ: QAT plan (one layer group at a time)

**Status: not to be started as the v3 PQ track.** The go / no-go in `README.md` is NO-GO for the pre-registered target: the bit rate that clears the chunk-4 speed gate (1 bpw) loses +22.8 pp as PTQ, and 2 bpw does not reach 1.2× at chunk 4 even with a 2× faster table build. This plan is kept for the **conditional low-latency-only follow-up** (chunk 1 / 2 at 2 bpw) and for any later model trained for codebooks from the start. Entry conditions: the kernel lane lands round-to-nearest LUT entries (R5) and a SIMD table build for K > N, measured quiet on xbabe0, and the 2-bpw joint PTQ result (README R7) sits in the repairable range.

## Format
- H24 `pqscan` (`ultra_kernels::CodebookMatrix`): 16 int8 codewords per 4-D subspace, shared by all rows of a matrix; per-row f32 gain; one int8 scale per matrix for all codebooks (the kernel sums all subspaces and stages directly); per-frame 7-bit LUT with one shift (the deployed rounding, bit-exact to the Rust kernel per R0).
- Conservative point: R = 2 (2 bpw). One-bit points only for groups that pass their screen: s = 8 R = 2 (same lookups and table MACs as s = 4 R = 1, lower error in this wave) or s = 4 R = 1.
- Layers that stay INT8 regardless (this wave): `joiner.out` (+2.67 pp at 2 bpw for one 160 KB layer) and `joiner.enc_proj` (+0.40 pp), and, under the current LUT contract, every post-SiLU input (`ff*.w2`, `conv.pw2`, `stem.out`: 7-bit LUT rounding of centred codes costs them 2–10× their weight error, R3) until the kernel gets FWHT16 plus a LUT fix. PQ candidates: the 80 LN-input GEMMs (`ff1/ff2.w1`, `att.qkv`, `conv.pw1`, `att.out`), 33.0 M weights = 59% of the INT8 bytes.

## Module (PyTorch, `train/pq.py`, owner trainer-lane with h51-pq; no engine code)
- `PQLinear`: buffers `codes (R, C, N)`; parameters `codebooks (R, C, 16, s)` and `gain (N)`, initialised from this wave's **GPTQ-style fit** (`h51lib.PQFitGPTQ`, Σ from TRAIN traces): 31% lower held-out output error than activation-weighted k-means at 2 bpw, and a joint PTQ start of **+1.56 pp** for the 80 LN-input GEMMs (k-means: +2.25 pp). That is the loss the QAT has to recover.
- Forward = the deployed arithmetic with straight-through estimators: centred per-row int8 inputs (as `qat.Int8Linear`), int8 rounding of the codebooks with the global scale, and in the last phase the exact LUT rounding (`acc_int` path) — R0 shows the emulation equals the kernel bit for bit.
- Codes re-assigned every 500 steps to the nearest codeword under the running per-subspace Σ_c (EMA of batch second moments), so codes follow the codebooks.
- Quant-Noise ramp (ULTRA_FRONTIER HPC-3): quantise a random share p of the group's subspaces per step, p 0.1 → 1.0 over the first half of the group's fine-tune, straight-through elsewhere; the rest of the run at p = 1 with the exact LUT rounding on.

## Loss and data
- A1's recipe (pruned RNN-T + CTC 0.3, dp 0.005 as adopted), plus self-distillation from B0: KL on CTC posteriors and an L2 on encoder outputs (weights fixed on the first group, then frozen). 960 h; no new data in this wave.

## Order and gates (pre-register before the first QAT run)
1. Group order = least sensitive per byte first, from the per-kind table (README R6; single-layer 2-bpw ΔWER means): (G1) `conv.pw1` + `att.qkv` (+0.06 / +0.08 pp per layer, 32 layers, 11.8 M weights); (G2) `att.out` (+0.09, 16 layers, 2.4 M); (G3) FFN-in `ff1.w1` + `ff2.w1`, the most sensitive LN-input kind (+0.12 / +0.14, 32 layers, 18.9 M). If the kernel lands the round-to-nearest LUT (R5), the post-SiLU FFN-out layers join **first**: they are the least sensitive with exact products (+0.05 / +0.07 per layer) and the largest K. Every group starts at 2 bpw (s4R2). 1 bpw is not attempted (+10 pp even with GPTQ, R7).
2. Per group: PTQ init → QAT ~1–2 epoch-equivalents (A1 runs ~2.2k audio-h/h per A100, so ≈ 0.5–1 A100-hour) → panel WER vs B0 INT8 with ultra-score's paired bootstrap.
   - Pass: ΔWER ≤ +0.05 pp point and the 95% upper bound ≤ +0.15 pp; cumulative ΔWER ≤ +0.10 pp.
   - Fail: the group stays INT8 (or goes back to 2 bpw if it was a 1-bpw attempt); continue with the next group.
3. After all groups: one joint epoch, then full DEV at chunk 4 **and** chunk 1 (and 2): ΔWER within the four-run spread (0.09 pp) on both, or INCONCLUSIVE.

- Cross-check against ULTRA_FRONTIER HPC-3's gates: ≤ +0.3 pp pooled vs the INT8 parent after 8,000 recovery steps (kill at > +0.5 pp), ≥ 1.8× fewer CPU-s per audio-s at chunk 4 and chunk 1. This plan's per-group gates are tighter; HPC-3's are the outer envelope.
- Do not add single-layer ΔWER as if independent (Final CPU plan §9.3): every candidate assignment is verified jointly (this wave's group runs, then QAT).

## Kill rules
- Two consecutive groups fail at 2 bpw → stop (format too lossy for this model size).
- The PQ share of encoder GEMM bytes falls below 70% → re-run the projection; stop if chunk 1 < 1.5× or chunk 4 < 1.2× with table build included (swarm #6's real-shape numbers, quiet xbabe0).

## Then (runtime-lane, after a passing QAT model exists)
- `ultra.r2-codebook/v0` prepared artifact; native pqscan path for the PQ layers with the INT8-style epilogue and the `[n][frame] → [frame][n]` layout; cross-ISA exactness (scalar == AVX2 == NEON) and engine-vs-PyTorch 730 / 730 panel identity; quiet xbabe0 CPU-s/s and W per stream at chunk 1 / 2 / 4 via bench-open.

## Budget
≈ 5 groups × (0.5–1 A100-h QAT + 0.1 h eval) + 1 joint epoch ≈ 4–7 A100-hours, plus CPU panel decodes. Small next to the trunk runs; can use any A100 gap.

## Addendum A: int4 weight QAT (v3.1 option; not started)
Context: README Addendum A.
- **Measured so far:** the K2 kernel projects 1.55× at chunk 4. PTQ is the limit: per-row int4 costs +0.54 pp with GPTQ and +0.60 pp with RTN. The g128 fallback was stopped before its decode.
- **Run QAT only if** A4-G128 and A4-GM (README A-V next steps 1–2) both miss the +0.30 pp rule.
- **Format:** the screen's best format. Activations stay centred per-row int8; joiner `enc_proj` / `out` stay INT8.
- **Module (PyTorch, owner trainer-lane with h51-pq):** `Int4Linear` = `qat.Int8Linear`'s activation path plus weights fake-quantised to int4 with a straight-through estimator.
  - Scales per row or per group are learnable (LSQ-style), initialised at max|w|/7 from the GPTQ fit.
  - The latent fp32 weights start from the GPTQ dequantised values (`int4_sens.py`, ~10 min on one CPU core).
- **Loss and data:** as above (A1's recipe plus self-distillation from B0), 960 h.
- **Budget:** 8,000 recovery steps (HPC-3's envelope), about 1 A100-hour; panel checks at 2k / 4k / 8k steps.
- **Gates (pre-register before the first run):**
  - Pass: joint panel ΔWER vs B0 INT8 ≤ +0.15 pp point and ≤ +0.30 pp upper bound at 8k steps.
  - Kill: > +0.5 pp at 8k steps, or a 4k → 8k gain below 0.05 pp while still above +0.30.
