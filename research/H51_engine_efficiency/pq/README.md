# H51-PQ: staged go / no-go for a 1–2-bit codebook (pqscan) encoder, before any QAT

Status: **PRE-REGISTERED 2026-11-23 ~06:25Z** (lane h51-pq; results appended below the line, never edited above it).
Owner: h51-pq (experiments, analysis, QAT plan, verdict). Kernels and microbenchmarks: runtime-lane swarm #6 (`feat/h51-pqscan-vnni-kernels`); quiet timing: bench-open's xbabe0 queue. Plan: `JSR_Final_Engineering_Plan.md` §7.3, approved v3 plan "Phase 7", kill table §11.3 ("PQ enables a better model", "Sparsity helps").

## 1. Question
The single-stream INT8 engine streams 55.6 MB of int8 weights from DRAM every step (GEMM ≈ 70–90% of a step; a 40 ms step costs about what a 160 ms step does). A Zen 2 CCX has 16 MB of L3. Can a gain-shape PQ weight set of ~7–14 MB (1–2 code bits per weight, `pqscan` kernels from H24) stay L3-resident, beat INT8 on the real shapes **including the table build**, and keep WER after a bounded QAT? This wave decides whether QAT is worth starting. It trains nothing.

## 2. Model, data, arithmetic (fixed)
- Model B0 = A1 step 49,000 EMA (`A1-49000-ema-as-model.pt`, sha256 `4055c989…3a70c`); arch from `/mnt/xbabe3-external/jsr/evals/eval-lane/export/A1-49000-ema/int8/arch.json` (dense, d 384, 16 blocks, chunk 4, left 128).
- Target GEMMs = the engine's int8 set (130 Linears): per block `ff1.w1/w2, att.qkv, att.out, conv.pw1, conv.pw2, ff2.w1/w2`; plus `stem.out`, `joiner.enc_proj`, `joiner.out` (stem conv 2 traced only).
- Baseline arithmetic = engine INT8 emulated in PyTorch (per-row centred int8 activations, per-channel int8 weights, stem conv 2 as int8 im2col GEMM).
- PQ arithmetic = exact emulation of the `pqscan` integer contract (`crates/ultra-kernels/src/pqscan.rs`): centred int8 activations; int8 codewords with one global scale per matrix; per-frame 7-bit LUT with the minimal shift and the kernel's round bias; dequant epilogue `acc·sa·g[n] + ca·wsum[n] + b[n]` (the INT8 epilogue form).
- PQ fit = gain-shape additive PQ, codebooks shared by all rows of a matrix (the kernel layout), weighted k-means per subspace under the TRAIN input second moment Σ_c (activation-weighted distortion), closed-form per-row gain under the full Σ, 3 outer rounds, seed 0.
- Data: 300 TRAIN utterances (1.076 h, 30 speakers; 5 row groups × 3 train splits, seed 51): first 250 for Σ, last 50 held out for distortion. WER only on the 730-utt DEV panel (`docs/ultra/data/dev_selection_panel.json`, manifest `76fead74…`), chunk 4, greedy (max 8 symbols). **Test sets are sealed.**

## 3. Measurements and arms
1. **Speed (swarm #6 numbers, not re-measured here):** INT8 `linear_i8` vs pqscan incl. table build, real shapes, M = 1 / 2 / 4 / 16, cache-hot and whole-encoder sweep; quiet values only from xbabe0. Plus a quiet xbabe0 engine phase profile (`ultra-engine-eval --profile`, chunk 1 / 2 / 4) for the Amdahl fractions.
2. **Activation traces (TRAIN):** for every GEMM input, the fraction of all-zero groups (widths 2 / 4 / 8 and the pqscan skip unit, a pair of subspaces = 2s) per row, as 4-row chunk intersections and as random 16-row batches; quantisers: deployed centred int8 and zero-anchored symmetric int8; rotations: none, block-16 Hadamard (FWHT16), seeded random orthogonal. Crest factor and centre offset per row.
3. **Per-layer sensitivity (primary arm, 2 bits/weight):** s = 4, R = 2 stages, one matrix at a time, everything else INT8. Per layer: bytes, held-out relative output MSE (exact kernel arithmetic; INT8 reference; no-LUT-rounding ablation), ΔWER vs the INT8 baseline (pooled, clean, other; `ultra-score` paired bootstrap, 10,000 replicates, seed 20261122).
4. **Secondary arms (same protocol, run after 3):** 1 bit/weight (s = 4, R = 1) per layer; 2-bit groups jointly (all FFN, all attention, all conv, everything); ablations on a subset: unweighted k-means, s = 2 R = 1.

## 4. Pre-registered predictions
- P1 (sparsity): with the deployed centred quantiser, < 5% of s = 4 pairs are all-zero in the post-SiLU inputs (`ff*.w2`, `conv.pw2`) and ≈ 0% elsewhere; 4-row intersections are lower still; any rotation lowers them further. Sparse skipping is then not a lever for B0.
- P2 (quality): most single layers at 2 bits move panel WER by ≤ 0.10 pp; the outer blocks / `att.out` / `joiner.*` are the most sensitive; all-2-bit PTQ costs > 1 pp (so QAT is required).
- P3 (speed): at chunk 1 the INT8 step is DRAM-bound and an L3-resident code set gives a ≥ 2× GEMM gain; at chunk 4 the gain is smaller and the table build is a visible share.

## 5. Decision rules (fixed now)
- **GO to QAT** only if all hold:
  - (a) Projected whole-engine speedup ≥ 1.5× at chunk 1 **and** ≥ 1.2× at chunk 4, by Amdahl from the quiet xbabe0 profile and swarm #6's measured pqscan / INT8 ratios (table build included), for the proposed mixed-precision assignment. The per-step GEMM time is max(compute, bytes / bandwidth), with L3 bandwidth only if the full working set (codes + codebooks + side data + KV cache + conv state + other params) ≤ 12 MB, DRAM bandwidth otherwise. *(Amended 07:10Z, before any sensitivity run: the 06:25Z draft required ≤ 12 MB outright; at chunk 1 the byte reduction alone matters, so residency enters the bandwidth term instead of being a hard gate.)*
  - (b) That assignment covers ≥ 70% of encoder GEMM bytes with PQ, using layers whose individual 2-bit (or 1-bit) PTQ ΔWER ≤ +0.10 pp.
  - (c) The joint PTQ ΔWER of that assignment is ≤ +2.0 pp (repairable range for a bounded QAT; larger means the QAT budget is unknown → INCONCLUSIVE, not GO).
- **NO-GO** (stop, per §11.3) if the measured pqscan incl. table build is not faster than INT8 on the whole-encoder sweep at M = 4, or the projection in (a) fails even with an optimistic 2× table-build reduction, or (b) cannot be met.
- **INCONCLUSIVE** otherwise; state what measurement would settle it.
- Sparse skipping is adopted only if real TRAIN traces show ≥ 30% skippable pairs at the M the mode runs (P1 predicts not).

## 6. Compute
CPU only on H0 (nice 10, ≤ 12–24 threads, cores 44–63), the 3090 only in gaps with a chat claim. Traces ~10 min; one panel decode ~2–8 min (CPU, load-dependent); 130 layers × 2 arms ≈ 4–10 CPU-hours. No GPU training, no xbabe0 use except through bench-open.

---
## Results (appended after the runs)

**Outcome (09:00Z): NO-GO for the pre-registered target (R8). One conditional low-latency option remains, behind two kernel measurements (R9).** Every WER number is DEV 730-utt panel, chunk 4, native engine, ultra-score paired speaker-cluster bootstrap (10,000 replicates); test sets were never opened.

### R0. Verification of the instruments (before results)
- **Engine baseline reproduces eval-lane exactly:** `ultra-engine-eval` (e61121e, sha256 `968d3ea7…`), A1-49000 INT8, 32 streams, 160 ms packets: 7.385% pooled (S 850 / D 94 / I 123 of 14,448), **730 / 730 hypotheses identical** to eval-lane's `ultra-cold-wav` panel run.
- **Int8 packing:** my packer reproduces `weights.int8.safetensors` (q, scale, wsum) on all 131 GEMMs, incl. `stem_out`'s engine column order `f·ch + c`.
- **pqscan emulation is bit-exact:** PyTorch `PQLinear.acc_int` equals the Rust `CodebookMatrix::gemv` (scalar and AVX2) on real held-out activations for s = 4 R = 2 (`b8.ff1.w2` 1536→384, `b8.ff1.w1` 384→1536), s = 4 R = 1 (`b3.att.out`) and s = 8 R = 2 (`b12.conv.pw2`), 64 frames each.

### R1. Method actually run (deviations from §3 stated)
- **WER instrument.** Panel decodes use the native engine instead of a PyTorch decode (a PyTorch fp32 panel pass costs ~30 CPU-min on loaded H0; the engine costs ~5 CPU-min with 32 synchronised streams). Each PQ layer's dequantised weights `W' = g·cb_scale·Σ_r cb_int[r, c, code]` are written through the engine's own int8 packing into a variant of `weights.int8.safetensors` (new hash in a variant `arch.json`; every other byte identical). Two arithmetic gaps versus a native pqscan engine are therefore present and both are measured per layer on held-out TRAIN rows: the int8 requantisation of `W'` (present here, absent in pqscan) and the 7-bit LUT rounding (absent here, present in pqscan).
- **Trace.** 300 TRAIN utterances in six parallel parts (2 threads each): Σ from utterances 0–249, held-out rows (24 per call) from 250–299. The deployed variant (centred int8, no rotation) is counted on every row; the other quantiser / rotation variants on 16 aligned 4-row blocks per call (≤ 64 rows), stem conv 2 on 256 im2col rows per utterance.
- **Other hosts:** the per-layer sweep ran on x9 (EPYC 7H12) until the lead moved it off the training host at 07:53Z, then briefly on x6 (Xeon 8358, over its CPU cap, moved at 08:04Z), then on xbabe2 (3995WX, 8 workers under `fleet_guard`) plus three reverse-order H0 workers with a 90 s two-way row sync. x9 / x6 used a static musl build of the same source (`ultra-engine-eval-musl`, sha256 `88a18659…`). INT8 panel decodes on x9, x6 and xbabe2 are all **730 / 730 identical** to H0's (7.385%), so rows from every host are comparable.
- **Count:** there are 131 target Linears (128 block GEMMs + `stem.out` + `joiner.enc_proj` + `joiner.out`); §2's "130" is a miscount.

### R2. Activation traces: real group sparsity (TRAIN, 300 utts, 1.076 h)
Files: `results/trace/trace_sparsity.md`, `trace_summary.json` (per layer), `sparsity_by_kind.json`.

| GEMM input (K) | zero codes, centred | zero pairs s=4, centred, 1 row | symmetric: 1 row / 4-row / 16-row | centred + FWHT16 | centred + rand-orth | crest none / FWHT16 / rand | centre / half-range |
|---|---:|---:|---|---:|---:|---|---:|
| ff1/ff2.w1, att.qkv, conv.pw1 (384, LN inputs) | 0.75–1.06% | 0.00% | 0.00 / 0.00 / 0.00% | 0.00% | 0.00% | 4.5–5.4 / 3.3–3.4 / 3.1–3.2 | 0.12–0.20 |
| att.out (384) | 1.19% | 0.00% | 0.00 / 0.00 / 0.00% | 0.00% | 0.00% | 4.1 / 3.9 / 3.2 | 0.10 |
| ff1.w2 / ff2.w2 (1536, post-SiLU) | 0.01% | 0.00% | 10.1–11.3 / 4.5–5.1 / ≤0.003% | 2.5% | 0.00% | 14.1 / 4.9 / 3.6 | 0.88–0.89 |
| conv.pw2 (384, post-SiLU) | 0.02% | 0.00% | 0.01 / 0.00 / 0.00% | 0.00% | 0.00% | 9.9 / 3.8 / 3.2 | 0.87 |
| stem.out (2432) | 0.02% | 0.00% | 1.6 / 0.6 / 0.00% | 0.00% | 0.00% | 9.2 / 15.4 / 3.6 | 0.83 |
| stem conv 2 (1152, im2col) | 0.05% | 0.00% | 26.4 / n/a / 13.8% | 0.35% | 0.00% | 9.6 / 9.6 / 3.4 | 0.67 |
| joiner.enc_proj (384) / joiner.out (320) | 0.90 / 0.22% | 0.00% | 0.00% | 0.00% | 0.00% | 3.7 / 1.4 | 0.11 / 0.00 |

- With the deployed centred quantiser no pqscan skip unit is ever all-zero (max over all blocks 0.00%). Post-SiLU rows are one-sided, so the row centre sits at ~0.88 of the half-range and code 0 is mid-range.
- A zero-anchored quantiser would expose 10–11% zero pairs in the FFN-out inputs per row, 4.5–5% at M = 4 and ~0 at M = 16: far below the pre-registered 30% bar. Rotations remove what little there is (random orthogonal) or create a little by recentring (FWHT16, 2.5%).
- **P1 confirmed. Sparse skipping is not a lever for B0; dense PQ is.** Real sparsity would need ReLU² FFNs with zero-anchored codes and a retrain.

### R3. The 7-bit LUT and the centred activation codes (found while verifying distortion; affects the kernel contract)
- Held-out relative output MSE, block 8, exact pqscan arithmetic vs the same PQ weights with exact products (`results/gptq/lut_diag_b8.json`):

| layer (input) | s4R1 no-LUT → LUT | s4R2 no-LUT → LUT | s4R2 with FWHT16 rotation |
|---|---|---|---|
| ff1.w1 (LN output) | 0.258 → 0.258 | 0.074 → 0.077 | 0.080 → 0.082 |
| att.out (attention output) | 0.196 → 0.198 | 0.052 → 0.056 | 0.051 → 0.054 |
| **ff1.w2 (post-SiLU)** | 0.234 → **0.303** | 0.066 → **0.337** (5×) | 0.074 → 0.146 |

- Sweep rows show the same on every post-SiLU input: b0.ff1.w2 0.038 → 0.170, b0.conv.pw2 0.056 → 0.090, stem.out 0.013 → 0.139 (10×).
- **Mechanism:** a post-SiLU row is one-sided, so its centred codes sit near −127 for every "zero" activation. Every table entry then carries a large common offset (≈ −127·Σcb_j), the per-frame shift is set by it, and the rounding errors of the 2·C lookups add up. INT8 cancels the offset exactly in the epilogue (`c·wsum`); the rounded LUT cannot.
- **Consequence:** under the H24 contract as written (centred codes), the FFN-out, conv.pw2 and stem.out layers (≈ 47% of the INT8 bytes) lose most of their 2-bpw accuracy to table rounding. Fix candidates (measured next): zero-anchored symmetric codes for post-SiLU inputs (the kernel already takes signed codes; the epilogue drops the centre term), FWHT16 (halves it), per-subspace shifts or 16-bit tables (kernel changes). *(Outcome in R5: zero-anchored codes make it worse; round-to-nearest table entries fix it.)*
- The engine WER sweep uses PQ weights through int8 requantisation, whose error equals the exact-product PQ error (`pq_int8requant` = `pq_nolut` to 4 digits on every layer so far). It therefore measures PQ **with the LUT problem fixed**; the LUT rows say what the unfixed contract would add.
- **Activation encodings under the exact H24 LUT** (`results/gptq/act_diag.json`, relative output MSE, exact LUT / exact products):

| layer, format | centred (deployed) | zero-anchored symmetric | FWHT16 + centred | FWHT16 + symmetric |
|---|---|---|---|---|
| b8.ff1.w2, s4R2 | 0.337 / 0.066 | **6.01** / 0.066 | 0.146 / 0.074 | 0.173 / 0.074 |
| b8.ff1.w2, s4R1 | 0.303 / 0.235 | 1.03 / 0.235 | 0.282 / 0.264 | 0.284 / 0.264 |
| b0.conv.pw2, s4R2 | 0.090 / 0.056 | 0.688 / 0.056 | 0.069 / 0.060 | 0.070 / 0.060 |
| b0.conv.pw2, s4R1 | 0.222 / 0.207 | 0.290 / 0.207 | 0.230 / 0.227 | 0.230 / 0.227 |

  Zero-anchored codes are worse, not better: most subspaces of a post-SiLU row then have a range far below the frame's shift, so all their entries truncate to 0 and the contract's global round bias (`(M·C·(2^sh − 1)) >> 1`) adds back half a step per lookup that was never there. **FWHT16 + centred codes is the best encoding the current contract allows** (conv.pw2 almost recovered; FFN-out still 2× its exact-product error at 2 bpw). LUT variants (8-bit entries, round-to-nearest without the global bias) are measured next (`contract_diag.json`).

### R4. Speed: quiet Amdahl split and projection (`results/projection/`)
- **Quiet profile** (bench-open 14h, xbabe0, 3 reps on CPUs 4 / 24 / 28, spread < 1.5%; *provisional*: four idle runner VMs were up, re-run pending). Per step:

| chunk | step | encoder INT8 GEMM | share | attention | rest |
|---|---:|---:|---:|---:|---:|
| 4 (M 4) | 5.48 ms | 4.19 ms | 76.6% | 0.67 ms | 0.62 ms |
| 2 (M 2) | 4.74 ms | 3.88 ms | 81.8% | 0.50 ms | 0.36 ms |
| 1 (M 1) | 4.73 ms | 4.06 ms | 85.9% | 0.42 ms | 0.25 ms |

  The GEMM time does not grow with M: INT8 is DRAM-bound at all three chunks (55.6 MB / 4.06 ms = 13.7 GB/s). `--kv-int8` does not help (attention 0.42 → 0.51 ms at chunk 1).
- **PQ kernel inputs:** swarm #6's receipt (`swarm6_h0_pqscan_microbench.json`, H0 non-quiet, staged 4-centroid format, scalar LUT build: 1536→384 spends 81–91% in the LUT and regresses to 0.80× at 1 bpw) and H24's pqscan ratios (SIMD table build, s4R1; R2 taken as half). Per layer, time = max(table + scan, PQ bytes / 13.7 GB/s); INT8 layers keep their share of the measured 4.1 ms.
- **Projection** (`projection.json`, whole engine, speedup vs INT8):

| assignment | swarm format, scalar LUT | swarm format, LUT ×4 (SIMD, hypothetical) | H24 format, SIMD table |
|---|---|---|---|
| all GEMMs 2 bpw | 1.94 / 1.04 / 0.63 | 2.38 / 1.96 / 1.32 | 2.10 / 1.50 / 1.07 |
| all GEMMs 1 bpw | 2.97 / 1.96 / 1.24 | 3.45 / 2.96 / 2.13 | 3.16 / 2.36 / 1.71 |
| LN-input 2 bpw, rest INT8 | 1.52 / 1.29 / 0.99 | 1.55 / 1.49 / 1.29 | 1.55 / 1.33 / 1.10 |
| LN-input 1 bpw, rest INT8 | 1.73 / 1.55 / 1.29 | 1.74 / 1.67 / 1.52 | 1.74 / 1.59 / 1.38 |

  (chunk 1 / chunk 2 / chunk 4.) Working sets: 2 bpw 18.0 MB of weights (+ 6.7 MB fp32 KV) is not L3-resident; 1 bpw 9.6–11.1 MB is borderline with an INT8 KV cache (14–15 MB), so no row assumes residency.
- **Reading:** the chunk-1 gate (≥ 1.5×) is met by every PQ assignment, even LN-input-only at 2 bpw; the chunk-4 gate (≥ 1.2×) needs most bytes at 1 bpw (or the hypothetical SIMD LUT build). FFN-out (K > N) is the bottleneck twice: table build (speed) and LUT rounding on post-SiLU inputs (accuracy, R3).

### R5. LUT-contract variants (`results/gptq/contract_diag.json`)
Relative output MSE on held-out TRAIN rows, s4R2 unless noted (exact products → H24 7-bit truncate + global bias → 7-bit round-to-nearest, no global bias → 8-bit round-to-nearest):

| layer | centred codes | FWHT16 + centred | zero-anchored |
|---|---|---|---|
| b8.ff1.w2 (post-SiLU) | 0.066 → **0.337** → **0.109** → 0.079 | 0.074 → 0.146 → 0.083 → 0.076 | 0.066 → 6.01 → 1.06 → 0.115 |
| b0.conv.pw2 (post-SiLU) | 0.056 → 0.090 → 0.071 → 0.059 | 0.060 → 0.069 → 0.063 → 0.061 | 0.056 → 0.688 → 0.303 → 0.064 |
| stem.out (post-SiLU conv features) | 0.013 → **0.139** → 0.032 → 0.024 | 0.016 → 0.039 → 0.018 → 0.017 | n/m |
| b8.ff1.w1 (LN input, control) | 0.074 → 0.077 → 0.076 → 0.074 | 0.080 → 0.082 → 0.081 → 0.080 | n/m |
| b8.ff1.w2, s4R1 | 0.235 → 0.303 → 0.252 → 0.239 | 0.264 → 0.282 → 0.267 → 0.265 | 0.235 → 1.03 → 0.336 → 0.240 |

**Recommendation to the kernel owner:** in the table builder, round to nearest (`e = min(127, (v − mn + 2^(sh−1)) >> sh)`, offset = Σ mn, shift chosen so `(range + 2^(sh−1)) >> sh ≤ 127`). That is one add per entry and leaves the scan unchanged; it removes ~80% of the post-SiLU penalty. FWHT16 on post-SiLU inputs removes most of the rest. Do not switch to zero-anchored codes.

### R6. Per-layer 2-bpw sensitivity (s4R2, activation-weighted k-means; engine WER = weights-only PQ)
Full ranked table: `results/engine_sens/s4r2/table.md` / `table.json` (one row per GEMM: bytes, held-out relative output MSE for INT8, PQ with exact products, PQ under the H24 LUT, PQ through int8 requantisation, and ΔWER pooled / clean / other with the 95% paired speaker-cluster interval). Rows were produced on H0, x9, x6 and xbabe2 (decode identity verified on each); the sweep was moved twice at the lead's request to keep training hosts free.

**49 of 131 layers measured** (WER); most / least sensitive:

| layer | INT8 B → PQ B | rel MSE no-LUT / H24-LUT | ΔWER pooled [95%] |
|---|---|---|---|
| joiner.out | 160,320 → 56,332 | 0.055 / 0.055 | +2.665 [+2.345, +2.986] |
| joiner.enc_proj | 122,880 → 46,848 | 0.075 / 0.080 | +0.401 [+0.251, +0.557] |
| blocks.3.ff1.w1 | 589,824 → 178,176 | 0.056 / 0.062 | +0.235 [+0.130, +0.345] |
| blocks.15.ff1.w1 | 589,824 → 178,176 | 0.077 / 0.080 | +0.228 [+0.093, +0.365] |
| blocks.14.ff2.w1 | 589,824 → 178,176 | 0.080 / 0.085 | +0.221 [+0.089, +0.361] |
| blocks.2.att.out | 147,456 → 53,760 | 0.042 / 0.047 | +0.173 [+0.084, +0.271] |
| blocks.2.ff2.w2 | 589,824 → 201,216 | 0.069 / 0.312 | +0.132 [+0.042, +0.227] |
| blocks.15.conv.pw1 | 294,912 → 95,232 | 0.078 / 0.083 | +0.132 [+0.015, +0.247] |
| … | | | |
| blocks.1.conv.pw1 | 294,912 → 95,232 | 0.032 / 0.039 | -0.014 [-0.099, +0.074] |
| blocks.15.ff1.w2 | 589,824 → 201,216 | 0.057 / 0.207 | -0.014 [-0.082, +0.056] |
| blocks.4.ff1.w2 | 589,824 → 201,216 | 0.064 / 0.187 | +0.021 [-0.061, +0.105] |
| blocks.3.ff1.w2 | 589,824 → 201,216 | 0.067 / 0.372 | +0.021 [-0.063, +0.110] |
| blocks.0.ff1.w1 | 589,824 → 178,176 | 0.033 / 0.042 | +0.028 [-0.065, +0.118] |
| blocks.2.ff1.w1 | 589,824 → 178,176 | 0.053 / 0.057 | +0.028 [-0.079, +0.139] |

| kind | layers | mean ΔWER | mean rel MSE no-LUT | mean rel MSE H24-LUT |
|---|---:|---:|---:|---:|
| ff1_w2 | 6 | +0.044 | 0.053 | 0.272 |
| conv_pw1 | 5 | +0.060 | 0.043 | 0.049 |
| ff2_w2 | 6 | +0.069 | 0.052 | 0.302 |
| conv_pw2 | 5 | +0.075 | 0.057 | 0.109 |
| att_qkv | 6 | +0.078 | 0.043 | 0.057 |
| stem_out | 1 | +0.083 | 0.013 | 0.139 |
| att_out | 6 | +0.088 | 0.044 | 0.048 |
| ff1_w1 | 6 | +0.118 | 0.054 | 0.059 |
| ff2_w1 | 6 | +0.120 | 0.058 | 0.066 |
| joiner_enc_proj | 1 | +0.401 | 0.075 | 0.080 |
| joiner_out | 1 | +2.665 | 0.055 | 0.055 |

Coverage (rule b): layers with ΔWER ≤ +0.10 pp hold **68.6%** of the measured INT8 bytes.

**Output distortion for all 131 GEMMs** (`results/distortion/s4r2.jsonl`; held-out TRAIN rows, 2 bpw s4R2 k-means; relative output MSE by kind):

| kind | layers | INT8 | PQ exact products | H24 LUT | round-to-nearest LUT |
|---|---:|---:|---:|---:|---:|
| ff1.w1 / ff2.w1 | 32 | 0.00013 | 0.065 / 0.066 | 0.069 / 0.071 | 0.067 / 0.068 |
| att.qkv | 16 | 0.00009 | 0.060 | 0.091 | 0.069 |
| conv.pw1 | 16 | 0.00011 | 0.061 | 0.068 | 0.064 |
| att.out | 16 | 0.00008 | 0.049 | 0.053 | 0.051 |
| ff1.w2 / ff2.w2 (post-SiLU) | 32 | 0.0010 / 0.0009 | 0.065 / 0.065 | **0.331 / 0.383** | 0.140 / 0.131 |
| conv.pw2 (post-SiLU) | 16 | 0.00021 | 0.066 | 0.121 | 0.088 |
| stem.out | 1 | 0.00008 | 0.013 | 0.139 | 0.032 |
| joiner.enc_proj / joiner.out | 2 | 0.0001 | 0.075 / 0.055 | 0.080 / 0.055 | 0.077 / 0.055 |
| **all 131** | 131 | 0.00033 | **0.062** | **0.147** | **0.084** |

At 2 bpw the PQ error is ~200× the INT8 error on every kind. H24's LUT multiplies it by 2.4 on average and by 5–6 on FFN-out; round-to-nearest leaves 1.36× (FFN-out 2×, which FWHT16 closes, R5).



### R7. Joint PTQ runs (native engine, panel 730, chunk 4; `results/engine_sens/joint_*`, `joint_summary.json`)
| assignment (rest INT8) | layers | INT8 → PQ bytes | panel WER | ΔWER pooled [95%] | Δclean / Δother |
|---|---:|---|---:|---|---|
| **LN-input GEMMs at 1 bpw** (s4R1, k-means) | 80 | 33.03 → 5.65 MB | **30.16%** | **+22.77 [+21.87, +23.71]** | +19.56 / +27.37 |
| **LN-input GEMMs at 2 bpw** (s4R2, k-means) | 80 | 33.03 → 10.27 MB | **9.64%** | **+2.25 [+1.87, +2.63]** | +1.73 / +2.99 |

**Better PTQ (GPTQ/GPTVQ-style error feedback, same deployed format; `h51lib.PQFitGPTQ`).** Across all 80 LN-input GEMMs, held-out relative output MSE with exact products: 2 bpw 0.0603 (k-means) → **0.0417** (GPTQ, −31%); 1 bpw 0.2156 → **0.1539** (−29%) (`results/diagnostics/gptq_vs_kmeans_lnin.json`). Under the H24 LUT GPTQ can be worse on post-SiLU layers (block 8 `ff*.w2`), because error feedback widens subspace ranges; with round-to-nearest entries that interaction should go away (not yet measured).
| LN-input GEMMs at 1 bpw, **GPTQ** fits | 80 | 33.03 → 5.65 MB | 17.66% | +10.28 [+9.53, +11.06] | +8.22 / +13.22 |
| LN-input GEMMs at 2 bpw, **GPTQ** fits | 80 | 33.03 → 10.27 MB | **8.94%** | **+1.56 [+1.25, +1.88]** | +1.13 / +2.17 |
| post-SiLU GEMMs at 2 bpw (ff*.w2, conv.pw2, stem.out; k-means; weights-only = assumes the R5 LUT fix) | 49 | 22.17 → 7.61 MB | 8.22% | +0.84 [+0.63, +1.05] | +0.91 / +0.74 |
| **all 129 encoder GEMMs at 2 bpw** (k-means; weights-only = assumes the R5 LUT fix) | 129 | 55.20 → 17.89 MB | **10.85%** | **+3.47 [+3.02, +3.93]** | +2.83 / +4.37 |
| all 129 encoder GEMMs at 2 bpw, **GPTQ** fits (assumes the R5 LUT fix) | 129 | 55.20 → 17.89 MB | **9.75%** | **+2.37 [+2.02, +2.72]** | +1.91 / +3.03 |

### R8. Verdict against the pre-registered rules (§5)
| assignment | (a) chunk 1 ≥ 1.5× | (a) chunk 4 ≥ 1.2× (optimistic 2× table build) | (b) ≥ 70% of bytes, single-layer ≤ +0.10 pp | (c) joint PTQ ≤ +2.0 pp |
|---|---|---|---|---|
| LN-input at 1 bpw | ✓ 1.73–1.74× | ✓ 1.41–1.43× | ✗ 59% of bytes, and 1-bpw single layers were not screened | **✗ +22.8 pp (k-means), +10.3 pp (GPTQ)** |
| LN-input at 2 bpw | ✓ 1.52–1.55× | **✗ 1.13–1.18×** | ✗ 59% of bytes by construction | ✓ +1.56 pp (GPTQ) / ✗ +2.25 (k-means) |
| all encoder GEMMs at 2 bpw (needs the R5 LUT fix) | ✓ 2.25–2.37× | ✗ / borderline 0.97–1.20× | ✗ 68.6% of the measured bytes (49 / 131 layers) | ✗ +3.47 (k-means), +2.37 [+2.02, +2.72] (GPTQ) |

**Decision: NO-GO** for the pre-registered target (one PQ encoder of B0, ≥ 1.5× at chunk 1 **and** ≥ 1.2× at chunk 4, within a bounded QAT):
- the bit rate that buys the chunk-4 speed (1 bpw) costs **+10 to +23 pp** as PTQ, which is training a new model, not repairing one;
- at 2 bpw the chunk-4 gain stays at **≤ 1.2×** even with a table builder twice as fast as measured. FFN-out (K > N) is both the table-build bottleneck and, under H24's LUT, the accuracy bottleneck;
- sparse skipping is not available (R2).

**What survives (not a GO, a conditional option):** a **low-latency-only 2-bpw variant** (chunk 1 / 2) projects 1.5× (LN-input only) to ~2.3× (all GEMMs with the R5 LUT fix and a SIMD table build) at chunk 1. Its PTQ loss is at the edge of the repairable range: +1.56 pp for the 80 LN-input layers and +2.37 pp for all 129 encoder GEMMs, both with GPTQ fits. It is worth exactly one QAT pilot, and only after the kernel lane has measured, quiet on xbabe0, a whole-encoder 2-bpw sweep with round-to-nearest LUTs and a SIMD table build that confirms ≥ 1.8× at chunk 1 (QAT_PLAN.md, entry conditions).

### R9. Next steps
1. **No PQ QAT for v3.** Spend v3's speed effort on the sleep gate (H47) and multi-stream batching. One stream's INT8 GEMM is DRAM-bound at every chunk (R4), so small-M kernels cannot beat bandwidth.
2. **Kernel lane (runtime-lane, PR #89):** (a) round-to-nearest LUT entries without the global bias, plus FWHT16 on post-SiLU inputs (R5; in progress); (b) a SIMD table builder for K > N (FFN-out), since swarm #6's scalar LUT spends 81–91% of 1536→384; (c) then a **quiet xbabe0 whole-encoder 2-bpw sweep** at M 1 / 2 / 4 with the table build included. This is the one measurement that could reopen the low-latency case (needs ≥ 1.8× at chunk 1).
3. **bench-open:** 14h re-run with the runner VMs paused is queued as **14hr (~10:45Z)**. It finalises the Amdahl split; the provisional numbers are 3 reps within 1.5%.
4. **Only if 2(c) passes:** a single 2-bpw QAT pilot on the low-latency model (one group, GPTQ init, ~1 A100-hour, gate ≤ +0.05 pp per group; QAT_PLAN.md).
5. **Capacity:** for this model PQ does not free chunk-4 CPU, so the "wider encoder at equal watts" plan should be budgeted on the sleep gate's savings, not on PQ.
6. To finish the per-layer table: `launch_x9.sh s4r2 4 2 8` on any staged host resumes from the done rows (`engine_sens.py` skips finished layers on every host).

---
## Addendum A — one-day 4-bit weight screen (PRE-REGISTERED 2026-11-23 09:10Z, before any 4-bit fit, decode or kernel run)
Requested by the lead (independent audit). It is not a PQ format: it is plain 4-bit weights on the same INT8 data path.

**Question.** The engine is DRAM-bound (~4.1 ms per step streaming 55.6 MB, R4). Can 4-bit weights (27.8 MB) buy ≥ 1.2× at chunk 4, where 2-bpw PQ could not, at ≤ +0.30 pp WER?

**Arms (fixed now).**
- **Format:** weight-only int4, symmetric per output channel (q ∈ [−7, 7], one f32 scale per row). Activations stay the engine's centred per-row int8, and the epilogue stays the INT8 one (`acc·sa·s[n] + ca·wsum[n] + b[n]`), so a kernel only has to unpack nibbles in front of the existing `vpmaddubsw` tile.
- **Fits:** (A4-G) GPTQ, column by column with lazy block updates, H = TRAIN input second moment from the H51 trace (250 utterances) + 1% mean-diagonal damping, static column order, per-row scale = max|w|/7. (A4-R) round-to-nearest control with the same scales. Fallback arm, run **only if A4-G fails the WER rule**: (A4-G128) GPTQ with per-128-column group scales; its kernel cost is noted separately.
- **Layers:** all 129 encoder GEMMs (stem.out + 128 block GEMMs). `joiner.enc_proj` / `joiner.out` stay INT8 (R6: +0.40 / +2.67 pp alone at 2 bpw).
- **WER:** one joint native-engine decode per arm on the 730-utt DEV panel (chunk 4, greedy, 32 streams, 160 ms packets). int4 values go through the engine's own int8 packing; that requant error is ≤ 1/36 of a 4-bit step and is reported on held-out rows. ΔWER vs the INT8 baseline (7.385%) uses ultra-score's paired bootstrap (10,000 replicates, seed 20261122). Test sets stay sealed.
- **Speed:** a scratch AVX2 int4-weight × int8-activation small-M kernel (nibble-packed panels; unpack, then the INT8 tile's `vpmaddubsw` / `vpmaddwd` path and the same epilogue). It must be bit-exact to a scalar reference and to `gemm_i8` on the same int4-valued weights. It is timed against the engine's `gemm_i8` on the real shapes at M = 1 / 2 / 4, both cache-hot and as a whole-encoder sweep (129 distinct matrices, DRAM-streamed), on one pinned core of an idle Zen 2 CCX, interleaved INT8 / int4. Labelled **non-quiet**; a quiet assay is staged for bench-open.
- **Projection:** whole engine at chunk 4 / 2 / 1 from the 14h quiet profile, with T_gemm4(M) = max(measured hot compute at M, 27.8 MB / 13.7 GB/s). Non-GEMM time is unchanged.

**Stop rules (lead's, fixed now):** STOP if the best arm's joint ΔWER > +0.30 pp, **or** the projected chunk-4 whole-engine speedup < 1.2×.

**Predictions (pre-registered):**
- P4a: A4-G costs +0.10 to +0.30 pp; A4-R is clearly worse.
- P4b: about 1.6–1.8× at chunk 1 and 1.2–1.4× at chunk 2, but only **0.9–1.1× at chunk 4**. INT8 at M = 4 is already about as compute-bound as it is bandwidth-bound on Zen 2's single `vpmaddubsw` port (hot ~17–19 ps/MAC ≈ 3.8–4.2 ms vs 4.06 ms of DRAM time), and nibble unpacking only adds work. P4b predicts **STOP on the chunk-4 rule**.

*Amendment A.1 (09:11Z, before any measurement; the kernel run started 09:15:54Z).* A second kernel arm is added. **(K2)** keeps the weights as offset-binary nibbles (w + 8 ∈ 1..15) and uses them directly as the unsigned operand of `vpmaddubsw` against the broadcast signed activation codes. That removes the INT8 tile's per-row `abs` / `sign`. Because |a|·15·2 ≤ 3,810, i16 sums of 8 groups cannot overflow, so `vpmaddwd` is needed only once per 8 groups. The offset is removed exactly in the epilogue as 8·Σa per row. **(K1)** is the naive arm: unpack to signed int8, then the unchanged INT8 tile path. P4b's 0.9–1.1× at chunk 4 was reasoned for K1. For K2 the prediction is **1.3–1.6× at chunk 4**: it halves the `vpmaddubsw`-port work per MAC, so at M = 4 the step becomes bandwidth-bound at 27.8 MB. Both arms must be bit-exact to the scalar reference.

### A-R1. int4 kernel microbenchmark (H0 core 36, an idle CCX, **non-quiet**; `results/int4/bench_h0_core36.jsonl`, binary sha256 `2bafd36d…`)
Both int4 kernels are **bit-exact** on the same int4-valued weights for all 8 shapes at M = 1 / 2 / 4. Run 1 (core 36, build `2bafd36d…`) checked them against the engine's AVX2 `gemm_i8`. Run 2 (core 60, build `5a07473c…`, `results/int4/bench_h0_core60_scalarcheck.jsonl`) adds the pre-registered scalar-reference check: K1, K2 and the AVX2 `gemm_i8` all match the engine's portable scalar `gemm_i8` bit for bit in all 24 shape × M cases.
- **Run 2 reproduces run 1's hot timings:** median ratios 1.00 / 0.99 / 0.98 (INT8 / K1 / K2), and K2 : INT8 at M = 4 is 0.602 in both runs.
- **Run 2's sweep is not used:** it ran while H0's DRAM was far more contended (INT8 166 ms per step).
- **Cache-hot compute (GMAC/s):** INT8 `gemm_i8` 52–55 at M = 1 and 57–60 at M = 4. K1 (naive unpack) 39–51, slower than INT8. **K2 (offset-binary nibbles, i16 sums over 8 groups) 73–76 at M = 1 and 96–104 at M = 2 / 4: 1.4–1.75× INT8**, because the `vpmaddubsw` port does half the work per MAC.
- **Whole-encoder sweep** (129 distinct matrices, 55.2 → 27.6 MB per step, DRAM-streamed on loaded H0, interleaved, medians of 30 steps):

| M | INT8 | K1 | K2 |
|---:|---:|---:|---:|
| 1 | 20.06 ms | 10.32 ms (1.94×) | 8.82 ms (2.27×) |
| 2 | 28.36 ms | 15.83 ms (1.79×) | 15.02 ms (1.89×) |
| 4 | 43.89 ms | 24.33 ms (1.80×) | 25.34 ms (1.73×) |

  H0's DRAM is contended (its INT8 sweep is ~5× slower than quiet xbabe0), so every arm there is bandwidth-bound. The projection below takes compute from the hot numbers and bandwidth from the quiet profile instead.
- **Projection, whole engine** (14h quiet split; GEMM = max(hot compute, bytes / 13.7 GB/s); `results/int4/projection_int4.json`):

| chunk | INT8 step | K1 (naive) | **K2** |
|---|---:|---:|---:|
| 4 | 5.48 ms | 0.95× (compute-bound, 4.47 ms GEMM) | **1.55×** (2.24 ms GEMM; compute ≈ bandwidth 2.03 ms) |
| 2 | 4.74 ms | 1.52× | **1.64×** |
| 1 | 4.73 ms | 1.75× | **1.75×** (bandwidth-bound) |

  K1 behaves as P4b predicted (≤ 1.0× at chunk 4); K2 clears the 1.2× chunk-4 rule.
  - **A bound that needs no H0 clock assumption.** H0 (Threadripper PRO 3995WX) and xbabe0 are both Zen 2, so the K2 : INT8 hot-compute ratio should carry over: 2.24 / 3.78 = 0.59 at M = 4. xbabe0's whole INT8 chunk-4 GEMM took 4.19 ms, so its INT8 compute is at most that, and K2 compute is at most 0.59 × 4.19 = 2.48 ms. The chunk-4 step is then ≤ 1.285 + max(2.48, 2.03) = 3.77 ms, i.e. **≥ 1.45×**; a purely bandwidth-bound K2 gives the ceiling, 1.285 + 2.03 = 3.32 ms, **1.65×**.
  - **Overlap.** The model assumes compute hides under streaming, as it does for INT8 at chunk 4: there compute ≈ bandwidth, yet the measured 4.19 ms is within 3% of the max() model's 4.06 ms. Charging K2 the same 3% gives 1.52×. With no overlap at all, the step would be 0.99×; the INT8 measurement rules that case out. A quiet xbabe0 assay is staged for bench-open (`bench/h51-int4/`: `h51_int4.sh`, binary, source, SHA256SUMS).

### A-R2. Joint WER, per-row int4 (`results/int4/int4_{gptq,rtn}.json`, `.score.md`, `.score-input.jsonl`)
Native-engine decodes of the 730-utt DEV panel at chunk 4. All 129 encoder GEMMs are int4; the joiner stays INT8. Scored with ultra-score against INT8 (7.385%), 10,000 replicates, seed 20261122.

| arm | pooled WER | ΔWER (95% CI) | dev-clean Δ | dev-other Δ | held-out rel. MSE (mean of 129) | rule |
|---|---:|---|---:|---:|---:|---|
| INT8 (reference) | 7.385% | — | — | — | 0.00033 | |
| A4-R: RTN, per-row | 7.987% | +0.60 pp [+0.40, +0.81] | +0.68 | +0.49 | 0.0180 | STOP |
| **A4-G: GPTQ, per-row** | **7.925%** | **+0.54 pp [+0.34, +0.74]** | +0.65 | +0.39 | 0.0113 | **STOP** |

- **P4a is refuted:** A4-G costs +0.54 pp, not +0.10–0.30. GPTQ cuts held-out output error by 37% but WER by only 0.06 pp.
- **The requant is not the cause.** Held-out rel. MSE through the engine's int8 packing is 0.01128, against 0.01133 for native int4.
- **The error sits in the GEMMs that read post-nonlinearity inputs**, `ff*.w2` and `conv.pw2`, and grows with depth (blocks 11–14 are the worst).

| GEMM kind (count) | INT8 | RTN | GPTQ | GPTQ / RTN |
|---|---:|---:|---:|---:|
| stem.out (1) | 0.00008 | 0.0103 | 0.0025 | 0.25 |
| ff1.w1 / ff2.w1 (32) | 0.00013 | 0.0174 | 0.0077 | 0.44 |
| att.qkv (16) | 0.00009 | 0.0116 | 0.0077 | 0.66 |
| att.out (16) | 0.00008 | 0.0106 | 0.0051 | 0.48 |
| conv.pw1 (16) | 0.00011 | 0.0126 | 0.0074 | 0.59 |
| conv.pw2 (16) | 0.00021 | 0.0175 | 0.0126 | 0.72 |
| ff1.w2 / ff2.w2 (32) | 0.00098 | 0.0289 | 0.0215 | 0.75 |

  Inputs to `ff*.w2` are post-SiLU (heavy-tailed), and those layers already carry ~10× the INT8 error of the others.
- **As pre-registered, A4-G failed the WER rule, so the A4-G128 fallback was run** (A-R4). It uses GPTQ with 128-column group scales, each taken lazily from the error-updated weights when its group starts, as reference GPTQ's `groupsize` does. The per-row arms are unaffected by that code path (`int4_sens.py` 25bc5837…).

### A-R3. Kernel cost of 128-column group scales (`results/int4/bench_g128_h0_core52.jsonl`; H0 core 52, **non-quiet**; build `8179cd63…`)
- **K2g:** the K2 nibble stream plus one f32 scale per (row, 128-column group).
  - Per group, the i32 dot product (offset removed with that group's 8·Σa) is converted to f32 and FMA'd with the group's 16 column scales.
  - The epilogue is y = facc·sa + ca·wsum + b.
  - Bit-exact to a scalar reference with the same float order. K2 is again bit-exact to `gemm_i8`.
- **Cost:** hot compute is 4.6% above K2 at M = 4, 9.7% at M = 2 and 5.4% at M = 1. The scales add 1.72 MB (f32) or 0.86 MB (f16) to the 27.6 MB stream.
- **Projection** (run-1 K2 compute × the K2g / K2 ratio, the same model as A-R1):

| chunk | K2 (per row) | K2g128, f32 scales | K2g128, f16 scales |
|---|---:|---:|---:|
| 4 | 1.55× | **1.51×** | **1.51×** |
| 2 | 1.64× | 1.57× | 1.60× |
| 1 | 1.75× | 1.67× | 1.71× |

  Group scales do not threaten the speed rule.
  - **Clock-independent bound, as in A-R1:** K2g : INT8 hot compute at M = 4 is 0.613 in this run, so on xbabe0 K2g compute ≤ 0.613 × 4.19 = 2.57 ms. The chunk-4 step is then ≤ 1.285 + max(2.57, 2.16) = 3.85 ms, **≥ 1.42×**.

*Amendment A.2 (pre-registered 09:45Z, while A4-G128 was still fitting; no A4-G128 number had been seen).*
- **Trigger:** one follow-up arm runs **only if A4-G128 also fails the WER rule**.
- **The arm, A4-GM:** A4-G's settings (GPTQ, per-row int4) on the 81 encoder GEMMs other than `ff1.w2`, `ff2.w2` and `conv.pw2`. Those 48 post-nonlinearity GEMMs stay INT8.
- **Why these 48:** A-R2's distortion table, not any WER. They carry the largest int4 error (0.0126–0.0215 against ≤ 0.0077) and 61% of the summed GPTQ output error.
- **Bytes and projected speed:** 38.2 MB; 1.33× at chunk 4, 1.29× at chunk 2, 1.36× at chunk 1 (K2 + INT8 hot compute, the same model as A-R1).
- **Prediction:** +0.15 to +0.30 pp. That is A4-G's +0.54 scaled by the 39% of output error that remains.
- **Status:** exploratory. It is post hoc relative to A-R2 and is never the screen's primary result. The lead's two rules are applied to it as written.
- **Procedure (09:48Z, A4-G128 still unread):** to save time, the A4-GM fit started at once. The fit sees no WER. Its decode waits for A4-G128's result file and runs only if that arm's pooled ΔWER > +0.30 pp (`int4_sens.py --decode-if-fails`, sha256 69a544ea…). Otherwise the variant is deleted undecoded.
- **Outcome: not run.** At 09:51:59Z I stopped A4-GM by its process group during the fit, which sees no WER. This follows the owner's 09:50Z freeze on exploratory launches (relayed by codex-full-fleet at 09:51:23Z). No variant was written, nothing was decoded, and no A4-GM number exists. The command stays below for after the release.
  `int4_sens.py --arm gptq --threads 1 --keep-int8 '\.(ff[12]\.w2|conv\.pw2)$' --tag int4_gptq_gm`

### A-R4. A4-G128: stopped before its decode
- **What happened:** A4-G128 (GPTQ with 128-column group scales, lazy scales, `int4_sens.py` 25bc5837…) started at 09:32:45Z on one H0 core. At 09:53:59Z I stopped it by its process group, in the last blocks of its fit (block 13 or later of 16), on the lead's relay of the owner's directive: "wrap up ALL research … H51 stops now."
- **What exists:** no variant was written, nothing was decoded, and **no A4-G128 number exists**.
- **What it would have cost:** its speed cost is measured (A-R3: 1.51× at chunk 4).

### A-V. Status and verdict of the 4-bit screen (closed 2026-11-23 09:55Z)

| arm | bytes | ΔWER vs INT8 (95% CI) | chunk 4 | chunk 2 | chunk 1 | lead's rules |
|---|---:|---|---:|---:|---:|---|
| A4-R: RTN, per-row int4 | 27.6 MB | +0.60 pp [+0.40, +0.81] | 1.55× | 1.64× | 1.75× | speed pass, **WER STOP** |
| A4-G: GPTQ, per-row int4 | 27.6 MB | +0.54 pp [+0.34, +0.74] | 1.55× | 1.64× | 1.75× | speed pass, **WER STOP** |
| A4-G128: GPTQ, 128-col groups | 29.3 MB | not measured (stopped) | 1.51× | 1.57× | 1.67× | speed pass, WER unknown |
| (A4-GM: mixed, A.2) | 38.2 MB | not run (freeze) | 1.33× | 1.29× | 1.36× | — |

Speeds are projected whole-engine step ratios against INT8 (14h quiet profile split, H0 non-quiet hot compute, K2 / K2g kernels). The chunk-4 floor needs no clock assumption: ≥ 1.45× (K2) and ≥ 1.42× (K2g).

**Verdict.**
- **Speed: PASS.** Halving the weight bytes does buy chunk 4, but only with the right kernel. K2 uses offset-binary nibbles as `vpmaddubsw`'s unsigned operand with i16 deferral; it is bit-exact to the engine's AVX2 and scalar `gemm_i8` and projects 1.55× at chunk 4 and 1.75× at chunk 1. The naive unpack kernel K1 projects 0.95× at chunk 4.
- **Accuracy: FAIL as measured.** Per-row int4 costs +0.54 pp with GPTQ and +0.60 pp with RTN, both over the +0.30 pp rule. The pre-registered g128 fallback was stopped before its decode, so whether group scales rescue int4 PTQ is **unknown**.
- **Overall: not GO.** By the lead's pre-registered rules the screen STOPs on the measured arms. int4 is deferred to v3.1 (lead's relay, ~09:53Z).

**Predictions, scored.**
- P4a (+0.10–0.30 pp for A4-G): **refuted**, +0.54.
- P4b (≤ 1.1× at chunk 4, STOP on speed): **right for K1** (0.95×), **wrong for K2**.
- A.1's K2 prediction (1.3–1.6× at chunk 4): **held** at 1.55×, not yet confirmed quiet.

**What carries over to v3.1.**
- **The kernel.** K2 / K2g are the first weight format in this wave that clears chunk 4 on single-stream Zen 2, with the engine's own epilogue and exact integer arithmetic. The sources are `bench/h51_int4_bench.rs` and `bench/h51_int4g_bench.rs`; the quiet assay `bench/h51_int4.sh` was queued as bench-open 14j and deprioritised at the freeze.
- **Where the error is.** It is concentrated in the post-nonlinearity GEMMs:
  - `ff*.w2` rows have a crest factor (max / rms) of 4.10, against ~3.2 for the other kinds; 128-column groups bring it to 3.05 (`results/int4/weight_crest.json`).
  - `conv.pw2` rows are ordinary (3.21), so its larger error enters through its inputs.
  - Error grows with depth (blocks 11–14).

**v3.1 next steps, in cost order** (each pre-registered before it runs; none launched now):
1. **A4-G128 decode.** About 25 min of one CPU core: fit plus one native decode. This is the screen's open question.
2. **A4-GM** (A.2): the 48 post-nonlinearity GEMMs stay INT8; 38.2 MB, projected 1.33×. One fit plus one decode.
3. **Stronger PTQ:** act-order GPTQ, plus sequential calibration (Σ re-collected after earlier blocks are quantised). This wave used neither.
4. **int4 QAT** (`QAT_PLAN.md`, Addendum A): about 8k recovery steps from the GPTQ init, ~1 A100-hour, after the release train frees GPUs.
5. **Engine path:** for whichever variant passes (runtime-lane / Codex), using the spec below, then full DEV at chunk 4 / 2 / 1. Sealed sets only with the lead's approval.

**Engine-path spec (for v3.1; not a patch).**
- **Export:** `weights.int4.safetensors` beside the INT8 file. Per encoder GEMM:
  - `<name>.q4`: u8. Per 16-column panel, one 64-byte block per pair of K-groups. Nibbles are offset-binary (w + 8): the low nibble is group 2·gp, the high nibble group 2·gp + 1, each in the INT8 block's byte position.
  - `<name>.scale`: f32 per row, padded to 16; or `<name>.gscale`: one scale per row and 128-column group, laid out per panel.
  - `<name>.wsum`: f32, Σ_k of the dequantised row.
  - Bias and the joiner stay in the INT8 file.
- **Kernels:** K2 / K2g for M ≤ 4. For larger M (chunk-16 final pass, batch decode), unpack once into the INT8 tile.
- **Activations:** `quantize_row` as now, plus Σa per row (K2) or per row and group (K2g).
- **Parity:** an int4 export can be checked on today's INT8 engine. For per-row int4, feeding the int4 codes as int8 values with the int4 scale makes K2 and `gemm_i8` bit-identical. The requant used in this screen gave held-out rel. MSE 0.01128 against 0.01133 native.
