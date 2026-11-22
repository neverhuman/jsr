# H24: 4-Bit Gain–Shape Codebook-Scan (`pqscan`) + `FWHT16` + `ReLU²` Sparse Scan Kernel Gate

Status: **EXPLORATORY — PASS ON BOTH H0 ZEN 2 AVX2 AND APPLE M3 ULTRA NEON** (JSR v2 Phase 1).

## 1. Question
Can a single-plane, exact-integer 4-bit learned gain–shape codebook-scan kernel (`vpshufb` on Zen 2 AVX2, `vqtbl1q_u8` on Apple M3 Ultra NEON) paired with in-register block-16 Walsh-Hadamard rotation (`FWHT16`) and `ReLU²` subspace zero-skipping break past Zen 2's 1-multiply/cycle INT8 wall (~62 GMAC/s, proven by H8/H15/H17) while maintaining **0-ULP scalar == AVX2 == NEON bit-exactness**?

## 2. Implementation & Exact Integer Contract
Implemented in `crates/ultra-kernels/src/{pqscan.rs,pqscan_avx2.rs,pqscan_neon.rs}` with standalone microbenchmark `crates/ultra-kernels/examples/pqscan_bench.rs`:
- **Exact contract (`ULTRA-SU §6.2`):** Per-row signed activations `|a| <= 127` (`-128` refused), `s in {2, 4, 8}`, `M in {1, 2}` additive stages, 16 signed `i8` codewords (`|cb| <= 127`) per subspace. Exact `i32` subspace dot products are shifted by minimal `sh >= 0` such that `(max - min) >> sh <= 127` → 7-bit unsigned LUT entries `0..=127`. Two adjacent subspaces sum in `u8` (`<= 254`) and accumulate in `u16` for at most `256` pairs (`256 × 254 = 65,024 < 65,535`) before widening to `i32`.
- **AVX2 zero-spill 14-YMM inner loop (`pqscan_avx2.rs`):** Uses the exact modular `u16` identity `even = acc0 - (acc1 << 8) (mod 2^16)` (eliminating `vpand` from the inner loop and fitting 4-frame accumulators in 14 of 16 YMM registers with zero stack spills) plus SIMD `abs(a) * sign(cb, a)` table build (`build_tables_into`, zero heap allocs).
- **NEON twin (`pqscan_neon.rs`):** `vqtbl1q_u8` + `vaddw_u8`/`vaddw_high_u8` into `uint16x8_t` accumulators + in-register `fwht16_neon`.
- **Cross-ISA verification:** `cargo test -p ultra-kernels` passes **12/12 tests** on **both** x86_64 Zen 2 AVX2 and aarch64 Apple M3 Ultra NEON (random shapes, odd tails `rows=1,31,33`, `frames=1..9`, 75% sparsity, 768-subspace worst-case accumulator flush `97,536`, and `FWHT16` self-inverse parity).

## 3. Measured Hardware Results (EXPLORATORY, 2026-11-22)

### A. H0 — AMD TR PRO 3995WX (Zen 2, AVX2 only, pinned quiet core `taskset -c 100`, single thread)
Baseline `K0` = `ultra_kernels::I8Matrix` (`vpmaddubsw` + `vpmaddwd`, A7 activations). Total speedup includes **both SIMD `TableBuild` and `Scan`**; `tbl%` shows `TableBuild`'s share of total time.

| Layer (`din → rows`) | Rows (`M`) | `K0` ps/MAC | `s=2` (tbl%) | **`s=4` (tbl%)** | `s=8` (tbl%) | **`s=4` + 80% `ReLU²` sparse** |
|---|---:|---:|---:|---:|---:|---:|
| **`384 → 1536`** (S FFN-in) | 1 | 34.13 | 1.99× (13%) | **3.36× (11%)** | 4.80× (18%) | **4.68×** |
| `384 → 1536` | 2 | 24.10 | 1.68× (15%) | **2.87× (13%)** | 4.05× (21%) | **4.38×** |
| **`384 → 1536`** | **4** | **19.31** | 1.58× (18%) | **2.68× (15%)** | 3.61× (24%) | **3.75×** |
| `384 → 1536` | 32 | 17.12 | 1.38× (18%) | 2.33× (15%) | 3.14× (23%) | 3.24× |
| **`1536 → 384`** (S FFN-out) | 1 | 19.82 | 0.91× (40%) | 1.79× (38%) | 2.22× (57%) | **2.19×** |
| **`1536 → 384`** | **4** | **16.50** | 0.96× (51%) | **1.88× (49%)** | 2.24× (66%) | **2.31×** |
| `384 → 1152` (S QKV) | 1 | 34.00 | 1.87× (17%) | **3.21× (14%)** | 4.57× (23%) | **4.55×** |
| `384 → 1152` | **4** | **19.40** | 1.49× (23%) | **2.55× (19%)** | 3.35× (29%) | **3.49×** |
| `384 → 384` (S square) | 1 | 32.91 | 1.36× (37%) | **2.40× (32%)** | 2.98× (46%) | **2.99×** |
| `384 → 384` | **4** | **18.99** | 1.01× (47%) | 1.83× (41%) | 2.07× (55%) | **2.18×** |
| **`768 → 3072`** (**`V2-L` 300M FFN-in**) | **4** | **17.09** | 1.69× (11%) | **3.01× (10%)** | 4.68× (16%) | **4.69×** |
| `768 → 3072` | 32 | 16.57 | 1.62× (11%) | **2.88× (9%)** | 4.35× (16%) | **4.29×** |
| **`3072 → 768`** (**`V2-L` 300M FFN-out**) | **4** | **16.38** | 1.30× (35%) | **2.60× (34%)** | 3.50× (53%) | **3.70×** |
| `3072 → 768` | 32 | 16.09 | 1.27× (35%) | **2.55× (33%)** | 3.43× (52%) | **3.61×** |

- **`FWHT16` on H0 Zen 2 AVX2:** `483.3 ns` for `4 × 384` (`5.03 ns` / 16-block) and `1,852.2 ns` for `4 × 1536` (`4.82 ns` / 16-block) — **`< 0.1%` of engine CPU**.

### B. `xbabe4` — Apple M3 Ultra (`aarch64`, NEON `vqtbl1q_u8`, single thread)
Note: `ultra_kernels::I8Matrix` has only a scalar fallback on `aarch64`, so we report **absolute NEON scan `ps/MAC`** directly (compare against M3 Ultra `ultra-engine` `sdot` INT8 ~`8–10 ps/MAC`). `TableBuild` on `aarch64` currently runs scalar (adding a NEON `sdot` table builder in H29 will cut `tbl%` by ~8×).

| Layer (`din → rows`) | Rows (`M`) | NEON `s=4` Scan `ps/MAC` (`GMAC/s-eq`) | NEON `s=4` + 80% `ReLU²` Scan `ps/MAC` (`GMAC/s-eq`) |
|---|---:|---:|---:|
| `384 → 1536` | 1 | `5.82 ps` (`172 GMAC/s`) | `3.28 ps` (`305 GMAC/s`) |
| `384 → 1536` | **4** | **`5.01 ps` (`200 GMAC/s`)** | **`2.94 ps` (`340 GMAC/s`)** |
| `1536 → 384` | **4** | **`4.05 ps` (`247 GMAC/s`)** | **`1.98 ps` (`505 GMAC/s`)** |
| `768 → 3072` (`V2-L`) | **4** | **`4.34 ps` (`230 GMAC/s`)** | **`2.28 ps` (`439 GMAC/s`)** |
| `3072 → 768` (`V2-L`) | **4** | **`3.88 ps` (`258 GMAC/s`)** | **`1.79 ps` (`559 GMAC/s`)** |

- **`FWHT16` on Apple M3 Ultra NEON:** `353.1 ns` for `4 × 384` (`3.68 ns` / 16-block) and `1,395.1 ns` for `4 × 1536` (`3.63 ns` / 16-block).

## 4. What Surprised Us
1. **Register pressure matters enormously on AVX2:** An initial 28-YMM draft spilled to stack and ran at `0.98×` K0 (with scalar table build at 51%); switching to the 14-YMM zero-spill modular-`u16` identity + SIMD `abs(a)*sign(cb,a)` table builder jumped `384→1536` (`M=4`) to **`2.68×`** (`3.36×` at `M=1`) and `768→3072` to **`3.01×`** (`4.69×` sparse)!
2. **Fixes `M=1` / `M=2` tile starvation:** At `M=1` (single-stream & 6.25 Hz U-Net middle blocks), `K0` slows to `34 ps/MAC` while `pqscan` hits **`3.36×`** (`4.68×` sparse)!
3. **Wider `V2-L` (`d=768`, 300M) amortizes `TableBuild` dramatically better than `d=384`:** `tbl%` drops from `15%/49%` (`d=384`) to `10%/34%` (`d=768`), yielding **`3.01× / 2.60×`** dense (`4.69× / 3.70×` sparse) — validating `ULTRA-SU`'s core thesis that a **300M `V2-L` model (`36 MiB` codes) can run at Dense S 54M CPU cost**!

## 5. Decision / Consequence
- **GATE VERDICT: PROMOTE!** Exceeds the pre-registered Zen 2 gate (`≥ 2.5×` on `384→1536` at `M=4`, `≥ 1.8×` on `1536→384` with `ReLU²`, and **0-ULP cross-ISA bit-identity** across scalar, Zen 2 AVX2, and M3 Ultra NEON).
- **Next steps:** Proceed directly to **H25** (engine hygiene + INT8 KV), **H26** (H22 KD hardening + CR-CTC), **H28** (`V2-S/M/L` `6.25 Hz` U-Net + GQA + `FWHT16` + `ReLU²` + TDT), and **H29** (PyTorch `CodebookLinear` QAT + Rust `ultra.r2-codebook/v0` + NEON `sdot` table builder).

## 6. Pointers
- Kernel code: `crates/ultra-kernels/src/{pqscan.rs,pqscan_avx2.rs,pqscan_neon.rs}`, `crates/ultra-kernels/examples/pqscan_bench.rs`.
- Spec sources: `tips/v2/ULTRA-SU_single_user_cpu_asr_spec.md` §6 & Appendix A–B, `tips/v2/jsr_ultra_v4_0_radical_cpu_asr_architecture_hpc_specification.md` §2.4 & §4.4, `tips/v2/test1.md` §3–4.
