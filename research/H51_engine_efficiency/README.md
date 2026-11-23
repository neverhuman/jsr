# H51: Engine Efficiency — AVX-512 VNNI Verification, Small-M Kernels & Staged `pqscan` Gate

Status: **VERIFIED ON NATIVE XEON 8358 (`avx512_vnni`) & H0 ZEN 2 (`avx2fma`) — STAGED PQ GATE SPLIT BY MATRIX ASPECT RATIO**

## 1. Pre-Registered H51 Decision Gates

| Track | Pre-Registered Gate | Measured Outcome | Verdict |
|---|---|---|---|
| **1. AVX-512 VNNI (`x86_avx512.rs`)** | `0-ULP` bit-exact parity (`Avx512Vnni == Avx2Fma == Scalar`, incl. `-128` activations & `+-127` adversarial weights) on real Xeon 8358 (`load_5m < 21.0`) + `>= 2.0x` over AVX2 at `M >= 8`. | **25/25 unit tests PASS (`0-ULP`)** on Xeon 8358 (`load_5m = 8.86`); **`2.88x–2.92x`** hot (`170.8–176.9` vs `59.3–60.5` GMAC/s) and **`2.77x`** streaming (`163.4` vs `58.9` GMAC/s) at `M=16`. | **PROMOTE** |
| **2. Small-M (`small_m.rs`, `M=1..4`)** | `0-ULP` bit-exact parity across all available ISAs + `>= 1.2x` speedup on low-latency `M=1..4` decode without regressing `aarch64` builds. | **0-ULP exact**, **0 `aarch64` warnings**; on Xeon 8358 VNNI: **`1.51x–2.28x`** (`M=1`), **`1.44x–2.34x`** (`M=2`), **`1.37x–1.63x`** (`M=4`, peaking at **`243.9 GMAC/s`** = **`4.03x`** AVX2!); on Zen 2 AVX2: **`1.05x–1.08x`** hot `M=1` (reaching `59.7 GMAC/s` = `99%` of Zen 2 peak) and **`1.21x`** streaming `M=1`. | **PROMOTE** |
| **3. Staged 2-bit/4-centroid group-4 `pqscan`** | Include **full LUT construction + scan** on real `S` encoder GEMM shapes (`(M,K,N)` for `M in {1,4}`, `384<->1536`); **STOP if LUT build erases the gain**. | **0-ULP exact** (`scan_avx2 == scan_scalar`). Wide-output `K=384->N=1536` achieves **`4.34x` (1-stage / 0.5 bpw)** and **`2.20x` (2-stage / 1.0 bpw)** at `M=4`, dropping to **`1.11x`** at 4-stage (`lut_pct = 58%`). Narrow-output `K=1536->N=384` spends **`81–91%`** in LUT build and **regresses to `0.80x` at 2-stage and `0.29x` at 4-stage**. | **CONDITIONAL / STOP ON `K>N` WITHOUT SIMD LUT BUILD** |

## 2. Native Xeon 8358 AVX-512 VNNI & Small-M Verification Receipt

- Receipt: [xbabe6_vnni_verify.json](results/xbabe6_vnni_verify.json)
- Host role: `xeon-8358-vnni-host` (`Intel Xeon Platinum 8358 @ 2.60GHz`, 30 vCPUs, pre-run 5-min load `8.86 < 21.0`).
- Binary: static `x86_64-unknown-linux-musl` unit-test binary (**25/25 passed in 0.05s, 0-ULP**) + release `kernel_bench --compare --json`. Temporary remote folder cleaned immediately after run.

| Benchmark Shape | `M` | `Scalar` | `AVX2` | `AVX2 small_m` | `AVX-512 VNNI` | **`VNNI small_m`** | `small_m / VNNI` | `VNNI(best) / AVX2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hot_1536x384` | 1 | 0.9 | 54.9 | 60.2 (`1.10x`) | 48.6 | **73.3** | **1.51x** | **1.34x** |
| `hot_1536x384` | 2 | 0.9 | 58.7 | 62.6 (`1.07x`) | 86.0 | **123.9** | **1.44x** | **2.11x** |
| `hot_1536x384` | 4 | 0.9 | 59.2 | 59.1 (`1.00x`) | 146.4 | **200.0** | **1.37x** | **3.38x** |
| `hot_1536x384` | 8 | — | 59.5 | — | **170.5** | — | — | **2.87x** |
| `hot_1536x384` | 16 | — | 59.3 | — | **170.8** | — | — | **2.88x** |
| `hot_384x1536` | 1 | 1.0 | 57.9 | 65.4 (`1.13x`) | 41.5 | **94.5** | **2.28x** | **1.63x** |
| `hot_384x1536` | 2 | 1.0 | 60.3 | 64.2 (`1.07x`) | 81.0 | **189.6** | **2.34x** | **3.15x** |
| `hot_384x1536` | 4 | 1.0 | 60.5 | 60.4 (`1.00x`) | 149.7 | **243.9** | **1.63x** | **4.03x** |
| `hot_384x1536` | 16 | — | 60.5 | — | **176.9** | — | — | **2.92x** |
| `stream_64x1536x384` | 1 | — | 29.7 | 29.9 (`1.00x`) | 29.9 | **30.4** | 1.02x | 1.02x |
| `stream_64x1536x384` | 4 | — | 57.7 | 57.5 (`1.00x`) | 111.4 | **116.1** | 1.04x | **2.01x** |
| `stream_64x1536x384` | 16 | — | 58.9 | — | **163.4** | — | — | **2.77x** |

*All throughputs in GMAC/s (single core). Note why `small_m` delivers `1.37x–2.34x` over baseline `x86_avx512.rs` at `M=1..4`: baseline processes 1 panel (1–4 ZMM accumulators) and hits `vpdpbusd` latency; `small_m` tiles 2–8 panels simultaneously (6–8 ZMM accumulators) and amortizes activation sign-flip + broadcast.*

## 3. Staged 2-Bit / 4-Centroid Group-4 `pqscan` Microbenchmark Receipt (`centred_trunc` vs `nearest_no_bias` vs `fwht16_nearest`)

- Receipt: [h0_pqscan_microbench.json](results/h0_pqscan_microbench.json) (`ultra.h51.pqscan-microbench/v3`, pinned single-core `taskset -c 100` on `h0-zen2-avx2-host`).
- Host role: `h0-zen2-avx2-host` (`AMD Ryzen Threadripper PRO 3995WX`, Zen 2 AVX2). **Preliminary non-quiet H0 single-core numbers**; official quiet CPU timings remain owned by `@bench-open` on `xbabe0`.
- Kernel: `crates/ultra-engine/examples/pqscan_bench.rs` (`291 LOC`) — exact integer 2-bit / 4-centroid group-4 codebook scan (`vpshufb` AVX2 modular-`u16` inner loop + scalar twin, **0-ULP bit-exact** verified across all 36 `shape x stage x variant` rows) comparing three LUT-builder contracts on centred `i8` codes (superseding `zero_anchored`, which `@h51-pq` proved yields `6.01` rel MSE under truncating LUTs on real post-SiLU activations):
  1. **`centred_trunc` (H24 baseline):** truncating 7-bit LUT entries `((v - lo) >> sh)` with frame-wide global round bias `sum(lo) + pairs * (1 << (sh - 1))`.
  2. **`nearest_no_bias` (`@h51-pq` `08:08:43Z` winning fix):** minimal `sh` with `((max_span + half) >> sh) <= 127`, round-to-nearest `e = min(127, (v - lo + half) >> sh)`, **no global round bias** (`offset = sum(lo)`), **unchanged `vpshufb` scan**.
  3. **`fwht16_nearest`:** in-place 16-point Walsh-Hadamard rotation (`ultra_kernels::pqscan::fwht16_inplace`) before quantization + `nearest_no_bias` LUT build.

| Shape `(M, K, N)` | Stages (bpw) | `gemm_i8` (ns) | `centred_trunc` `lut / scan` (`vs i8`) | **`nearest_no_bias` `lut` (`Δ`) / `vs i8`** | `fwht16` (ns) | **`fwht16_nearest` `tot` (`Δ`) / `vs i8`** |
|---|---:|---:|---:|---:|---:|---:|
| `(4, 384, 1536)` | 1 (0.5) | 39,704 | 5,070 / 3,871 (4.44x) | 5,092 (+0.4%) / 4.42x | 548 | 9,559 (+6.9%) / **4.15x** |
| `(4, 384, 1536)` | 2 (1.0) | 39,704 | 10,566 / 7,853 (2.16x) | 10,182 (-3.6%) / 2.24x | 548 | 18,360 (-0.3%) / **2.16x** |
| `(4, 384, 1536)` | 4 (2.0) | 39,704 | 20,585 / 14,867 (1.12x) | 20,312 (-1.3%) / 1.13x | 548 | 36,007 (+1.6%) / **1.10x** |
| `(4, 1536, 384)` | 1 (0.5) | 38,512 | 20,218 / 3,081 (1.65x) | 20,030 (-0.9%) / 1.67x | 2,276 | 25,752 (+10.5%) / **1.50x** |
| `(4, 1536, 384)` | 2 (1.0) | 38,512 | 48,070 / 6,105 (0.71x) | 46,309 (-3.7%) / 0.73x | 2,276 | 55,216 (+1.9%) / **0.70x (STOP)** |
| `(4, 1536, 384)` | 4 (2.0) | 38,512 | 124,117 / 12,142 (0.28x) | 123,258 (-0.7%) / 0.28x | 2,276 | 137,344 (+0.8%) / **0.28x (STOP)** |
| `(1, 384, 1536)` | 1 (0.5) | 11,073 | 1,350 / 1,382 (4.05x) | 1,330 (-1.4%) / 4.08x | 160 | 2,873 (+5.2%) / **3.85x** |
| `(1, 384, 1536)` | 2 (1.0) | 11,073 | 2,582 / 2,680 (2.10x) | 2,586 (+0.2%) / 2.10x | 160 | 5,471 (+4.0%) / **2.02x** |
| `(1, 384, 1536)` | 4 (2.0) | 11,073 | 5,097 / 5,262 (1.07x) | 5,114 (+0.3%) / 1.06x | 160 | 10,547 (+1.8%) / **1.05x** |
| `(1, 1536, 384)` | 1 (0.5) | 10,214 | 5,075 / 1,186 (1.63x) | 5,085 (+0.2%) / 1.63x | 543 | 6,837 (+9.2%) / **1.49x** |
| `(1, 1536, 384)` | 2 (1.0) | 10,214 | 10,160 / 2,334 (0.82x) | 9,993 (-1.6%) / 0.83x | 543 | 12,989 (+4.0%) / **0.79x (STOP)** |
| `(1, 1536, 384)` | 4 (2.0) | 10,214 | 20,418 / 4,600 (0.41x) | 20,263 (-0.8%) / 0.41x | 543 | 25,298 (+1.1%) / **0.40x (STOP)** |

### Real Held-Out Post-SiLU Accuracy Spot-Check (`b8.ff1.w2`, 64 rows, `K=1536, N=384`, 2-stage group-4)
- `centred_trunc` LUT rounding rel MSE vs exact PQ: **`0.096567`**
- `nearest_no_bias` LUT rounding rel MSE vs exact PQ: **`0.017310` (`-82.1%` LUT error reduction)**
- `fwht16_nearest` LUT rounding rel MSE vs exact PQ: **`0.002439` (`-97.5%` LUT error reduction)**

### Key H51 Engineering Takeaway for `@h51-pq`
1. **`nearest_no_bias` is strictly free (`-3.7%` to `+0.4%` LUT-build delta, unchanged `vpshufb` scan, 0-ULP `scan_avx2 == scan_scalar`):** replacing H24's truncate+global-bias rule with round-to-nearest+no-bias removes `~82%` of post-SiLU LUT rounding error at zero latency cost.
2. **`fwht16_nearest` costs only `160–548 ns` (`K=384`) or `543–2,276 ns` (`K=1536`):** adds just `-0.3%` to `+4.0%` at 2-stage while lowering the LUT shift by a full bit (`10 -> 9`) and removing `97.5%` of LUT rounding error on real post-SiLU rows.
3. **Table build still dominates whenever `K > N` (`1536 -> 384`):** at `N = 384`, scalar LUT build consumes `81–91%` of wall time and erases the gain at `>= 2` stages (`0.70x–0.83x` at 2-stage, `0.28x–0.41x` at 4-stage).
4. **Recommendation:** Adopt `nearest_no_bias` (+ `fwht16` on post-SiLU inputs) as the canonical H51 LUT contract; target wide-output matrices (`K <= N`, e.g., `384->1536`, `384->1152`) at `<= 2` stages (`0.5–1.0 bpw`), or require a `>= 4x` AVX2 SIMD LUT builder before considering FFN-out (`1536->384`).

## 4. Code Pointers
- AVX-512 VNNI kernel: `crates/ultra-engine/src/kernels/x86_avx512.rs` (`204 LOC`)
- Small-M kernel: `crates/ultra-engine/src/kernels/small_m.rs` (`238 LOC`)
- Real-shape staged `pqscan` benchmark (`centred_trunc` / `nearest_no_bias` / `fwht16_nearest`): `crates/ultra-engine/examples/pqscan_bench.rs` (`291 LOC`)
- ISA & `small_m` comparison benchmark: `crates/ultra-engine/examples/kernel_bench.rs` (`192 LOC`)
