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

## 3. Staged 2-Bit / 4-Centroid Group-4 `pqscan` Microbenchmark Receipt (Centred vs Zero-Anchored Asymmetric)

- Receipt: [h0_pqscan_microbench.json](results/h0_pqscan_microbench.json) (`ultra.h51.pqscan-microbench/v2`, pinned single-core `taskset -c 100` on `h0-zen2-avx2-host`).
- Host role: `h0-zen2-avx2-host` (`AMD Ryzen Threadripper PRO 3995WX`, Zen 2 AVX2). **Preliminary non-quiet H0 single-core numbers**; official quiet CPU timings remain owned by `@bench-open` on `xbabe0`.
- Kernel: `crates/ultra-engine/examples/pqscan_bench.rs` (`289 LOC`) — exact integer 2-bit / 4-centroid group-4 codebook scan (`vpshufb` AVX2 modular-`u16` inner loop + scalar twin, **0-ULP bit-exact** verified on every shape, stage, and quantization variant before timing) evaluated on synthetic post-SiLU one-sided activations (`silu(z)`) comparing:
  - **Centred (`i8`):** engine's `quantize_row` (`c = (max+min)/2`, maps near-zero post-SiLU entries near `-127`).
  - **Zero-anchored asymmetric (`zero_anchored`, `i16` with exact `0.0 -> 0` anchor and full 256 asymmetric levels):** eliminates the post-SiLU `-127` offset & centre epilogue term (addressing `@h51-pq`'s `07:50:40Z` `2–10x` post-SiLU quantization error finding) while feeding the identical 7-bit `vpshufb` scan kernel.

| Shape `(M, K, N)` | Stages (bpw) | `gemm_i8` (ns) | Centred `lut / scan` (ns) | Centred vs `i8` | **`zero_anchored` `lut / scan` (ns)** | **`za` LUT Overhead** | **`zero_anchored` vs `i8`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(4, 384, 1536)` | 1 (0.5) | 39,771 | 4,763 / 3,984 | 4.55x | **4,850 / 3,980** | +1.8% | **4.50x** |
| `(4, 384, 1536)` | 2 (1.0) | 39,771 | 9,506 / 7,700 | 2.31x | **9,482 / 7,631** | -0.2% | **2.32x** |
| `(4, 384, 1536)` | 4 (2.0) | 39,771 | 18,722 / 15,062 | 1.18x | **18,971 / 15,096** | +1.3% | 1.17x |
| `(4, 1536, 384)` | 1 (0.5) | 38,935 | 18,791 / 3,117 | 1.78x | **18,976 / 3,132** | +1.0% | 1.76x |
| `(4, 1536, 384)` | 2 (1.0) | 38,935 | 38,023 / 6,228 | 0.88x | **38,197 / 6,223** | +0.5% | **0.88x (STOP)** |
| `(4, 1536, 384)` | 4 (2.0) | 38,935 | 112,880 / 12,414 | 0.31x | **116,132 / 12,501** | +2.9% | **0.30x (STOP)** |
| `(1, 384, 1536)` | 1 (0.5) | 17,637 | 1,216 / 1,531 | 6.42x | **1,233 / 1,555** | +1.4% | **6.33x** |
| `(1, 384, 1536)` | 2 (1.0) | 17,637 | 2,394 / 2,970 | 3.29x | **2,405 / 2,978** | +0.5% | **3.28x** |
| `(1, 384, 1536)` | 4 (2.0) | 17,637 | 4,740 / 5,932 | 1.65x | **4,777 / 5,926** | +0.8% | 1.65x |
| `(1, 1536, 384)` | 1 (0.5) | 12,768 | 4,733 / 1,333 | 2.11x | **4,832 / 1,365** | +2.1% | 2.06x |
| `(1, 1536, 384)` | 2 (1.0) | 12,768 | 9,410 / 2,622 | 1.06x | **9,476 / 2,629** | +0.7% | 1.05x |
| `(1, 1536, 384)` | 4 (2.0) | 12,768 | 18,691 / 5,271 | 0.53x | **18,923 / 5,353** | +1.2% | **0.53x (STOP)** |

### Key H51 Engineering Takeaway for `@h51-pq`
1. **Zero-anchored asymmetric LUTs are essentially free (`-0.2%` to `+2.9%`, median `+1.1%` LUT-build delta, identical `vpshufb` scan time, 0-ULP `scan_avx2 == scan_scalar`):** fixing the post-SiLU `-127` offset quantization error adds negligible runtime overhead while removing the centre epilogue term.
2. **Scan throughput is enormous (`2.6x–12.5x` faster than INT8)**, proving `vpshufb` memory bandwidth & ALU scaling works as expected.
3. **Table build still dominates whenever `K > N` (`1536 -> 384`)**: LUT cost scales as `O(stages * M * K)` while scan scales as `O(stages * M * K * N / 16)`. At `N = 384`, scalar LUT build consumes `78–90%` of wall time and **erases the gain at `>= 2` stages (`0.88x` at 2-stage, `0.30x` at 4-stage for `M=4`)**.
4. **Recommendation:** Only pursue 2-bit / staged PQ QAT (with zero-anchored asymmetric activations on post-SiLU layers) for **wide-output matrices (`K <= N`, e.g., FFN-in `384->1536`, QKV `384->1152`, or wider `V2-L`)** at `<= 2` stages (`0.5–1.0 bpw`), OR require an AVX2 SIMD LUT builder (`>= 4x` faster `build_lut`) before considering FFN-out (`1536->384`).

## 4. Code Pointers
- AVX-512 VNNI kernel: `crates/ultra-engine/src/kernels/x86_avx512.rs` (`204 LOC`)
- Small-M kernel: `crates/ultra-engine/src/kernels/small_m.rs` (`238 LOC`)
- Real-shape staged `pqscan` benchmark (centred + `zero_anchored`): `crates/ultra-engine/examples/pqscan_bench.rs` (`289 LOC`)
- ISA & `small_m` comparison benchmark: `crates/ultra-engine/examples/kernel_bench.rs` (`192 LOC`)
