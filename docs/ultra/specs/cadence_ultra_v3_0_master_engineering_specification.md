# CADENCE-ULTRA (v3.0)
## Master Engineering Specification: Two-Track, Event-Clocked, Ternary-Exact Streaming ASR for CPUs

**Date:** 2026-11-20 · **Target Repository:** `neverhuman/JopeDime` (or standalone workspace) · **Scope:** English v1 · **Inference:** 100% CPU-Only · **Training:** GPU + Massive CPU Fleet  
**Status:** Frozen Engineering Specification v3.0 — Nothing in Track B has been trained; all unmeasured targets and projections are explicitly tagged.

---

## 0. Executive Decision & Epistemic Contract

**The Mission.** Deliver the strongest measured English CPU-only streaming ASR system across both **word error rate (WER)** and **CPU cost (`core-seconds per audio-second`)**, while guaranteeing strict real-time streaming deadlines, zero silent transcript corruption, and reproducible receipts.

**The Strategy (Two Tracks, One Measured Frontier).**
1. **Track A — Shippable Pretrained Frontier (Weeks 1–3, Zero Training Risk):** Immediately build the frozen benchmark harness, irrevocable streaming API, and versioned adapters for the strongest existing CPU recognizers (Nemotron EN 0.6B, Moonshine v2 S/M, pinned Streaming Zipformer, FUTO ASR4ALL M/L, Phonon-2.5, Granite 5.0 TurboCTC, Parakeet Unified, Parakeet Redux). Measure every candidate on the actual deployment machine (`H0`) and ship the measured Pareto winners for **LIVE**, **EFFICIENT**, and **ACCURATE** operating profiles.
2. **Track B — `CADENCE-ULTRA` Breakthrough Student (Waves U00–U18):** Build and progressively prove a custom **Three-Clock, Event-Driven, Ternary Integer-Exact** recognizer that attacks the empirically measured **$5.0\times$ DRAM fixed-cost bottleneck** of streaming CPU ASR. Track B earns deployment **only** by beating Track A's measured Pareto frontier on `H0` under paired cluster bootstrap and frozen promotion margins. Every novel mechanism is isolated behind an independent kill gate with a shippable fallback.

**Epistemic Labels Used Throughout:**
| Tag | Meaning |
|---|---|
| `[VERIFIED-SOURCE]` | Checked against a primary external model card, paper, or benchmark file (still publisher-reported until reproduced on `H0`) |
| `[REPO-ARCHIVE]` | Verified in the 2026-11-20 `neverhuman/JopeDime` code-only archive; live checkout confirmation required in `P00` |
| `[SANDBOX-MEASURED]` | Measured empirically on two Intel Xeon AVX-512 VNNI/VBMI cloud hosts via `wstream.c` (Appendix A) |
| `[DERIVED]` | Reproducible arithmetic from stated dimensions or published measurements (`sizing_ultra.py`, Appendix B) |
| `[TARGET]` | Pre-registered engineering goal; not a forecast or achieved result |
| `[HYPOTHESIS]` | Novel mechanism gated by an explicit wave and kill/fallback rule (§10) |

---

## 1. Empirical Cost Law & Governing Principles

### 1.1 The Two-Term CPU Streaming Cost Law
Low-latency single-stream CPU ASR is dominated by **weight traversal across memory**, not arithmetic FLOPs:
1. **Incumbent Decomposition `[DERIVED]` from `[VERIFIED-SOURCE]`:** NVIDIA's `NeMo-Speech.cpp` CPU benchmark for Nemotron 0.6B (`Q8_0`, 8 threads, i7-11700K) reports $27.0\text{ ms}$ per $160\text{ ms}$ (2-frame) chunk and $42.7\text{ ms}$ per $1,120\text{ ms}$ (14-frame) chunk:
   $$\text{Marginal cost/frame} = \frac{42.7 - 27.0}{14 - 2} = 1.31\text{ ms}, \qquad \text{Fixed cost/chunk} = 27.0 - 2(1.31) = 24.38\text{ ms}\;(90.3\%\text{ of chunk time}).$$
   Streaming $\approx 618\text{ MB}$ of `int8` weights in $24.4\text{ ms}$ implies $\approx 25.3\text{ GB/s}$ DDR4 throughput. Wave `U00` verifies this directly with hardware LLC-miss and memory-bandwidth counters.
2. **Empirical 8-Point Kernel Fit `[SANDBOX-MEASURED]`:** Running `wstream.c` across $F \in \{1,2,3,4,6,8,12,16\}$ frames per pass over $1,536\text{ M}$ DRAM-resident weights (Appendix A) fits the two-term law $t_{\text{pass}}(W, F) = W \cdot (a + c \cdot F)$ with $R^2 = 0.9993$:
   * **DRAM `int8` (`vpdpbusd`):** $a = 96.38\text{ ps/wt}$, $c = 10.50\text{ ps/MAC}$ ($R^2 = 0.967$).
   * **DRAM 2-bit Packed Ternary (`vpmultishiftqb` + `vpdpbusd`):** $a = 19.36\text{ ps/wt}$ (**$5.0\times$ lower fixed fetch cost!**), $c = 10.98\text{ ps/MAC}$ ($R^2 = 0.9993$).
   * **Replication:** A second run on a 2.3 GHz Xeon reproduced all 18 checksums exactly and confirmed $2.08\times$–$2.56\times$ out-of-cache speedups at $F \in \{1, 2\}$.

### 1.2 Six Governing Design Laws
1. **Minimize Fixed Fetches First:** Pack large matrices at 2 bits (or $1.67\text{ bits}$ via `T-LUT3-Split`), amortize events per audio chunk in **one** weight sweep, and batch concurrent server streams into **one** weight sweep (driving per-stream cost down to the marginal term $c \cdot F$ with bit-exact batch invariance!).
2. **Preserve Dense Acoustic Ground Truth & State Integrity:** Never place CTC on a compressed $5\text{ Hz}$ grid ($T \ge U + R$ fails). Never skip upper-layer KV updates per event (corrupts attention caches). Keep precision and depth **utterance-fixed** by default.
3. **One Unambiguous Output Contract:** `Commit` means **truly irrevocable** within the API stream. A later pass may replace only mutable `Preview` text, or emit a separate `CorrectionSuggested` event—never silently overwrite `Commit` or `Final`.
4. **Spend Extra Compute Where Error Reduction Is Predicted:** Invoke the bidirectional utterance refiner (`Echo`) conditionally via a calibrated error-reduction controller trained on real $\Delta\text{edit\_errors}$, charging 100% of its actual CPU time and deadline slack.
5. **Cross-ISA Integer Exactness Without Saturation Bugs:** Enforce the `A7`/`A7z` 7-bit activation rule (with explicit $-64 \sum w_i$ zero-point correction) on `int8` layers; test `U8` 8-bit inputs on ternary layers (max pair sum $2 \times 255 = 510 \ll 32,767$); use exact 16-byte `T-LUT2` or dual 16-byte `T-LUT3-Split` LUTs on AVX2/NEON/WASM; use exact `max - score` softmax LUTs.
6. **Honest Data, Honest Accounting, Honest Adaptation:** Deduplicate all audio via a canonical recording-interval identity graph; store compressed authoritative audio with bounded local PCM caches; keep self-training OFF by default; adapt committed outputs only from user-confirmed labels under exact **sample-block LFO** $E_B^{\text{LFO}} = (I - S_{BB})^{-1} E_B$ and later-session WER holdouts (`recall` label).

---

## 2. Operating Profiles, Hardware Matrix & Promotion Rules

### 2.1 Hardware Profiles
`P00` binds `H0` to the owner's actual deployment machine before any comparison runs. Unmeasured hosts are marked `UNTESTED` (never extrapolated).

| Profile | Role | Required Capture | Thread Sweep |
|---|---|---|---|
| **H0** | Actual deployment CPU (primary selection) | Exact CPU/SoC, P/E/SMT topology, L1/L2/L3 sizes, RAM, OS, governor, ISA flags, thermal state | 1, 2, 4 physical P-cores |
| **H1** | Portable x86 laptop (AVX2, no AVX-512) | Must compile without AVX-512 flags | 1, 2, 4 P-cores |
| **H2** | High-throughput x86 desktop/server (i7-11700K / i9-12900K / Zen 5) | HW LLC-miss & memory-BW counters | 1 core, 2/4 cores, full socket (multi-stream) |
| **H3** | ARM desktop/laptop (Apple M-series, CPU-only verified) | Verify zero Metal/ANE dispatch | 1, 2, 4 P-cores |
| **H4** | Small ARM edge (Raspberry Pi 5 class) | Sustained 60-min thermal & backlog | 1, 2, 4 cores |

### 2.2 Four Operating Profiles `[TARGET]`
Initial targets are bound to `H0` before confirmatory evaluation. `CPU-s/audio-s` is **measured process-tree user+system CPU seconds divided by input audio seconds** (counting all helpers, refiners, and background threads—never `wall_ms * configured_threads`).

| Profile | Workload | CPU Cost Limit | Latency & Backlog Deadlines | Peak RSS Limit |
|---|---|---|---|---|
| **LIVE** | Interactive live captions / voice UI | $\le 4$ P-cores, $\le 0.40\text{ CPU-s/s}$ (Track B target: $\le 0.15$) | First correct partial p95 $\le 450\text{ ms}$; stable-word p95 $\le 700\text{ ms}$; finalization p95 $\le 750\text{ ms}$; backlog $< 1$ chunk | $\le 1.5\text{ GiB}$ (Track A) / $\le 320\text{ MiB}$ (Track B L) |
| **EFFICIENT** | Background desktop / Pi 5 coexistence | 1 P-core preferred, $\le 0.20\text{ CPU-s/s}$ (Track B M target: $\le 0.08$) | Stable-word p95 $\le 1.5\text{ s}$; finalization p95 $\le 1.2\text{ s}$; zero growing backlog | $\le 512\text{ MiB}$ (Track A) / $\le 150\text{ MiB}$ (Track B M) |
| **ACCURATE** | Highest-accuracy dictation / finalized text | $\le 4$ P-cores, $\le 1.0\text{ CPU-s/s}$ (Track B L+Echo: $\le 0.20$) | Stable-word p95 $\le 1.5\text{ s}$; finalization p95 $\le 1.2\text{ s}$; file mode reported separately | $\le 1.5\text{ GiB}$ (Track A) / $\le 350\text{ MiB}$ (Track B L) |
| **COMMIT** | Voice agents acting on irreversible words | Same as LIVE | Irrevocable `Commit` events scored directly; zero mutations permitted | Same as LIVE |

### 2.3 Track B Accuracy & Speed Gates (`ULTRA-L` on Frozen 8-Set Macro Average & Fresh Domain Set `[TARGET]`)
| Gate | Committed WER ($160\text{ ms}$) | Committed WER ($560\text{ ms}$) | Final WER (Selective Echo) | 1-Core CPU-s/s on H2 (i9-12900K) | p99 / $160\text{ ms}$ Chunk (1 / 2 thr) | Cross-ISA Determinism |
|---|---:|---:|---:|---:|---:|---|
| **G1 (Viable Ship)** | $\le 7.6\%$ | $\le 6.9\%$ | $\le 6.4\%$ | $\le 0.22$ | $\le 55 / 32\text{ ms}$ | Exact (`0` diff) |
| **G2 (Pareto Leader)** | $\le 7.1\%$ | $\le 6.4\%$ | $\le 5.9\%$ | $\le 0.15$ | $\le 45 / 26\text{ ms}$ | Exact (`0` diff) |
| **G3 (Stretch)** | $\le 6.6\%$ | $\le 5.9\%$ | $\le 5.4\%$ | $\le 0.10$ | $\le 32 / 19\text{ ms}$ | Exact (`0` diff) |

* **Win Condition:** `ULTRA-L` passes **G2** at both latencies while `ULTRA-M` matches the Track A quality reference within $+0.30$ percentage points at $\le \frac{1}{3}$ of its measured process-tree CPU-seconds on `H0`.
* **Frozen Promotion Rules (Confirm Partition, $m$ Confirmatory Comparisons, One-Sided Bootstrap Confidence $1 - 0.05/m$):**
  * *Quality-preserving speedup:* $\ge 10\%$ measured full-pipeline CPU-s reduction; upper bound on WER degradation $\le +0.10$ percentage points; critical slices & SLOs pass.
  * *Accuracy promotion:* $\ge 3\%$ relative WER reduction with upper bound $< 0$; added CPU-s $\le 10\%$ (unless selecting `ACCURATE`).
  * *Compression / thinning promotion:* $\ge 15\%$ full-pipeline CPU-s reduction within the non-inferiority bound (smaller file size alone never passes).

---

## 3. Track A: Pretrained Baseline Tournament (Weeks 1–3)

Before spending a single GPU-hour on Track B, build the common benchmark and versioned process-isolated adapters for the pinned first-wave catalog:
1. **Nemotron Speech Streaming EN 0.6B** (`NeMo-Speech.cpp`, `Q8_0`, 160 ms & 560 ms, CPU forced) — primary cache-aware quality anchor `[VERIFIED-SOURCE]`.
2. **Moonshine v2 Small & Medium** (shipped quantized CPU runtime, actual update cadence) — compact streaming challengers `[VERIFIED-SOURCE]`.
3. **Pinned Streaming Zipformer** (`sherpa-onnx`, exact checkpoint + training corpus + chunk config pinned) — native Rust-binding & reproducible recipe control `[VERIFIED-SOURCE]`.
4. **FUTO ASR4ALL Medium & Large** (`ONNX`/`ExecuTorch`, default & high-context tiers, correction pass explicitly charged & ablated) — single-core CPU efficiency challengers `[VERIFIED-SOURCE]`.
5. **Phonon-2.5 English, Granite 5.0 TurboCTC, Parakeet Unified EN 0.6B, Parakeet Redux** — low-bit, offline/buffered, and existing ternary challengers (evaluating both file mode and bounded-window live mode, explicitly capping replay amplification $r L^2 / (2h)$).
6. **Optional ecosystem control:** `faster-whisper` `small.en` `INT8`.

Every candidate is locked in `CANDIDATES.lock` (exact repo commit, artifact SHA-256, tokenizer SHA-256, compiler flags, forced CPU verification, license status). Two views are reported: **Controlled Recognition View** (common audio, normalizer, context) and **Product View** (each runtime's best native VAD/endpointing/formatting path under identical user deadlines).

---

## 4. Track B: `CADENCE-ULTRA` Architecture (Bug-Free Synthesis)

```
 16 kHz int16 PCM (stateful resampler if 44.1/48 kHz)
   │
 ┌─▼─────────────────────────────────────────────────────────────────────────────┐
 │ CLOCK 0 · COCHLEA (100 Hz)                                                    │
 │ Control: strict-IEEE float32 log-mel (zero subnormals, -ffp-contract=off)     │
 │ Candidate (U02): 100% integer Q31 DC-block/preemph/FFT → Q15 80 Mel           │
 │   → shift EMA → exact isqrt64 PCEN(α=1, r=1/2) [+ Goertzel wire-61 mod bands] │
 └─┬─────────────────────────────────────────────────────────────────────────────┘
   │ 100 Hz A7[80] (4 stacked frames → 25 Hz)
 ┌─▼─────────────────────────────────────────────────────────────────────────────┐
 │ CLOCK 1 · REFLEX (Always-On, Strictly Causal, Ternary + Integer TimeMix)      │
 │ Stem ↓2 (50 Hz) → ↓2 (25 Hz) → R1 (DWConv k=9 + TimeMix{2,4,6} + Ternary FFN) │
 │ → ↓2 (12.5 Hz) → R2 (+ causal SWA-32 every 4th block)                         │
 │ ──► DENSE AUX CTC HEAD (1,025 pieces @ 12.5 Hz, T >= U+R verified!)           │
 │     (Supplies frame-dense CTC posteriors, fallback text & gate features)      │
 └─┬─────────────────────────────────────────┬───────────────────────────────────┘
   │ h_t (R16[d_r])                          │ p(blank), H, top-4 tokens, flux
 ┌─▼─────────────────────────────────────────▼───────────────────────────────────┐
 │ ADAPTIVE EVENT GATE · GP-ASR `(sym raw ...)` (Verified Binary `step(a, c)`)   │
 │ 8 Q7 features (col(5)=Biasing-Trie Active, col(7)=dial θ ∈ [-1, 1])           │
 │ Mass-weighted pooling (explicitly lossy summary); max speech gap 160–320 ms   │
 │ Runtime-only triggers (ZERO reference-dependent fallbacks!)                   │
 └─┬─────────────────────────────────────────────────────────────────────────────┘
   │ Chunk-amortized events (fixed audio grid C ∈ {1,2,4,7,14} frames = 80..1120 ms)
   │ [Server mode: cross-stream batched into ONE weight sweep, bit-exact!]
 ┌─▼─────────────────────────────────────────────────────────────────────────────┐
 │ CLOCK 2 · CORTEX (Event-Clock Ternary Transformer, Utterance-Fixed Precision) │
 │ Primary: Dense Ternary (`T1` + 15% `T2`) · Optional Branch (U06): Sparse MoE  │
 │ 12–20 blocks: Chunk-Causal SWA (128 ev & 25.6 s horizon, TimeBucket-32 bias)  │
 │   + Causal Event DWConv(k=5) + FFN (d → 4d → d, ReLU6)                        │
 │ → Stateless 2-Token Integer ReLU6 RNN-T (4,096 cased+punct, FULL projection!) │
 │   [Optional Branch (U11): FastEmit TDT (d ∈ {0..4}) with duration debt]       │
 │ ──► LIVE EDITABLE: `Preview`  |  COMMIT MODE: Irrevocable `Commit`            │
 └─┬─────────────────────────────────────────────────────────────────────────────┘
   │ Cached Cortex events h_k^cortex (+ optional 12.5 Hz Reflex states)
 ┌─▼─────────────────────────────────────────────────────────────────────────────┐
 │ CLOCK 3 · SELECTIVE ECHO (2–4 Bidir Ternary Layers, Calibrated Gate q ≈ 0.25) │
 │ Triggered ONLY when predicted Δedit_errors justifies measured CPU & deadline! │
 │ ──► EDITABLE: `Final` (closes segment)  |  COMMIT: `CorrectionSuggested`      │
 └───────────────────────────────────────────────────────────────────────────────┘
   │ User-confirmed corrections + Phrase Biasing Trie (Self-training OFF!)
 ┌─▼─────────────────────────────────────────────────────────────────────────────┐
 │ APEX-EQ DOSSIER (Background Thread, Confirmed Labels Only, `recall` label)    │
 │ Biasing Trie (boosts Gate col(5)=1) · D1 prior · D2 r-dim projected ridge     │
 │ Guarded by EXACT sample-block LFO: E_B^LFO = (I - S_BB)^-1 E_B + H460 QR      │
 └───────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Verified Presets & Exact Sizing Table `[DERIVED]`
Computed by `sizing_ultra.py` (Appendix B) using the fitted AVX-512 Xeon DRAM ternary law ($a = 19.36\text{ ps/wt}$, $c = 10.98\text{ ps/MAC}$, `P2 = 1.15`, `EV = 5.6/s`, `RHO160 = 3.75 wake-ups/s`, Reflex at true $12.5\text{ Hz}$ across $6.25$ wake-ups/s, Selective Echo at $q = 0.25$, and a **conservative $1.8\times$ end-to-end system allowance** for attention $QK^\top/AV$, RMSNorm, softmax, DWConv, and joiner steps).

| Quantity | **ULTRA-S** (Edge / Pi 5) | **ULTRA-M** (Laptop Default) | **ULTRA-L** (Dense Flagship) | **ULTRA-L-MoE** (Wave U06 Branch) |
|---|---:|---:|---:|---:|
| Reflex ($d_s / d_r / n_r$) | $192 / 256 / 8$ | $256 / 384 / 12$ | $256 / 512 / 12$ | $256 / 512 / 12$ |
| Cortex ($d / n / \text{heads}$) | $512 / 12 / 8$ | $1024 / 16 / 16$ | $1280 / 20 / 16$ | $1280 / 20 / 16$ ($1\text{sh}+4\text{ses}+4\text{ev}$) |
| Echo blocks ($e$) / Joiner ($J$) | $2 / 384$ | $3 / 512$ | $4 / 640$ | $4 / 640$ |
| Reflex params (`tern` / `i8`) | $5.91\text{ M}$ / $0.44\text{ M}$ | $18.87\text{ M}$ / $0.72\text{ M}$ | $32.64\text{ M}$ / $0.93\text{ M}$ | $32.64\text{ M}$ / $0.93\text{ M}$ |
| Cortex Total / Active ternary | $37.75\text{ M}$ / $37.75\text{ M}$ | $201.33\text{ M}$ / $201.33\text{ M}$ | $393.22\text{ M}$ / $393.22\text{ M}$ | **$786.43\text{ M}$ / $393.22\text{ M}$** |
| Echo ternary / Heads+IO `i8` | $6.29\text{ M}$ / $5.96\text{ M}$ | $37.75\text{ M}$ / $10.11\text{ M}$ | $78.64\text{ M}$ / $12.89\text{ M}$ | $78.64\text{ M}$ / $12.89\text{ M}$ |
| **Total Parameters** | **$56.35\text{ M}$** | **$268.78\text{ M}$** | **$518.32\text{ M}$** | **$911.53\text{ M}$** |
| Packed Weight RAM (`T1`+15% `T2`) | $20.8\text{ MB}$ | $85.0\text{ MB}$ | $158.9\text{ MB}$ | $271.9\text{ MB}$ |
| Disk Download (5 trits/byte) | $17.9\text{ MB}$ | $70.2\text{ MB}$ | $129.9\text{ MB}$ | $220.3\text{ MB}$ |
| Per-Stream State Arena (incl. 30 s PCM) | $2.9\text{ MB}$ | $5.9\text{ MB}$ | $8.5\text{ MB}$ | $8.5\text{ MB}$ |
| Cortex Fixed Fetch / Wake-Up | $0.72\text{ ms}$ (L3) | $4.48\text{ ms}$ | $8.75\text{ ms}$ | $8.75\text{ ms}$ |
| Cortex Marginal / Event | $0.54\text{ ms}$ | $2.66\text{ ms}$ | $5.11\text{ ms}$ | $5.11\text{ ms}$ |
| **Projected 1-Core RTF @ 160 ms** (`kernel` / `$1.8\times$`) | $0.008$ / **$0.014$** | $0.038$ / **$0.069$** | $0.073$ / **$0.131$** | $0.073$ / **$0.131$** (+routing) |
| **160 ms Chunk, 1-ev / 3-ev ($1.8\times$, 1 core)** | $2.8$ / $4.7\text{ ms}$ | $14.5$ / $24.1\text{ ms}$ | $27.8$ / $46.2\text{ ms}$ | $28.5$ / $47.5\text{ ms}$ |
| **10 s Echo Pass (1 core, `kernel` / `$1.8\times$`)** | $4.6$ / $8.3\text{ ms}$ | $27.5$ / $49.6\text{ ms}$ | $57.4$ / $103.3\text{ ms}$ | $57.4$ / $103.3\text{ ms}$ |
| **Batched Server Streams / Core ($1.8\times$)** | **$\approx 138$** | **$\approx 30$** | **$\approx 16$** | **$\approx 15$** |

### 4.2 Detailed Component Specifications & Bug Fixes
1. **Clock 0 (Cochlea):**
   * *Control arm:* Standard 80-band float32 log-mel (`-ffp-contract=off`, flush-to-zero disabled, explicit tiny noise floor $\epsilon = 10^{-10}$ preventing x86 subnormal stalls).
   * *Candidate arm (Wave U02):* 100% integer Q31 DC-block (`pole_q15=32735`) $\to$ pre-emphasis (`coef_q15=31785`) $\to$ $25\text{ ms}/10\text{ ms}$ Hann $\to$ 512-pt Q31 FFT $\to$ Q15 80-mel `u64` energies $E_b[t]$ $\to$ shift EMA $M_b[t] = M_b[t-1] + ((E_b[t]-M_b[t-1]) \gg 5)$ $\to$ **exact integer `isqrt64` $\sqrt{\cdot}$-PCEN** ($\alpha=1, r=\frac{1}{2}$):
     $$R_b[t] = \left\lfloor \frac{E_b[t] \cdot 2^{20}}{\epsilon + M_b[t]} \right\rfloor + \delta_{\text{q20}}, \qquad P_b[t] = \operatorname{isqrt64}(R_b[t] \cdot 2^{20}) - \operatorname{isqrt64}(\delta_{\text{q20}} \cdot 2^{20})$$
     giving full 64-bit dynamic range with zero LUT quantization loss, zero transcendentals, and zero floats. Wave U02b tests adding 4-band envelope modulation power ($2,4,8,16\text{ Hz}$) via JopeDime's `goertzel_bank` (`wire 61` admitted on `SampleTime`) for noise resilience.
2. **Clock 1 (Reflex + Dense Auxiliary CTC):**
   * Strictly causal, zero lookahead. Each block combines causal `DWConv1d(k=9)` + `TimeMix` shift-add integrators ($m_s[t] = m_s[t-1] - (m_s[t-1] \gg s) + (x[t] \gg s)$, $s \in \{2,4,6\}$) + ternary FFN ($d \to 3d \to d$, `ReLU6`), with 4-head causal SWA-32 every 4th block.
   * **CTC Feasibility Fix:** The Auxiliary CTC head ($1,025$ pieces) sits strictly on the **dense $12.5\text{ Hz}$ Reflex output** (never on the thinned event grid!). Every training utterance checks $T \ge U + R$ (where $R$ is adjacent identical token pairs); infeasible sequences are explicitly counted and masked on the auxiliary term only (never silently zeroed).
3. **Adaptive Event Gate (Content-Defined Pooling):**
   * Uses verified **binary** `step(a, c)` (`1` iff $a > c$) over 8 Q7 features (`col(0)` $= 1-p(\text{blank})$, `col(1)` $= H/\ln 1025$, `col(2)` $= \text{frames\_since\_event}/10$, `col(3)` $= \text{spectral flux}$, `col(4)` $= \text{top non-blank prob}$, `col(5)` $= \text{Biasing-Trie active flag}$, `col(6)` $= \text{smoothed speech activity}$, `col(7)` $= \text{dial }\theta \in [-1, 1]$).
   * Default rule (15 nodes, depth 4—fits kernel2 search caps!):
     ```lisp
     (gate (sym raw "max(step(col(0)+col(1)+0.3*col(5), 0.1+col(7)), step(col(2), 0.16)*step(col(6), 0.5))")
           max_speech_gap_ms=160 silence_heartbeat_ms=1000 pool=mass)
     ```
   * **Mass-Weighted Pooling & Honest Bounds:** Non-kept frames accumulate weighted by $1 + 127(1-p(\text{blank}))$ into the next event vector `[h_event ; pool_mean] + Emb(gap) + sum_{k<=4} p_k Emb(tok_k)`. This is explicitly modeled as lossy temporal compression (never claimed lossless). Dial $\theta = -1$ keeps every frame (exact same-weights control for `K-EVENT`). Max speech gap begins at $160\text{ ms}$ (expanding to $240/320\text{ ms}$ only if Wave U04 passes). `u64` sample counters prevent overflow across multi-hour silence.
4. **Clock 2 (Cortex + Stateless 4,096-Vocab Integer RNN-T):**
   * **Chunk-Causal Execution:** Processes all events in the current fixed audio-time chunk ($C \in \{1,2,4,7,14\}$ frames) in **one batched sweep** (zero wake-up on empty chunks). Chunk boundaries are fixed across layers so future context never leaks progressively deeper.
   * **Bounded Attention Horizon:** Capped simultaneously at 128 events AND $25.6\text{ s}$ source-time horizon (matched to dense context in Wave U04 controls!). Uses exact 32-bucket log-time-gap position bias table (Wave U05 compares against integer `Time-RoPE`).
   * **Utterance-Fixed Depth & Precision:** Precision (`T1` vs `T1+T2`) and layer depth are **fixed for an utterance** by default—eliminating KV-cache holes! If mid-segment dense recovery is triggered by runtime-observable uncertainty/cap-hits, the runtime replays affected upper-encoder/decoder state from retained $12.5\text{ Hz}$ Reflex features under the locked-prefix invariant and charges the replay cost.
   * **Stateless 2-Token Integer `ReLU6` RNN-T (Full 4,096 Projection):** Predictor concatenates the last two non-blank 256-d token embeddings (`512 -> J`), joiner computes `ReLU6(W_e h + W_p p) -> 4096`. Because $4096 \times J$ is $< 1\%$ of Cortex MACs, **no approximate shortlist is used** (eliminating shortlist recall failures!). Non-blank expansions per event are capped at `max_sym = 3` (forced advance + telemetry counter; ablated against `max_sym = 8`).
   * **Optional Isolated Branches:**
     * *Wave U06 (Sparse Ternary MoE):* Replace Cortex FFN with $1\text{ shared} (2d) + E_s\text{ session} (d) + E_e\text{ event} (d)$ ternary experts (aux-loss-free load balancing + chunk-affinity bonus $\beta=0.15$). Must beat iso-active-bytes dense Cortex by $\ge 3\%$ rel WER at $\le +10\%$ measured CPU-s (no "L3 pinned" assumptions).
     * *Wave U11 (FastEmit TDT):* Add duration outputs $d \in \{0,1,2,3,4\}$ only if profiling shows joiner calls exceed $10\%$ of CPU time. Strict streaming rules: zero-duration blank forbidden, cross-chunk duration debt preserved, end-of-stream flush defined.
5. **Clock 3 (Selective Error-Reduction `Echo` — Waves U07–U08):**
   * 2–4 bidirectional ternary Transformer layers running once at segment pause over cached Cortex events (and optionally $12.5\text{ Hz}$ Reflex features in Wave U07b!).
   * **Calibrated Benefit Controller:** Rather than running Echo blindly on every pause, train a compact calibrated regressor/classifier on held-out dev utterances predicting $\text{benefit}_j = \text{edit\_errors}(\text{fast}_j) - \text{edit\_errors}(\text{echo}_j)$ from pre-Echo observables (Cortex/Reflex agreement, token rarity, biasing hit, SNR proxy, speech rate, segment length).
   * **Deadline Admission:** Predict service time $T_{\text{echo}}(K)$ + queue wait; skip Echo if remaining finalization slack is insufficient, logging `echo_skipped_deadline` in the receipt. Total CPU accounting charges 100% of invoked, late, and canceled Echo jobs.

---

## 5. Airtight Cross-ISA Integer Numeric Contract & Multi-ISA Kernels

### 5.1 Tensor Formats & Zero-Point / Saturation Proofs
| Format | Range / Zero Point | Exact Arithmetic Rule |
|---|---|---|
| `A7` | `u8` $\in [0, 127]$, zp $= 0$ | Post-`ReLU6` / front-end features. |
| `A7z` | `u8` $\in [0, 127]$, zp $= 64$ | Signed normalized inputs: $\sum_j w_j (a_j - 64) = \big(\sum_j w_j a_j\big) - 64 \big(\sum_j w_j\big)$, with `row_sum` precomputed in `RowMeta`. |
| `I8` | `i8` $\in [-127, 127]$ | First/last layers, embeddings, DWConv, joiner. Max AVX2 `vpmaddubsw` pair sum $= 2 \times 127 \times 127 = 32,258 < 32,767$ (**zero saturation**). |
| `U8-Tern` *(Wave U10b)* | `u8` $\in [0, 255]$ into ternary `{-1,0,+1}` | Max AVX2 pair sum $= 2 \times 255 \times 1 = 510 \ll 32,767$! Gives a free extra activation bit into ternary matrices (ARM/WASM use `a' = a - 128` + `128 * row_sum`). |
| `T1` / `T2` | Ternary $\{-1,0,+1\}$ | `T2` rows store additive residual plane $w = M_1 t_1 + M_2 t_2$ ($15\%$ sensitive rows, fixed per utterance). Invalid 2-bit code `3` rejected at load. |
| `ACC` | `i32` | Build prover asserts $K \cdot \max|a| \cdot \max|w| < 2^{31}$ for every row before bundle emission. |

### 5.2 Exact Integer Operations (Zero Bugs)
* **Requantization:** `y = clamp_i16(((acc_i64 * M_i32) + (1i64 << (s - 1))) >> s)` (explicit `s == 0` guard; half-up rounding).
* **Integer RMSNorm:** `n = sum(r_i^2)` (`i64`) $\to$ `rms = isqrt64(n / d)` $\to$ `inv = (1u64 << 30) / max(rms, 1)` $\to$ `clamp(((r_i * g_i * inv) >> s_n) + 64, 0, 127)`.
* **Exact Integer Softmax (Attention Only):**
  $$\text{dist}_i = \operatorname{clamp}(\max_j s_j - s_i,\; 0,\; 255), \qquad e_i = \text{EXP\_LUT}_{u16}[\text{dist}_i], \qquad Z = \sum_i e_i\;(u32), \qquad w_i = \left(e_i \cdot \left\lfloor \frac{2^{32}}{Z} \right\rfloor\right) \gg 24$$
  with $\text{EXP\_LUT}[0] = 65535 > 0$ (zero-denominator impossible).

### 5.3 Multi-ISA Ternary Kernel Bake-Off (Wave U01 — Zero Byte-Layout Bugs)
Every SIMD kernel MUST match the zero-`unsafe` `cadence-core` scalar oracle bit-for-bit (`max_abs_diff == 0`) on randomized extreme-operand property tests AND 1,000 golden utterances, AND deliver a measured end-to-end streaming speedup before admission:
1. **`T2B-VNNI` (AVX-512 VBMI + VNNI):** 4 trits/byte unpacked via `vpmovzxwq` + `vpmultishiftqb` $\to$ `vpdpbusd` with $U = 4$ independent accumulators (empirically proven in Appendix A).
2. **`T-LUT2` (AVX2 `vpshufb`, ARM NEON `vtbl`, WASM `i8x16.swizzle`):** Group **2 trits** ($3^2 = 9$ states: 3-bit magnitude index $0..4$ + 1 sign bit $= 4\text{ bits/pair}$, exact `2 bits/wt`). For any `A7` pair $(a_0, a_1) \in [0,127]^2$, the 5 non-negative partial sums $\{0, a_0, a_1, |a_0 - a_1|, a_0 + a_1\}$ lie in **$[0, 254]$**—fitting cleanly as **5 single `u8` bytes inside one 16-byte shuffle register** with zero multiplications!
3. **`T-LUT3-Split` (AVX2 / NEON / WASM, $1.67\text{ bits/wt}$ in RAM):** Group **3 trits** ($3^3 = 27$ states: 4-bit magnitude index $0..13$ + 1 sign bit $= 5\text{ bits/triple}$). The 14 partial sums $\in [0, 381]$ are stored across **two** 16-byte register tables (`lut_lo: [u8; 16]` for bits `0..7` and `lut_hi: [u8; 16]` for bit `8`), using two shuffles + `i16` combine (mathematically exact; replaces CADENCE v1's broken single-16B claim).
4. **Packed Bitmask Add/Sub (`FairyFuse` / `BitNet.cpp` style):** Tested alongside `T-LUT2` and `T-LUT3-Split` in Wave U01; winning bit-exact kernel is selected per ISA automatically.

---

## 6. Irrevocable Streaming API, Bundle & Receipt Contracts

### 6.1 Output Event Contract (Zero Contradictions)
| Event | Semantics | Can Earlier Text Mutate? |
|---|---|---|
| `Preview` | Replaces mutable suffix after `immutable_prefix_utf8_bytes` (segment-local UTF-8 byte offset, always on a valid char boundary) | Yes, suffix only |
| `Commit` | Appends text to the segment's immutable prefix; monotonically advances `immutable_prefix_utf8_bytes` | **NEVER** |
| `Final` | Closes `segment_id` exactly once and commits any remaining suffix | **NEVER** |
| `CorrectionSuggested` | Optional post-final or post-commit alternative from `Echo` | **NEVER** mutates committed/final ledger |
| `Gap` | Explicit lost/discontinuous PCM sample range; closes affected segment as incomplete, preserves immutable prefix, resets state, starts new segment | No invented speech |
| `Status` | Backlog, dial $\theta$ change, overload rung, Echo skip, endpoint trigger | No text mutation |

* **Editable Dictation Mode:** Live Cortex emits `Preview`; selective `Echo` emits `Final` once at pause (replacing only mutable `Preview`!).
* **COMMIT Mode:** Live Cortex emits irrevocable `Commit`; `Echo` refines only the uncommitted suffix (strictly preserving the exact UTF-8 bytes of the locked prefix) or emits `CorrectionSuggested`.
* **Segment Caps & Ownership:** Soft cap $20\text{ s}$ (prefer low-energy/word boundary), hard cap $30\text{ s}$. Each finalized word carries stream-global ownership sample intervals so words straddling a forced cut are owned **exactly once**.
* **Overload Ladder (Cache-Safe):** (1) Skip optional formatting & `Echo`; (2) reduce cross-stream batch wait ($\le 5\text{ ms}$ cap); (3) raise gate dial $\theta$ at chunk boundary; (4) widen chunk grid at segment boundary; (5) drop `T2` at **segment boundary** (never mid-utterance!); (6) reject new streams; (7) emit `Gap` if PCM ring overflows.

### 6.2 Core Rust Trait ABI (`cadence-core`)
```rust
pub trait AsrBackend: Send + Sync {
    fn capabilities(&self) -> BackendCaps;
    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, AsrError>;
    /// Optional server path: processes pending events across multiple streams in ONE weight pass.
    fn batch_step(&self, streams: &mut [&mut dyn AsrStream]) -> Result<(), AsrError>;
}

pub trait AsrStream: Send {
    /// Lifecycle: ACCEPTING -> DRAINING -> FINISHED (or CANCELLED / FAILED).
    fn push_pcm(&mut self, start_sample: u64, pcm_i16: &[i16]) -> Result<PushReport, AsrError>;
    fn poll_event(&mut self) -> Result<Option<TranscriptEvent>, AsrError>;
    fn finish_input(&mut self) -> Result<(), AsrError>; // idempotent
    fn cancel(&mut self) -> Result<(), AsrError>;       // idempotent
    fn set_biasing_phrases(&mut self, phrases: &[&str]) -> Result<(), AsrError>;
    fn metrics(&self) -> StreamMetrics;
    fn export_receipt(&self) -> RunReceiptV1;
}
```
* **FFI / IPC Contract:** Track A integrations start as supervised length-delimited process adapters (preventing dependency/panic/crash propagation; worker crash preserves immutable prefix and reports explicit error). Native Rust Track B crates expose an opaque-handle C17 FFI (`cadence-capi`, never unwinding panics) and `cadence-wasm`.
* **Bundle & Receipt:** Content-addressed `models/sha256-<manifest-digest>/` verifying every payload SHA-256, schema, ISA, overflow proof, and `program.dsl` round-trip at load. Every reported benchmark number requires a `cadence.run-receipt.v1` JSON binding `source_commit`, `bundle_sha256`, `hardware_manifest_sha256`, `audio_manifest_sha256`, `scorer_sha256`, `cpu_only_verified`, `process_tree_cpu_seconds`, `configured_threads`, `observed_worker_threads`, `honesty_label`, and `prediction_events_sha256`.

---

## 7. `APEX-EQ Dossier`: Contextual Biasing & Repaired Closed-Form Adaptation

1. **Always-Safe First Release — Gate-Boosted Phrase Biasing:** User-supplied names/acronyms/code symbols compiled to a BPE trie. Active prefix matches apply a bounded beam-4 bonus AND set **Event Gate `col(5) = 1`** (allocating higher frame resolution across rare user terms). Evaluated strictly with **unspoken distractor phrases** to cap false insertions!
2. **Self-Training OFF by Default:** Model-generated pseudo-labels NEVER adapt committed or final outputs.
3. **Confirmed-Label `APEX-EQ` D2 Readout (Wave U13 — Opt-In, `recall` Label Only):**
   * Triggered ONLY by user-confirmed corrections or explicit 30–60 s user enrollment (stored locally, user-deletable, per-profile isolated).
   * **Repaired Primal Math (Fixing CADENCE v1 & TACTUS):** Project Cortex encoder states $h_k \in \mathbb{R}^d$ through a fixed orthogonal basis $P \in \mathbb{R}^{d \times r}$ ($r = 64$ or $128$, offline SVD) to compact features $x_k = P^\top h_k \in \mathbb{R}^r$. Solve for a compact encoder-shift matrix $V \in \mathbb{R}^{r \times r}$ mapping $h_k' = h_k + \eta P (V^\top x_k)$ toward confirmed-alignment target representations (or Echo states on confirmed-improved utterances):
     $$A = \tilde{X}^\top \tilde{X} + \lambda I_r \in \mathbb{R}^{r \times r}, \qquad V^* = A^{-1} \tilde{X}^\top \tilde{Y}$$
     Because $A$ is $64 \times 64$ or $128 \times 128$, Cholesky via `feat-math::kernel::ridge_fast::Factor` takes **$< 0.5\text{ ms}$**! Adapting the encoder input affects all 4,096 vocabulary pieces without joiner-predictor coupling breakage or 4,096-way sparsity.
   * **Exact Sample-Block Leave-Fold-Out (LFO) Guard:** Group confirmed events into $K = 5$ **whole-utterance** sample blocks $B_1 \dots B_5$. For each block $B$ of size $m_B$, compute the exact block smoother $S_{BB} = \tilde{X}_B A^{-1} \tilde{X}_B^\top \in \mathbb{R}^{m_B \times m_B}$ and exact block LFO residual:
     $$E_B^{\text{LFO}} = (I_{m_B} - S_{BB})^{-1} E_B$$
     Choose $\eta \in \{0, 0.25, 0.5, 1.0\}$. Adopt $\eta > 0$ ONLY if: (a) $\ge 4$ of $5$ utterance blocks improve LFO error by $\ge 1\%$ relative; (b) incremental column-pivoted QR span novelty $\rho_{\text{innov}} \ge 10^{-4}$; (c) zero `RidgeRefusal` pivots. $\eta = 0$ is bit-identical to the unadapted model.
   * **No False Guarantees:** Lower LFO representation error is a necessary gate, NOT a WER no-harm theorem—Wave U13 validates actual WER and per-speaker harm rate ($\le 2\%$) on **held-out later sessions** (`asr-adapt-v1`).

---

## 8. `GP-ASR` Dialect & Verified JopeDime Integration

* **Verification-First Rule (`P00`):** Inspect the live `neverhuman/JopeDime` checkout (`AGENTS.md`, `Cargo.toml`, `docs/adr/ADR-0006-rust-only-stack.md`, `PYTHON_EXCEPTIONS.md`, `crates/feat-math/registry/operators.toml`) before editing. Track A uses typed YAML configs immediately; Track B adds the `(asr ...)` dialect cleanly in `crates/feat-audio` without breaking `tests/compat_freeze.rs` or appropriating reserved wires `53–75` (request coordinated wire allocation via Owner Ruling `O7` `[REPO-ARCHIVE: docs/contracts/operator-registry-v2.md]`).
* **Reused JopeDime Primitives `[REPO-ARCHIVE]`:**
  * Canonical S-expression lexer/printer (`dsl.rs`), `(sym raw "...")` genomes (`genome.rs`), binary `step(a, c)` (`math_expr/parser.rs`).
  * `ordered_axis.rs` (`Axis::SampleTime` + new `Axis::EventTime`, `FitScope::{Pretrained, Profile}`) + `timescale_nabla` `state_codec` pattern for bit-exact stream checkpoint/resume.
  * Lowering `(asr ...)` to a typed DAG following `feat_math::kernel::ir` (`jope.selfplay.graph-effects/v1` pattern), with static cost parity modeled on `tokens.rs`.
  * `ridge_fast.rs` (B11), `goertzel.rs` (wire 61), `savgol.rs` (wire 58), `lexicase.rs` (B10), `pareto.rs` (B18), e-graph/LM refinement (B17/B20), and `H443–H472` empirical rules (A/A resolution floor, no optimizer zoo, ESS floor on percentile regret, H446 train/serve channel parity, H461 per-checkpoint threshold refitting, H472 full-softmax training, H94 fresh-audio confirmation of all cached policy wins).
* **Canonical `ULTRA-L` Program (verified arity & caps):**
```lisp
(asr
  (meta lookahead_ms=0 name=cadence-ultra-l sample_rate=16000 version=3)
  (stream cochlea
    (dc_block pole_q15=32735) (preemph coef_q15=31785) (frame hop=160 win=400 window=hann_q31)
    (fft_power_q31 n=512) (melbank_q15 bins=80 hi=7600 lo=60)
    (pcen_isqrt delta_q20=2097152 eps_q20=1 shift=5) (quantize_a7 scale_q16=1572864) (stack_frames n=4))
  (stack reflex
    (pipe cochlea
      (conv_stem d_s=256 d_r=512)
      (reflex_blocks attn_every=4 d_r=512 expand=3 fmt=t1 kernel=9 layers=12 plane2=0.15 timemix="2 4 6" window=32))
    (ctc delay_penalty=0.3 fmt=i8 name=aux_dense vocab=reflex-bpe1024))
  (gate (sym raw "max(step(col(0)+col(1)+0.3*col(5), 0.1+col(7)), step(col(2), 0.16)*step(col(6), 0.5))")
        max_speech_gap_ms=160 pool=mass silence_heartbeat_ms=1000)
  (stack cortex
    (pipe events
      (event_embed d=1280 gap=16 topk=4)
      (cortex_blocks cache=128 conv=5 d=1280 ffn=4 fmt=t1 heads=16 horizon_ms=25600 layers=20 plane2=0.15 pos=timebucket32))
    (rnnt context=2 fmt=i8 joiner=640 max_sym=3 name=main vocab=ultra-bpe4096))
  (echo bidir=true d=1280 fmt=t1 gate=calibrated_benefit heads=16 layers=4 max_events=256 plane2=0.15)
  (endpoint (sym raw "max(step(col(0), 6), max(step(col(0), 3)*step(col(2), 0.5), step(col(3), 2.56)))"))
  (dossier bias_beam=4 d1_prior=true d2_lfo_rank=64 folds=5 source=confirmed))
```

---

## 9. Data Identity Graph, Honest Storage, Training & CPU Fleet

### 9.1 Canonical Audio Identity Graph & Storage Realism
Never add overlapping dataset headlines (LibriSpeech, MLS, Libriheavy, and Libri-Light all share LibriVox audio; Loquacious dev/test shares LibriSpeech dev/test `[VERIFIED-SOURCE]`).
* **Identity Graph:** `source_collection -> canonical_recording_id -> [start_sample, end_sample) -> pcm_sha256` with separate `transcript_record`, `license_record`, and `split` fields. Report gross vs. exact-duplicate vs. overlap-merged vs. **unique admitted hours**.
* **Honest PCM Storage Math `[DERIVED]`:** $16\text{ kHz}$ mono `int16` PCM $= 115.2\text{ MB/hour}$ ($960\text{ h} = 110.6\text{ GB}$; $25\text{k h} = 2.88\text{ TB}$; $150\text{k h} = 17.28\text{ TB}$; $300\text{k h} = 34.56\text{ TB}$). **Rule:** Store compressed authoritative audio (Opus/FLAC) in object storage; materialize decoded `mmap` PCM shards into a bounded local cache ($\le 25\%$ free disk or explicit quota) with SHA-256 verification and `READY` sentinels.
* **Data Progression:** `D0` (smoke overfit) $\to$ `D1` (LibriSpeech `clean-100`) $\to$ `D2` (LibriSpeech `960 h`) $\to$ `D3` (Loquacious qualified `clean` $13\text{k}$ + Common Voice + VoxPopuli) $\to$ `D4` (deduplicated MLS/Libriheavy + conversational domain data) $\to$ `D5` (licensed unlabeled pool: YouTube-Commons/YODAS2/Libri-Light/MOSEL, strictly gated by `P14` pilot!). SPGISpeech stays evaluation-only unless commercial rights are cleared.

### 9.2 Pilot-First Architecturally Independent Teacher Assay (Job `J2`)
* **Pilot First:** Transcribe 500 stratified hours (across accents, noise, rates) and run a human error audit BEFORE launching a 100k+ hour factory.
* **True Independence:** Teachers MUST come from distinct lineages (e.g., Granite TurboCTC [Conformer-CTC] + Moonshine/Whisper [Encoder-Decoder] + Parakeet/Nemotron [FastConformer-RNNT] — NEVER Phonon + Redux together, as both derive from Parakeet!). Verify every teacher's license permits distillation/pseudo-labeling.
* **Tiers & Canary:** Separate `provenance` from `agreement`. Validate that a 200-utterance **label-poison canary** is rejected at $\ge 95\%$ AND measure real-error rates on Loquacious `clean`. Retain a stratified disagreeing reserve for human review so accented/rare speech is not systematically erased.

### 9.3 Training Recipe, Fixed-Point Fleet GPTAQ & Three Resource Envelopes
* **Resource Envelopes:**
  * `E1` (CPU workstation): Full Track A tournament, runtime optimization, PTQ calibration, controller fitting, evaluation.
  * `E2` (1× 24 GB GPU + workstation): `D1`/`D2` reproduction, `ULTRA-S`/`M` fine-tuning, distillation, QAT pilots (duration-bucketed microbatches, measured peak VRAM).
  * `E3` (Multi-GPU + massive CPU fleet): Full `ULTRA-L`, `D3`–`D5` scale, Event-Clock Text Injection, fleet GPTAQ. **Scale Gate:** Measure actual audio-hours/GPU-hour in `D1` before committing multi-thousand-GPU-hour budgets (planning range $300$–$600\times$ RT/GPU, NOT naive FLOP ceilings!).
* **Unified Loss (Staged Introduction):**
  $$\mathcal{L} = 1.0\,\mathcal{L}_{\text{RNNT}}^{\text{cortex}} + 0.30\,\mathcal{L}_{\text{CTC}}^{\text{reflex,dense}}(T \ge U+R) + 0.45\,\mathcal{L}_{\text{RNNT}}^{\text{echo}} + 0.15\,\mathcal{L}_{\text{MCR}} + \lambda_{\text{txt}}\,\mathcal{L}_{\text{TextInj}} + \lambda_{\text{kd}}\,\mathcal{L}_{\text{KD}}$$
  * **Valid Distillation:** Use sequence-level teacher text across mismatched tokenizers, or frame/event top-$K$ KL (with explicit residual tail-mass bucket!) ONLY from an in-house same-vocabulary Oracle.
  * **Optimizer:** Plain AdamW ($\beta=(0.9,0.98)$, WD $10^{-2}$, clip $1.0$), warm-up + cosine + H456 raised-cosine valleys (4 EMAs, atomic `ckpt_valley_NNN`), 28-family causal audio augmentation (keeping clean anchor + $120\text{ s}$ multi-utterance carried-state sessions).
* **Progressive QAT + Fixed-Point Integer-Associative Fleet GPTAQ (Wave U10):**
  1. `bf16` valley checkpoint $\to$ `A7`/`A7z` activation QAT with integer RMSNorm/softmax simulation $\to$ ternary weight ramp $\lambda: 0 \to 1$ + `T2` on top $15\%$ sensitive rows.
  2. **Repaired Fleet `int64` GPTAQ (`J5`–`J6`):** Quantize **both** student inputs $x \in [0,127]$ AND teacher/fp32 reference activations $x_{\text{ref}}$ to fixed-point **Q15 `i32`** (`|x_q15| <= 2^20`), guaranteeing exact order-independent `i64` associativity for both $H = \sum x x^\top$ and cross-Gram $G = \sum x x_{\text{ref}}^\top$ across 1,000+ CPU workers! Solve closed-form asymmetric rounding per row block via `ridge_fast::Factor`, followed by 1 healing valley.
  3. **Export Gate:** Python integer simulator and Rust scalar/SIMD runtimes MUST agree with **`max_abs_diff == 0`** across 1,000 golden utterances.

### 9.4 Massive CPU Fleet Orchestration (`J1`–`J13`)
Follows JopeDime's proven lock-free fleet contract `[REPO-ARCHIVE: steal_distributed_tails_cpu.py]`: POSIX-atomic `os.mkdir` claims, heaviest-first + reverse-depth tail stealing, abort checks, OS cgroups (`MemoryHigh`/`MemoryMax`), `taskset`, `nice -n 19`, `OMP/MKL/OPENBLAS/RAYON <= 4`, and create-only SHA-256 S3 uploads.
* `J1` Ingest/Identity Graph · `J2` Diverse Teacher Assay (pilot $\to$ scale) · `J3` Oracle Forced Alignments · `J4` Live Augmentation Staging · `J5` Fixed-Point `i64` Gram/Cross-Gram Stats · `J6` Parallel B11 GPTAQ Rounding · `J7` Zero-Interrupt Valley Scoring & Soups · `J8` Free-Gene Lexicase Search (fresh-confirmed!) · `J9` Percentile-Regret Hard Mining (with ESS floor) · `J10` Multi-ISA Parity Farm · `J11` Text-Event Prep · `J12` Kernel Bake-Off (`H0`–`H4`) · `J13` Track A Baseline Census.

---

## 10. Evaluation, Statistical Protocol & Pre-Registered Waves (`U00`–`U18`)

### 10.1 Frozen Suites & Statistical Discipline
* **Suites:** `TRAIN`, `CALIBRATION` (train-only), `DEV` (adaptive), `CONFIRM` (**single-use per decision round**—once read to pick a winner, retired to `DEV`!), `PUBLIC-TEST` (8-set leaderboard + exact normalizer + separate product-fidelity view), `DOMAIN-SEALED` (fresh 10–20 h multi-speaker target-domain mic recordings), `ADAPT-TEST` (paired enrollment + later sessions), `PARITY` (1,000 golden utterances).
* **Honesty Labels:** `fair`, `in-corpus`, `recall` travel with every number.
* **Statistics:** Paired cluster bootstrap ($B = 5,000$–$10,000$) over speakers/recordings preserving dataset strata and linked recordings; explicit $1 - 0.05/m$ multiplicity bounds; 3 independent training seeds for final arms; null seed arm; **label-poison canary**; silence reported as **insertions/hour** (never $0/0$ WER); ABBA alternating timing with qualification lock.

### 10.2 Research Waves & Kill Gates
| Wave | Question / Gate | Control vs. Treatment | Pass Gate | Kill / Fallback |
|---|---|---|---|---|
| **U00** | Instrument, A/A noise floor & Nemotron HW-counter check | Known-edit fixtures; 2-seed A/A; Nemotron LLC/BW counters | Scorer exact; floor $<1.5\%$ rel; fixed-cost source measured | Fix harness / resize `DEV` |
| **U01** | Track A Baseline Tournament + Multi-ISA Kernel Bake-Off | Full §3 catalog on `H0`; `T2B-VNNI`, `T-LUT2`, `T-LUT3-Split`, `int8` + interleaved LLC test | Track A Pareto winners shipped; winning bit-exact kernel per ISA | Update `sizing_ultra.py` with measured `H0` law |
| **U02** | Zero-Float Integer `isqrt64`-PCEN (+ Goertzel `wire 61`) | Float32 log-mel vs Q31 `isqrt64`-PCEN vs +Goertzel mod | $\Delta\text{WER} \le +0.10$ clean; $\ge 3\%$ rel gain in noise; 0-bit ISA diff | Keep float32 log-mel control |
| **U03** | Track B Dense Frame-Clock Student Control (`D1`/`D2` $\to$ `D3`) | Reproduced streaming Zipformer/Conformer recipe | Matches recipe; establishes `ULTRA` dense baseline | Continue Track A optimization |
| **U04** | **`K-EVENT`** (Adaptive Event Gate + Mass Pooling) | Same-weights $\theta=-1$ vs $\theta=0$ AND separately trained matched-context dense | $\ge 15\%$ full-pipeline CPU-s drop; $\Delta\text{WER} \le +0.10$ pp; recall $\ge 99.5\%$ | 2 rule revisions; else ship dense or fixed $2\times$ |
| **U05** | Multi-Grid Chunk Sampling & Time-Bucket vs `Time-RoPE` | Grid-specific vs sampled; TimeBucket-32 vs integer `Time-RoPE` | Sampled within $2\%$ rel; pick simpler pos bias unless RoPE wins $\ge 2\%$ | Ship two chunk checkpoints |
| **U06** | **Sparse Ternary MoE Branch** (`ULTRA-L-MoE`) | Iso-active-bytes Dense Cortex vs $1\text{sh}+E_s+E_e$ MoE | $\ge 3\%$ rel WER reduction at $\le +10\%$ measured CPU-s | Keep Dense Ternary Cortex |
| **U07** | Echo Utterance Refiner Opportunity | Never-Echo vs Always-Echo (Cortex-only vs Cortex+Reflex) | $\ge 5\%$ rel Final WER gain; repair/harm table positive | Drop Echo |
| **U08** | Selective Error-Reduction Controller | Random/fixed gate vs calibrated $\Delta\text{edit}$ controller | $\ge 20\%$ CPU-s savings vs Always-Echo at non-inferior WER, or $\ge 3\%$ WER gain at $\le +15\%$ CPU-s | Always-Echo (`ACCURATE`) or Never-Echo (`LIVE`) |
| **U09** | Event-Clock Text Injection | $0\%$ vs $15\%$ vs $25\%$ text batches (with Reflex confusion table) | $\ge 5\%$ rel rare-word/entity gain; macro unchanged | Drop text stream |
| **U10** | **`K-TERN`** + `U8`-Ternary (U10b) + Fixed-Point Fleet GPTAQ (U10c) | `bf16` vs `A7`+`T1` vs `T1+T2` vs `U8-Tern` vs `+GPTAQ` | $\Delta\text{WER} \le +0.25$ pp clean, $\le +0.40$ pp noise; 0-bit export parity | Ladder: `T2` $25\%$ $\to$ `i8` bottom 4 blocks $\to$ `i8` |
| **U11** | FastEmit TDT Branch (Conditional on Joiner Profile) | Stateless RNN-T vs FastEmit TDT ($d \in \{0..4\}$) | Lower CPU-s / latency at non-inferior WER; duration-debt tests pass | Keep Stateless RNN-T |
| **U12** | Gate-Boosted Phrase Biasing (`col(5)=1`) | Bias off vs on, WITH unspoken distractor phrases | Entity recall up; false insertions bounded; macro intact | Lower bonus / off |
| **U13** | Confirmed-Label `APEX-EQ` D2 (Sample-Block LFO Guard) | Off vs D1 vs D1+D2 (guard on vs off; null shuffled-label arm) | Null arm $\eta=0 \ge 99\%$; later-session WER $\ge 3\%$ rel better; harm $\le 2\%$ | Ship Biasing + D1 only |
| **U14** | Multi-Stream Server Batching (`batch_step`) | 1 vs 4 vs 16 streams/core on `H2` | Bit-exact batch invariance; $\ge 70\%$ projected stream density | Single-stream only |
| **U15** | Fleet Free-Gene Lexicase Search (H94 Confirmed) | Hand defaults vs searched (confirmed on fresh `CONFIRM` split!) | $\ge 10\%$ CPU-s drop at equal WER or $\ge 1\%$ rel WER gain | Keep hand defaults |
| **U16** | Data Scale (`D3` $\to$ `D4` $\to$ `D5` Pilot-Validated) | Equal extra compute on current data vs added unique hours | Confirmed learning curve justifies GPU/storage cost | Stop scaling; target error slices |
| **U17** | Distillation to `ULTRA-M`/`S` + Valley Soups | Teacher `ULTRA-L` $\to$ `M`/`S`; single valley vs soup | Tier gates pass; soup beats best single valley on fresh split | Ship best single valley |
| **U18** | **`K-ACC`** Single Pre-Registered Sealed Release Look | Frozen Track A & Track B finalists on `PUBLIC-TEST` + `DOMAIN-SEALED` | Release Gates (§2.3) & §11.4 checks | Publish exact measured frontier with receipts |

---

## 11. Autonomous Agent Takeover Manual (`P00`–`P21`)

### 11.1 Workspace Map
```
crates/cadence-core/          # zero-unsafe scalar oracle, integer contract, event/stream traits, bundle verifier
crates/cadence-kernels/       # avx512_vnni.rs, avx2_tlut2.rs, avx2_tlut3_split.rs, neon_tlut.rs, wasm_simd.rs
crates/cadence-frontend/      # float32 strict-IEEE log-mel + Q31/isqrt64 integer Cochlea
crates/cadence-runtime/       # Track A process adapters + Track B Reflex/Gate/Cortex/Echo/Dossier/Batcher
crates/cadence-eval/          # paced replay, process-tree CPU/RSS/PSS, scorer, cluster bootstrap, receipts
crates/cadence-gp/            # (asr ...) dialect, binary step(a,c) checker, static cost estimator, emitters
crates/cadence-cli/           # inventory, transcribe, stream, bench, score, compare, bundle verify, doctor
research/cadence_ultra/       # PyTorch trainer (ADR-0006 exception), identity graph, fleet J1–J13, waves/U00..U18/
docs/cadence_ultra/           # STATUS.md, DECISIONS.md, RULINGS.md, CANDIDATES.lock, DATA.lock, LOOK_LEDGER.jsonl
```

### 11.2 Ordered Work Packages & Machine-Checkable Acceptance Criteria
* **Track A Critical Path (Deliverable: Qualified Shippable CPU Recognizer in Weeks 1–3):**
  * `P00` **Live Checkout & Hardware Inventory:** Verify repo commit, `Cargo.toml`, `AGENTS.md`, `H0` specs (`HARDWARE.json`), file Owner Rulings `O1–O8` (ADR-0006 Python exception, rights, compute budget, H0–H4 access, wire IDs). *Accept:* `cargo test` green; unknowns explicitly recorded.
  * `P01` **Scorer & Statistical Engine (`cadence-eval`):** Implement normalized & product-fidelity WER, silence insertions/hr, paired cluster bootstrap, multiplicity bounds, poison canary. *Accept:* Known S/D/I fixtures & poison canary pass.
  * `P02` **Irrevocable Event & Adapter Contract (`cadence-core`):** Implement `AsrBackend`, `AsrStream`, `Preview`/`Commit`/`Final`/`CorrectionSuggested`/`Gap`, UTF-8 byte offsets, 20s/30s caps. *Accept:* State-machine & crash-recovery tests pass.
  * `P03` **Paced Replay & Process-Tree Accounting:** Feed 10/20/37 ms packets at real audio arrival rate; capture full process-tree CPU-s, RSS/PSS, observed worker threads, GPU-zero check. *Accept:* Packetization invariance & idle-wait exclusion verified.
  * `P04–P06` **Track A Adapters & Wave U00/U01 Baseline Tournament:** Integrate Nemotron Q8, Moonshine S/M, pinned Zipformer, FUTO M/L, Phonon-2.5, Granite, Parakeet Unified/Redux; run `wstream_ultra` multi-ISA bake-off + Nemotron HW-counter check. *Accept:* Frozen Pareto table on `H0`; `wstream_ultra` scalar=SIMD `0`-bit diff.
  * `P07` **Package Track A Local Release:** Qualified bundle, CLI, 1-hour sustained stream test, rollback path. *Accept:* All applicable release gates pass on `H0`.
* **Track B Breakthrough Path (`CADENCE-ULTRA` Student):**
  * `P08` **Audio Identity Graph & Bounded Shard Cache:** LibriSpeech (`D1`/`D2`) + Loquacious (`D3`) with LibriVox/8-gram/fingerprint exclusion & quarantine gate. *Accept:* Sealed shard refused; unique-hour union reconciled.
  * `P09` **Cochlea & Numeric Core (`U02`):** Zero-float `isqrt64`-PCEN + Rust/Python bit-exact parity + overflow prover. *Accept:* 100-file hash equality across targets.
  * `P10` **`GP-ASR` Parser & Static Cost Estimator:** Implement `(asr ...)`, binary `step(a,c)` verification, `sizing_ultra.py` Rust port. *Accept:* 10k fuzz round-trips pass; 1-arg `step` rejected.
  * `P11` **Trainer & Dense Control (`U03`):** H456-style PyTorch trainer with per-stage timers, measured throughput pilot, `D1`/`D2` dense `ULTRA-S`/`M` control. *Accept:* Simulation matches Rust `max_abs_diff == 0`; GPU re-budget note filed.
  * `P12–P14` **Core Architectural Waves (`U04`–`U09`):** Execute `K-EVENT` (`U04`), chunk/pos (`U05`), optional MoE (`U06`), Echo & Selective Controller (`U07`–`U08`), Event-Clock Text Injection (`U09`). *Accept:* Pre-registered verdicts & fallbacks logged in `EXPERIMENTS.jsonl`.
  * `P15` **Quantization & Fixed-Point Fleet GPTAQ (`U10`–`U11`):** Progressive QAT + `J5`/`J6` fixed-point `i64` GPTAQ + `U8`-ternary check + optional TDT. *Accept:* `K-TERN` passes; 1,000-file cross-ISA parity `0` diff.
  * `P16–P18` **Runtime Hardening, Biasing, `APEX-EQ` & Gene Search (`U12`–`U15`):** Multi-stream `batch_step`, Gate-Boosted Biasing (with distractors), confirmed-label block-LFO D2, fresh-confirmed lexicase free-gene search. *Accept:* Zero alloc after `create_stream`; 1-hour stream & harm gates pass.
  * `P19–P21` **Pilot-Validated Data Scale, Distillation & Sealed Look (`U16`–`U18`):** 500 h teacher pilot $\to$ `D4`/`D5` scale if ROI holds $\to$ distill `M`/`S` $\to$ single pre-registered sealed look by independent scorer $\to$ release with receipts.

### 11.3 Agent Authority & Stop-and-Ask Triggers
* **Proceed autonomously on:** Read-only repo/HW inspection; isolated implementation & unit/parity tests; running any committed `PREREG.md` wave within authorized budget; taking an explicitly listed fallback after a failed kill gate; recording `NO_LIFT` verdicts.
* **Stop affected work & escalate to owner on:** Any paid cloud/GPU spend beyond stage budget; ambiguous dataset/teacher license; `ADR-0006` exception or wire-ID allocation; any read of `PUBLIC-TEST` or `DOMAIN-SEALED`; changing any gate threshold; a null-arm "win" (instrument broken); `K-EVENT` and `K-TERN` both failing. Always continue independent authorized work while one decision awaits sign-off.

### 11.4 Copyable Agent Takeover Prompt
> You are taking charge of **CADENCE-ULTRA (v3.0)**. Read `AGENTS.md`, inspect the live repository state, and inventory deployment machine `H0` (`P00`) before assuming any path, wire ID, or hardware speed. Maintain `STATUS.md`, `DECISIONS.md`, `CANDIDATES.lock`, `DATA.lock`, and `EXPERIMENTS.jsonl`. Execute **Track A** first (`P00`–`P07`) so a measured, reproducible pretrained CPU recognizer ships within weeks 1–3 on `H0` under the irrevocable `Preview`/`Commit`/`Final`/`Gap` contract, measuring true process-tree CPU-seconds, paced word latencies, RSS/PSS, and WER. In parallel, run Wave `U00`–`U01` (`wstream_ultra` multi-ISA bake-off + Nemotron HW counters) to bind the real two-term cost law on `H0`. Advance **Track B** (`P08`–`P21`) strictly wave by wave with committed `PREREG.md` files, dense `12.5 Hz` Aux CTC ($T \ge U+R$), utterance-fixed precision/depth, exact `A7`/`A7z` and `T-LUT2`/`T-LUT3-Split` scalar parity (`max_abs_diff == 0`), exact sample-block LFO for confirmed-label adaptation, compressed authoritative audio with an identity graph, and retired single-use confirmation splits. Never report projections as measurements, never weaken a failing acceptance test, never mutate irrevocable `Commit` text, and never touch sealed audio outside the Look Ledger.

---

## Appendix A. Measured `wstream.c` Two-Term Cost Law Data `[SANDBOX-MEASURED]`

Measured on Cloud Intel Xeon (AVX-512 VNNI + VBMI, 1 thread, `gcc -O3 -march=native`), `DIN = 1024`, $U = 4$ accumulators, 7-bit activations, identical checksums between `int8` and 2-bit packed ternary:
* **DRAM `1536M` Ternary Fit ($F \in \{1,2,3,4,6,8,12,16\}$):** $a = 19.36\text{ ps/wt}$, $c = 10.98\text{ ps/MAC}$, $R^2 = 0.9993$, max relative error $5.0\%$.
* **DRAM `1536M` `int8` Fit:** $a = 96.38\text{ ps/wt}$, $c = 10.50\text{ ps/MAC}$, $R^2 = 0.9667$ ($4.98\times$ higher fixed fetch cost!).
* **L3 `64M` Ternary Fit:** $a = 16.46\text{ ps/wt}$, $c = 10.43\text{ ps/MAC}$, $R^2 = 0.9979$.
* **L2 `1M` Fit:** `int8` $a = 5.36\text{ ps/wt}$ beats ternary $a = 14.14\text{ ps/wt}$ in-cache due to unpack latency, proving small cache-pinned layers prefer `int8` while DRAM-swept layers (including Cortex-evicted Reflex) strongly favor ternary.

Wave `U01` extends `wstream.c` into `crates/cadence-kernels/benches/wstream_ultra.rs` adding: AVX2 `T-LUT2`, AVX2 `T-LUT3-Split`, ARM NEON `T-LUT2`/`T-LUT3-Split`, WASM relaxed SIMD, **interleaved Reflex+Cortex LLC eviction**, and `H0`–`H4` hardware counters.

---

## Appendix B. Reproducible Sizing Script (`sizing_ultra.py`)

```python
# CADENCE-ULTRA v3.0 exact sizing & two-term cost model (reproduces §4.1).
A_T_DRAM, A_T_L3, C_MAC = 19.36e-12, 16.46e-12, 10.98e-12  # [SANDBOX-MEASURED]
P2, DERATE = 1.15, 1.80                                    # 15% T2 rows; conservative 1.8x end-to-end allowance
EV, RHO160, Q_ECHO = 5.6, 3.75, 0.25                       # events/s; non-empty 160ms wake-ups/s; selective Echo rate
VR, VC = 1025, 4096

CFGS = {
    "S": dict(ds=192, dr=256, nr=8,  d=512,  n=12, e=2, J=384),
    "M": dict(ds=256, dr=384, nr=12, d=1024, n=16, e=3, J=512),
    "L": dict(ds=256, dr=512, nr=12, d=1280, n=20, e=4, J=640),
}

for name, c in CFGS.items():
    ds, dr, nr, d, n, e, J = (c[k] for k in ("ds", "dr", "nr", "d", "n", "e", "J"))
    r_t = 9*dr*dr*nr + 4*dr*dr*(nr//4) + 18*ds*ds
    r_i = 9*nr*dr + 18*ds + 320*ds + 2*ds*dr + VR*dr
    c_t = 12*d*d*n
    c_i = 2*dr*d + VR*d + 64*d + 5*d*n
    e_t = 12*d*d*e
    h_i = VC*d + d*J + VC*256 + (VC + 512)*J
    tern, i8 = r_t + c_t + e_t, r_i + c_i + h_i
    ram_mb = (tern*0.25*P2 + i8) / 1e6
    dl_mb  = (tern*0.20*P2 + i8) / 1e6
    a_c = A_T_L3 if c_t*0.25*P2 < 16e6 else A_T_DRAM
    fixed_ms = c_t*P2*a_c*1e3
    ev_mac = c_t*P2 + 2*dr*d + VC*d + 2*(d*J + VC*J)
    per_ev_ms = ev_mac*C_MAC*1e3
    r_mac_s = (18*ds*ds*25 + (r_t - 18*ds*ds)*12.5)*P2 + (320*ds*25 + (r_i - 320*ds)*12.5)
    refl_rtf = 6.25*r_t*P2*A_T_DRAM + r_mac_s*C_MAC
    echo_10s_ms = e_t*P2*(A_T_DRAM + 56*C_MAC)*1e3
    kern_rtf = refl_rtf + RHO160*(fixed_ms/1e3) + EV*(per_ev_ms/1e3) + Q_ECHO*(echo_10s_ms/1e4)
    streams = 1.0 / (DERATE * (r_mac_s*C_MAC + EV*per_ev_ms/1e3))
    print(f"{name}: total={(tern+i8)/1e6:.2f}M ram={ram_mb:.1f}MB dl={dl_mb:.1f}MB "
          f"fixed={fixed_ms:.2f}ms ev={per_ev_ms:.2f}ms kern_rtf={kern_rtf:.4f} "
          f"derated_rtf={kern_rtf*DERATE:.3f} echo10s={echo_10s_ms:.1f}ms streams/core={streams:.0f}")
```