# H11: Cross-ISA bit-exact portable log-mel frontend (G4)

**Status:** `EXPLORATORY` (verified across Linux x86_64 AVX2 and macOS ARM64 NEON; awaiting G4 merge before `bench-freeze-v1`).

## Question
Can we eliminate the 1-ULP log-mel drift between Linux x86_64 (AVX2+FMA + glibc `libm`) and macOS Apple M3 Ultra (scalar + Apple `libSystem`) so that `ultra-engine` emits **bit-identical tokens from raw PCM16** across ISAs—even for INT8 and R4T ternary weights—without breaking B2 trainer-parity or regressing WER?

## Setup
- **Code:** `crates/ultra-frontend/src/portable_math.rs` (merged in `2306e2f`), enabled by default in `lane/runtime-g4` (`2c9863d`), removing the `x86_64`-only skip guard in `crates/ultra-engine/tests/output_continuity.rs` and regenerating `output_continuity.json` (`encoder_continuity.json` untouched).
- **Root causes removed:**
  1. Replaced platform `libm` calls (`cos`, `sin`, `hypot`, `f32::ln`) with pure IEEE-754 unfused `f64` math (`sin_cos_portable` Cody-Waite + degree-14/15 Taylor, exact `(re*re + im*im) as f32` power, `ln_portable_f32` mantissa/exponent odd polynomial accurate to `<= 0.5 ULP`, and precomputed 512-pt FFT twiddles).
  2. Replaced `_mm256_fmadd_pd` FMA contraction with unfused `_mm256_add_pd` + `_mm256_mul_pd` and matched 8-lane DC-mean reduction order identically between AVX2 and scalar/NEON.
- **Hosts:**
  - `xbabe3` (`H0`): AMD TR PRO 3995WX (Linux x86_64, `avx2fma`).
  - `xbabe4`: Apple M3 Ultra Mac Studio (macOS ARM64, `neon`).
- **Data:**
  - B2 trainer parity: 5 synthetic edge fixtures + 50 stratified `dev-clean` utterances (31,772 frames) against Python `train.bundle_infer.BundleRecognizer`.
  - DEV selection panel (`docs/ultra/data/dev_selection_panel.json`, manifest `76fead746d1f8b8ae57a30aa341e81364538eb69759af813d6f13d6d1f57fc36`, 730 utterances: 400 `dev-clean` + 330 `dev-other`) across 5 checkpoints × 2 precisions (FP32, prepared INT8).
  - DEV400 (`dev-clean.npz --limit 400`) old→new frontend comparison on `d1-7800` and `d1mr-s18200`.

## Results

### 1. Trainer parity & unit bit-identity
- **B2 trainer-parity (31,772 frames):** `max_abs_diff = 7.153e-7` (`<= 1e-4` gate; unchanged from legacy path).
- **AVX2 vs scalar/NEON frame identity:** **0 ULP** (`to_bits()` identical across every frame and mel bin; `u0_portable_sha256 = 0bf57fd99d0a62c8698c48c7b1d681315ee19a34a3e66c4556c2cb940e9e7157`).
- **`output_continuity` (PCM16 → encoder proj SHA-256 + tokens, 4 utts × FP32/INT8):** **2/2 PASS** on BOTH `xbabe3` (x86_64) and `xbabe4` (M3 Ultra ARM64, 2.87 s) with the `x86_64`-only skip guard removed. **0 token changes** in `output_continuity.json`; `encoder_continuity.json` **unchanged**.

### 2. Cross-ISA 730-utterance DEV panel (x86_64 AVX2 vs Apple M3 Ultra NEON)

| Checkpoint | Precision | Pre-G4 x86 WER | Post-G4 x86 WER | Old→New Δ (pp) | Pre-G4 x86==Mac | **Post-G4 x86==Mac** | Post-G4 INT8 Gate (vs FP32) |
|---|---|---:|---:|---:|---:|---:|---|
| `d1mr-s18200` (S_MR) | FP32 | 30.9939% | 30.9939% | +0.0000 | 730/730 | **730/730** | — |
| `d1mr-s18200` (S_MR) | INT8 | 31.0562% | 30.9731% | −0.0831 | 717/730 | **730/730** | **PASS** (−0.0208 pp, upper +0.1510) |
| `d2-45000` (S dense) | FP32 | 14.1681% | 14.1681% | +0.0000 | 730/730 | **730/730** | — |
| `d2-45000` (S dense) | INT8 | 14.1681% | 14.1888% | +0.0208 | 725/730 | **730/730** | **PASS** (+0.0208 pp, upper +0.1567) |
| `d3mrb-20000` (S_MR_I8S) | FP32 | 21.5255% | 21.5255% | +0.0000 | 730/730 | **730/730** | — |
| `d3mrb-20000` (S_MR_I8S) | INT8 | 21.4839% | 21.5116% | +0.0277 | 720/730 | **730/730** | **PASS** (−0.0138 pp, upper +0.1291) |
| `r4t-12000` (ternary) | FP32 | 25.3045% | 25.3045% | +0.0000 | 730/730 | **730/730** | — |
| `r4t-12000` (ternary) | INT8 | 25.2561% | 25.2284% | −0.0277 | 636/730 | **730/730** | **PASS** (−0.0761 pp, upper +0.0352) |
| `d1-7800` (S dense) | FP32 | — | 38.5659% | — | — | **730/730** | — |
| `d1-7800` (S dense) | INT8 | — | 38.5867% | — | — | **730/730** | **PASS** (+0.0208 pp, upper +0.2110) |

**Total cross-ISA token agreement across all 5 checkpoints × 2 precisions:** **7,300 / 7,300 (100.000%)**.
FP32 old→new token agreement is **730/730 (100%)** on every baseline checkpoint (`+0.0000 pp`). INT8 old→new deltas are all within `[-0.0831, +0.0277] pp` (well inside the `±0.10 pp` gate).

### 3. DEV400 (`dev-clean.npz --limit 400`) old→new continuity (x86_64)

| Checkpoint | Precision | Pre-G4 WER | Post-G4 WER | Old→New Δ (pp) | Old→New Token Agreement |
|---|---|---:|---:|---:|---:|
| `d1-7800` (continuity fixture) | FP32 | 29.7343% | 29.7343% | +0.0000 | 400/400 |
| `d1-7800` (continuity fixture) | INT8 | 29.7343% | 29.7343% | +0.0000 | 370/400 |
| `d1mr-s18200` (S_MR) | FP32 | 23.6811% | 23.6811% | +0.0000 | 400/400 |
| `d1mr-s18200` (S_MR) | INT8 | 23.6811% | 23.7071% | +0.0259 | 350/400 |

## What went wrong or surprised us
- Ternary R4T had previously drifted on **94/730** utterances (`636/730`) between x86 and Mac solely due to 1-ULP log-mel tie-breaking in per-row INT8 activation quantisation; switching to portable log-mel eliminated **100% of those 94 mismatches** (`730/730`).
- Compiling on `aarch64` caught two `x86_64`-only unused imports in `trainer_mel_avx2.rs` (`#[cfg(target_arch = "x86_64")]` added).

## Decision / consequence
- **G4 satisfied (Option a):** portable log-mel becomes the default in `TrainerMelConfig`, eliminating the "frontend-dependent" caveat on ARM rows ahead of `bench-freeze-v1`, and enforcing PCM→tokens `output_continuity` on both `x86_64` and `aarch64`.

## Pointers
- Commits: `2306e2f` (portable math backend), `lane/runtime-g4` (`2c9863d`, default flip + fixture).
- Data-drive artifacts (not committed):
  - `/mnt/xbabe3-external/jsr/runs/engine/g4-panel-x86-20261121/`
  - `/mnt/xbabe3-external/jsr/runs/engine/g4-panel-mac-20261121/`
  - `/mnt/xbabe3-external/jsr/runs/engine/g4-dev400-x86-20261121/`
  - Baseline comparison: `/mnt/xbabe3-external/jsr/runs/engine/panel-fd0b1a3/`
