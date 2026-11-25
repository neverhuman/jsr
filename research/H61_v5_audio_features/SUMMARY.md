# H61 — V5 Cheap CPU Audio Features & Signal-Processing Closeout Ledger

**Status:** `EXPLORATORY` (selection-split `S` and development pilots; `PUBLIC-TEST` untouched; `C`-half untouched in all V5 runs)
**Scope:** Receipt-backed closeout of all tested, closed, parked, and in-progress concepts from both V5 proposals:
- **Spec 1:** `tips/v5/isosyll-cheap-audio-features-spec.md` (`ENV`, `FINE`, `REF`, `EXC`, `Z1`, `Z2`, `Z3`, `P0-a`–`P0-d`, §4.4 schema/export guard)
- **Spec 2:** `tips/v5/JSR_CPU_Audio_Features_Engineering_Spec.md` (`FLOOR`/band reliability, `MOD`/multiscale contrasts, `PCEN`, `SHAPE`, capture quality/LM routing, low-energy residual, rank-8 adapters, distillation, exclusions)

**Release Baseline & Pinned Scorer:**
- Signed release pin: **`v3.1.0` (RC2)** (`REGctl-boundary.pt`, EMA, `d=384`, `56.73M` params, `chunk=4`, `left_context=128`, native INT8).
- Selection split: `dev_split_v2.json:selection` (`S`-half only: **2,782 utterances**, **36 speakers** [20 `dev-clean`, 16 `dev-other`], **52,196 reference words**, `sha256: 002ac7aee51519b4657d56b887b859660cfda51cf925c76aae61574150f29cac`).
- Signed RC2 `S`-half C4 greedy baseline: **3,626 / 52,196 edits = 6.9469%** (`dev-clean` 3.4959%, `dev-other` 10.8289%; matched Python FP32/EMA diagnostic = 3,622 edits / 6.9392%).
- Pinned scorer: `ultra-score` (`sha256: d784f8ca65a0df52c2dc2f58aa0d22912428f3595effa0625367d6e8aad063e1`, `ultra-en-wer-norm/v1`, seed `20261122`, `10,000` speaker-cluster bootstrap replicates).

---

## 1. Executive Verdict (`v3.2.0` Release Disposition)

1. **`v3.2.0` is a runtime-only release; signed `v3.1.0` `RC2` INT8 remains the public weight pin.** No neural weight update ships in `v3.2.0`. Probe `P0-b` failed on clean speech (`0/19` feature arms reached `−1.00%` Block-4 phone CE), `D1` failed under all three stressed conditions (`−30 dB`, MUSAN `10 dB`, MUSAN `5 dB`; `0/4` pre-registered groups passed), `H61` 3.5k / 8k Core-40 `ALL-ON` and the 3-seed `H61 Step-2` noise+clean-tail follow-up all missed signed `RC2` on clean `S`-half C4 (`7.13–7.27%` / `7.1698%` vs `6.9469%`), and Track M1-v2 / M1-v3 short-horizon runs (`M1v2_FULL_s7919`, `M1v2_CTL_s7919`, `M1v3_CLEAN`, `M1v3_CTL`) all missed the clean `RC2` bar (`7.0714–7.1327%` EMA vs `6.9469%`).
2. **`v3.2.0` Runtime Scope (`R10` + `Z1` + Export Guard; `Z2` OUT of `v3.2.0`):**
   - **`R10` runtime path** (already on `main`) + signed `v3.1.0` `RC2` INT8 weights.
   - **`Z1` (`PR #152` MERGED → `main` `a374a546`, head `f04027c`; `PR #154` MERGED → `main` `6d95f37d`, CLI `--commit-rms-norm`):** Opt-in whole-segment RMS gain in the final-text commit pass (`113.0%` recovery at `−20 dB`, `96.9%` at `−30 dB`, full-`S` clean `M5` `−0.0671 pp` [`2,838 → 2,803` edits, upper `−0.0175 pp`], bit-exact x86 oracle over `53,034,000` samples on `f04027c`, and `0`-diff 50/50 AVX2/Scalar/NEON cross-ISA parity on both `de0443a` and `f04027c`), exposed opt-in via `--commit-rms-norm` on `ultra stream --commit-pass` (`PR #154` MERGED `6d95f37d`).
   - **Standalone Fail-Closed `side_mode` Export Guard (Spec 1 §4.4):** Extracted into `r1_to_safetensors.py` and `export.py` (with `test_export_side_guard.py`) so `ultra.r1-weights/v1` and ONNX export refuse any checkpoint with `side_mode != ""`, `side_film_dim > 0`, or `encoder.side.*` / `*.side_film.*` weights.
   - **`Z2` Arm B & TRAIN-Calibrated Arm B-cal (`feat/v32-z2-agc` `a2d9a3a`) — Stays OUT of `v3.2.0`:** Uncalibrated Arm B failed both strict full `164/164` (`27.56 h`) `C3` gates (`+0.091 pp` vs `±0.087 pp` spread; `9` vs `8` `dup_inserted`; `@codex` replay `+0.09052 pp`), despite cutting noise-gap false updates `586×` (`99.729% → 0.1703%`) and `0.000%` silence-gap updates. The sealed `06:26Z` **TRAIN-calibrated Arm B-cal** (`63/63` `SHA256SUMS` OK; frozen TRAIN-only voiced target `0.084848661` / `−21.4271 dBFS`, 1,000 LibriSpeech TRAIN utts) resolved the `−3.68 dB` bias and **PASSED** the full `164/164` `C3` WER spread (`11.144%`, **`+0.032 pp`** vs `off`, `|Δ| ≤ 0.087 pp`, **`−92.8%`** regression cut vs `v3.0.0`), while holding `>100%` `−30 dB` recovery and lowering MUSAN-100 gain drift (`2.659 → 1.817 dB`), but **FAILED** strict `Δdup ≤ 0` (`11` vs `8` `dup_inserted`, `+3` mid-utterance greedy transducer flips, `0` gap hallucinations). The sealed `06:18Z` `Z1 × Z2` 4-cell policy matrix (`136/136` `SHA256SUMS` OK) confirmed zero double-gain overshoot or extra clipping (`2/38.25M` = `0.05 ppm`, `Δclip = 0`). Because strict `C3` fails on `Δdup`, **`Z2` stays OUT of `v3.2.0`**.
3. **Closed & Separate Post-`v3.2.0` Tracks:**
   - **`H61 Step-2` Noise+Clean-Tail Follow-Up (`A4`/`A3`) — `FAIL` (`H61` Additive Side Features Permanently CLOSED as a Release Lever):** Primary variant **EMA** (`7.1698%`, `11,227` edits vs raw `7.3141%`, selected before any noise read). **Leg 1 (Clean `c4`, `ALL-ON − signed RC2` `6.9469%`):** **`+0.223 pp`** (`95% CI [+0.098, +0.352]`, one-sided upper **`+0.331 pp > 0` → FAIL**). **Leg 2a (`MUSAN-TEST 5 dB`, `ALL-ON − RC2`):** **`−8.853 pp`** (`[−10.08, −7.67]`, upper `−7.844 < 0`, PASS). **Leg 2b (`MUSAN-TEST 5 dB`, `ALL-ON − CAP-CTRL`):** **`−0.114 pp`** (`[−0.231, +0.010]`, upper `−0.010`, marginal; two-sided CI includes `0`). ~98.7% (`−8.74 pp`) of the held-out real-noise gain comes from real-noise training (`musan_h61_train`), not the 40-dim side features (`s2_gate.json` `9df36708`).
   - **`A2` `h62_warm` Part 2 (`G1`/`G2`/`G2n`):** **CLOSED** — no frozen-model guard clears both `WARM_NAT` and `WARM_SHUF` (`G2n` TRAIN `J = 0.095 < 0.30`).
   - **Track M1-v2 (`S4v11`, 12k) & `M1v3_CLEAN` vs `M1v3_CTL` (6k Zero-Factor-E Diagnostic):** **SEALED** — `M1v2_CTL` isolates that `turn_switch` adds zero clean tax vs that matched control (`−0.002 / −0.098 pp`) and cuts wrong-speaker collapse **85–90%**. The sealed `M1v3_CLEAN` vs `M1v3_CTL` diagnostic (`272/272` OK, `gain_aug_p=0, noise_aug_p=0`, 6k 1-GPU / `1,580` audio-h; `M1v3_CTL = 7.1327%` [`+0.186 pp` vs `RC2`], `M1v3_CLEAN = 7.1155%` [`−0.017 pp` EMA / `−0.081 pp` raw vs `CTL`], `DIFF_SPK_5S −68%`, `WARM_SHUF −72%`, `WARM_NAT −0.362 pp`) proves the clean offset came from cold-restart + short horizon, **NOT** Factor-E.
   - **Separate Post-`v3.2.0` Full-Horizon Schema-`v1` Tracks:** `M1v4` (`M1v4_MIXED_COOL_s7919`, full-horizon cooldown, zero aug, in flight) and frozen `H63` / `M1v5` (`M1v5_NOISE_COOL_s7919`, full-horizon cooldown + real-noise training `musan_h61_train` with clean tail, `PREREG.md` `cfa7acef62e7a05c128ddd554f379ecb279ad2aac44be16f543c732d3855b4b2`); neither gates `v3.2.0`.

---

## 2. Gate 0 Probes & Zero-Retraining Fixes (`P0-a`–`P0-d`, `Z1`, `Z2`, Router)

### 2.1 `P0-a` — Warm Audio-Prefix Speaker Adaptation Headroom & Turn-Switch Cliff (`PASS / KEEP`)
- **Protocol:** Full `S`-half (2,782 utts, 52,196 words), signed `v3.1.0` INT8 `c4_greedy`, 5.0 s active-RMS-matched prefix + 0.3 s zero silence, causal trace boundary slicing at `5.300 s`. `ALONE` matched archived `RC2` with **0 edit diff** (`3,626` edits, `6.9469%`).
- **Results (`B − A`, 2-sided 95% speaker-cluster CI):**
  - `SAME_SPK_5S − ALONE`: **−0.2050 pp** `[-0.3787, -0.0284]` (1-sided upper `−0.0557`; `dev-clean` `+0.0181`, `dev-other` **`−0.4560 pp`** `[-0.7424, -0.1645]`; Words 1–5 `−0.669 pp` pooled / **`−1.264 pp`** `dev-other`; Words 6+ `−0.039 pp` null). 86% of the gain (`−92` of `−107` edits) occurs in target words 1–5; worst-decile speakers Words 1–5 drop `24.41% → 21.89%` (`−2.528 pp`).
  - `SAME_SPK_5S − DIFF_SPK_5S`: **−3.0749 pp** `[-3.7777, -2.4554]` (36/36 speakers improve; `dev-other` `−5.3859 pp`).
  - `DIFF_SPK_5S − ALONE` *(wrong-speaker fragility)*: **+2.8700 pp** `[+2.3189, +3.5289]` (35/36 speakers hurt; `dev-other` `+4.9300 pp`; Words 1–5 `dev-other` `+12.295 pp`, `12.95% → 25.24%`).
- **Decision:** Confirms real warm speaker-adaptation headroom on `dev-other` cold starts and exposes severe wrong-speaker KV contamination. Motivated `A2` (`h62_warm`) and Track M1 (`turn_switch`).
- **Receipt:** `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/p0a_prefix/summary.json` (`sha256: a51dffaa7a61e039c102c55f7c8d67e9bcbff2db663244d6ad83d30449366c2b`).

### 2.2 `P0-b` — Clean Frozen RC2 Block-4 & Stem MFA Linear Phone-CE Probe + `EXC` Audit (`FAIL` on Gate)
- **Protocol:** 5-fold speaker-disjoint CV on `S`-half (458,110 25 Hz frames, 71 ARPAbet classes, convex GPU L-BFGS, L2=`1e-3`, 2,000-rep speaker bootstrap). Gate: `≥ 1.00%` relative Block-4 CE reduction (`Δ% ≤ -1.00%`) with 95% CI `< 0`.
- **Results (Block-4 BASE CE = `0.9610` nats; Stem BASE CE = `1.9003` nats):**
  - **0 / 19 feature arms passed** the Block-4 `-1.00%` gate (all 22 serialized rows incl. Gaussian controls `CTRL_GAUSS_48` `+0.18%` and `CTRL_GAUSS_160` `+0.74%` failed).
  - Best Block-4 arm: `G6_MOD_32_STACK` (`−0.25% [-0.31, -0.19]` pooled, `−0.75%` fricatives). `G8b_PCEN_32_STACK` `−0.11%`, `G5_FINE_48_STACK` `−0.05%`, `G1_FLOOR_8` `−0.04%`, `G4_EXC_4` `−0.04%`, `LEAD_40_STACK` `+0.00%`, `G7_SHAPE_8` `+0.02%`, `G8a_REF_80` `+0.08%`, `G2b_ENV8_COLD_8` `+0.10%`, `G3a_ENV35_ONLINE` `+0.56%`, `G3b_ENV35_UTT` `+0.93%`.
  - On the **Stem** (before Conformer blocks), features carry strong local signal (`LEAD_40_COLD_STACK` `−4.49%`, `ALL_FEATURES` `−5.95%`, `G3b_ENV35_UTT` `−3.36%`, `G6_MOD_32_STACK` `−2.30%`, `G8a_REF_80` `−1.99%`), showing the first 4 Conformer blocks already absorb nearly all clean-speech information from 80-bin log-mel.
- **Real-Speech `EXC` Audit (100 Hz integer NCCF):**
  - Voiced (`φ ≥ 0.5`): MFA sonorants `90.57%` (`mean φ = 0.8269`), vowels `90.58%`, nasals `93.52%`, voiced obstruents `67.80%`, unvoiced stops/frics `37.66%`, silence `24.30%`.
  - **Critical false-voicing finding:** raw `φ ≥ 0.5` fires on **26.124%** of `musan_noise` frames (100 cuts, 95,226 frames) and **75.885%** of `musan_music` frames (40 cuts, 39,875 frames). Adjacent-voiced octave-jump rate (`|Δlog2 F0| > 0.7`): `7.705%` at 100 Hz, `14.406%` at 25 Hz. Blocked raw `EXC`-only AGC and drove the RMS+SNR+`EXC` gate in `Z2` and M2.
- **Receipts:** `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/p0b_mfa_probe/summary.json` (`sha256: f8b0d756db0007213c75f36200945ce703f6387217d66a9b7fa15cc6f67abae1`) and `p0b_table.md` (`sha256: 42d924f6f6cb33428330c27d6278368b4cf7bae627f414fc41b17afae82f7f9c`).

### 2.3 `P0-c` — Filterbank Warp / VTLN (`CLOSED` for Global Warps; 7-Bank Oracle `PARKED`)
- **Protocol:** 144 `S`-half clips (36 speakers × 4 utts, 3,051 words), signed native INT8 decoder (`avx2fma`).
- **Results:** Neutral `5.4408%` WER; global `warp095` `5.5064%` (`+0.0656 pp [-0.245, +0.426]`); global `warp105` `5.5719%` (`+0.1311 pp [-0.174, +0.437]`). In-sample hindsight selection across `{neutral, 0.95, 1.05}` (chosen 28 / 5 / 3 speakers): `5.0803%` (`−0.3605 pp [-0.548, -0.186]`, non-deployable hindsight only).
- **Receipt:** `/home/ubuntu/jsr/target/v5-feature-probes/warp/summary.json` (`sha256: a57d26836c6309519fab9d1016a4207b314c4bfb8d1622181c406298fdb46e95`).

### 2.4 `P0-d` / `Z1` — Whole-Segment RMS Normalisation in Final-Text Commit Pass (`MERGED` — `PR #152` → `main` `a374a546`, `PR #154` → `main` `6d95f37d`)
- **Implementation:** **`PR #152` MERGED to `main` (`a374a546`, branch head `f04027cfb9d7d13b39506d153b14b6b79b6d88a9`, review fixes `972bda5`)** and **`PR #154` MERGED to `main` (`6d95f37d`, commit `9c30bfc`, opt-in `--commit-rms-norm` on `ultra stream --commit-pass`):** `ultra_frontend::level`, target RMS `1928`, gain clamp `[-12, +40] dB`, half-even rounding, int16 clip, applied strictly before 320 ms tail padding; default `rms_norm: false` bit-exact to `v3.1.0` (`serde`-skipped on `commit_rms_norm` when off, folded into `EffectiveDecodeConfig::policy_sha256` only when enabled; unused LM hook dropped per `@grok` review).
- **Results (signed `v3.1.0` `ultra-engine-commit`, `chunk=16`, `beam=16`, `M5.ulm`):**
  - **144-clip `S` attenuation screen (3,051 words):**
    - `0 dB`: `4.4248% → 4.3592%` (`−0.0656 pp [-0.279, +0.169]`, 135 → 133 edits).
    - `−20 dB`: `5.1786% → 4.3265%` (**`−0.8522 pp [-1.610, -0.264]`**, 158 → 132 edits, **113.0% recovery**).
    - `−30 dB`: `5.4736% → 4.4576%` (**`−1.0161 pp [-1.770, -0.408]`**, 167 → 136 edits, **96.9% recovery**).
  - **Full `S`-half normal-level guard (`@grok` reviewed, 2,782 utts, 52,196 words, `18,333.7` audio-s):** `5.4372%` (`2,838` edits) → **`5.3701%`** (`2,803` edits), **`Δ = −0.0671 pp [-0.1216, -0.0175]`**, clearing the `+0.087 pp` upper guard cleanly. Preprocessing CPU: `72.51 µs/audio-s` (`0.036%` of commit decode CPU); 147 / 293,339,054 samples clipped (`0.000050%`).
  - **Exact `f04027c` Native Kernel & Cross-ISA Parity:**
    - `@codex` exact `f04027c` 53,034,000-sample x86 oracle (`z1-native-f04027c/receipt.json`, `sha256: de451fba...`, `level.rs` `2d55e786...`): **`0` PCM diffs** across all `+0 / −20 / −30 dB` × 144 clips (`17,678,000` samples each).
    - Sealed `de0443a` 50/50 AVX2/Scalar/NEON cross-ISA + H47 Tier-A real-endpoint witness (`v5_z1_witness/SHA256SUMS` `sha256: fc1b4373...`, `z1_witness_report.json` `sha256: ee1e0aa7...`).
    - Sealed `f04027c` 50/50 AVX2/Scalar/NEON cross-ISA recheck (`v5_z1_witness_f04027c/cross_isa_report.json` `sha256: 74c66be0...`, `SHA256SUMS` `sha256: eba22cbc...`): **`0` full-tuple diffs** across AVX2, Scalar, and Apple Silicon NEON on both `rms_norm=false` and `rms_norm=true`, and **`0` committed transcript diffs** vs `@codex` oracle.
- **Note:** Voiced-frame `Z1` remains a separate unimplemented research hypothesis.
- **Receipts:**
  - Attenuation: `/home/ubuntu/jsr/target/v5-feature-probes/z1-rc2/summary.json` (`sha256: 12310cb8073d56e06b02336c88f96d0d24713e65a8bda1e89237ded410d0d2a3`)
  - Full-`S` guard: `/home/ubuntu/jsr/target/v5-feature-probes/z1-clean-guard/summary.json` (`sha256: ea479df52f2592798c09f8978131a757186d2da2979c1ca4264a10dff0c20543`)
  - `f04027c` 53,034,000-sample x86 oracle: `/home/ubuntu/jsr/target/v5-feature-probes/z1-native-f04027c/receipt.json` (`sha256: de451fba6463b34a3639d6fae44dcf62c2594131fc4a20a1f5847550cdbe831e`)
  - `de0443a` 50/50 cross-ISA + H47 endpoint witness: `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_witness/SHA256SUMS` (`sha256: fc1b43737655c7838dbf6a5c336eb69345f7cca1fa834da931d3396da81ce0e5`)
  - `f04027c` 50/50 AVX2/Scalar/NEON witness: `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_witness_f04027c/cross_isa_report.json` (`sha256: 74c66be0155d1e3a19c5cda3497582235712625a2fa64a26f2805f789e571756`)

### 2.5 `Z2` — RMS/SNR + `EXC` Voicing-Gated Warm AGC & `Z1 × Z2` Policy Matrix (`C3 FULL 164/164 FAIL` — Stays OUT of `v3.2.0`)
- **Implementation:** Branch `feat/v32-z2-agc` commit `a2d9a3a` (`ultra_frontend::ExcVoicingTracker`, `#![forbid(unsafe_code)]`, 0 `libm`, 0 FMA, `2,064 B < 2.5 KiB` state, opt-in `#[serde(default)] voicing_gate: false` bit-exact to `origin/main`, bit-exact cross-ISA scalar parity). Tested **Arm B** (`--warm-agc --tau-s 5.0 --voicing-gate --snr-gate-db 6.0 --voicing-thresh 0.5 --speech-floor-dbfs -65.0`, default `target_rms = 0.058884366` / `−24.6 dBFS`) and **TRAIN-Calibrated Arm B-cal** (`--target-rms 0.084848661` / `−21.4271 dBFS`).
- **Sealed Full `164/164` (`27.56 h`, `106,054` ref words) `C3` Confirmation (binary `ultra-engine-agc` `sha256: 41744b9c5b705e17b633458213233e5ec6900e5162ad3227c794a1a44ad4bdf7`, `63/63` `SHA256SUMS` OK):**
  - `off` (`--no-warm-agc`): **`11.112%`** (`8,689 S / 1,996 D / 1,100 I`, `8` dup inserted repeats; **`164/164` exact match vs archived `v3.0.0`**, `0.0 pp` drift).
  - `v3.0.0` ungated warm AGC: **`11.559%`** (`+0.447 pp`, `10` dup repeats = `+2` new; gap false-update rate `99.7291%` on noise gaps, `49.376%` all gaps → **FAIL**).
  - **Uncalibrated Arm B (`−65 dBFS`, default `target_rms`):** **`11.203%`** (`8,773 S / 2,033 D / 1,075 I`, **`+0.091 pp`** vs `off` / `−0.356 pp` vs `v3.0.0` [80% regression cut], **`9` vs `8` dup inserted repeats** [`+1` new vs `≤ 0`]) → **STRICT C3 FAIL** on both gates (`+0.091 pp > ±0.087 pp` spread AND `+1` new inserted repeat; median gain `−3.68 dB`, `p95 −1.84 dB`, `min −10.57 dB`).
  - **Independent `@codex` Transcript Replay (`z2-c3-full-independent/receipt.json`, `sha256: b5b05975...`):** Verified exact uncalibrated Arm B counts (`164/164` pieces, `0` off mismatches vs archived `v3.0.0`), **`Δ = +0.09052 pp`** (`absolute_wer_within_0_087_pp: false`), **`9` vs `8` inserted repeats**.
  - **Gap False-Update Audit (`gap_audit_full164.json`, `164/164` exact cache parity, `0.0 dB` max diff):** Noise-gap false updates cut **`586×`** (`99.7291% → 0.1703%`), silence-gap false updates **`0.000%`** (`0 / 3,097,142` frames), all-gap **`0.0843%`** (`5,171 / 6,134,186`), utterance update rate `40.938%` (`15.64%` overall).
  - **Sealed `06:26Z` TRAIN-Calibrated Arm B-cal (`criterion3_full164_cal.json`, `PASSES` spread / `FAILS` `Δdup ≤ 0` → `Z2` stays OUT of `v3.2.0`):**
    - **Frozen TRAIN-Only Calibration (`train_voiced_rms_calibration.json`):** Measured strictly on `1,000` LibriSpeech TRAIN utterances (`500` `train.clean.100` + `500` `train.other.500`, `750` speakers, `39.844%` voiced frames): frozen voiced target **`0.084848661`** (**`−21.4271 dBFS`**, `+3.1729 dB` above default `−24.6 dBFS`).
    - **Full `164/164` `C3` Result:** **`11.144%`** (`8,727 S / 1,996 D / 1,096 I`, **`+0.032 pp`** vs `off` `11.112%` — **PASSES `|Δ| ≤ 0.087 pp`**, **`−0.415 pp` / `−92.8%`** regression cut vs `v3.0.0` `+0.447 pp`; median gain `−0.51 dB`, `p95 +1.34 dB`, `min −7.40 dB`, `max +4.81 dB`, `15.64%` update rate), but **FAILS strict `Δdup ≤ 0`** (**`11` vs `8` `dup_inserted`, `+3` new inserted repeats** from mid-utterance greedy transducer flips in pieces 18/53/82 [with `−1` fixed in piece 39], **`0` gap hallucinations**, total insertions `I = 1,096` vs `1,100` `off`).
    - **Arm B-cal Gain Recovery (`shalf_gain_cal_summary.json`, 360 `S`-half utts, 0 `C`-half) & MUSAN-100 (`musan100_cal_summary.json`):** Holds `>100%` `−30 dB` recovery at `0 / −20 / −30 dB` (**`7.772% / 7.787% / 7.728%`**, `−0.044 / −0.029 / −0.088 pp` vs `off 0 dB` `7.816%`; median gains `−0.14 / +19.74 / +29.65 dB`), and lowers MUSAN-100 mean `|gain drift|` **`2.659 → 1.817 dB`** (median gain `−0.021 dB`; false update rate unchanged `9.0404%` mean / `3.70%` cut median).
    - **Disposition:** Strict `C3` fails on `Δdup` (`+3 > 0`); code untouched at `a2d9a3a` and **`Z2` stays OUT of `v3.2.0`**.
- **Sealed `06:18Z` `Z1 × Z2` 4-Cell Policy Interaction Matrix (`v5_z1_z2_matrix/SUMMARY.json`, `136/136` `SHA256SUMS` OK):**
  - **Protocol:** `360` `S`-half utts (`36` speakers, `6,755` ref words, `38,252,642` PCM samples, `0` `C`-half utts), `Z1` (`a374a546`) × `Z2` Arm B (`a2d9a3af`) across `0 / −20 / −30 dB`, commit decode (`chunk=16, beam=16, M5.ulm`) and live streaming `c4`.
  - **Zero Double-Gain Overshoot or Extra Clipping:** Combining `Z1 + Z2-ArmB` produces **`2 / 38,252,642` clipped samples (`0.05 ppm`, `Δclip = 0` vs `Z1/OFF`)** at all three gains (`0 / −20 / −30 dB`), with `0.0 dB` median output RMS and net-gain delta vs `Z1/OFF` (`−24.61 dBFS` target).
  - **Final `M5` Commit WER (`0 / −20 / −30 dB`) & Live `c4` WER:**
    - `OFF/OFF`: `4.811% / 4.885% / 5.551%` (live `c4`: `6.203% / 6.603% / 7.713%`).
    - `Z1/OFF`: `4.693% / 4.782% /` **`4.796%`** (`−30 dB` **`−0.755 pp`** `[-1.103, -0.412]`).
    - `OFF/Z2-ArmB`: `4.663% / 4.841% / 4.885%` (`−30 dB` `−0.666 pp` `[-1.056, -0.292]`).
    - `Z1/Z2-ArmB`: `4.752% / 4.870% /` **`4.870%`** (`−30 dB` **`−0.681 pp`** `[-1.033, -0.340]`).
    - **Live `c4` `Z2-ArmB`:** `6.129% / 6.292% /` **`6.321%`** (`−30 dB` **`−1.392 pp`** `[-1.898, -0.911]`).
- **Supporting Pilot & Uncalibrated Gain-Recovery Screens (historical, superseded by full `164/164` `C3` above):**
  - 24-piece `C3` pilot (4 h, 15,810 ref words): `off` `10.683%`, `v3.0.0` `11.012%` (`+0.329 pp`), Arm B `10.639%` (`−0.044 pp`, `0` dup repeats — did not generalize to full 164 pieces due to `−3.68 dB` voiced-target bias); Arm A (`−45 dBFS`) `10.576%` (`1` dup repeat, FAIL).
  - 360-utt `S`-half gain recovery (0 `C`-half utts): `off` `0 / −20 / −30 dB` = `7.816% / 8.705% / 10.466%`; uncalibrated Arm B `7.802% / 7.772% / 7.802%` (`−0.014 pp` vs `off 0 dB` = **100% recovery**, median gain `+26.48 dB`); Arm A `−30 dB` `8.497%` (`+0.681 pp`, incomplete recovery).
  - 100 MUSAN noise cuts (94,620 frames): false update rate drops `95.14%` (`v3.0.0`) / `26.1%` (raw `EXC`) → **`9.04%`** mean (**median `3.70%`**; stationary 60 Hz hum `0.0%`, white/pink `< 1%`).
- **Receipts:**
  - Sealed `Z2` summary & manifest (`63/63` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/SUMMARY.json` (`sha256: 7e7c32fc14b5e6daaf047a34e2df365109e5574cb50482470b5ec4c4757b9644`) and `SHA256SUMS` (`sha256: 7b5aeb9576b65b52bc98cd97f5e6ddcfe161c2c03244dadcb868e831331a14f1`)
  - Arm B-cal TRAIN calibration, full `164/164` `C3`, `S`-half gain & MUSAN-100: `calibration/train_voiced_rms_calibration.json` (`sha256: 167274338e14fd940001af46beafcd8ab3e67135504ab646795296ceb1a6b7cb`), `c3/criterion3_full164_cal.json` (`sha256: a1f16e12b9aaeff855c2e436cd7bce731270139886f9eb079915a11237e01748`), `shalf_gain/shalf_gain_cal_summary.json` (`sha256: 01c546371865e022d0c0259f36727ca542ac70dd976f9c7493422d17eb9045fb`), `musan100/musan100_cal_summary.json` (`sha256: cdb8808f7c19426e9c1de79f90cf29a3452e6fa439cc94eaba6a3c013c5129b6`)
  - Uncalibrated full `164/164` `C3` & gap audit: `c3/criterion3_full164.json` (`sha256: 52589f4810e0c826ba3731ad3c73607b3c91e7141494d1a6f7dda9f1e21f1d45`), `c3/gap_audit_full164.json` (`sha256: a1c7f70b80187acdfb6c438ceedc6030352d704ad0458c24398e3d10280489c`)
  - Independent `@codex` full-164 replay: `/home/ubuntu/jsr/target/v5-feature-probes/z2-c3-full-independent/receipt.json` (`sha256: b5b0597527ec5fe278550644e7c471bdf89533893f4eddea4f383e0251a3f9b9`)
  - Sealed `Z1 × Z2` 4-cell matrix (`136/136` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_z2_matrix/SUMMARY.json` (`sha256: eb6cc78f76925ab8fe86a5effdbab32ec1d6bf84f548d21dd0d259b9fa201894`) and `SHA256SUMS` (`sha256: 4444b886bb2f7e8f825dd66985d89ed0e270e7b62c5486017af160508504f2dd`)

### 2.6 First-Chunk Feature-Augmented LM Router (`CLOSED`)
- **Result (2,782 `S` utts, `2,560`-sample prefix, fused C4+LM `3,267` edits vs greedy `3,626`):** Mel-only router chose LM on 100% of utterances (`0` greedy); adding reliability/MOD/quality selected 11 greedy (`+0.0038 pp [-0.0037, +0.0117]`); adding `FINE` selected 26 greedy (`+0.0019 pp [-0.0084, +0.0121]`). Null; closed.
- **Receipt:** `/home/ubuntu/jsr/target/v5-feature-probes/router/summary.json` (`sha256: 9bd1493656d1fe80c502c4bfbf012b141ddb55f9314f86f392de4745ac679662`).

---

## 3. H61 Core-40 Side-Feature Path (`D1`, `D2`, `long8k`, Owner `Step-2`, Rust `side.rs`)

### 3.1 `D1` — Stressed `P0-b` Probe (`FAIL` — Closes H61 Core-40 for `v3.2.0`)
- **Protocol:** Identical 5-fold speaker-disjoint `S`-half Block-4 linear phone-CE probe (`458,110` frames, 71 phones, `train/side_features.py` `sha256: 617b59ac95b9...`) on exact stressed int16 PCM under 3 conditions: `−30 dB` downscale, TRAIN-only MUSAN `noise/` @ `10 dB`, and MUSAN `noise/` @ `5 dB`. Gate: `≥ 1` pre-registered arm (`MOD_32`, `FLOOR_8`, `ENV8_COLD`, `LEAD_40_STACK`) achieves `Δ Block-4 CE ≤ −1.00%` (pooled or any target phone class, 95% CI `< 0`).
- **Official Verdict: `FAIL` (0 / 12 pre-registered arm×condition cells passed):**

| Condition | Block-4 BASE CE | `MOD_32` Pooled / Best Class | `FLOOR_8` Pooled | `ENV8_COLD` Pooled | `LEAD_40_STACK` Pooled / Best Class | `CTRL_GAUSS_160` |
|---|---:|---:|---:|---:|---:|---:|
| `gain_neg30db` | 1.0102 | `-0.28%` / `-0.51%` (frics) | `-0.04%` | `+0.22%` | `+0.06%` / `-0.81%` (frics) | `+0.73%` |
| `musan_10db` | 1.2894 | `-0.22%` / `-0.32%` (frics) | `-0.02%` | `+0.00%` | `-0.07%` / `-0.30%` (frics) | `+0.60%` |
| `musan_5db` | 1.5207 | `-0.23%` / `-0.35%` (frics) | `-0.02%` | `-0.04%` | `-0.16%` / `-0.36%` (frics) | `+0.52%` |

- *(Diagnostic Stem BASE CE `1.9501 / 2.3270 / 2.4944`: `LEAD_40_STACK` achieves `-4.70% / -5.45% / -5.51%`, confirming the stem-to-Block-4 absorption pattern persists under stress.)*
- **Receipts:** `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/d1_stressed_p0b/summary.json` (`sha256: a6ec3576774f6d2897dad155745cd19374ecf3eaa0bf74da3d34771afec3d935`) and `d1_table.md` (`sha256: 7d916fd88a1df33a54e96626691ed5d2eba9fb539b74a265d83cc86842e70b8d`).

### 3.2 `D2` — 3-Seed `ALL-ON` vs Parameter-Matched `CAP-CTRL` (3,500 Steps, Seeds `7919/7920/7921`)
- **Setup:** Code `54d4e6ec` (`v5/side-features`, Core-40 `FLOOR|ENV|EXC|FINE|MOD` via zero-init `Linear(160→384)` vs parameter-matched `melproj` `CAP-CTRL` and mel-only `CTRL-AUG`, all trained 3.5k steps with Factor-E gain/noise aug from `REGctl-boundary.pt`).
- **Independent Matched-Pair Replay (`S`-half, C4 greedy, 3-seed mean `ALL-ON − CAP-CTRL`, 1-sided 95% speaker bootstrap):**

| Condition | `ALL-ON` 3-Seed WER (7919 / 7920 / 7921) | `CAP-CTRL` 3-Seed WER (7919 / 7920 / 7921) | Mean `Δ` (`ALL-ON − CAP-CTRL`) | Speaker 1-Sided 95% Bounds | Seed+Spk Sensitivity |
|---|---|---|---:|---:|---:|
| `clean` (0 dB) | 7.238% / 7.148% / 7.267% | 7.261% / 7.150% / 7.225% | `+0.0057 pp` | `[-0.0242, +0.0353]` | `[-0.0405, +0.0509]` |
| `g-30db` | 7.970% / 7.953% / 7.974% | 8.071% / 7.999% / 8.035% | `-0.0696 pp` | `[-0.1273, -0.0125]` | `[-0.1405, +0.0006]` |
| `g-30db + Z1` (C4) | 7.253% / 7.253% / 7.322% | 7.294% / 7.219% / 7.267% | `+0.0166 pp` | `[-0.0305, +0.0624]` | `[-0.0501, +0.0851]` |
| `g-30db + Z1` (C16) | 6.807% / 6.840% / 6.824% | 6.828% / 6.805% / 6.820% | `+0.0057 pp` | `[-0.0452, +0.0526]` | `[-0.0555, +0.0664]` |
| `white10` (10 dB) | 22.992% / 22.735% / 22.856% | 24.017% / 23.509% / 23.860% | **`-0.9343 pp`** | `[-1.1065, -0.7748]` | `[-1.1429, -0.7304]` |
| `musan5` (5 dB) | 22.439% / 22.442% / 22.331% | 22.891% / 22.755% / 22.933% | **`-0.4553 pp`** | `[-0.5717, -0.3422]` | `[-0.6323, -0.2943]` |

- **Same-Engine Python FP32/EMA `RC2` Diagnostic (C4 greedy):** `RC2` scored `25.0920%` on `white10` and `24.3716%` on `musan5`. `ALL-ON` improved by **`−2.2307 pp`** on `white10` (all 3 seeds upper `≤ −1.620 pp`) and **`−1.9676 pp`** on `musan5` (all 3 seeds upper `≤ −1.624 pp`). However, after `−30 dB + Z1`, `ALL-ON` has zero advantage over `CAP-CTRL` (`+0.0166 pp` C4, `+0.0057 pp` C16), every clean `ALL-ON` seed misses signed `RC2` (`6.9469%`), and `D1` failed — closing H61 for `v3.2.0`.
- **Receipts:**
  - Registered 3-seed matched pairs: `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-independent-registered-matched-pairs.json` (`sha256: 9fc317e73bc54fb7c7229f32b1afcbcc1ccaa6d6970125efba159f286d941ac8`)
  - Same-engine RC2 stress diagnostic: `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-python-rc2-stress-diagnostic.json` (`sha256: 42c75c402670cba50bf50b29d4cb6b1882f8d114baa3e1a970f90dd4dce9dd94`)
  - Post-`Z1` sensitivity: `h61-d2-independent-post-z1-c4.json` (`sha256: d937483240b7cdf4b7f914f3cef398954f53936ab24a90feab2176bea1418a02`) and `h61-d2-independent-post-z1-c16.json` (`sha256: 9010beb9a55d60e3f578c69217c9bb0b025177bebcbce2913ddd2b371f644b05`)
  - 3.5k `S`-half harvests: `capctrl/summary.json` (`3a4a7053...`), `allon_s7919_and_ctrlaug_s7920/summary.json` (`cfd661e2...`), `allon_s7920/summary.json` (`38e96a6f...`).

### 3.3 `long8k` — 8,000-Step Seed-7919 Clean/Level Pair (`INCONCLUSIVE → CLOSED`)
- **Setup:** `H61_ALLONLONG_s7919` (`ckpt-00008000.pt`, `sha256: 83f9d830...`) vs `H61_CAPCTRLLONG_s7919` (`sha256: e9b488c9...`), `S`-half, `0/64` batch16-vs-batch1 parity diffs.
- **Paired Results (`ALLONLONG − CAPCTRLLONG`, 1-sided 95% speaker bootstrap):**
  - `c4 0 dB`: `7.1308%` (3,722 edits) vs `7.2228%` (3,770 edits), `Δ = −0.0920 pp [-0.2064, +0.0151]` (still `+0.1839 pp` worse than signed `RC2` `6.9469%`; RNN-T NLL `+0.00105` adverse).
  - `c8 0 dB`: `6.7879%` vs `6.8185%`, `Δ = −0.0307 pp [-0.1192, +0.0525]`.
  - `c16 0 dB`: `6.6748%` vs `6.6308%`, `Δ = +0.0441 pp [-0.0560, +0.1455]` (adverse).
  - `c4 −30 dB`: `7.6079%` vs `7.6481%`, `Δ = −0.0402 pp [-0.1407, +0.0562]`; `c16 −30 dB`: `+0.0153 pp [-0.0681, +0.0978]`.
- **LOGO (Leave-One-Group-Out, `c4`, `drop_G − all_on`):**
  - At `0 dB`: `MOD` `+0.0402 pp`, `FLOOR` `+0.0192 pp`, `FINE` `+0.0134 pp`, `ENV` `−0.0172 pp`, `EXC` `−0.0421 pp` (all CIs span zero).
  - At `−30 dB`: `ENV` **`+0.0920 pp [+0.0483, +0.1369]`**, `FLOOR` **`+0.0613 pp [+0.0056, +0.1168]`**, `FINE` `+0.0613 pp [-0.0078, +0.1295]`, `EXC` `+0.0115 pp`, `MOD` `−0.0690 pp`.
- **Receipts:**
  - Independent replay: `/home/ubuntu/jsr/target/v5-feature-probes/h61-long-independent-replay/summary.json` (`sha256: d6895c610e2229bfc1b7a94e0af4ade8efb7b39c91302c0223d7ba4422ab2d65`)
  - Full harvest & LOGO: `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_harvest/long8k/summary.json` (`sha256: b67dad552f0df672651845cf0f1410f25f7a6b2e78c6a6c4fe89c79ad9fe9293`)
  - Archived nonfinal checkpoints: `/mnt/xbabe3-external/jsr/archive/h61-nonfinal/SHA256SUMS` (`sha256: 4512638f1a43060f4443be91f6cd1388bd753ccc5c84738b3cef2beace0c1b87`).

### 3.4 Owner `H61 Step-2` Noise + Clean-Tail Follow-Up (`A4`/`A3` — `FAIL`, `H61` Additive Side Features Permanently CLOSED as a Release Lever)
- **Authorization & Pre-Registration:** Separate Owner-authorized follow-up (`PREREG_STEP2.md`, `sha256: 2aab2eb301aecc1af04f123fe4e3ba5bf0115ee276e2a3090b4e933e806eb93d`) lifting the `D1` stop-rule for one noise-augmented + clean-tail retrain (`8,000` steps from `REGctl-boundary.pt`: Phase A steps `0–7000` `noise_aug_p=0.5`, `0–20 dB`, `gain_aug_p=0.15`; Phase B steps `7000–8000` clean-only; Addendum A `PREREG_STEP2_ADDENDUM_A.md`, `sha256: ffc5f2ce3b16573eda19e9abbc4cd185c9bf004b8b60bcf688fb4a505fcdcecb`, adds the 3rd seed pair `7921` for `7919/7920/7921`). Explicitly does **not** waive the clean `≤ RC2` gate (`1`-sided 95% upper `≤ 0`) and targets `v4.0.0` only — **out of `v3.2.0` scope**.
- **Source-Disjoint MUSAN Noise Split (`h61-step2-musan-split-audit.json`, `sha256: f34d1043...`):**
  - TRAIN: `musan_noise_train.txt` (`sha256: 2d1e65a2859979b4796571ae638aa312360e27fa32412d8843ef2804e0a592f5`, `1,021` pieces / `324` sources).
  - TEST: `musan_noise_test.txt` (`sha256: 7db5c352feee2d2feb4deb9dc47b29584d2f7b13933323c474993da2b7ef8c88`, `1,190` pieces / `358` sources).
  - Audited `0` piece overlap, `0` source overlap, `0` content overlap.
- **Binding 3-Seed Gate Verdict (`FAIL` — `A3` / `@claude-lead`, `s2_gate.json` `9df36708...`):**
  - **Primary variant (selected before any noise read, `s2_variant.json`):** **EMA** (`7.1698%`, `11,227 / 156,588` edits across seeds `7919/7920/7921`: `7.1136% / 7.1749% / 7.2209%`, vs raw `7.3141%` / `11,453` edits; `@codex` independent `s7919` clean replay `h61-step2-s7919-independent-clean.json` `696e7a24...` matched `3,713 / 52,196` = `7.1136%`).
  - **Leg 1 (Clean `c4`, `ALL-ON − signed RC2` `6.9469%`):** **`+0.223 pp`** (`+0.2229 pp`, 95% CI **`[+0.098, +0.352]`**, one-sided upper **`+0.331 pp > 0` → `FAIL`**; `ALL-ON − Python RC2` `+0.231 pp`; `CAP-CTRL` seeds `7.1691% / 7.0982% / 7.0810%`, `ALL-ON − CAP-CTRL` `+0.054 pp`). **`H61` additive side features are permanently CLOSED as a release lever.**
  - **Leg 2a (`MUSAN-TEST 5 dB` `c4`, `ALL-ON − RC2` `24.7509%`):** **`−8.853 pp`** (`−8.8525 pp`, 95% CI **`[−10.08, −7.67]`**, one-sided upper **`−7.844 < 0` → `PASS`**; `ALL-ON` seeds `15.9265% / 15.9093% / 15.8595%`).
  - **Leg 2b (`MUSAN-TEST 5 dB` `c4`, `ALL-ON − CAP-CTRL`):** **`−0.114 pp`** (`−0.1143 pp`, 95% CI **`[−0.231, +0.010]`**, one-sided upper `−0.010`, marginal; two-sided CI includes `0`; `CAP-CTRL` seeds `16.0587% / 15.9514% / 16.0280%`, `CAP-CTRL − RC2` **`−8.738 pp`**).
  - **Key Scientific Takeaway:** **~98.7% (`−8.74 pp`)** of the held-out real-noise gain comes from real-noise training (`musan_h61_train`), not the 40-dim side features.
- **Receipts:**
  - Binding gate verdict: `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/s2_gate.json` (`sha256: 9df3670891782419df81cadad5b2def32c772369e6a00fea1080d02c5e740f84`)
  - Full 3-seed results: `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/s2_results.json` (`sha256: 2a60ff8c53c81e9c7dbb318aa4fae1d972f644b469da9fd05251575db610c1df`)
  - Score manifest: `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/SHA256SUMS` (`sha256: 661f6180e1677a56f4980bfc5361407b7ad5348afa1177b71b9bcecc23f6cccb`)
  - `@codex` `s7919` clean replay: `/home/ubuntu/jsr/target/v5-feature-probes/h61-step2-s7919-independent-clean.json` (`sha256: 696e7a2435d1e93c777dcd7e2685fc10b97d756c6e49315a97ae449d356344ac`)

### 3.5 Rust `side.rs` H0 Zen 2 CPU Microbench (`COST PROVEN`, Neural Integration Parked)
- **Implementation:** `.agent/worktrees/v5-audio-features` (`crates/ultra-frontend/src/side.rs` + `examples/side_bench.rs`), `#![forbid(unsafe_code)]`, `0 libm`, `0 FMA`, `size_of::<SideExtractor>() = 5,264 B ≤ 8 KiB`, `13/13` unit + `3/3` integration tests PASS (0-ULP taps, packet invariance `160/320/592/73`, 0-bit causality).
- **Measured H0 Zen 2 CPU Cost (vs `v3.1.0` live `30.6 ms/audio-s`):**
  - **Core-40 extraction:** `2,106 ns/frame` = `0.2106 ms/audio-s` = **`0.688%` live** (`EXC-4` `1,489 ns` [71%], `ENV` `403 ns`, `FINE-12` `146 ns`, `MOD-8` `20 ns`, `FLOOR-8` `18 ns`).
  - **All 8 groups (`+ ENV-35 + SHAPE-8 + REF-80`):** `2,318 ns/frame` = `0.2317 ms/audio-s` = **`0.757%` live** (`SHAPE-8` `148 ns`, `REF-80` `7 ns`).
  - **25 Hz `Linear(160→384)` adapter:** Naive 1-frame scalar spills 240 KiB out of L1D (`3.225%` live), whereas chunk-4 batched GEMM (`[4,160]@[160,384]^T`) achieves `28.4 GMAC/s` = `0.0540 ms/s` (`0.177%` live) → **Core-40 + c4-GEMM total = `0.2646 ms/s` (`0.865%` live)** (`Rank-16` `0.800%`; ALL+FiLM `1.242%`).
  - **Independent standalone `FINE-12` H0 CPU-45 microbench** (110,237 real `S` frames, median of 7 reps): `184 ns/frame` = `18.37 µs/audio-s` (`/home/ubuntu/jsr/target/v5-feature-probes/fine-bench/bench.log`, `sha256: f1767c5e529f99d465d23c6bb7247146886797abd4329531d602b0fb7360f573`).

---

## 4. Warm Context & Architecture Tracks (`A2` `h62_warm`, Track M1 / M1-v2, Track M2)

### 4.1 `A2` `h62_warm` — Warm Context Carry & Boundary Guards (`Part 1` & `Part 2` `G1`/`G2`/`G2n` `CLOSED`)
- **Protocol:** `S`-half (2,782 utts, signed `v3.1.0` INT8 `c4_greedy`), raw predecessor tail (`≤ 5.0 s + 0.3 s` zeros), chronological same-chapter (`WARM_NAT`) vs seeded shuffled speakers (`WARM_SHUF`). Thresholds fit strictly on TRAIN (13,085 utts). Success: `WARM_NAT` 1-sided 95% upper `< 0` AND `WARM_SHUF` guard delta `≤ +0.087 pp`.
- **Part-1 Results (Δ vs `ALONE` `6.9469%`):**
  - `WARM_NAT(b)` (always carry): `6.5407%`, **`Δ = −0.4062 pp [-0.6190, -0.2011]`** (`dev-other` `−0.7002 pp`, Words 1–5 `dev-other` `−2.2962 pp`).
  - `WARM_SHUF(b)` (always carry): `9.4892%`, **`Δ = +2.5423 pp [+1.9641, +3.2350]`** (2,696 wrong-speaker carries).
  - Spectral guard `(d)` (`t=12.73`): `NAT −0.3372 pp`, **`SHUF +0.6974 pp`** (756 wrong carries → **FAIL** vs `+0.087 pp`).
  - Hybrid guard `(e)` (`t1=13.74, t2=6.50 dB`): `NAT −0.3583 pp`, **`SHUF +0.6916 pp`** (618 wrong carries → **FAIL**).
  - Exploratory `(d2)` (energy-gated 5 s vs 1 s, `t=9.81`): `NAT −0.2835 pp`, **`SHUF +0.1360 pp`** (214 wrong carries → **FAIL**).
  - Oracle `(o)` (reference ceiling): `NAT −0.4062 pp`, `SHUF −0.0153 pp`.
- **Part-2 Model-Signal Guards (`PREREG2.md` + Addendum A — `CLOSED`):**
  - On TRAIN (1,200 same vs 1,200 diff pairs), `G1` (cold encoder cosine) scored Youden `J = 0.0425` and `G2` (first 1.0 s warm-vs-cold joiner posterior) scored `J = 0.0792` (blank-dominated).
  - On `S`-half (`summary2.json` `sha256: 300122e5...`; independent `@codex` replay `h62-part2-independent-replay.json` `sha256: 2c9710b4...`): **`advancing_rules: []`** (`G1`: `NAT −0.010 pp [upper +0.016] / SHUF +0.023 pp`; `G2`: `NAT −0.172 pp / SHUF +1.096 pp`; `G2c`: `NAT −0.019 pp / SHUF +0.121 pp`; `H`: `NAT −0.011 pp [upper +0.006] / SHUF +0.008 pp`).
  - Addendum-A `G2n` (first 6 emitted non-blank tokens; independent `@codex` replay `h62-g2n-train-independent.json` `sha256: fe456077...`): TRAIN Youden **`J = 0.095 < 0.30`** minimum gate (`advance_to_s: false`).
  - **Verdict:** **CLOSED** — no frozen-model guard clears both `WARM_NAT` and `WARM_SHUF`.
- **Receipts:** `/mnt/xbabe3-external/jsr/claude-lead/h62_warm/outputs/summary.json` (`sha256: 52b2bff42d28b5963175653764713da58191a333b13c0ed55797887117211762`), `outputs/summary2.json` (`sha256: 300122e556afec27c7526f3051f6bacff72a00d76bd2788f08681accff0daada`), `h62-part2-independent-replay.json` (`sha256: 2c9710b4d50e8c08b28f60df95d1bb13a814fe467c4d4cc72d316cff71de1799`), `h62-g2n-train-independent.json` (`sha256: fe4560771eab1dabc164043948e4f6bbd426901da4de7bc90f0c876a3a6c3a9f`), `warm-turn-audit/summary.json` (`sha256: 0c62f910c543d081dda0285f9d45b962e7ab77a449d31ef2dd13b1fd77f73270`).

### 4.2 Track M1 (3.5k Screen `SEALED`), Track M1-v2 Full-Data (`SEALED`), & `M1v3_CLEAN` Zero-Factor-E Diagnostic (`SEALED`)
- **Implementation:** Local commit `f42a394986e624f331122d6b5c9df1446b6eda72` (`feat/v32-m1-turnswitch`), mel-only schema `v1`, `stream_ctx_mode=turn_switch` (`p=0.20`, `gap_s=0.32`) + gentle Factor-E (`0.15/0.15`).
- **3.5k `librilight.small` Screen (seeds `7919`/`7920`, sealed `160/160` `SHA256SUMS` OK; independent replay verified `18/18` inputs):**
  - **Clean C4 greedy:** Both `M1_s7919` (`ckpt sha256: 0740a926...`) and `M1_s7920` (`83dcd779...`) scored **3,755 / 52,196 = 7.1940%** vs signed `RC2` **6.9469%** (**`+0.2471 pp`**, 1-sided 95% upper `+0.3401 / +0.3326 pp`) → missed release screen (`+0.247 pp` matched `H61_CTRLAUG` `7.1825%`).
  - **Turn-switch & warm robustness (Δ vs own `ALONE`):** `P0-a DIFF_SPK_5S` `RC2_EMA_ctrl` `+2.8834 pp` → `M1` **`+2.1937 / +2.2147 pp`**; `WARM_SHUF(b)` `+2.5098 → +1.8986 / +1.9216 pp`; `shuf_d2` `+0.1341 → +0.1207 / +0.1054 pp`; `WARM_NAT(b)` `−0.4157 → −0.4905 / −0.4866 pp`.
- **Track M1-v2 Full-Data (`S4v11` + `incumbent.tsv`, 12k steps, seed `7919` — BOTH Arms `SEALED`):**
  - **`M1v2_FULL_s7919`** (`turn_switch` + Factor-E `0.15/0.15`; `172/172` `SHA256SUMS` OK `166c81a0...`; `v5_m1v2_full_s7919/summary.json` `907f5faf...`; 12k ckpt `c88e92fb...`, 8k ckpt `72a8075b...`):
    - **Cold `c4 0 dB`:** **`7.0714%` EMA** (`3,691` edits, **`+0.1245 pp`** vs `RC2`) / **`7.0159%` raw** (`3,662` edits, **`+0.0690 pp`** vs `RC2`) → **MISSES clean bar** (`c8` `6.7802%`, `c16` `6.6269%`).
    - **Level (`c4 −30 dB`):** `7.5753%` (**`+0.5039 pp`** vs own `0 dB` — cuts `RC2`'s `−30 dB` degradation ~84%).
    - **Wrong-speaker & warm carry:** 12k EMA `DIFF_SPK_5S` **`+0.3966 pp`**, `WARM_SHUF(b)` **`+0.7261 pp`**, `nat_d2 −0.0958 pp` (upper `+0.0174`), `shuf_d2 +0.0192 pp` (upper `+0.0619`); **8k EMA** (`7.1040%` cold `c4`) achieves **`nat_d2 −0.2855 pp`** (upper `−0.1717 pp < 0`) AND **`shuf_d2 +0.0211 pp`** (upper `+0.0642 pp ≤ +0.087 pp`), clearing **both** `H62` warm-carry legs.
  - **Matched Control `M1v2_CTL_s7919`** (identical `S4v11` + Factor-E `0.15/0.15`, **no** `turn_switch`; `171/171` `SHA256SUMS` OK `e64c670d...`; `v5_m1v2_ctl_s7919/summary.json` `e628609a...`; 12k ckpt `b780585a...`, 8k ckpt `e2ead3c9...`):
    - **Cold `c4 0 dB`:** **`7.0733%` EMA** (`3,692` edits, `+0.1264 pp` vs `RC2`) / **`7.1136%` raw** (`3,713` edits, `+0.1667 pp` vs `RC2`); `c8` `6.8090%`, `c16` `6.5867%`, `c4 −30 dB` `7.5906%`.
    - **Wrong-speaker & warm carry (12k EMA):** **`DIFF_SPK_5S +3.9237 pp`**, **`WARM_SHUF(b) +4.7705 pp`**, **`shuf_d2 +0.1590 pp`** (upper `+0.2094 pp`, FAIL; `nat_d2 −0.2625 pp`).
  - **Causal Finding:** Matched `FULL` vs `M1v2_CTL` isolates that `turn_switch` adds **zero clean tax** versus that matched control (`−0.0019 pp` EMA / `−0.0977 pp` raw vs `CTL`) and cuts wrong-speaker collapse by **85–90%** (`+3.924 → +0.397 pp` on `DIFF_SPK_5S` [−90%], `+4.771 → +0.726 pp` on `WARM_SHUF(b)` [−85%], `+0.159 → +0.019 pp` on `shuf_d2`); however, `M1v2_CTL` alone does not separate Factor-E (`0.15/0.15`) from the rest of the shared continuation recipe, and both arms miss signed `RC2` (`6.9469%`).
- **Sealed `M1v3_CLEAN` vs `M1v3_CTL` Zero-Factor-E Diagnostic (`v5_m1v3_clean_pair/`, `272/272` `SHA256SUMS` OK):**
  - **Setup:** Zero Factor-E (`gain_aug_p=0, noise_aug_p=0`), 6k steps 1-GPU (`1,580` audio-h), seed `7919` (`M1v3_CLEAN` ckpt `668eef3dada3...`, `M1v3_CTL` ckpt `623e1b2d6496...`).
  - **Results (`S`-half `c4` greedy):** **`M1v3_CTL = 7.1327%`** EMA (`3,723` edits, **`+0.186 pp`** vs `RC2`; raw `7.1998%` / `3,758` edits); **`M1v3_CLEAN = 7.1155%`** EMA (`3,714` edits, **`−0.017 pp`** [`−0.0172 pp`] EMA / **`−0.081 pp`** [`−0.0805 pp`, `7.1193%` vs `7.1998%`] raw vs `CTL`). Turn-switch cuts wrong-speaker collapse (**`DIFF_SPK_5S` `+2.7971 → +0.8870 pp` [−68%]**, **`WARM_SHUF(b)` `+2.5941 → +0.7204 pp` [−72%]**, `shuf_d2` `+0.1590 → +0.0766 pp`) while retaining **`WARM_NAT(b) −0.362 pp`** (`−0.3621 pp`; `nat_d2 −0.2625 pp`).
  - **Causal Resolution:** Proves the clean offset vs signed `RC2` was caused by cold-restart + short horizon, **NOT** Factor-E.
- **Separate Post-`v3.2.0` Full-Horizon Schema-`v1` Tracks (Out of `v3.2.0` Scope):**
  - **`M1v4` (`M1v4_MIXED_COOL_s7919`):** Full-horizon cooldown (`19,500` steps, world 4, zero aug) — **in flight** to test the horizon fix alone.
  - **`H63` / `M1v5` (`M1v5_NOISE_COOL_s7919`):** Frozen pre-registration (`/mnt/xbabe3-external/jsr/claude-lead/h63_full_horizon_noise/PREREG.md`, `sha256: cfa7acef62e7a05c128ddd554f379ecb279ad2aac44be16f543c732d3855b4b2`) adding real-noise training (`musan_h61_train` + clean tail) to the byte-identical `M1v4` full-horizon recipe.
- **Receipts:**
  - 3.5k screen (`160/160` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1_turnswitch/summary.json` (`sha256: 28fdb9d36d75c8c37f5053b1f233891e7f3c7b94f05981d820683a1852d4fec4`) and `SHA256SUMS` (`sha256: d79cd477842ec866a425fb0979c0c9cadb3b03097a27db7087f2cb33547fbcfe`)
  - 3.5k independent replay: `/home/ubuntu/jsr/target/v5-feature-probes/m1-independent-pair-replay.json` (`sha256: d3c8843db51a9b982fed61f8dd3bb8609bc3a4b7c3bfb1ece030cb8922c79162`)
  - `M1v2_FULL_s7919` (`172/172` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v2_full_s7919/summary.json` (`sha256: 907f5faf5d16cc0863832ef5efb9b1ce5ee057584dcebdc8ab4b2da05bdb5747`) and `SHA256SUMS` (`sha256: 166c81a0f2964a25afc13f51b6813439d2ee449bfa690664364facee32e350cc`)
  - `M1v2_CTL_s7919` (`171/171` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v2_ctl_s7919/summary.json` (`sha256: e628609a9e5abe16e1729be7c4768a3973a0bae2851593b5ad307e28668eadfb`) and `SHA256SUMS` (`sha256: e64c670deadaf4714bacb02ada1be0f08e15d1cb9310f1864d68d66e907944a0`)
  - `M1v3_CLEAN` vs `M1v3_CTL` (`272/272` OK): `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v3_clean_pair/summary.json` (`sha256: 8131bd052124eb1a01aaeb405034e48055448dc5990762de03c58044d656827e`) and `SHA256SUMS` (`sha256: 8019cecc3b0518afce1419ef8fccebaba4df4ac6874115682d1eb85782e7597d`)
  - `H63` / `M1v5` frozen prereg: `/mnt/xbabe3-external/jsr/claude-lead/h63_full_horizon_noise/PREREG.md` (`sha256: cfa7acef62e7a05c128ddd554f379ecb279ad2aac44be16f543c732d3855b4b2`).

### 4.3 Track M2 — Spec-1 Rank-1 `ENV-35` Multiplicative FiLM + RMS-Gated `EXC` (`WITNESSED`, Training `PARKED`)
- **Implementation:** Local commit `de8c0158a525ce96069bdf49bc0a31e01e61e972` (`feat/v5-m2-env35-film`): zero-init `EnvFilmAdapter` (`x * (1 + γ) + β`, `35 → 384`), chunk-tied regime (`c4` causal `ENV[4t]`, `c16/chunk=0` end-value broadcast, `c8` 50/50 train / causal eval), SpecAugment tail exemption, TRAIN-only disjoint half-split `EnvWarmStartBank` (`p=0.30`), and RMS + 5×25 MinStat SNR + spectral-tilt gated `EXC`.
- **Verified Witnesses (`7/7` `test_v5_m2_env35_film.py`, `122/122` combined trainer regression suite):**
  - Gated `EXC` fixes the `P0-b` MUSAN false-voicing bug: `−50 dBFS` 60 Hz hum `100% → 0.0%`; 100 real MUSAN noise cuts `22.96% → 2.46%` (strided) / `32.09% → 1.53%` (first-100); 50 LibriSpeech `train-clean-100` utts retain `87.34%` voiced recall.
  - Bit-exact (`0.0` `max_abs`) causality & truncation; single-thread 10 s speed `4.30 ms` median with precomputed `log_mel` (`10.87 ms` standalone).
- **Disposition:** GPU training held behind Track M1 / `D1` closeout; parked for post-`v3.2.0`.

---

## 5. Remaining Spec 1 & Spec 2 Concepts (Parked / Excluded)

| Concept | Spec Ref | Evidence & Disposition |
|---|---|---|
| `REF-80` relative-spectrum channel | Spec 1 §5.3 | `P0-b` Block-4 CE `+0.08%` (`-0.17%` frics), Stem `-1.99%`; **PARKED** (needs 2-channel stem + INT8 parity). |
| `PCEN` side view / 80-bin challenger | Spec 2 §4.4 | `P0-b` Block-4 CE `-0.09%` (8-d) / `-0.11%` (32-d); **PARKED** (whole-frontend replacement excluded for RC2 warm start). |
| `SHAPE-8` pre-mel FFT centroid/concentration | Spec 2 §4.5 | `P0-b` Block-4 CE `+0.02%`, `LEAD_40+SHAPE` `+0.04%`; `side.rs` cost `148 ns/frame`; **PARKED**. |
| `aux20`/`aux24`/`aux32`/`sentinel20` rank-8 packets | Spec 2 §6.2 | Superseded by H61 Core-40 screen + `D1` fail; **PARKED**. |
| Low-energy compression residual (`ε=1e-8/1e-10`) | Spec 2 §4.6 | No WER receipt (`grok-residual` tested onset CE proxy only); superseded by `Z1` + `Z2`; **PARKED**. |
| `Z3` endpoint features | Spec 1 §7 | No endpointer WER receipt; **PARKED** pending live-endpointer benchmark. |
| `LPC`, formants, 128-mel, 2–16 Hz modulation | Spec 1 §6.1, Spec 2 §5 | No trained V5 receipt; **PARKED**. |
| PNCC, RASTA-PLP, denoising, MFCC/ZCR bags | Spec 1 §6.2, Spec 2 §5.4–5.5 | Explicitly **EXCLUDED** (incompatible with RC2 warm start or redundant with Conformer). |
| Feature-enriched teacher → mel-only student | Spec 2 §8.4 | Contingent on a winning feature teacher (none qualified); **PARKED**. |
| Fail-closed `side_mode` export guard | Spec 1 §4.4 | **IMPLEMENTED** in `r1_to_safetensors.py` + `export.py` + `test_export_side_guard.py` (ships in `v3.2.0`). |

---

## 6. Verified Artifact Digest Index

| Artifact | Path | Verified SHA-256 |
|---|---|---|
| `P0-a` summary | `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/p0a_prefix/summary.json` | `a51dffaa7a61e039c102c55f7c8d67e9bcbff2db663244d6ad83d30449366c2b` |
| `P0-b` summary | `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/p0b_mfa_probe/summary.json` | `f8b0d756db0007213c75f36200945ce703f6387217d66a9b7fa15cc6f67abae1` |
| `P0-c` warp summary | `/home/ubuntu/jsr/target/v5-feature-probes/warp/summary.json` | `a57d26836c6309519fab9d1016a4207b314c4bfb8d1622181c406298fdb46e95` |
| `Z1` attenuation (144) | `/home/ubuntu/jsr/target/v5-feature-probes/z1-rc2/summary.json` | `12310cb8073d56e06b02336c88f96d0d24713e65a8bda1e89237ded410d0d2a3` |
| `Z1` full-`S` guard | `/home/ubuntu/jsr/target/v5-feature-probes/z1-clean-guard/summary.json` | `ea479df52f2592798c09f8978131a757186d2da2979c1ca4264a10dff0c20543` |
| `Z1` `f04027c` 53M x86 oracle | `/home/ubuntu/jsr/target/v5-feature-probes/z1-native-f04027c/receipt.json` | `de451fba6463b34a3639d6fae44dcf62c2594131fc4a20a1f5847550cdbe831e` |
| `Z1` `de0443a` cross-ISA+H47 | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_witness/SHA256SUMS` | `fc1b43737655c7838dbf6a5c336eb69345f7cca1fa834da931d3396da81ce0e5` |
| `Z1` `f04027c` 50/50 cross-ISA | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_witness_f04027c/cross_isa_report.json` | `74c66be0155d1e3a19c5cda3497582235712625a2fa64a26f2805f789e571756` |
| `Z2` sealed summary (`63/63`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/SUMMARY.json` | `7e7c32fc14b5e6daaf047a34e2df365109e5574cb50482470b5ec4c4757b9644` |
| `Z2` manifest (`63/63` OK) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/SHA256SUMS` | `7b5aeb9576b65b52bc98cd97f5e6ddcfe161c2c03244dadcb868e831331a14f1` |
| `Z2` TRAIN voiced RMS cal | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/calibration/train_voiced_rms_calibration.json` | `167274338e14fd940001af46beafcd8ab3e67135504ab646795296ceb1a6b7cb` |
| `Z2` Arm B-cal full-164 C3 | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/c3/criterion3_full164_cal.json` | `a1f16e12b9aaeff855c2e436cd7bce731270139886f9eb079915a11237e01748` |
| `Z2` Arm B-cal `S`-half gain | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/shalf_gain/shalf_gain_cal_summary.json` | `01c546371865e022d0c0259f36727ca542ac70dd976f9c7493422d17eb9045fb` |
| `Z2` Arm B-cal MUSAN-100 | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/musan100/musan100_cal_summary.json` | `cdb8808f7c19426e9c1de79f90cf29a3452e6fa439cc94eaba6a3c013c5129b6` |
| `Z2` full `164/164` C3 FAIL | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/c3/criterion3_full164.json` | `52589f4810e0c826ba3731ad3c73607b3c91e7141494d1a6f7dda9f1e21f1d45` |
| `Z2` full-164 gap audit | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z2_agc/c3/gap_audit_full164.json` | `a1c7f70b80187acdfb6c438ceedc6030352d704ad0458c24398e3d10280489c` |
| `Z2` `@codex` full-164 replay | `/home/ubuntu/jsr/target/v5-feature-probes/z2-c3-full-independent/receipt.json` | `b5b0597527ec5fe278550644e7c471bdf89533893f4eddea4f383e0251a3f9b9` |
| `Z1 × Z2` 4-cell matrix | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_z2_matrix/SUMMARY.json` | `eb6cc78f76925ab8fe86a5effdbab32ec1d6bf84f548d21dd0d259b9fa201894` |
| `Z1 × Z2` manifest (`136/136`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_z1_z2_matrix/SHA256SUMS` | `4444b886bb2f7e8f825dd66985d89ed0e270e7b62c5486017af160508504f2dd` |
| `D1` stressed `P0-b` | `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_probes/d1_stressed_p0b/summary.json` | `a6ec3576774f6d2897dad155745cd19374ecf3eaa0bf74da3d34771afec3d935` |
| `D2` 3-seed matched pairs | `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-independent-registered-matched-pairs.json` | `9fc317e73bc54fb7c7229f32b1afcbcc1ccaa6d6970125efba159f286d941ac8` |
| `D2` same-engine RC2 | `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-python-rc2-stress-diagnostic.json` | `42c75c402670cba50bf50b29d4cb6b1882f8d114baa3e1a970f90dd4dce9dd94` |
| `D2` post-`Z1` C4 | `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-independent-post-z1-c4.json` | `d937483240b7cdf4b7f914f3cef398954f53936ab24a90feab2176bea1418a02` |
| `D2` post-`Z1` C16 | `/home/ubuntu/jsr/target/v5-feature-probes/h61-d2-independent-post-z1-c16.json` | `9010beb9a55d60e3f578c69217c9bb0b025177bebcbce2913ddd2b371f644b05` |
| `long8k` harvest | `/mnt/xbabe3-external/jsr/runtime-lane/v5_h61_harvest/long8k/summary.json` | `b67dad552f0df672651845cf0f1410f25f7a6b2e78c6a6c4fe89c79ad9fe9293` |
| `long8k` replay | `/home/ubuntu/jsr/target/v5-feature-probes/h61-long-independent-replay/summary.json` | `d6895c610e2229bfc1b7a94e0af4ade8efb7b39c91302c0223d7ba4422ab2d65` |
| `H61 Step-2` prereg | `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/PREREG_STEP2.md` | `2aab2eb301aecc1af04f123fe4e3ba5bf0115ee276e2a3090b4e933e806eb93d` |
| `H61 Step-2` Addendum A | `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/PREREG_STEP2_ADDENDUM_A.md` | `ffc5f2ce3b16573eda19e9abbc4cd185c9bf004b8b60bcf688fb4a505fcdcecb` |
| `H61 Step-2` MUSAN split | `/home/ubuntu/jsr/target/v5-feature-probes/h61-step2-musan-split-audit.json` | `f34d10437198c79ccd7796b572bc817b9b94aa1fa1b5d8b2dd3eead833524865` |
| `H61 Step-2` `s7919` clean | `/home/ubuntu/jsr/target/v5-feature-probes/h61-step2-s7919-independent-clean.json` | `696e7a2435d1e93c777dcd7e2685fc10b97d756c6e49315a97ae449d356344ac` |
| `H61 Step-2` gate (`FAIL`) | `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/s2_gate.json` | `9df3670891782419df81cadad5b2def32c772369e6a00fea1080d02c5e740f84` |
| `H61 Step-2` full results | `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/s2_results.json` | `2a60ff8c53c81e9c7dbb318aa4fae1d972f644b469da9fd05251575db610c1df` |
| `H61 Step-2` score manifest | `/mnt/xbabe3-external/jsr/claude-lead/h61_step2/score/SHA256SUMS` | `661f6180e1677a56f4980bfc5361407b7ad5348afa1177b71b9bcecc23f6cccb` |
| Track M1 3.5k (`160/160`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1_turnswitch/summary.json` | `28fdb9d36d75c8c37f5053b1f233891e7f3c7b94f05981d820683a1852d4fec4` |
| Track M1 independent replay | `/home/ubuntu/jsr/target/v5-feature-probes/m1-independent-pair-replay.json` | `d3c8843db51a9b982fed61f8dd3bb8609bc3a4b7c3bfb1ece030cb8922c79162` |
| `M1v2_FULL_s7919` (`172/172`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v2_full_s7919/summary.json` | `907f5faf5d16cc0863832ef5efb9b1ce5ee057584dcebdc8ab4b2da05bdb5747` |
| `M1v2_CTL_s7919` (`171/171`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v2_ctl_s7919/summary.json` | `e628609a9e5abe16e1729be7c4768a3973a0bae2851593b5ad307e28668eadfb` |
| `M1v3_CLEAN/CTL` (`272/272`) | `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v3_clean_pair/summary.json` | `8131bd052124eb1a01aaeb405034e48055448dc5990762de03c58044d656827e` |
| `M1v3_CLEAN/CTL` manifest | `/mnt/xbabe3-external/jsr/runtime-lane/v5_m1v3_clean_pair/SHA256SUMS` | `8019cecc3b0518afce1419ef8fccebaba4df4ac6874115682d1eb85782e7597d` |
| `H63` / `M1v5` frozen prereg | `/mnt/xbabe3-external/jsr/claude-lead/h63_full_horizon_noise/PREREG.md` | `cfa7acef62e7a05c128ddd554f379ecb279ad2aac44be16f543c732d3855b4b2` |
| `A2` `h62_warm` Part-1 seal | `/mnt/xbabe3-external/jsr/claude-lead/h62_warm/outputs/summary.json` | `52b2bff42d28b5963175653764713da58191a333b13c0ed55797887117211762` |
| `A2` `h62_warm` Part-2 seal | `/mnt/xbabe3-external/jsr/claude-lead/h62_warm/outputs/summary2.json` | `300122e556afec27c7526f3051f6bacff72a00d76bd2788f08681accff0daada` |
| `A2` Part-2 `@codex` replay | `/home/ubuntu/jsr/target/v5-feature-probes/h62-part2-independent-replay.json` | `2c9710b4d50e8c08b28f60df95d1bb13a814fe467c4d4cc72d316cff71de1799` |
| `A2` `G2n` TRAIN replay | `/home/ubuntu/jsr/target/v5-feature-probes/h62-g2n-train-independent.json` | `fe4560771eab1dabc164043948e4f6bbd426901da4de7bc90f0c876a3a6c3a9f` |
| `A2` turn-specific audit | `/home/ubuntu/jsr/target/v5-feature-probes/warm-turn-audit/summary.json` | `0c62f910c543d081dda0285f9d45b962e7ab77a449d31ef2dd13b1fd77f73270` |
| LM prefix router | `/home/ubuntu/jsr/target/v5-feature-probes/router/summary.json` | `9bd1493656d1fe80c502c4bfbf012b141ddb55f9314f86f392de4745ac679662` |
| `FINE-12` H0 CPU bench | `/home/ubuntu/jsr/target/v5-feature-probes/fine-bench/bench.log` | `f1767c5e529f99d465d23c6bb7247146886797abd4329531d602b0fb7360f573` |
| Archived H61 nonfinal | `/mnt/xbabe3-external/jsr/archive/h61-nonfinal/SHA256SUMS` | `4512638f1a43060f4443be91f6cd1388bd753ccc5c84738b3cef2beace0c1b87` |
