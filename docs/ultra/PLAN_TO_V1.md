# PLAN_TO_V1: from here to v1.0.0 (2026-11-21 ~17:30Z)

**Controlling plan for all lanes.** It supersedes the schedule in BENCH_PLAN_v1 §4; the BENCH_PLAN protocol still applies. Owner definition (O-rulings): **v1.0.0 = R0 release + R1 measured beside it.** Owner instructions:
- the first "shocking" claim is **stream density**;
- the fleet never exceeds 90% mean CPU/GPU;
- coordinate through `AGENT_CHAT.md`, with frequent git commits and jankurai audits.

---

## 1. Where we are

### 1.1 Built and on `main` (2888f20)

| Area | State |
|---|---|
| Contracts / eval | §33 event contract, receipts, WER/CQ/FC, paired cluster bootstrap. `ultra-measure` pins a disjoint observer and fails closed on unknown topology. |
| R0 runtime | sherpa Zipformer int8, streaming, Rust |
| R1 ORT runtime | Experimental; FP32 parity 400/400; static-INT8 v1.1 bundle (batch-invariant, +0.066 WER) |
| **ultra-engine** (pure Rust) | FP32 and per-row centred INT8 (batch-invariant, +0.000 WER); prepared INT8 (259 MiB RSS); dense S, S_MR, ternary R4; AVX2 + NEON (sdot/i8mm), bit-identical across ISAs; output-continuity tests; optional GEMM split |
| Bench tooling | `ultra bench-suite` (C4 slices, silence probe, MFA latency joined after inference, job manifests, cost_scope); bridge whole-service M1/M2 with ORT and fused-engine adapters |
| Trainer | dense, S_MR, S_MR_T (ternary QAT), S_MR_I8S (int8-QAT stem, speed perturbation); divergence guard; exports v1.1 and r4-ternary |
| Data | LibriSpeech 960 h + dev/test; MFA word timings; sealed OOD VoxPopuli EN + AMI IHM; frozen slices; silence probe |

### 1.2 Best measured results so far (all exploratory, DEV, shared host)

| System | dev-clean WER | Whole-service CPU-s/s | Density (A1 rule) |
|---|---:|---:|---|
| R0 Zipformer int8 | 2.85% | 0.078 | 192 machine-wide (Python harness) |
| R1 D2 snapshot @5k, engine INT8 | 21.80% | **0.041** | — |
| R1 D1MR (S_MR, 100 h), engine INT8 | 23.7% (engine 400) | — | **136/CCX Zen 2; 84/core M3** |
| ORT R1 (same harness) | — | 0.175 (FP32) | < 32/CCX |
| Nemotron 560 ms | 2.16% | 0.472 | — |

### 1.3 Training (live; dev-clean400 unless noted)

| Run | Host | Latest | Planned end | ETA (at current rate) |
|---|---|---|---|---|
| D2 dense S, 960 h | xbabe3 (H0) | **9.99%** @40k (dev-other 21.9%) | 102k | **~2026-11-23 01:00Z** |
| D2MR S_MR, 960 h | xbabe2 | 12.3% @40k | 50k | **~2026-11-21 19:30Z** (tonight) |
| D3MRb S_MR + int8 stem + speed perturbation | xbabe1 | 15.0% @20k (full dev curve 15.3/29.6 @10k) | 140k | **~2026-11-22 17:30Z** |
| R4T ternary (100 h), done | — | 17.98% engine | — | — |

**Gap to R0:** 2.85% against our best 9.99%. Accuracy, not speed, is the main open question.

## 2. What is left (gates for v1.0.0)

| # | Gate | Status | Owner |
|---|---|---|---|
| G1 | Final R1 checkpoints selected on DEV (dense and S_MR) | waiting on training | trainer |
| G2 | Engine prepared-INT8 artifacts of the finalists, quality gate (≤ +0.10 vs FP32), continuity | per finalist | engine |
| G3 | Same-harness ORT density row (to support "×N density") | open | runtime, or reassigned |
| G4 | Platform-deterministic log-mel (cross-ISA PCM → tokens) | open | runtime, or reassigned |
| G5 | FREEZE.md: identities, scorer/normaliser hashes, settings, protocol, speaker-stratified DEV panel | open | **lead** |
| G6 | Qualifying window on H0: quiet host, observer pinned, ≥5 ABAB blocks for finalists, per-CCX and full-machine density, 1 h sustained, silence probe, CQ/FC latency | open | lead + engine + codex |
| G7 | One PUBLIC-TEST read (test-clean/other) + sealed OOD read (VoxPopuli, AMI) after the `bench-freeze-v1` tag | open | lead |
| G8 | Independent review of the engine and of the freeze | partial | codex, or a fresh reviewer |
| G9 | Release: bundles, model cards, SBOM, FRONTIER_v1.0.0.md + Artifact page, CHANGELOG, tag v1.0.0 | open | lead |

## 3. Execution plan

### Phase A: now → tonight (D2MR finishes ~19:30Z)
- **A1 (lead):** build the speaker-stratified DEV selection panel (all 40 dev-clean and 33 dev-other speakers) and commit its manifest.
- **A2 (lead):** draft FREEZE.md, the candidate list and protocol (§4), and update CANDIDATES.lock with the engine candidates.
- **A3 (trainer):** score D2MR's final checkpoint on full dev-clean and dev-other. Copy it to H0 with its hash.
- **A4 (engine):** prepared-INT8 for D2MR-final: quality gate, continuity, single-stream and per-CCX density on Zen 2 and M3.
- **A5 (lead):** a dress rehearsal of the full qualifying script on **xbabe2**, once D2MR frees it. Same 3995WX SKU, exploratory label. It finds script and protocol bugs before the real window.

### Phase B: tomorrow (11-22), training continues
- **B1 (engine):** the same as A4 on the D2 snapshot at 60–80k and on D3MRb when it finishes (~17:30Z). It also does any remaining engine speed work, all of it continuity-preserving.
- **B2 (trainer):** DEV curves. Select the S_MR finalist (D2MR-final vs D3MRb-final) on the stratified DEV panel.
- **B3 (lead):** the G3 ORT density row in the same harness. It's a run, not a code edit, so the lead can run runtime-lane's merged `bench-suite`. G4 (log-mel) is reassigned to engine-lane if runtime-lane stays silent, with notice in AGENT_CHAT.
- **B4 (reviewer):** an independent review of the freeze draft and the engine. Codex if active; otherwise a fresh-context Claude reviewer with no shared conclusions.

### Phase C: 11-23 (D2 finishes ~01:00Z)
- **C1 (trainer):** D2-final DEV scores → select the dense finalist.
- **C2 (engine):** prepared INT8 + gate for D2-final.
- **C3 (lead):** the freeze commit, then tag `bench-freeze-v1`.
- **C4 (all):** the **qualifying window on H0**. All three GPU jobs will be done; announce it ≥1 h ahead, take the lease, and pause other lanes' CPU work.
  - **Quality:** PUBLIC-TEST and OOD read once, for R0, Nemotron, R1-dense and R1-MR.
  - **Timing:** ≥5 ABAB blocks.
  - **Density:** per-CCX and full machine.
  - **Sustained:** 1 h run.
  - **Latency:** CQ/FC.
  - **Silence:** the silence probe.
- **C5 (lead):** FRONTIER_v1.0.0.md + a published Artifact page with the frontier chart, model cards, SBOM, CHANGELOG, tag `v1.0.0`.

## 4. Freeze candidate list (draft)

| Id | System | Path | Precision | Label |
|---|---|---|---|---|
| R0 | Zipformer en 2023-06-26 | sherpa | int8 | in-corpus |
| N560 / N160 | Nemotron 0.6B Q8_0 | NeMo-Speech.cpp | q8 | unknown overlap |
| R1-dense | D2 final (selected on DEV) | ultra-engine | prepared INT8 (+ FP32 control) | in-corpus |
| R1-MR | best of D2MR-final / D3MRb-final | ultra-engine | prepared INT8 (+ FP32 control) | in-corpus |
| R1-T (research row) | R4T step 12k (100 h) | ultra-engine | ternary | in-corpus, not a finalist |
| context | Moonshine v2 base, Whisper small.en | sherpa offline | int8 | unknown overlap |

**Pre-registered claims (unchanged in spirit, numbers set at the freeze):**
- density of R1-MR (engine) vs R0 and vs Nemotron on H0;
- R1 cost (CPU-s/s) vs R0 at its measured WER;
- the WER gap stated plainly;
- the OOD degradation row.

## 5. Risks
- **Accuracy:** R1 may finish around 5–8% against R0's 2.85%. v1.0.0 still ships, with the gap stated. Data expansion is an owner decision.
- **Host noise:** A/A replay varied +25%. The qualifying window needs a quiet host and blocks.
- **Agents:** codex and runtime-lane (external) have been silent since about 04:00Z. Their items get fallbacks (B3, B4).
- **Divergence:** auto-stop is in place (69a9bf2).

## 6. Owner decisions still open
1. MFA 20-utterance listening check.
2. TED-LIUM 3 rights. The default is to run the 2 OOD sets.
3. Data expansion beyond LibriSpeech (accuracy).
4. Restart the codex and Antigravity sessions, if you want them back for G3, G4 and G8.
