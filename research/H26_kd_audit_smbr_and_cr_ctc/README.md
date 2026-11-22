# H26: H22 KD Forensic Audit, sMBR N-Best Sequence Distillation & CR-CTC Auxiliary Loss

Status: **PRE-REGISTERED** (JSR v2 Phase 1 — zero-inference-cost accuracy jump).

## 1. Question
Can we close the remaining streaming WER gap toward R0 Zipformer (`2.85%` dev-clean) with **zero extra inference compute** by:
1. Hardening H22 frame-level distillation (`kd.py` + `teacher_cache.py`) against subword-path and physical-delay mismatches,
2. Adding **sMBR top-8 N-best sequence-level distillation** (completely immune to tokenizer/clock differences), and
3. Adding **CR-CTC (Consistency-Regularized CTC)** auxiliary supervision (published: Zipformer-L `2.00/4.38 → 1.88/3.95`, plus sharper CTC posteriors for H27 speculative finalization)?

## 2. Forensic Audit of Current `python/ultra/train/kd.py`
- **Confirmed already correct on `main`:** `kd.py:82–84` already computes the exact collapsed-support complement KL term `q_other * (log(q_other) - p_other)` (no top-k renormalization distortion!).
- **Implemented in PR #24 (ready for matched 100 h / `H22-kd` ablations):**
  1. **Exact non-blank emitted path verification (`python/ultra/train/kd.py`):** Added `verify_emitted_path(emitted, rnnt_tgt)` to catch subword-segmentation mismatches where text normalization matches (`normalise(hyp) == normalise(text)`) even though non-blank emitted IDs differ from `rnnt_tgt`.
  2. **CR-CTC two-view consistency (`python/ultra/train/asr.py` & `python/ultra/train/trainer.py`):** Implemented masked symmetric stop-gradient KL consistency (`cr_ctc_weight`, default `0.0` = bit-exact no-op; active-rank weighted in DDP; logged as `cr_ctc` in interval train JSONL), covered by `python/ultra/tests/test_train_cr_ctc.py`.
- **Remaining H26 hardening items:**
  1. **Physical time-support metadata:** Record `frontend_delay_samples` and `chunk_right_context_samples` in `teacher_cache.py` metadata to audit R0 vs R1 frame-lag alignment (`Δt ∈ {-1, 0, +1}` shift ablation on 100 h).
  2. **Hard-example coverage audit:** Log fraction of accepted utterances binned by SNR/speaking-rate/length so teacher-perfect filtering never silently drops hard speech; keep supervised human RNN-T + CTC anchor on 100% of batches.
  3. **CR-CTC 100 h control:** Run matched 100 h control on `xbabe2` (equal GPU compute: two augmented views at half batch/epochs) before enabling on 960 h `H22-kd` / `V2` runs.
  4. **sMBR top-8 sequence KD:** Cache offline CR-CTC Zipformer-L (or R0) top-8 N-best hypotheses + scores on TRAIN splits only to inject stronger sequence-level priors without frame-alignment errors.

## 3. Decision / Promotion Rule
- Promote each component individually (KD alignment fix, CR-CTC, sMBR sequence KD) only if it delivers **`≥ 5%` relative `dev-other` WER reduction** (or `≥ 0.3 pp` absolute) at **matched GPU hours** on a 100 h control before scaling to 4–6× A100 960 h runs.

## 4. Pointers
- Sources: `tips/v2/JSR_CPU_ASR_BREAKTHROUGH_RESEARCH_SPEC.md` §5, `tips/v2/ULTRA-SU_single_user_cpu_asr_spec.md` §5.1, `tips/v2/jsr_ultra_v4_0_radical_cpu_asr_architecture_hpc_specification.md` §5.1, `research/H22_r0_distillation/README.md`.

## CR-CTC 100 h arms run by trainer-lane on 2026-11-22 (written after the results)

**Process note:** these arms were launched at 17:40Z without a gate section in this README, and this section was written after they ended. They do not measure this README's gate (dev-other at matched GPU hours): they are matched on steps and scored on dev-clean400 greedy. Treat them as exploratory.

- **Protocol:** H23a's exactly (preset S, SentencePiece-512, pruned RNN-T, AdamW lr 1e-3, 2,000 warmup steps, seed 1234, budget 300 s / 64, 24,000 steps on 100 h, SpecAugment from 4,000, EMA) plus `cr_ctc_weight`. H23a is the control.
- **Loss variant:** PR #24's. The second view receives only the consistency term, not a supervised CTC loss. PR #32 changes that.
- **Arms:** weight 0.1 on xbabe1 (RTX 4090), completed. Weight 0.2 on xbabe6 (A10), stopped at step 17,582 when the fleet plan changed.

**dev-clean400, greedy, raw / EMA:**

| step | H23a (control) | CR-CTC 0.1 | CR-CTC 0.2 |
|---:|---|---|---|
| 10,000 | 24.538 / 21.412 | 23.496 / 21.164 | 23.639 / 21.529 |
| 16,000 | 18.312 / 16.827 | 17.400 / 16.424 | 17.947 / 16.697 |
| 20,000 | 17.023 / 16.228 | 15.877 / 15.486 | not reached |
| 23,034 | 16.450 / 16.319 | 15.395 / 15.277 | not reached |
| 24,000 | 16.202 / 16.371 | 15.642 / 15.290 | not reached |

- **Reading:** weight 0.1 ends 0.56 pp raw and 1.08 pp EMA better than H23a at step 24,000. That is not distinguishable from run-to-run spread:
  - H23e, which has no CR-CTC, ends at 15.251 / 15.343, also about 1 pp better than H23a;
  - two identical 100 h runs differ by about 0.57 pt on this panel.
- **Cost:** CR-CTC runs a second encoder pass. On the A10 it trained at 417 audio-h/h against H23a's 984 (guard targets 0.83 and 0.88).
- **Verdict:** not adopted on this evidence. A further test belongs in the H41 replay protocol, with both views supervised.

