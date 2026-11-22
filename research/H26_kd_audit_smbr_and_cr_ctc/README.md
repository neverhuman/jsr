# H26: H22 KD Forensic Audit, sMBR N-Best Sequence Distillation & CR-CTC Auxiliary Loss

Status: **PRE-REGISTERED** (JSR v2 Phase 1 — zero-inference-cost accuracy jump).

## 1. Question
Can we close the remaining streaming WER gap toward R0 Zipformer (`2.85%` dev-clean) with **zero extra inference compute** by:
1. Hardening H22 frame-level distillation (`kd.py` + `teacher_cache.py`) against subword-path and physical-delay mismatches,
2. Adding **sMBR top-8 N-best sequence-level distillation** (completely immune to tokenizer/clock differences), and
3. Adding **CR-CTC (Consistency-Regularized CTC)** auxiliary supervision (published: Zipformer-L `2.00/4.38 → 1.88/3.95`, plus sharper CTC posteriors for H27 speculative finalization)?

## 2. Forensic Audit of Current `python/ultra/train/kd.py`
- **Confirmed already correct on `main`:** `kd.py:82–84` already computes the exact collapsed-support complement KL term `q_other * (log(q_other) - p_other)` (no top-k renormalization distortion!).
- **Remaining H26 hardening items:**
  1. **Exact non-blank emitted path verification:** `kd.py:46` checks `normalise(hyp) == normalise(text)`, which can pass even when `sum(emitted != 0) != len(rnnt_tgt)` (different subword segmentation -> mismatched lattice row `u_t`). Verify exact subword sequence identity (`r0_to_student(emitted[emitted != 0]) == rnnt_tgt`) before marking `ok = True`.
  2. **Physical time-support metadata:** Record `frontend_delay_samples` and `chunk_right_context_samples` in `teacher_cache.py` metadata to audit R0 vs R1 frame-lag alignment (`Δt ∈ {-1, 0, +1}` shift ablation on 100 h).
  3. **Hard-example coverage audit:** Log fraction of accepted utterances binned by SNR/speaking-rate/length so teacher-perfect filtering never silently drops hard speech; keep supervised human RNN-T + CTC anchor on 100% of batches.
  4. **CR-CTC two-view consistency:** Run matched 100 h control on `xbabe1`/`xbabe2` (equal GPU compute: two augmented views at half batch/epochs) before enabling on 960 h `H22-kd` / `V2` runs.
  5. **sMBR top-8 sequence KD:** Cache offline CR-CTC Zipformer-L (or R0) top-8 N-best hypotheses + scores on TRAIN splits only to inject stronger sequence-level priors without frame-alignment errors.

## 3. Decision / Promotion Rule
- Promote each component individually (KD alignment fix, CR-CTC, sMBR sequence KD) only if it delivers **`≥ 5%` relative `dev-other` WER reduction** (or `≥ 0.3 pp` absolute) at **matched GPU hours** on a 100 h control before scaling to 4–6× A100 960 h runs.

## 4. Pointers
- Sources: `tips/v2/JSR_CPU_ASR_BREAKTHROUGH_RESEARCH_SPEC.md` §5, `tips/v2/ULTRA-SU_single_user_cpu_asr_spec.md` §5.1, `tips/v2/jsr_ultra_v4_0_radical_cpu_asr_architecture_hpc_specification.md` §5.1, `research/H22_r0_distillation/README.md`.
