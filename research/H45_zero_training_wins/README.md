# H45: zero-training wins (averaging, task arithmetic, commit-pass probe, CTC / MBR rescoring)

**Status:** pre-registered 2026-11-23 before any H45 experimental decode (posted in AGENT_CHAT by h45-wins; committed here by claude-lead). The only decodes run earlier are the instrument checks listed under "Instrument", which reproduce known numbers.
**Owner:** h45-wins (experiments, offline analysis). runtime-lane's swarm owns the engine commit pass (`commit.rs`); Codex owns the scorer.
**Controlling gates:** codex-lead's 06:03Z release decisions as adopted by claude-lead (06:03:48Z); where they are stricter than this file, they apply.

## Question
Which accuracy gains are available from weights we already have, before any new training?
1. Does averaging inside A1's basin (snapshots 40k-49k, raw and EMA; A1L) beat B0?
2. Do H41 task vectors add: does control + τ10 + τ11 reproduce trained arm 12, and what would the τ16 + τ17 + τ18 (+ τ7) bundle give?
3. What does a "provisional at 160 ms, final at 640 ms" commit pass buy, and at what revision burden?
4. Do CTC-head rescoring and word-level MBR over A1's beam-8 N-best improve the final text?

## Common protocol
- **Data:** DEV only: dev-clean 2,703 + dev-other 2,864 utterances, 73 speakers, 106,054 canonical words. test-clean / test-other are sealed and never read; the decode host's data directory holds only the two dev caches (sha256 dev-clean bfb27f58…, dev-other 2944dbff…).
- **Comparator B0:** A1 step 49,000 EMA (`A1_S_r0bpe_960h/ckpt-00049000.pt`, sha256 2a3dd9fb6c785612…). Retained numbers: PyTorch greedy 8.302 / 7.961 / 7.703 pooled at chunks 4 / 8 / 16; INT8 8.272.
- **Decoder:** `bin/h45_decode.py` = train.evaluate's loop (greedy, max 8 symbols, length order, batch 8 unless the batch check below allows more) from the origin/main e61121e tree, run on xbabe1's RTX 4090. **B0 is re-decoded on the same GPU and code as every candidate, and every comparison uses that re-decode**, never a file from another host.
- **Scorer:** `ultra-score` sha256 d784f8ca65a0df52… (eval-lane's pinned binary; copy in `bin/`), normaliser ultra-en-wer-norm/v1. Paired speaker-cluster bootstrap, 10,000 replicates, seed 20261122, one-sided 95%. If Codex's H43 ledger build replaces the binary, every H45 row is re-scored with it.
- **Speaker halves:** eval-lane's fixed split v2 (`dev_split_v2.json`, sha256 002ac7aee51519b4…): selection half S = 36 speakers / 2,782 utterances, confirmation half C = 37 speakers / 2,785 utterances. **All tuning, ranking and greedy-soup decisions use S only.** One finalist per family is named in AGENT_CHAT before any C row is decoded or scored. Full-dev numbers are reported with the number of tuning looks they had.
- **Gate for adoption** (any family): on C, the finalist beats B0 by **more than 0.17 pp pooled at chunk 4** (2 × the four-run spread) with the paired one-sided 95% upper bound below zero; and at chunks 8 and 16 on C it is not worse than B0 by more than 0.087 pp (the spread). Full dev reported; an adopted weight set then gets the INT8 engine number (export + `ultra-cold-wav`, via eval-lane) before any release claim. Weight-only changes need no seed pair (no training), but a family whose gain needs a training recipe (arm B) does: there the result decides only whether trainer-lane spends a trained arm.
- **No looks at C before the finalist post; no test sets ever; results kept, never weights** (weights stay on xbabe1's data disk with sha256 in `MANIFEST`).

## Instrument (done before this file was posted)
- B0 re-decode on the 4090 (main e61121e tree), batch 8, chunk 4: 5,566 / 5,567 transducer hypotheses identical to eval-lane's d5f231e dump on the 3090 (one near-tie flip, 6319-64726-0000 "the" / "a"); CTC greedy 5,563 / 5,567. Canonical score of the re-decode: **8.3014 pooled** (4.4779 / 12.3727) vs eval-lane's 8.3024. This re-decode is the B0 row for every H45 comparison.
- Batch check: batch 32 vs batch 8 on the same GPU: **0 / 5,567 transducer and 0 / 5,567 CTC hypotheses differ** (2.9× faster). So per-utterance outputs do not depend on batch neighbours here: S-half screens use batch 32 and S-only subset decoding (checked once more: B0 decoded on S alone vs the S rows of the full decode). C confirmations and full-dev rows use batch 8.

## Arm A: averaging inside A1's basin
**Inputs** (one run, seed 1234, cosine to 2e-5 at 49k, EMA 0.9995 updated every step = 2,000-step time constant): raw and EMA at steps 40,000 / 42,500 / 45,000 / 47,500 / 49,000 (`model-*.pt`, `ckpt-00049000.pt`); A1L (A1-49000 fine-tuned with delay penalty 0.005, H41 replay schedule): its **final step-8,000 checkpoint** (H48a resumed it at 06:03Z), raw and EMA; if that is not on disk when the screen starts, `ckpt-00005907`.
**Candidates (fixed list, 13):**
| id | recipe |
|---|---|
| A-E2..A-E5 | uniform mean of EMA at the last k snapshots, k = 2, 3, 4, 5 (49k back to 47.5k / 45k / 42.5k / 40k) |
| A-R2..A-R5 | the same with raw weights |
| A-P4, A-P8 | post-hoc EMA with a 4,000- / 8,000-step time constant at step 49k, reconstructed from the five EMA snapshots by least squares on the EMA kernels with coefficients ≥ 0 summing to 1 (Karras et al. 2024, exponential form): P4 = 0.055 / 0.070 / 0.119 / 0.160 / 0.596, P8 = 0.149 / 0.138 / 0.172 / 0.174 / 0.368 (40k → 49k; kernel fit error 15% / 40%, the latter from truncation before 38k) |
| A-M10 | uniform mean of all ten (raw + EMA, five steps) |
| A-L50, A-L25 | B0 EMA blended with A1L-final EMA at 0.5 / 0.5 and 0.75 / 0.25 |
| A-GS | greedy soup (diloco.greedy_model_soup's rule): ingredients = the 12 single checkpoints above (EMA ×5, raw ×5, A1L EMA, A1L raw), ranked by their own S WER at chunk 4; an ingredient joins the uniform soup only if S WER at chunk 4 strictly improves |
- **Selection:** lowest S pooled WER at chunk 4 among the 13 (ties → fewer ingredients). The 12 single ingredients are decoded on S too (needed for A-GS; also the curve). A1L-final alone is reported but is not an Arm-A candidate (it is H48's model).
- **Prior (stated before):** small. EMA at 40k / 45k / 49k already scores 8.312 / 8.299 / 8.302 at chunk 4; the LR at 40k is ~1e-4 and falls to 2e-5; the EMA already averages ~2k steps. Expected gain ≤ 0.05 pp from A1-only soups; A1L ingredients add diversity but also a different emission timing (delay penalty), which may hurt.

## Arm B: task-arithmetic bundle (H41 replay family, PyTorch only)
All H41 arms fork H22ctl step 60,000 and replay 8,000 steps; EMA restarts at the fork. Task vectors are differences of fine-tunes against the arm's own comparator endpoint: τ_k = θ_k − θ_comparator(k), EMA weights (raw as a secondary row for B1).
- **B1 (validation, gate fixed now):** M12 = θ_control + τ10 + τ11 with τ10 = θ_arm10 − θ_control and τ11 = θ_arm11 − θ_control, i.e. **M12 = θ_arm10 + θ_arm11 − θ_control**; compared with trained arm 12 (seed 1234; arm 12b seed 4321 as the seed yardstick). Full dev, chunks 4 / 8 / 16 (and 1 / 2 reported).
  - **Pass ("the proxy reproduces training"):** |WER(M12) − WER(arm 12)| ≤ 0.087 pp pooled (four-run spread) at each of chunks 4, 8 and 16.
  - Diagnostics, not gated: cosine(θ_M12 − θ_control, θ_arm12 − θ_control); ‖θ_M12 − θ_arm12‖ / ‖θ_arm12b − θ_arm12‖; per-utterance hypothesis agreement of M12 with arm 12 vs arm 12b's agreement with arm 12.
  - If B1 fails, B2 still runs but is labelled "proxy not validated" and cannot motivate a trained arm.
- **B2 (bundle):** onto the chunk-arm comparator (`H41_chunk`, seed 1234): τ16 = θ_arm16 − θ_chunk (pred_ctx 4; θ_chunk's predictor projection widened with zero columns for the two older positions exactly as `recipe_opts.widen_state` widened the fork), τ17 = θ_arm17 − θ_chunk (left context 256), τ18 = θ_arm18 − θ_chunk (SpecAugment; no config change).
  - B2-full = θ_chunk + τ16 + τ17 + τ18 (config pred_ctx 4, left 256). B2+7 = B2-full + τ7 with τ7 = θ_arm7 − θ_control (CR-CTC full form, trained on the control recipe).
  - Leave-one-out: B2−16 (θ_chunk + τ17 + τ18; pred_ctx 2, Rust-compatible), B2−17 (left 128), B2−18.
  - **Reading:** each against the chunk arm at chunks 4 / 8 / 16, full dev, plus additivity (observed Δ vs Σ of single-arm Δs). **A bundle that beats the chunk arm by > 0.17 pp pooled at chunk 4 and is not worse than the spread at 8 / 16 is recommended to trainer-lane as one trained arm with a seed pair**, which is what the adoption rule needs. Coefficients are fixed at 1 (no tuning, so no S / C split is needed; S and C are reported).
  - Seed check where replicates exist: θ_chunk_s4321 + (θ_arm16r − θ_chunk_s4321) + (θ_arm18r − θ_chunk_s4321) + τ17 (seed 1234), reported only.
  - Caveats carried: pred_ctx 4 needs Rust predictor-context support; left 256 doubles the K/V cache and holds more of a preceding gap in always-on use.

## Arm C: commit-pass probe (offline, CPU; eval-lane's and my dumps)
- **Model:** each dev utterance stands for one endpoint-delimited segment. Provisional text = B0 greedy at chunk 4 (160 ms). Final text = the same weights re-decoding the segment at chunk 16 (640 ms) at the endpoint.
- **Reported:** final vs provisional WER (pooled, per set, S / D / I, S and C halves); revision burden (share of segments whose final differs; word edits between provisional and final per 100 words; edits that fix vs introduce errors, from paired per-utterance alignments to the reference); first-word and last-word error rates; by utterance length.
- **Variants:** final at chunk 8; final = beam 8 at chunk 16; final + the Arm-D rescoring tuned on S; final + h46-lm's A1 + LM decode when it lands (chunk 4 and chunk 16 if h46 produces both).
- **Cost (estimated here, measured later by runtime-lane):** one chunk-16 encoder pass over the segment at the endpoint plus a decode; reported as encoder-seconds per audio-second from the PyTorch timing on the same host, and as the final-text delay this implies on one core.
- **No adoption gate:** the probe sets targets for `commit.rs`; the engine path is gated by runtime-lane / bench-open.

## Arm D: rescoring screens over A1's beam-8 N-best
- **N-best:** modified beam search, width 8, B0 at chunks 4 and 16 (`h45_decode.py --beam 8 --nbest --ctc-score`), full dev. Each entry carries the transducer log-probability (merged-path score from the beam) and the CTC head's log-likelihood of the hypothesis on the same encoder output; hypotheses are **re-tokenised with the checkpoint's CTC tokenizer** (here the same R0 bpe.model, re-encoded from text, never the transducer ids), and an impossible CTC path is recorded as null and ranked last (never zero cost). N-best oracle WER reported as headroom.
- **D1, CTC rescoring:** score = log p_RNNT + λ · log p_CTC + β · words, λ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, β ∈ {−1, −0.5, 0, 0.5, 1}; tuned on S per chunk.
- **D2, word-level MBR:** choose the N-best entry minimising expected word edit distance under posteriors ∝ exp(score / T), T ∈ {0.5, 1, 2, 4}; with the transducer score alone and with the D1-tuned score. Tuned on S.
- **Finalists:** one per family (D1, D2) per chunk, named before C is scored. Gate: the adoption gate above against the comparator at the same chunk (chunk 4: B0 greedy; chunk 16: B0 greedy chunk 16, the commit-pass final). Rescoring needs the whole segment, so a chunk-4 win is a commit-pass component, not a streaming one.

## Compute (xbabe1 RTX 4090 unless trainer-lane takes it for H48b; then H0's 3090 by agreement with kd-cache)
| item | GPU time (batch 8 estimate) |
|---|---|
| instrument checks | ~10 min (done) |
| Arm A: 25 S-half decodes at chunk 4 + greedy-soup trials (≤ 11) + finalist on C at 4 / 8 / 16 + full dev | ~60 min |
| Arm B: B1 (EMA + raw) and 6 bundles × 3 chunks, full dev | ~50 min |
| Arm D: beam-8 N-best at chunks 4 and 16, full dev | ~60-90 min (Python beam; CPU-bound, 1 process) |
| Arm C, D scoring and tuning | CPU only, H0 cores, minutes |
Kill rules: a family is stopped early only if all its members are worse than B0 on S; nothing else stops early. Caps: GPU 5-min mean ≤ 90%, memory ≤ 85% (≤ 2 processes at 40%), CPU ≤ 80%; other projects' processes untouched.

## Files
- `bin/`: h45_decode.py, h45_merge.py, pull scripts, ultra-score copy. `recipes/`: one JSON per merge. `results/`: score reports, verdict tables, probe tables. `MANIFEST`: every checkpoint and derived weight file with sha256 (weights themselves stay on xbabe1 `/mnt/amphora-backup/h45-wins/weights/`).

## Amendment 1 (posted in AGENT_CHAT before any of its decodes): Arm C2, final-pass settings
- **Why:** the commit pass re-decodes a closed segment offline, so it is not bound to the streaming chunk. B0 was trained with chunk modes {4, 8, 16} and left context 128 frames; a longer final-pass chunk or left context may help (or hurt: out of distribution).
- **Grid (greedy, B0 EMA, S half only, batch 32):** final chunk ∈ {16, 24, 32, 48, 64} frames (640 ms … 2.56 s) × left context ∈ {128, 256} frames, plus full context (whole segment, non-causal) = 11 settings. c16 / left 128 is the pre-registered final.
- **Selection:** lowest S pooled WER (ties → the setting closer to c16 / left 128). One finalist named here before C is decoded.
- **Confirmation:** on C, finalist vs c16 / left 128 (paired bootstrap). Adopted as the commit-pass final setting only if it beats c16 by > 0.17 pp pooled on C with the one-sided 95% upper bound below zero; otherwise c16 stays (its gain over provisional is already measured). Reported either way, with the full-dev number labelled "1 tuning look".

## Results (2026-11-23, DEV only; canonical ultra-score d784f8ca, paired speaker-cluster bootstrap 10,000 / seed 20261122, one-sided 95% bounds)
All decodes in this section ran on xbabe1's RTX 4090 or xbabe2's RTX 3090 with the e61121e tree; every paired comparison uses decodes made on the same GPU and code. Batch-size invariance was checked twice (B0 and M12, batch 8 vs 32: 0 / 5,567 hypotheses differ), so screens at batch 32 / 128 equal batch-8 rows.

### Arm C: commit-pass probe (no tuning; eval-lane's B0 / A1L dumps)
| provisional → final | WER prov → final | final − prov [95%] | segments revised | word edits prov→final / 100 ref words | fixes / breaks per 100 words | first-word err | last-word err |
|---|---|---:|---:|---:|---:|---|---|
| B0 c4 → B0 c16 | 8.302 → **7.703** | −0.600 [−0.700, −0.505] | 30.0% | 2.75 | 1.02 / 0.46 | 9.43 → 8.68% | 9.95 → 8.41% |
| B0 c4 → B0 c8 | 8.302 → 7.961 | −0.341 [−0.408, −0.276] | 23.8% | 2.07 | 0.68 / 0.38 | 9.43 → 9.16% | 9.95 → 8.87% |
| A1L c4 → A1L c16 | 8.426 → 7.895 | −0.531 [−0.633, −0.434] | 31.0% | 2.88 | 1.01 / 0.54 | 9.66 → 8.82% | 9.47 → 8.57% |
| **A1L c4 → B0 c16 (hybrid)** | 8.426 → **7.703** | −0.723 [−0.832, −0.613] | 38.2% | 3.86 | 1.43 / 0.76 | 9.66 → 8.68% | 9.47 → 8.41% |
- Halves (B0 c4 → c16): S 8.399 → 7.878, C 8.209 → 7.533. By reference length: 1-4 words 15.1 → 14.3, 5-9 11.2 → 9.8, 10-19 9.4 → 8.7, 20+ 7.4 → 6.9.
- A1L's own final text vs B0's final text: **+0.192 [+0.126, +0.259]**: A1L alone fails the v3 runtime margin (point ≤ +0.10, upper ≤ +0.20). The hybrid keeps B0's final text exactly and A1L's first token (164 vs 258 ms, eval-lane); adopted by claude-lead (06:59:55Z) as a v3 candidate operating point pending the engine path.

### Arm C2 (amendment 1): final-pass settings (B0 greedy)
- S half: c16 7.878 (pre-registered final), c24 7.824, **c32 7.786**, c48 7.803, c64 7.899; left 256 at every chunk +0.19 … +0.31 (out of distribution); full non-causal context 9.587 (+1.71).
- Finalist c32 / left 128, on C: 7.446 vs c16 7.533, **−0.087 [−0.177, +0.002]**: not significant, under 0.17. **Not adopted; chunk 16 stays the final pass.** Never extend the left context past the trained 128 frames.

### Arm D: CTC rescoring and word MBR over B0's beam-8 N-best
| chunk | comparator (B0 greedy, same GPU) | beam-8 1-best | D1 finalist (S-tuned) | D2 finalist (S-tuned) | N-best oracle (S) |
|---|---|---|---|---|---|
| 4: C half, paired vs greedy | 8.209 | −0.067 [−0.112, −0.021] | λ 0.3, β −0.5: **−0.123 [−0.191, −0.055]** | D2b T 0.5: −0.124 [−0.192, −0.056] | 6.274 |
| 4: full dev (1 look) | 8.301 | 8.218 (−0.084) | 8.136 (−0.166) | 8.136 (−0.165) | |
| 16: C half, paired vs greedy | 7.533 | −0.082 [−0.126, −0.037] | λ 0.1, β −1.0: **−0.091 [−0.142, −0.038]** | D2b T 1.0: −0.085 [−0.133, −0.036] | 5.899 |
| 16: full dev (1 look) | 7.703 | 7.613 (−0.090) | **7.574** (−0.129) | 7.575 (−0.127) | |
- S-half grids: `results/rescore/c{4,16}.S.grid.json`. MBR adds nothing over the CTC term; at chunk 16 the CTC term adds only ~0.01 over the beam on C (S suggested 0.07: selection shrinkage).
- **Verdict: real but below the 0.17 bar at both chunks: NOT ADOPTED.** As a commit-pass component (it needs the closed segment): full final pass c16 + beam 8 + CTC vs provisional greedy c4 = −0.728 [−0.842, −0.619] on full dev.
- CTC scoring re-tokenises each hypothesis text with the checkpoint's own CTC tokenizer (R0 bpe.model, SentencePiece encode); impossible paths would be null and ranked last (none occurred: 0 / 40,000+ entries).

### Arm A: averaging inside A1's basin
- **S half, chunk 4 (B0 re-decode on S 8.397):** A-L50 8.330 (−0.067); A-R2 = greedy soup (raw 49k + raw 47.5k) 8.351 (−0.046); raw 49k alone 8.353; EMA-uniform E2-E5 −0.013 … −0.012; post-hoc EMA P4 −0.013, P8 −0.010; A-R3 −0.019, A-R4 −0.010, A-R5 −0.002; all-ten mean 0.000; A-L25 −0.006. A1L-final alone +0.153 (EMA), +0.243 (raw). Greedy soup: kept raw 47.5k after raw 49k, rejected every EMA and both A1L ingredients. Table: `results/arm_a/selection_S.md`.
- **Finalist A-L50 (0.5 · B0 EMA + 0.5 · A1L-final EMA), posted before any C row:**

| chunk | C half: A-L50 vs B0 | Δ [95%] | full dev (1 look) | Δ [95%] |
|---:|---|---:|---|---:|
| 4 | 8.123 vs 8.209 | −0.085 [−0.158, −0.013] | 8.225 vs 8.301 | −0.076 [−0.133, −0.020] |
| 8 | 7.858 vs 7.841 | +0.017 [−0.052, +0.084] | 7.985 vs 7.961 | +0.024 [−0.031, +0.078] |
| 16 | 7.587 vs 7.533 | +0.054 [−0.007, +0.118] | 7.798 vs 7.703 | +0.095 [+0.034, +0.159] |
- **Verdict: NOT ADOPTED** (chunk-4 gain on C 0.085 < 0.17; non-inferior at 8 / 16 on C). A1's basin is flat at the end of its schedule: averaging is worth ≤ 0.05 pp from A1 alone and ≤ 0.09 pp with A1L. **Latency side door (eval-lane, full dev, paired vs B0 at 160 ms):** first token −13.4 ms [−14.4, −12.3], first correct word −14.6 ms, CQ300 +8.4 pp, coverage +0.1 pp; native INT8 WER 8.252. Real but small (about 1/6 of A1L's −76.7 ms), so not a substitute for the hybrid. Weights: `/mnt/xbabe3-external/jsr/h45-wins-weights/A-L50.pt` (sha256 fc9a01cd…, never committed).

### Arm B1: task-arithmetic validation (pre-registered gate |M12 − arm 12| ≤ 0.087 pp at chunks 4, 8, 16)
| chunk | M12 = θ10 + θ11 − θ_control | trained arm 12 | M12 − arm 12 [95%] | gate |
|---:|---:|---:|---:|---|
| 4 | 9.526 | 9.378 | +0.148 [+0.065, +0.231] | FAIL |
| 8 | 9.355 | 9.244 | +0.110 [+0.037, +0.182] | FAIL |
| 16 | 9.227 | 9.115 | +0.112 [+0.021, +0.204] | FAIL |
- **B1: FAIL.** Weight space (EMA, vs control): ‖arm11 − control‖ 101 ≈ ‖aa (seed 4321) − control‖ 102; ‖arm10 − control‖ 60 with identical batches; ‖M12 − arm12‖ / ‖arm12 − control‖ 1.14 vs 0.99 for the seed replicate arm 12b. The task vectors are dominated by trajectory noise; adding two of them adds their noise. Chunk-arm side: ‖τ16‖ 84, ‖τ17‖ 86, ‖τ18‖ 101, cos(τ16, τ17) 0.77 (a shared component: the chunk arm's own deviation, which a bundle subtracts once per vector).

### Arm B2: bundles onto the chunk arm ("proxy not validated", because B1 failed)
| model (EMA, same GPU, full dev, pooled) | c4 | c8 | c16 | Δ c4 vs chunk arm [95%] | additive prediction (sum of single-arm Δ, c4) |
|---|---:|---:|---:|---:|---:|
| chunk arm | 9.226 | 8.973 | 8.819 | 0 | |
| arm 16 / 17 / 18 alone | 9.156 / 9.149 / 9.177 | 8.988 / 8.944 / 8.945 | 8.810 / 8.780 / 8.826 | −0.070 / −0.076 / −0.048 | |
| B2-full = chunk + τ16 + τ17 + τ18 | 9.974 | 9.672 | 9.484 | +0.749 [+0.581, +0.925] | −0.194 |
| B2+7 | 10.127 | 9.940 | 9.711 | +0.901 [+0.737, +1.072] | |
| B2−16 / B2−17 / B2−18 | 9.351 / 9.363 / 9.444 | 9.140 / 9.126 / 9.159 | 8.970 / 8.989 / 9.020 | +0.125 / +0.138 / +0.219 | −0.124 / −0.118 / −0.146 |
| seed check (chunk_s4321 + τ16r + τ18r + τ17) vs chunk_s4321 | 9.429 | 9.195 | 9.047 | +0.177 [+0.086, +0.271] | |
- **Verdict:** every bundle is worse than the chunk arm; nothing adds in weight space. Weight arithmetic cannot screen H41 recipe bundles; a bundle question needs a trained arm (and each component was already NOT ADOPTED alone). Full table: `results/arm_b_report.md`.

### Commit pass in the engine (PR #76) and commit pass + LM
- **#76 INT8 parity, full dev (ultra-engine-commit d44b62e, sha f76b2872; B0 INT8 export weights.int8 4701787c…; chunk 16; tail pad 0; rows from x9 / xbabe2 / xbabe1 via panel manifests of the missing uids):** provisional 8.2722 = eval-lane's B0 INT8 8.2722 (identical texts); **committed 7.7187 vs PyTorch chunk-16 7.7027: +0.016 [−0.013, +0.045]**; committed − provisional −0.554 [−0.650, −0.461]; revised 30.73% (probe 30.0%); 429 utterances decoded twice across hosts / runs, identical text 429 / 429. **All three ACK criteria (fixed 07:59Z) pass: ACK posted 08:55Z.** Timing: x9's pre-park 0.034 wall-s per audio-s (p50 174 ms per segment, 2 threads, not quiet); the full-dev run's wall times are inflated by fleet_guard duty cycling and are not used. `results/int8_parity/full_pad0.parity.json`.
- **#76 at the shipped 320 ms tail pad, full dev:** provisional 8.2722 (unchanged); committed **7.6772** (4.120 / 11.466) vs pad 0 7.7187: −0.042 [−0.062, −0.022]; committed − provisional −0.595 [−0.702, −0.494]; revised 31.1%. `results/int8_parity/full_pad320.parity.json`.
- **LM pipeline instrument check:** T4 at chunk 4 decoded on my GPU-written cache = h46-lm's CPU-cache row exactly (6.9898; 5,567 / 5,567 identical hypotheses), so the chunk-16 LM rows below differ from h46-lm's chunk-4 rows only by the chunk.
- **Commit pass + LM (no tuning of mine):** h46-lm's pre-registered T4 (beam 4 + M5 + LODR, H42 ship weights 0.6 / 0.45, penalty 0), run unchanged with `train.lm_decode beam` (reference beam) on B0's chunk-16 encoder cache, which I wrote on the GPU in lm_decode's own format (utterance order, frame counts, speakers and checkpoint identical to h46-lm's chunk-4 cache; only the chunk differs).

| row | full dev | C half | paired Δ, full dev [95%] |
|---|---:|---:|---|
| provisional stream: B0 greedy c4 | 8.302 | 8.209 | |
| streaming LM: T4 at c4 (h46-lm's cache; reproduces their 6.990) | 6.990 | 6.902 | |
| commit pass, greedy c16 final | 7.703 | 7.533 | |
| **commit pass + LM: T4 at c16 final** | **6.510** (3.438 / 9.781; 6,904 edits) | **6.324** | −0.480 [−0.587, −0.380] vs streaming T4 c4; −1.193 [−1.333, −1.055] vs greedy c16; −1.793 [−1.995, −1.594] vs the greedy stream |

| provisional stream → final T4 c16 (full dev) | segments revised | word edits / 100 words | fixes / breaks per 100 | first-word err | last-word err |
|---|---:|---:|---:|---|---|
| greedy c4 | 40.4% | 4.17 | 2.10 / 0.44 | 9.43 → 7.56% | 9.95 → 7.62% |
| beam-4 c4 | 39.9% | 4.01 | 2.00 / 0.45 | 9.41 → 7.56% | 10.97 → 7.62% |
| T4 c4 (LM in the stream too) | 24.6% | 2.28 | 0.91 / 0.46 | 8.21 → 7.56% | 9.88 → 7.62% |
- **Reading:** the LM pays most in the final pass: 6.51% final text with an unchanged greedy stream, vs 6.99% with the LM in the stream. 6,904 edits is under the release gate's 7,895 line (gate bounds to be applied on the native path). Files: `results/lm_commit/`.

## Amendment 2 (lead 08:3xZ, posted in AGENT_CHAT before any of its decodes): do wide beams pay off, and only in the commit pass?
- **Final pass at chunk 16:** beam ∈ {4, 8, 16}, each without an LM and with M5 + LODR. LM weights per beam chosen on **S only** with FastBeam (`train.lm_decode beam --fast`): λ_lm ∈ {0.5, 0.65, 0.8, 0.95} × λ_lodr ∈ {0.45, 0.6, 0.75} (ties → nearest h46-lm's L4 0.8 / 0.6); h46-lm's pre-registered T4 (0.6 / 0.45) at beam 4 is kept as the transfer row. Blank penalty 0. Comparator: greedy chunk-16 final (7.703 full dev).
- **Provisional stream:** greedy chunk 4 (B0) and beam 4 chunk 4 (no LM).
- **Reported:** final-text WER on C (finalists named in AGENT_CHAT before C) and on full dev (labelled with its tuning look), paired against the greedy chunk-16 final; revision burden provisional → final for both provisional options; first-word error; **endpoint CPU of the final pass** from the INT8 engine (`ultra-engine-eval --chunk 16 --beam N [--lm M5.ulm]`, decode-thread CPU, 1 GEMM thread) on a fixed 200-utterance panel (100 dev-clean + 100 dev-other, the smallest sha256("h45-cpu-panel:" + uid) per subset, chosen before any result), per audio-second and per mean segment; non-quiet host, informational (quiet numbers via bench-open).
- WER rows are FastBeam (float32: rare near-tie flips vs `train.beam`; h46-lm verified identity without an LM); the operating point recommended to the scoreboard also gets its INT8 engine WER on C.
- **No adoption gate:** this maps the WER / CPU trade-off of the commit pass for v3-scoreboard; the default stays as decided by the lead.

### Amendment 2: beam width × LM in the commit pass (chunk-16 final)
- **S half (FastBeam, selection only):** no LM: beam 4 / 8 / 16 = 7.788 / 7.780 / 7.771 (flat). With M5 + LODR, S-best weights per beam: beam 4 6.669 (0.65 / 0.45; T4 0.6 / 0.45: 6.702), beam 8 6.445 (0.80 / 0.60), beam 16 6.345 (0.80 / 0.60). Grids: `results/amend2/dumps/sweepS.select.json`. Finalists posted 09:22:46Z before any C decode.
- **Engine CPU of the final pass** (INT8, chunk 16, 1 thread, decode-thread CPU, 200-utterance hash panel, xbabe2 not quiet): greedy 0.148, beam 4 0.156 / + LM 0.155, beam 8 0.162 / + LM 0.162, beam 16 0.155 / + LM 0.157 CPU-s per audio-s, i.e. ≤ ~10% over greedy at every width; ≈ 1.04-1.14 CPU-s per 7 s segment on this contended host; peak RSS 111 MB, 671-678 MB with the M5 v1 pack. `results/amend2/cpu/`.
- **C half and full dev (finalists posted before C; WER FastBeam: T4 FastBeam = reference beam on C, 6.324 both; revisions from the greedy-c4 stream):**

| final pass c16 | C half | Δ vs greedy c16 on C [95%] | full dev (1 look) | edits | CPU-s / audio-s | segments revised | word edits / 100 w | fixes / breaks | first-word err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 7.533 | 0 | 7.703 | 8,169 | 0.148 | 30.8% | 2.84 | 1.08 / 0.46 | 8.72% |
| beam 4 / 8 / 16 | 7.455 / 7.451 / 7.447 | −0.078 / −0.082 / −0.085 | 7.619 / 7.613 / 7.606 | 8,080 / 8,074 / 8,067 | 0.156 / 0.162 / 0.155 | ~32% | ~2.9 | ~1.17 / 0.45 | ~8.7% |
| beam 4 + LM (0.65 / 0.45) | 6.274 | −1.259 [−1.436, −1.083] | 6.468 | 6,860 | 0.155 | 41.4% | 4.29 | 2.23 / 0.46 | 7.40% |
| beam 4 + LM (T4) | 6.324 | −1.209 [−1.381, −1.041] | 6.510 | 6,904 | 0.155 | 41.2% | 4.20 | 2.18 / 0.44 | 7.47% |
| beam 8 + LM (0.80 / 0.60) | 6.098 | −1.435 [−1.656, −1.220] | 6.269 | 6,648 | 0.162 | 45.0% | 4.82 | 2.52 / 0.59 | 7.14% |
| **beam 16 + LM (0.80 / 0.60)** | **5.975** | −1.558 [−1.787, −1.332] | **6.157** | **6,530** | 0.157 | 45.4% | 4.94 | 2.63 / 0.59 | 7.00% |
- Paired on C: LM beam 8 vs 4 −0.176 [−0.259, −0.099]; 16 vs 8 −0.122 [−0.170, −0.076] (full dev −0.111 [−0.142, −0.081]).
- **Verdict (no gate; trade-off map):** wide beam pays only with the LM, and costs ≤ ~10% of the chunk-16 encoder pass at the endpoint, so it belongs in the commit pass; the stream stays greedy. The best PyTorch final-text row is beam 16 + LM at 6.157% full dev (−2.145 pp vs the greedy stream, −0.79 pp vs the native streaming L4 incumbent 6.951%), and the native INT8 engine reproduces it: `ultra-engine-eval --chunk 16 --beam 16 --lm M5.ulm 0.8 / 0.6` on the C half = **5.958%** vs PyTorch 5.975 (Δ −0.017 [−0.059, +0.028], 2,629 / 2,785 identical texts; 0.171 CPU-s per audio-s, xbabe2, not quiet). Remaining: wiring it into the engine commit pass (#96 path), with this row as its parity target; quiet timing via bench-open. `results/amend2/engine_C/`.

## Decision (h45-wins, 2026-11-23 ~09:45Z)
| family | verdict | headline |
|---|---|---|
| A: averaging in A1's basin | **not adopted** | best A1-only soup −0.046 on S; A-L50 (B0 ⊕ A1L) −0.085 on C (< 0.17); latency −13 ms only |
| B1: task-arithmetic validation | **fail** | M12 is 0.11-0.15 pp worse than trained arm 12 at every chunk |
| B2: recipe bundle by task vectors | **proxy not validated; bundles worse** | full bundle +0.75 pp vs the chunk arm; do not screen bundles by weight arithmetic |
| C: commit pass (same weights, c4 → c16) | **measured; engine parity ACKed (#76)** | 8.302 → 7.703 (PyTorch), 8.272 → 7.677 (INT8, 320 ms pad); 30% of segments revised |
| C hybrid: A1L provisional + B0 final | **adopted by the lead as a v3 candidate** | B0's final text exactly; A1L's first token: −94 ms on the 100 clips (258 → 164), −77 ms on full dev (312 → ~235, eval-lane) |
| C2: final-pass chunk / left context | **not adopted** | c32 −0.087 on C (n.s.); left 256 hurts |
| D: CTC rescoring / word MBR | **not adopted** | −0.12 (c4) / −0.09 (c16) on C, real but < 0.17 |
| Commit pass + LM (T4, no tuning) | **measured** | 6.510% full dev final text (−0.48 vs the streaming LM) |
| Amendment 2: beam × LM in the final pass | **trade-off mapped** | beam 16 + LM 6.157% full dev / 5.958% native on C; beam costs ≤ 10% of the final pass; wide beam only with the LM and only at the endpoint |

**Recommended next (for the lead):** (1) land the LM + beam in the engine commit pass (#96 path) with the parity targets above, then let eval-lane apply the release gate to its native full-dev row; (2) commit.rs optional separate final-pass model for the hybrid; (3) bench-open quiet timing of greedy / beam 4 + LM / beam 16 + LM final passes and of the RSS cost of the LM pack (+565 MB v1); (4) drop soups and weight-arithmetic bundles from v3.

## Changes
- No repository commits by h45-wins (no git lane). Tools in `bin/` (decode driver, merge, scorer wrapper, greedy soup, commit-pass probe, rescoring, LM-cache writer, LM-beam subset wrapper, INT8-parity scorer, amendment-2 tools); recipes in `recipes/`; pre-registration hashes in `prereg/PREREG.sha256`.
- `raw/` holds shard-level outputs (not for the repo); `results/` holds the summaries and score inputs; weights never leave xbabe1 / the source hosts (MANIFEST).
