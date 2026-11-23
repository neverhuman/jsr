## H48c: MFA-anchored latency loss (pre-registration draft, h48-latency, 2026-11-23; written before any H48c run)

Section for `research/H48_latency/README.md`. trainer-lane runs the arms (it owns the trainer and all training runs). eval-lane owns the latency metrics. Every result carries the label **"MFA-anchored"** (owner ruling, 05:4xZ). Revised at 06:51Z and 07:05Z, before any run: the fork and WER comparator are B0; trainer-lane's A1C is the control; an anchor-only arm runs beside the plan's arm, making a 2×2 with A1L.

### Question
Can a one-sided, MFA-anchored late-emission cost on the pruned RNN-T lattice reach **≤ 133 ms mean first-token latency on the end-of-first-word clock at 80 ms chunks** with **no WER loss against B0 beyond the spread**? It is tested alone and on top of `delay_penalty=0.005` (the plan's arm).

The uniform penalty already reaches that latency, but it costs WER:
- 13d (0.0075) gets 131 ms and 13a (0.01) 105 ms at 80 ms, at +0.17–0.20 pp at chunk 4;
- even dp 0.005 in A1L is +0.12 pp against B0 at chunks 4 / 8 (penalty and re-warm confounded).

### Hypothesis
k2's `delay_penalty` adds `dp·((T−1)/2 − t)` to every label arc. That rewards every earlier frame, including frames before the word has been heard, and it pushes on-time tokens as hard as late ones.

A cost that is exactly zero up to the frame where the word's acoustic evidence is complete (plus a margin), and positive only after it, should cut first-token and stable-word lag with a smaller WER cost:
- no gradient ever favours emitting before the evidence;
- tokens that are already on time feel nothing.

### Objective (code: `train/anchored_rnnt.py`; side file: `train/mfa_anchors.py`)
`L_κ = −log Σ_{π∈A(y)} exp{s(π) − κ·C(π)}`, where:
- `A(y)` is k2's exact regular RNN-T valid graph;
- `s(π)` includes k2's `delay_penalty`, applied exactly as k2 applies it;
- `C(π) = Σ_u min(cap, max(0, t_u(π) − a_u − m))`.

The terms of `C`:
- `t_u(π)` is the encoder frame at which path `π` emits token `u`.
- `a_u` is the **evidence frame** of the MFA end of the word holding token `u`: `max(0, ceil((τ_end·den/num − 25 ms)/40 ms))` at the batch's speed `num/den`. This is the first 25 Hz frame whose causal stem window reaches the word end. It was checked against the real LogMel + stem by perturbing one sample.
- `m` is the margin in frames and `cap` is in frames.

How the cost is applied:
- `C` is a constant (no gradient) added to k2's label log-probs `px`.
- It is applied to the pruned loss and, with `anchor_simple`, to the smoothed loss, so the pruning band follows the tilted posterior.
- Tokens without an anchor cost 0. Batch normalisation is unchanged: per-sequence sums ÷ transducer tokens.

**Unit-tested invariants** (24 tests):
- κ = 0 equals k2 bit for bit (values, occupancy gradients; dp 0 and 0.005), and the model's loss is unchanged;
- float64 gradcheck;
- on enumerated lattices: path mass, `C(S+T−1, S)` path counts, arc marginals, flow conservation, `dL/dκ = E_q[C]` and `dE_q[C]/dκ = −Var_q(C)`;
- late frames are strictly penalised;
- every frame at or before `a_u + m` keeps one common posterior factor, so earliness is never rewarded (k2's uniform dp is shown to reward it);
- the builder never opens dev / test members.

### Data
- **Source:** Zenodo 2619474 v1.0 (Lugosch et al. 2019, MFA, CC BY 4.0).
  - `librispeech_alignments.zip`: 623,023,332 bytes, md5 `2bab567d0ace651a4ba254e813629f46` (equal to the published value), sha256 `6ac1d003cf54dbf7968adc8f3a0ce59e0345d8458847477bb70d57f49c577b16`.
- **Train TextGrids:** 28,538 / 104,008 / 148,645 for train-clean-100 / 360 / other-500.
- **Side file:** `mfa_anchors_train960_v1.npz` (`ultra.mfa-anchors/v1`), keyed by (utterance id, speed ∈ {0.9, 1.0, 1.1}).
  - sha256 `58a8bd0f65231138cb0f10cd62d1b59a28c56c6247732ae4b59f570f13e60b2b`, 160,520,322 bytes.
  - Phone tier in `mfa_phones_train960_v1.npz`, sha256 `473b9fff…`.
- **Coverage against our parquet transcripts:** 281,127 of 281,241 utterances and 9,399,766 of 9,403,555 words (99.96%). The gaps:
  - 50 utterances have no TextGrid (the archive's own `unaligned.txt` lists);
  - 64 fail the word mapping (MFA dictionary splits around apostrophes).
  - All 114 train untilted.
  - No last word ends after the audio.
- **Real-data check with A1's tokenizer** (R0 BPE-500, train-clean-100 shard 0): 2,039 of 2,039 utterances anchored at every speed, anchors monotone within each utterance, the last token's anchor ≤ the last encoder frame + 1.
- **Use rules:**
  - TRAIN anchors are used only in training.
  - Dev MFA timings are used only by eval-lane's latency metrics.
  - Test stays sealed.

### Arms (revision 2 at 07:05Z, before any H48c run: A1C is the control and the arms match it)
**Revision 1 (06:51Z), after eval-lane's H48a verdict.** A1L (B0 + 8,000 replay steps with dp 0.005) fails its WER gate against B0 at chunks 4 and 8: +0.124 / +0.112 pp, with one-sided lower bounds +0.050 / +0.033. Penalty and LR re-warm cannot be separated. Forking from A1L would carry that cost into every H48c arm, and gating against A1L would hide it. The owner's target is "≤ 133 ms **without the WER cost**", a statement against B0. Per this draft's own contingency, H48c forks from **A1-49000 (B0)**.

**Revision 2 (07:05Z), after claude-lead's request at ~07:03Z.** trainer-lane's **A1C** (amendment 23) is H48c's control; no second control is run. A1C is B0 + 8,000 replay steps with dp 0, the same LR re-warm and the same data as A1L, launched on x9. The H48c arms copy A1C's command and add only the anchored loss. That includes A1L's `chunk_modes=4,8,16`: revision 1's chunk 2 is dropped from training so that the control is exact.

**Common settings (= A1C):**
- **Fork:** `ckpt-00049000.pt` (sha256 `2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146`), raw weights, EMA restarted.
- **Recipe:** A1's (dense S, R0 BPE-500, pruned RNN-T with prune range 5, CTC 0.3, speed perturbation 0.9 / 1.0 / 1.1).
- **Chunks:** `chunk_modes=4,8,16`.
- **Schedule:** 8,000 steps, 200 warmup steps to 2.476e-4, cosine to 3.0e-5, EMA 0.9995.
- **Seed and data:** seed 1234, A1L's data order.
- **Batch:** 650 s / 144, A1C's and A1L's (codex-lead, 07:06Z: A1C keeps A1L's batch, exposure, warmup and schedule; VRAM is filled by separate guarded co-tenants). Every cell of the 2×2 shares it.
- **Decodes:** end-of-run full-dev decodes at chunks **2 / 4 / 8 / 16** for A1C and every H48c arm (chunk 2 added to A1C as a decode only).

**The 2×2 at matched exposure:**

| Cell | `delay_penalty` | `anchor_kappa` | `anchor_margin` / `cap` / `simple` | Seed | Who runs it |
|---|---:|---:|---|---:|---|
| A1C (control) | 0 | 0 | – | 1234 | trainer-lane, amendment 23 |
| A1L (exists) | 0.005 | 0 | – | 1234 | H48a (same batch) |
| **H48c-a** (anchor only) | 0 | 0.01 | 1 / 12 / true | 1234 | stage 1 |
| **H48c-b** (anchor on top of dp 0.005, the plan's arm) | 0.005 | 0.01 | 1 / 12 / true | 1234 | stage 1 |
| H48c-r (confirm) | the pick | the pick | 1 / 12 / true | 4321 | stage 2, with A1C's seed-4321 twin if trainer-lane runs one; otherwise B0's four-run spread |
| H48c-c (stronger tilt) | the better arm's dp | 0.02 | 1 / 12 / true | 1234 | 1b, only under the rule below |

**What each comparison answers:**
- **a − A1C:** the anchored loss alone.
- **b − A1C:** the plan's arm.
- **b − a:** what dp 0.005 adds once late emissions are anchored.
- **b − A1L:** the anchored loss on top of dp 0.005.
- **A1L − A1C:** dp 0.005 alone.

Every comparison is paired over speakers.

eval-lane registered H48c-a vs A1C (non-inferiority, confirmation half, 0.087 / 0.094 / 0.360) at 07:07:38Z, before any result. h48-latency requests the same registration for H48c-b vs A1L.

**Why κ = 0.01.** On arm 10's family the uniform penalty moved the mean first token at 80 ms by ~9.4 ms per 0.001 (199 → 166 → 131 → 105 ms for dp 0 / 0.005 / 0.0075 / 0.01). Reaching ≤ 133 ms needs about 0.0075 of per-frame pressure on the tokens that are late. The anchored cost puts κ on exactly those tokens and nothing on the others, so κ = 0.01 gives late tokens a little more pressure than dp 0.0075 did, and on-time or early tokens none.

**Chunk 2 is a decode only.** All four cells and B0 were trained at {4, 8, 16}, so chunk 2 is equally out of distribution for every one of them, and paired comparisons at 80 ms stay fair. The latency gate is read at 80 ms (the plan) and at 160 ms (in distribution, same rules). If an anchored arm reaches ≤ 133 ms at **160 ms**, the default mode meets the target and no separate 80 ms mode is needed. A chunk-2 fine-tune in the style of H48b would follow only if the 80 ms mode is still wanted.

**Settings:**
- `margin = 1` frame (40 ms) covers MFA boundary uncertainty; MFA ends sit a few tens of ms after the RMS offset (C1 spot check), which errs late, the safe direction.
- `cap = 12` frames (480 ms) bounds the pull of a misaligned word.
- **Memory:** an anchored arm adds one (B, S, T+1) fp32 tensor (tens of MB at 650 s / 144) and one extra no-grad recursion; check its peak when placing a co-tenant.

**trainer-lane command:**
- A1C's command
- `+ --set mfa_anchors=<copy of mfa_anchors_train960_v1.npz> anchor_kappa=0.01 anchor_margin=1 anchor_cap=12 anchor_simple=true`
- `+ delay_penalty=0.005` for H48c-b only.
- The trainer prints a `mfa_anchors` line (path, sha256, utterances, speeds) at start. Each train record carries:
  - `anchor_risk`: `L_κ − L_0` per token, ≥ 0;
  - `anchor_cov`: the anchored-token share;
  - `anchor_counts`: anchored / missing / token-mismatch utterances.

### Measurements
All on DEV only, with the canonical scorer and eval-lane's pinned evaluator and pipeline.
1. **Accuracy:**
   - full-dev EMA greedy WER, pooled and clean / other, at chunks 2 / 4 / 8 / 16;
   - S / D / I;
   - first-word error rate;
   - insertions before the first word.
2. **Latency (eval-lane, H43 metrics):**
   - mean / p50 / p95 first-token latency on the **end-of-first-word clock** (design; primary), and on the onset clock;
   - first-correct-word with coverage;
   - correct-by-deadline at 150 / 300 / 500 ms, provisional and stable;
   - stable-word lag p50 / p95 over all words.
   - Taken at fixed 80 ms and fixed 160 ms chunks, and on the trained dense 40 / 40 / 80 → 160 ms ramp if the engine supports it then (reported only).
   - Paced replay on a quiet host for onset, first correct stable text and immutable commit (controlling plan).
3. **Training diagnostics:**
   - `anchor_risk` and `anchor_cov` curves;
   - emission frame minus evidence frame per token on dev (from forced alignment of the final EMA; dev MFA used for evaluation only).

### Gates (fixed now)
**Comparators:**
- WER: **B0 = A1-49000**, with eval-lane's registered per-chunk four-run spreads (`h43/margins.json`, 06:24Z): 0.087 / 0.094 / 0.360 at chunks 4 / 8 / 16.
- Attribution: A1C for both arms and A1L for H48c-b, paired. A1C is trainer-lane's amendment 23.
- Context only: H48b at matched chunks, paired. Not a gate.

**Stage-1 pass (per arm; EMA, pooled full dev, canonical scorer):**
- **WER:** ΔWER vs B0 ≤ the spread at each of chunks 4 / 8 / 16 (the H41 point rule). The v3 interval reading (one-sided 95% bound) is reported next to it. An INCONCLUSIVE interval is not a pass for adoption (stage 2).
- **First-word and deletions:**
  - first-word error rate within +0.50 pp of B0;
  - deletions not up by more than the spread (§8.3).
- **Latency:**
  - mean end-of-first-word FTL ≤ 133 ms at fixed 80 ms, or at 160 ms;
  - controlling rules vs B0 and vs A1C at the same chunk: lower median, paired p95 ≤ +20 ms, first-correct-word coverage within 0.50 pp;
  - first-word error at that chunk within +0.50 pp of B0's at the same chunk.

**Stage-1 → stage 2:**
- If both arms pass, confirm **H48c-a** (no uniform penalty; the simpler recipe).
- If only one passes, confirm that one.
- If neither passes WER: H48c is **negative**. Report the (FTL, WER at chunk 4) points of the 2×2 against the uniform-dp frontier: arms 10 / 13c / 13d / 13a, the chunk arm / 14, A1L and H48b. Stop.
- If an arm holds WER but misses 133 ms: run H48c-c (κ = 0.02, seed 1234, on that arm's dp) under the same gate, then stage 2. H48c-c is the last step.

**Adoption (stage 2):**
- Both seeds pass every stage-1 gate.
- The pair's mean passes the interval rule against B0 (upper bound ≤ spread).
- The result is reported as "MFA-anchored".

Promotion into H53's trunk recipe is claude-lead's ruling, on the same basis as dp 0.005.

### Kill rules (automated, by recorded PID)
| Rule | Condition |
|---|---|
| K1 | At step 2,000, dev-clean400 EMA at chunk 4 is more than 1.5 pp worse than A1L's step-2,000 dev-clean400 EMA at chunk 4 (4.650, so the threshold is 6.150). A1L is the same fork and schedule |
| K2 | Raw chunk-4 dev-clean400 WER above 20% at any eval |
| K3 | Divergence guard trips, or `anchor_risk` rises over 1,000 consecutive steps |
| K4 | Full-dev first-word error or deletions up by more than 0.50 pp. The arm fails; no confirmation |

### Compute
- **Stage 1:** 2R (H48c-a, H48c-b), the plan's budget. A1C is trainer-lane's run and shared with H48a.
- **Stage 2:** 1R. Stage 1b, only under its rule: 1R.
- **Decodes:** full dev at 4 chunks per arm, ~15 min on one GPU.
- **Latency traces:** eval-lane CPU.
- **Loss overhead:** on CPU, the anchored loss path measured within noise of the existing one (median ratio 0.96, loss part only, contended H0). The first 200 GPU steps are compared with A1C's step time and reported.

### Known limitations (stated before any run)
- **SpecAugment time masks:** up to ~10 encoder frames each, about two per 10 s. They can hide a word's evidence, and its token is still costed from the MFA anchor. k2's uniform dp has the same exposure.
  - `cap` bounds the cost at `κ·cap` per token: ≤ 0.12 nats at κ = 0.01 and 0.24 at κ = 0.02.
  - If deletions or insertions near masked regions rise, a refinement zeroes the cost of tokens whose word overlaps a mask. That needs `spec_augment` to return its masks. Not part of this arm.
- **Multi-piece words:** every piece is anchored at the end of its word. That is conservative for the first pieces of long words and never penalises them early. Per-piece anchors from the phone tier (`mfa_phones_train960_v1.npz`) are a follow-up, not part of this arm.
- **MFA alignment errors:** the margin and the cap bound their pull. 114 of 281,241 train utterances train untilted: 50 have no MFA alignment and 64 hit MFA apostrophe splits that do not map onto the reference words.

### Outputs
- `results/`: metrics JSON / JSONL, score summaries, latency receipts.
- `MANIFEST`: checkpoint paths and sha256, the side file's sha256 and the archive sha256. Never weights.
