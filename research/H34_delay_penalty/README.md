# H34: emission-delay penalty for streaming first-preview latency

**Status:** pre-registered on 2026-11-22, before any H34 run.

## Question
Live Transcribe shows the first preview **370–540 ms after speech onset**. Codex's direct-PCM measurement puts almost all of that in the model's emission delay; the pipeline accounts for only 18–27 ms.

Can k2's pruned RNN-T `delay_penalty` cut the first-preview horizon by **≥ 150 ms** at a cost of **≤ +0.3 pp** dev-clean400 WER?

## Design

Three arms, run sequentially on xbabe1's RTX 4090 under fleet_guard (target 0.85). Each arm is a 3,000-step fine-tune from **D2c final** (`ckpt-00010193`, raw weights; full-DEV beam 4 pooled 10.11).

| arm | `delay_penalty` λ |
|---|---|
| H34-ctl | 0 (control: same fine-tune, no penalty) |
| H34-a | 0.0015 |
| H34-b | 0.004 |

Everything except λ is matched across arms:
- **Model:** preset S, SentencePiece 4096 (D2's tokenizer).
- **Loss:** pruned RNN-T (prune range 5) + 0.3 CTC.
- **Data:** 960 h with speed perturbation 0.9/1.0/1.1, seed 1234, same data order.
- **Batch:** budget 650 s / max batch 144, chosen to fit 24 GB.
- **Schedule:** lr 1.1e-4 with warmup 100, cosine to 0 over 3,000 steps. That is D2c's 2,800-s peak (4.25e-4) scaled by √(650/2800) and halved for a fine-tune.
- **Other:** EMA 0.9995, qkv cap from step 100. Snapshots and evals every 1,000 steps.

- **Implementation:** the new `TrainConfig.delay_penalty` is passed to both `k2.rnnt_loss_smoothed` and `k2.rnnt_loss_pruned`, as icefall does. With 0 the calls are unchanged, which a unit test checks for bit-exactness.

## Measurements
1. **Accuracy:** dev-clean400 greedy WER, raw and EMA, at 3k for each arm. DEV only.
2. **Latency:** codex's first-preview horizon (ms after speech onset) with the Live Transcribe direct-PCM harness, on the same utterance set for all arms.
   - Inputs: each arm's 3k raw and EMA weights.
   - Also reported: the mean emission frame of the first non-blank token relative to forced-alignment word onsets.

## Decision rule (fixed now)
- **Pass:** an arm passes when, compared with H34-ctl, its first-preview horizon is **≥ 150 ms earlier** and its dev-clean400 WER is **≤ +0.3 pp** (EMA, or raw if EMA is not better).
- **Choice:** if both arms pass, the one with the smaller WER cost wins.
- **Handoff:** the winner goes to codex for the Live Transcribe registry.
- **No pass:** if neither arm passes, H34 is recorded as negative and λ stays 0 in production.
- **Kill:** an arm stops early if dev-clean400 at 1k is more than 1.5 pp worse than the control at 1k.

## Out of scope
- No production change happens without the gate.
- The penalty is not applied to D2e, H22 or L3 runs.
