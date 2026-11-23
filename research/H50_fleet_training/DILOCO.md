# H50: fixed-roster synchronous exact-FP32 DiLoCo vs synchronous DDP (pre-registered before any run)

Draft by h50-fleet for `research/H50_fleet_training/DILOCO.md`. trainer-lane launches; h50-fleet measures.
It follows the owner's baseline (fixed roster, synchronous rounds, exact FP32 deltas, whole-roster
commits), codex-lead's 06:03Z gates and codex-wave0-review's 06:11Z PR #74 review. It sits beside
`TOPOLOGY.md` (the DDP scaling ladder) and uses the same recipe, pinned commit and exposure.

## Question
At matched unique exposure, does DiLoCo with two single-A100 islands reach DDP's WER on two A100s,
and does it get there at least 10% sooner in wall time? Islands join production (D2, cross-site)
only if both hold.

## Preconditions (no DiLoCo GPU run before all hold)
1. The hub meets claude-lead's 06:08Z PR #73 comment and h50-fleet's 06:17Z review items 1-12. In particular:
   - tensor-only authenticated transport on loopback / ssh tunnels, never 0.0.0.0, never port 8766;
   - a blocking client barrier;
   - whole-roster commits only, with a fixed member order and uniform weights;
   - durable hub state (round, anchor sha256, outer velocity);
   - anchor-aligned checkpoints and evals.
2. The trainer patch with `data_rank` / `data_world` (h50-fleet, 06:39Z) is merged, so each island reads exactly its DDP rank's share.
3. Acceptance tests A1-A5 (below) pass at H=2 on x5.

## Common settings (identical to TOPOLOGY.md T2L unless stated)
- **Code and recipe:**
  - code: trainer-lane's merged H44 PR plus the patch above, checked on each host with the `LC_ALL=C` tree hash;
  - recipe: A1's recipe + `delay_penalty=0.005`, `chunk_modes=4,8,16`, fork of H22ctl step 60,000 (seed sha256 `d1233667…`), replay LR schedule by step.
- **Exposure:** `budget_s=750`, `max_batch=160`, `ddp_global_batches=true`, `total_steps=8000`.
  - 8,000 is a whole number of rounds for H = 2, 50 and 500.
  - That is about 1,667 audio-h per run.
- **Logging and saves:** `log_every=25`, `eval_every=0`, `weights_every=1000`, `ckpt_every_steps=1000`. Every checkpoint falls on a round boundary for H = 2, 50 and 500.
- **GPU cap:** 5-min mean <= 0.87; synchronised step-boundary throttle; never SIGSTOP.
- **Hosts:** x5 GPU0 + GPU1, one arm at a time. x5 is the only free 2-GPU host while a co-tenant project holds x7. Both arms then share GPUs, host and PCIe topology.

## Arms
| id | what | seeds | steps |
|---|---|---|---|
| D0 | DDP, 2 ranks (= T2L), FP32 gradient all-reduce. Run in the NCCL arm that wins TOPOLOGY.md's T2L comparison (expected "tuned": SHM memcpy). DiLoCo must beat the best DDP, not a handicapped one | 1234, 4321 | 8,000 |
| D1-H2 | Protocol / exactness run, H=2, outer lr 1.0, momentum 0 (plain averaging), then lr 0.7 / Nesterov 0.9 | 1234 | 400 |
| D1-H50 | Smoke, H=50, outer lr 0.7, Nesterov 0.9 | 1234 | 2,000 |
| D1-H500 | Matched trial, H=500, outer lr 0.7, Nesterov 0.9 | 1234, 4321 | 8,000 |

**DiLoCo specifics:**
- **Islands:** island i in {0, 1} runs one process with `data_rank=i data_world=2`. At step s it reads exactly D0 rank i's share of global step s. The union over islands equals D0's global step s, so unique audio and tokens match by construction.
- **Inner optimiser:** AdamW, schedule by step as D0. Each worker keeps its own AdamW state across rounds.
- **Rounds:** every member starts round t from anchor θ_t (sha256 pinned in the round record) and takes exactly H steps. It submits Δ_k = θ_t − θ_k (FP32) and blocks until θ_{t+1} is committed.
- **Outer step:**
  - Δ̄ = ½ Δ_0 + ½ Δ_1, summed in fixed order (island 0, then 1) in FP32.
  - v ← 0.9 v + Δ̄; θ_{t+1} = θ_t − 0.7 (Δ̄ + 0.9 v).
  - Trainable parameters only; buffers must be bit-equal across members (asserted).
  - Per-member token counts are recorded. If they differ by more than 2%, the round is flagged (equal shards make them ~equal).
- **Exposure is counted from verified whole-roster commits only.** An aborted round counts nothing and is rolled back to θ_t plus the members' round-t checkpoints.
- **Evaluated weights:** D1's anchor θ_t at steps 1,000 … 8,000 (rounds 2 … 16 at H=500). Each is evaluated after its commit; the island's local weights are never used.

## Acceptance tests at H=2 on x5 (before H=50 / 500)
- **A1 exactness:** every committed θ_{t+1} equals an offline FP32 recomputation from the two submitted deltas and θ_t, bit for bit. Checked for both outer settings over 200 rounds.
- **A2 dropout:** kill island 1 mid-round (by PID).
  - The hub never commits a subset.
  - Island 0 blocks, then aborts at the barrier timeout.
  - Both restart from round t (anchor + member checkpoints). The continued run's anchors equal an uninterrupted reference run's, bit for bit (deterministic data and RNG).
- **A3 corrupt delta:** a wrong key, shape or dtype, a NaN, or a body / hash mismatch is rejected with 4xx before queuing. A valid resubmission completes the round.
- **A4 interrupted commit:** kill the hub between receiving the last delta and publishing. After restart, round t+1 is either fully published (anchor + velocity + round counter, atomically) or not at all, and members resume consistently.
- **A5 stale / duplicate / future:** `base_round` ≠ current is rejected with 409, and so is a second submission with a different hash. An identical resubmission is idempotent.

## Endpoints and gates (decided on full-dev canonical scores; dev_split_v2 confirmation half for decisions)
- **Primary, non-inferiority:**
  - Comparison: D1-H500's final anchor against D0's final weights, both raw, at chunks 4 / 8 / 16. Pinned scorer, paired speaker bootstrap.
  - PASS if the paired upper bound of D1 − D0 is <= 0.087 pp (the registered four-run spread) at all three chunks, with both seeds agreeing.
  - Otherwise INCONCLUSIVE or FAIL under the v3 rule.
- **Gate (codex 06:03Z):** D1-H500's time to reach D0's final canonical WER is >= 10% lower at matched audio / tokens.
  - Scored with `h50_measure.py --wer-file … --cohort … --scorer … --weights raw` on both arms.
  - Both arms run on the same GPUs, so their wall clocks compare.
- **Reported, not gated:**
  - hub-side EMA of anchors (per-round decay 0.9995^H) against D0's EMA;
  - per-round delta norms and cosines;
  - per-round wall breakdown: compute, delta upload, barrier wait, outer step, anchor download;
  - steady-window throughput and joules per audio-h (three windows per arm, `h50_measure.py` protocol).
- **Kill rules:**
  - any non-finite delta;
  - an anchor train-loss more than 20% above D0 at the same step;
  - a hub round that commits a subset (a protocol bug: stop and fix).

## Prediction (written before the runs)
- **Per step:** on one host, DDP with the tuned SHM path exposes only ~5% of the step to communication (x5 probe 06:23Z: 119.8 vs 113.4 ms at 375 s per rank).
- **Per round:** an H=500 round costs ~1-2 s of exchange and outer step per ~60 s of compute.
- So D1 should be at most ~3-5% faster per step. DiLoCo's sample efficiency at matched tokens is typically equal or worse.
- **P(time-to-WER gate passes on x5) ≈ 0.1.** P(non-inferiority holds) ≈ 0.6.
- DiLoCo's real value is cross-site. Measured site <-> cloud is ~115 Mbit/s down and ~245 Mbit/s up, where per-step DDP is impossible. D2 tests exactly that.

## D2 (after D1 passes; outline only, pre-registered separately)
- **Members:**
  - the A100 DDP island (x5 ×2 as ONE member, internal DDP, rank 0 talks to the hub);
  - one on-prem 3090 island (xbabe2 or H0).
- **Per-member steps:** H_k is proportional to measured throughput, so every member's round takes the same wall time (still synchronous, whole-roster).
- **Weighting:** token-weighted outer average (equal-H weights would mis-weight unequal work).
- **Topology:** hub on H0, bound to 127.0.0.1. The cloud member reaches it through `ssh -N -R` started from H0 (the cloud firewall drops inbound traffic except :22). IPs only.
- **Transfer budget:** 232 MB FP32 delta down (~16 s at 115 Mbit/s) plus 232 MB anchor up (~8 s at 245 Mbit/s) per round. Choose H so that this is <= 10% of a round.
- **Gate:** >= 10% lower time to the same canonical WER than the A100 island alone, at matched unique exposure, with energy and S/D/I checks (codex 06:11Z).

## Compute
- D0: 2 × ~20 min.
- D1-H500: 2 × ~20 min.
- D1-H2 / H50: ~10 min.
- Acceptance tests: ~20 min.
- Total about 1.7 h of x5 ×2.
