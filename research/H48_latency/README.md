# H48 latency (ULTRA v3)

Wave H48 covers A1L, the low-latency mode, onset peek and the MFA-anchored latency loss (owners: trainer-lane, eval-lane, h48-latency). Each run below is pre-registered here before it starts. Results are kept (metrics, receipts); weights are not committed, only their paths and sha256 in `MANIFEST`.

## H48a: resume A1L (pre-registered 2026-11-23 before the resume)

**What A1L is** (trunk step, first pre-registered in AGENT_CHAT at 04:44Z):
- A1's final checkpoint `ckpt-00049000.pt` (sha256 `2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146`), fine-tuned on A1's recipe plus `delay_penalty=0.005`.
- A1's recipe: dense S, R0 BPE-500, `chunk_modes=4,8,16`, `eval_chunks=8,16`, speed perturbation 0.9 / 1.0 / 1.1.
- Schedule: the H41 replay schedule, 8,000 steps (200 warmup steps to 2.476e-4, cosine to 3.0e-5). Batch 650 s / 144, seed 1234, EMA 0.9995.
- The model starts from the raw weights; the EMA restarts from them.

**History:**
- Launched on xbabe8 at 04:46:50Z.
- Stopped at step 5,907 by the owner's 05:07Z stop. The trainer wrote `ckpt-00005907.pt` on SIGTERM.
- dev-clean400 greedy (raw / EMA):

| step | chunk 4 | chunk 8 | chunk 16 |
|---:|---|---|---|
| 2,000 | 4.962 / 4.650 | 5.053 / 4.506 | 5.040 / 4.298 |
| 4,000 | 4.936 / 4.702 | 4.780 / 4.506 | 4.506 / 4.376 |

- Final eval at the stop (step 5,907): raw chunk 4 4.845.

**Resume checks** (done 05:58-06:02Z, before the resume):
- **Checkpoint:** sha256 `2cdcd722cba8af61fcb3f416fae5f79e1cbb10a090562abb15e0d9eb4c9fc228`, which matches its sidecar; Codex verified the digest and contents independently. It holds model, EMA, optimizer, scheduler, position, RNG (python / numpy / torch / cuda), the data manifest and `watch_sig`.
- **Recipe:** `train_config` in the checkpoint equals the run's `config.json`; the only differences are list-vs-tuple types for `splits` and `betas`.
- **Cursor:**
  - step 5,907;
  - scheduler `last_epoch` 5,907 (lr 6.643e-5, continuing the cosine; step 5,900 logged 6.666e-5);
  - position epoch 1 / window 3 / batch 10, which is 7 batches after step 5,900's 1 / 3 / 3;
  - tokenizer sha256 `c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5`.
- **Host change, xbabe8 → xbabe9:** another project now holds xbabe8's GPU memory. Both hosts are A100 40 GB. The venvs are identical (torch 2.7.0+cu128, numpy 2.5.5, scipy 1.18.1).
- **Code identity, pinned:**
  - The resume runs the same tree as steps 0-5,907: commit `d5f231e399d9272a581e486a00641c52776bb4ba` plus xbabe8's local diff.
  - The diff's sha256 is `bf9f3cadb66d1385048c9a54cdec7ed659782257d8123c23740643853829ec88`. It covers `kd.py`'s exact speed key and its test, which A1L doesn't use, and two ops scripts.
  - On xbabe9 that tree is the worktree `.agent/worktrees/trainer-a1l`. Its `python/ultra` `*.py` content sha256 is `99bce400e81fbd6748c11b4603950c85b7bfc730abb9d5c9909672201ce04ff9`, identical to xbabe8's.
  - The launcher refuses to start if the hash differs.
- **Launch command:** the original A1L command, unchanged. `h41_replay.sh`'s settings with `INIT_FROM=ckpt-00049000.pt`, run name `A1L_dp0.005`, plus `chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty=0.005`. The trainer resumes from `LATEST` (`ckpt-00005907`) and does not re-initialise. Guard 0.87.

**Gate** (unchanged from the 04:44Z pre-registration):
- Full-dev EMA, pooled, canonical scorer, at chunks 4 / 8 / 16, must stay within the four-run spread of A1-49000's 8.302 / 7.961 / 7.703.
- First-token latency at chunks 4 / 8 is the reported outcome, against A1-49000's 258 / 332 ms (eval-lane's pipeline).
- Promotion requires arm 14 to pass; it did (eval-lane, 04:42Z).

**Kill rule:** raw chunk-4 dev-clean400 WER above 20% at any eval. The step-2,000 rule already passed (EMA 4.650 against a threshold of 6.163).

**Outputs:** `train.jsonl` evals, then full-dev EMA decodes at chunks 4 / 8 / 16 with eval-lane's pinned evaluator (hypotheses + WER JSON). Weights stay on xbabe9 and are listed in `MANIFEST`.

**Compute:** 2,093 steps, about 0.26R, plus about 15 min of decodes.
