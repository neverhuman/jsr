# H49 launch lines (trainer-lane launches; tree = PR #87 head 4d51bf2 + patches/h49-trainer.on-4d51bf2.patch)

Common: A1C's exact command (H41 amendment 23): `h41_replay.sh` settings (seed=1234 budget_s=650.0 max_batch=144
optimizer=adamw lr=0.0002476 warmup=200 total_steps=8000 min_lr_ratio=0.1212 ... ema_decay=0.9995 ... weights_every=2000),
INIT_FROM = A1-49000 `ckpt-00049000.pt` (sha256 2a3dd9fb...), `chunk_modes=4,8,16 eval_chunks=8,16`, delay_penalty 0.
Before launch: the 50-step bit-exactness check against A1C's tree if the trees differ (README, shared protocol).

## H49a (MWER), stage 2: lambda screen, seed 1234, read at step 2,000 (selection half)
INIT_FROM=$DATA/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt RUN_NAME=H49a_mwer_l0.3_s1234 \
  train/ops/h41_replay.sh $DATA <cpus> <gpu> h49a chunk_modes=4,8,16 eval_chunks=8,16 \
  mwer_weight=0.3 mwer_beam=4 mwer_utts=12 mwer_tau=1.0 mwer_scorer=<ultra-score --candidate-edits build>
INIT_FROM=... RUN_NAME=H49a_mwer_l1.0_s1234 ... mwer_weight=1.0 (otherwise identical)

## H49a stage 3: the winner continues (same run dir, resume) to 8,000; winner with seed=4321 from scratch
INIT_FROM=... RUN_NAME=H49a_mwer_l<win>_s4321 ... seed=4321 mwer_weight=<win> (otherwise identical)

## H49b (sequence KD), after the pilot's stop rule passes and the store exists
INIT_FROM=... RUN_NAME=H49b_seq_l0.3_s1234 ... seq_weight=0.3 seq_store=<ultra.seqkd/v1 store> seq_tau=1.0 seq_max_alt=3
INIT_FROM=... RUN_NAME=H49b_seq_l1.0_s1234 ... seq_weight=1.0 (otherwise identical)

## Comparator
A1C (seed 1234, running) and A1C-r = A1C with seed=4321 (requested; shared with H48c / H53 P1).
Kill rules K1-K4 as in the README (K1/K2 are the existing h41_killrule.sh / trainer rules).
