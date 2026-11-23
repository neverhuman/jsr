#!/usr/bin/env bash
# trainer-lane: gate (b) STAGE 2 (eval-lane Amendment 1; lead 11:20:26Z / 11:25:09Z), run on H0 ONLY if stage 1 did not pass.
# Pause S2b at an exact window boundary (all ranks), run the registered fast cooldown from BP-m = S2b-BP1's raw weights on x9
# alone, post its end checkpoint (path + sha256) for eval-lane, then resume S2b at world 6 from its own boundary checkpoint.
#   s2b_stage2.sh <bpm checkpoint name, e.g. ckpt-00005500>
set -u
D=/mnt/xbabe3-external/jsr/trainer-lane/s2; ST=$D/state; LOG=$ST/s2b_stage2.log
RUN=H53_S2b_medium_s7919 BPM=$1 FC=H53_FC_S2bBP1_s1234
ENVS=$(cat $ST/s2b_env.txt); eval "$ENVS"
P=/home/ubuntu/jsr-data/protected/$RUN/$BPM.pt SEED=/home/ubuntu/jsr-data/seeds/H53_S2b_BPm
SM=/home/ubuntu/jsr-data/librilight/shards/r0_v11/librilight.small MD=/home/ubuntu/jsr-data/librilight/shards/r0_v11/librilight.medium
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
# 1. BP-m to x9 under one hash (x5's protected file)
H=$(ssh -n -o BatchMode=yes xbabe5 "sha256sum $P | cut -c1-64"); [ ${#H} = 64 ] || { log "no BP-m $P"; exit 1; }
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
ssh -n -o BatchMode=yes xbabe9 "mkdir -p $SEED"
ssh -A -n -o BatchMode=yes xbabe5 "taskset -c 56-59 rsync -a -e 'ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR' $P ${P%.pt}.sha256 ubuntu@${X9_LAN:?}:$SEED/"
[ "$(ssh -n -o BatchMode=yes xbabe9 "sha256sum $SEED/$BPM.pt | cut -c1-64")" = "$H" ] || { log "BP-m copy on x9 differs"; exit 1; }
log "BP-m $BPM ${H:0:16} on x9"
# 2. pause S2b at an exact window boundary (all four nodes save and exit)
bash $D/boundary.sh $RUN s2b "$ENVS" | tee -a $LOG | grep -q 'BOUNDARY OK' || { log "ABORT: S2b did not pause at a boundary"; exit 1; }
# 3. the registered fast cooldown on x9 alone (h41_replay.sh at 90b7f4d; Amendment 5 recipe on S2b's mix)
ssh -n -o BatchMode=yes xbabe9 "cd ~ && INIT_FROM=$SEED/$BPM.pt RUN_NAME=$FC bash ~/jsr/.agent/worktrees/trainer-s2/python/ultra/train/ops/h41_replay.sh ~/jsr-data 0-27 0 FC \
  seed=1234 chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty=0.005 lr=0.0001241 min_lr_ratio=0.1709 \
  extra_splits=$SM,$MD extra_base_weight=$S2B_BW window_groups=$S2B_WG workers=20" 2>&1 | tail -2 | tee -a $LOG
F=/home/ubuntu/jsr-data/runs/train/$FC
for i in $(seq 360); do  # <= 3 h
  ssh -n -o BatchMode=yes xbabe9 "kill -0 \$(cat $F/PID) 2>/dev/null" || break; sleep 30
done
E=$(ssh -n -o BatchMode=yes xbabe9 "cat $F/LATEST"); EH=$(ssh -n -o BatchMode=yes xbabe9 "sha256sum $F/$E.pt | cut -c1-64")
log "cooldown ended: $F/$E.pt $EH"
$D/tools/post.sh "GATE (b) STAGE 2: fast cooldown from S2b-BP1 ($BPM) DONE on x9, end checkpoint for eval-lane @eval-lane @claude-lead" <<B
- **xbabe9:$F/$E.pt**, sha256 **$EH** (EMA in the payload). Init BP-m $BPM (sha256 ${H:0:16}…).
- Recipe: h41_replay.sh at 90b7f4d, 650 s / 144, 8,000 steps, 200 warmup → 1.241e-4, cosine min_lr_ratio 0.1709, EMA 0.9995, seed 1234, dp 0.005, S2b's mix (bw $S2B_BW, window_groups $S2B_WG). Run with HOST=x9, GATE=b.
- S2b resumes at world 6 now from its boundary checkpoint.
B
# 4. resume S2b at world 6 (each node resumes its own LATEST = the boundary checkpoint)
for s in 5 7 8 9; do ssh -n -o BatchMode=yes xbabe$s "$ENVS bash ~/x${s}_s2b.sh 2>&1 | tail -1" | sed "s/^/x$s: /" | tee -a $LOG & sleep 1; done; wait
log "S2b resumed at world 6"
