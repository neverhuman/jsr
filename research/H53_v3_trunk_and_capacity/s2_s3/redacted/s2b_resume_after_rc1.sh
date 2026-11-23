#!/usr/bin/env bash
# trainer-lane (lead 14:13:25Z): the island must not idle after RC1. When RC1's end checkpoint is in its protected dir (with
# its marker) and every RC1 torchrun has exited, resume S2b at world 6 exactly from its cut state (each node's LATEST =
# ckpt-00010818; same pin, config and window_groups 96), then log the resume record and the first train records.
set -u
D=/mnt/xbabe3-external/jsr/trainer-lane/s2; ST=$D/state; LOG=$ST/s2b_resume.log
RC=/home/ubuntu/jsr-data/runs/train/H53_RC1_s7919 SB=/home/ubuntu/jsr-data/runs/train/H53_S2b_medium_s7919
EN=$(printf 'ckpt-%08d' "$(sed -n 's/.*RC1_STEPS=\([0-9]*\).*/\1/p' $ST/rc1_env.txt)")
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
until ssh -n -o BatchMode=yes -o ConnectTimeout=20 xbabe5 "[ -e ~/jsr-data/protected/H53_RC1_s7919/$EN.pt ] && [ -e ~/jsr-data/protected/H53_RC1_s7919/$EN.sha256 ]" 2>/dev/null; do sleep 15; done
log "RC1 end checkpoint $EN protected; waiting for every RC1 torchrun to exit"
for spec in "xbabe5 0" "xbabe7 1" "xbabe8 2" "xbabe9 3"; do set -- $spec
  until ssh -n -o BatchMode=yes -o ConnectTimeout=20 $1 "! kill -0 \$(cat $RC/PID.node$2) 2>/dev/null" 2>/dev/null; do sleep 5; done
done
for spec in "xbabe5 0" "xbabe7 1" "xbabe8 2" "xbabe9 3"; do set -- $spec
  l=$(ssh -n -o BatchMode=yes $1 "cat $SB/LATEST"); [ "$l" = ckpt-00010818 ] || { log "ABORT: $1 S2b LATEST is $l, not ckpt-00010818"; exit 1; }
done
log "RC1 exited on all four nodes; every S2b LATEST = ckpt-00010818; resuming S2b at world 6"
ENVS=$(cat $ST/s2b_env.txt)
for s in 5 7 8 9; do ssh -n -o BatchMode=yes xbabe$s "$ENVS bash ~/x${s}_s2b.sh 2>&1 | tail -1" | sed -E "s/^/x$s: /; s/[0-9]+-[0-9]+-[0-9]+-[0-9]+/<host>/g" >> $LOG & sleep 1; done; wait
log "S2b relaunched (resume)"
for i in $(seq 60); do
  r=$(ssh -n -o BatchMode=yes xbabe5 "grep -c '\"kind\": \"train\"' $SB/train.jsonl; grep '\"kind\": \"resume\"' $SB/train.jsonl | tail -1 | cut -c1-200")
  n=$(echo "$r" | head -1); [ "${n:-0}" -ge 437 ] && break; sleep 10
done
log "resume record: $(echo "$r" | tail -1)"
log "S2b RESUMED"
