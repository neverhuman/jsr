#!/usr/bin/env bash
# trainer-lane (Amendment 8): the S3 switch. Pause the trunk (S2b) at its next exact window boundary, stage that checkpoint as
# S3's init under one hash on all four nodes (S2b KEEP-listed), and fire H53_S3_large_s7919 at world 6.
set -u
D=/mnt/xbabe3-external/jsr/trainer-lane/s2; ST=$D/state; LOG=$ST/s3_switch.log
SEED=/home/ubuntu/jsr-data/seeds/H53_S2b_s3cut
declare -A LAN=([xbabe7]=${X7_LAN:?} [xbabe8]=${X8_LAN:?} [xbabe9]=${X9_LAN:?})
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
OUT=$(bash $D/boundary.sh H53_S2b_medium_s7919 s2b "$(cat $ST/s2b_env.txt)" 2>&1); echo "$OUT" | sed -E 's/[0-9]+-[0-9]+-[0-9]+-[0-9]+/<host>/g' >> $LOG
L=$(echo "$OUT" | grep 'BOUNDARY OK' | tail -1); [ -n "$L" ] || { log "ABORT: no BOUNDARY OK"; exit 1; }
C=$(echo "$L" | awk '{for (i=1;i<=NF;i++) if ($i=="OK") {print $(i+1); exit}}'); H=$(echo "$L" | awk '{for (i=1;i<=NF;i++) if ($i=="OK") {print $(i+2); exit}}')
N=$(basename "$C"); I=$SEED/$N
log "trunk paused at $C ($H)"
ssh -n -o BatchMode=yes xbabe5 "mkdir -p $SEED && ln -f $C $I && cp ${C%.pt}.sha256 ${I%.pt}.sha256 && (grep -qx $N ~/jsr-data/protected/H53_S2b_medium_s7919/KEEP || echo $N >> ~/jsr-data/protected/H53_S2b_medium_s7919/KEEP)"
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
for h in xbabe7 xbabe8 xbabe9; do ssh -n -o BatchMode=yes $h "mkdir -p $SEED"
  ssh -A -n -o BatchMode=yes xbabe5 "taskset -c 56-59 rsync -a -e 'ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com' $I ${I%.pt}.sha256 ubuntu@${LAN[$h]}:$SEED/" & done; wait
for h in xbabe5 xbabe7 xbabe8 xbabe9; do
  got=$(ssh -n -o BatchMode=yes $h "sha256sum $I | cut -c1-64; tr -d ' \n' < ${I%.pt}.sha256" | tr '\n' ' '); read -r g m <<< "$got"
  [ "$g" = "$H" ] && [ "$m" = "$H" ] || { log "REFUSED: $h init copy differs ($g / $m)"; exit 1; }
done
log "init $N staged on all four nodes (${H:0:16})"
ENVS="S3_INIT=$I S3_INIT_SHA256=$H S3_BW=0.525641 S3_LARGE_SUM16=e6077dcba9b75b54 S2B_MEDIUM_SUM16=7e87f000be40a0d0"
echo "$ENVS" > $ST/s3_env.txt
for s in 5 7 8 9; do ssh -n -o BatchMode=yes xbabe$s "$ENVS bash ~/x${s}_s3.sh 2>&1 | tail -2" | sed -E "s/^/x$s: /; s/[0-9]+-[0-9]+-[0-9]+-[0-9]+/<host>/g" | tee -a $LOG & sleep 1; done; wait
log "S3 fired"
