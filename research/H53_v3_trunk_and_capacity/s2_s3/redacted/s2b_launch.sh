#!/usr/bin/env bash
# trainer-lane (new agent): stage S2b's init and fire H53_S2b_medium_s7919 at world 6 (Amendment 6), run on H0 AFTER
# s2a_boundary.sh printed BOUNDARY OK and medium is staged + verified on all four nodes.
#   s2b_launch.sh <boundary_ckpt_path_on_x5> <boundary_sha256> <medium_manifest_sha256> <extra_base_weight> <window_groups>
set -u
B=$1 BSHA=$2 MSUM=$3 BW=$4 WG=$5
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; LOG=$ST/s2b_launch.log
S2A=H53_S2_small_w6_s7919 RUN=H53_S2b_medium_s7919 SEED=/home/ubuntu/jsr-data/seeds/H53_S2a_boundary
N=$(basename "$B"); I=$SEED/$N
declare -A LAN=([xbabe7]=${X7_LAN:?} [xbabe8]=${X8_LAN:?} [xbabe9]=${X9_LAN:?})
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
# 1. one init file, one hash, the same absolute path on every node (x5: hard link of the boundary checkpoint + its marker)
ssh -n -o BatchMode=yes xbabe5 "mkdir -p $SEED && ln -f $B $I && cp ${B%.pt}.sha256 ${I%.pt}.sha256 && grep -qx $N ~/jsr-data/protected/$S2A/KEEP || echo $N >> ~/jsr-data/protected/$S2A/KEEP"
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
RSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com"
for h in xbabe7 xbabe8 xbabe9; do
  ssh -n -o BatchMode=yes $h "mkdir -p $SEED"
  ssh -A -n -o BatchMode=yes xbabe5 "taskset -c 56-59 rsync -a -e '$RSH' $I ${I%.pt}.sha256 ubuntu@${LAN[$h]}:$SEED/" &
done; wait
ok=1
for h in xbabe5 xbabe7 xbabe8 xbabe9; do
  got=$(ssh -n -o BatchMode=yes $h "sha256sum $I | cut -c1-64; tr -d ' \n' < ${I%.pt}.sha256" | tr '\n' ' ')
  read -r g m <<< "$got"; [ "$g" = "$BSHA" ] && [ "$m" = "$BSHA" ] && log "$h init OK $N ${g:0:16}" || { log "$h init BAD file=$g marker=$m"; ok=0; }
done
[ $ok = 1 ] || { log "REFUSED: init copies differ"; exit 1; }
# the env every S2b node script needs (also for the stage-2 pause / resume and any restart)
echo "S2B_INIT=$I S2B_INIT_SHA256=$BSHA S2B_BW=$BW S2B_MEDIUM_SUM16=${MSUM:0:16} S2B_WG=$WG" > $ST/s2b_env.txt
# 2. fire: node 0 first, then the others; each node's launcher re-checks pin, tree hash, init, both manifests
for s in 5 7 8 9; do
  ssh -n -o BatchMode=yes xbabe$s "$(cat $ST/s2b_env.txt) bash ~/x${s}_s2b.sh 2>&1 | tail -2" | sed "s/^/x$s: /" | tee -a $LOG &
  sleep 1
done; wait
log "fired; waiting for the DDP record"
for i in $(seq 90); do
  r=$(ssh -n -o BatchMode=yes xbabe5 "grep -h '\"kind\": \"ddp\"\|\"kind\": \"start\"\|\"kind\": \"train\"' ~/jsr-data/runs/train/$RUN/train.jsonl 2>/dev/null | head -4")
  echo "$r" | grep -q '"kind": "train"' && break
  sleep 10
done
echo "$r" > $ST/s2b_first_records.jsonl
log "first records saved to $ST/s2b_first_records.jsonl"
