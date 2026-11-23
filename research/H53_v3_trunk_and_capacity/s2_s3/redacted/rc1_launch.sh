#!/usr/bin/env bash
# trainer-lane: RC1 (Amendment 7) after the S2b cut. Run on H0 after `boundary.sh H53_S2b_medium_s7919 s2b "<S2b env>"` printed
# BOUNDARY OK. Computes D = max(10,000 audio-h, 0.15 x E_main) (E_main = S2a + S2b logged audio, the counted attempt only),
# total_steps = ceil(D / mean audio per step of S2b's last 1,000 logged steps), stages the cut checkpoint on all four nodes under
# one hash, and fires H53_RC1_s7919 at world 6.
#   rc1_launch.sh <cut_ckpt_path_on_x5> <cut_sha256>
set -u
C=$1 CSHA=$2
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; LOG=$ST/rc1_launch.log
SEED=/home/ubuntu/jsr-data/seeds/H53_S2b_cut N=$(basename "$C"); I=$SEED/$N
declare -A LAN=([xbabe7]=${X7_LAN:?} [xbabe8]=${X8_LAN:?} [xbabe9]=${X9_LAN:?})
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
read -r EA EB APS < <(ssh -n -o BatchMode=yes xbabe5 "python3 - <<'PY'
import json
def recs(n):
    return [json.loads(l) for l in open('/home/ubuntu/jsr-data/runs/train/%s/train.jsonl' % n) if '\"kind\": \"train\"' in l]
a = recs('H53_S2_small_w6_s7919'); b = recs('H53_S2b_medium_s7919')
ea = sum(r['audio_s'] for r in a) / 3600; eb = sum(r['audio_s'] for r in b) / 3600
tail = b[-40:]  # 40 records x 25 steps = the last 1,000 logged steps
aps = sum(r['audio_s'] for r in tail) / (tail[-1]['step'] - tail[0]['step'] + 25)
print('%.1f %.1f %.2f' % (ea, eb, aps))
PY")
D=$(python3 -c "print(max(10000.0, 0.15 * ($EA + $EB)))"); STEPS=$(python3 -c "import math; print(math.ceil($D * 3600 / $APS))")
log "E_main = S2a $EA + S2b $EB audio-h; D = $D audio-h; audio per step $APS s -> RC1 total_steps $STEPS"
EN=$(printf "ckpt-%08d.pt" "$STEPS"); ssh -n -o BatchMode=yes xbabe5 "mkdir -p ~/jsr-data/protected/H53_RC1_s7919 && touch ~/jsr-data/protected/H53_RC1_s7919/KEEP && (grep -qx $EN ~/jsr-data/protected/H53_RC1_s7919/KEEP || echo $EN >> ~/jsr-data/protected/H53_RC1_s7919/KEEP)" && log "RC1 end checkpoint $EN pre-registered in its protected KEEP"
[ "$(python3 -c "print(int($D == 10000.0))")" = 1 ] || { log "REFUSED: D > 10k audio-h, so the EMA (0.999134 = tau 2,500 audio-h) must be recomputed by hand"; exit 1; }
ssh -n -o BatchMode=yes xbabe5 "mkdir -p $SEED && ln -f $C $I && cp ${C%.pt}.sha256 ${I%.pt}.sha256 && (grep -qx $N ~/jsr-data/protected/H53_S2b_medium_s7919/KEEP || echo $N >> ~/jsr-data/protected/H53_S2b_medium_s7919/KEEP)"
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
for h in xbabe7 xbabe8 xbabe9; do
  ssh -n -o BatchMode=yes $h "mkdir -p $SEED"
  ssh -A -n -o BatchMode=yes xbabe5 "taskset -c 56-59 rsync -a -e 'ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com' $I ${I%.pt}.sha256 ubuntu@${LAN[$h]}:$SEED/" &
done; wait
for h in xbabe5 xbabe7 xbabe8 xbabe9; do
  got=$(ssh -n -o BatchMode=yes $h "sha256sum $I | cut -c1-64; tr -d ' \n' < ${I%.pt}.sha256" | tr '\n' ' '); read -r g m <<< "$got"
  [ "$g" = "$CSHA" ] && [ "$m" = "$CSHA" ] || { log "REFUSED: $h init copy differs ($g / $m)"; exit 1; }
done
log "cut $N staged on all four nodes (${CSHA:0:16})"
eval "$(cat $ST/s2b_env.txt)"
ENVS="RC1_INIT=$I RC1_INIT_SHA256=$CSHA RC1_STEPS=$STEPS S2B_BW=$S2B_BW S2B_MEDIUM_SUM16=$S2B_MEDIUM_SUM16 S2B_WG=$S2B_WG"
echo "$ENVS" > $ST/rc1_env.txt
for s in 5 7 8 9; do ssh -n -o BatchMode=yes xbabe$s "$ENVS bash ~/x${s}_rc1.sh 2>&1 | tail -2" | sed -E "s/^/x$s: /; s/[0-9]+-[0-9]+-[0-9]+-[0-9]+/<host>/g" | tee -a $LOG & sleep 1; done; wait
log "RC1 fired"
