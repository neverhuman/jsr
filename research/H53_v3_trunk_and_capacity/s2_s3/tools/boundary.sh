#!/usr/bin/env bash
# H0: stop S2a (world 6) at an exact sampler-window boundary (Amendment 6). SIGTERM node 0's torchrun (recorded PID, command
# line checked; the stop is all-reduced, every node saves), read the checkpoint's position, count the window's global batches at
# S2a's shape (window_len.py, the pin's packing), relaunch all four nodes with MAX_STEPS = batches left (the trainer saves at
# the max-steps stop), and verify the boundary. Prints BOUNDARY OK <ckpt> <sha256> step <S> (<E>, <W>, <B>).
# (Predecessor's draft, hardened and generalised by the new trainer-lane: every intermediate value is checked before use.)
#   boundary.sh <run_name> <node script suffix: s2w6 | s2b> ["<env assignments the node scripts need>"]
set -u
RUN=$1 SUF=$2 ENVS=${3:-}
R=/home/ubuntu/jsr-data/runs/train/$RUN
NODES="xbabe5:x5:0 xbabe7:x7:1 xbabe8:x8:2 xbabe9:x9:3"
LOG=/mnt/xbabe3-external/jsr/trainer-lane/s2/state/boundary_$RUN.log
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
wait_all() {  # every node's recorded torchrun PID gone (<= 10 min)
  local bad=0
  for spec in $NODES; do IFS=: read -r h s n <<< "$spec"
    r=$(ssh -n -o BatchMode=yes "$h" "P=\$(cat $R/PID.node$n); for i in \$(seq 300); do kill -0 \$P 2>/dev/null || break; sleep 2; done; kill -0 \$P 2>/dev/null && echo ALIVE || echo \"stopped LATEST \$(cat $R/LATEST)\"")
    log "$h node $n: $r"; [ "${r%% *}" = stopped ] || bad=1
  done
  return $bad
}
pos() { ssh -n -o BatchMode=yes xbabe5 "cd ~/jsr/.agent/worktrees/trainer-s2/python/ultra && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$PWD taskset -c 56-59 ~/jsr-data/venvs/train/bin/python -c \"
from train import checkpoint
p = checkpoint.load(checkpoint.latest('$R'), restore_rng=False); q = p['position']
print(p['step'], q['epoch'], q['window'], q['batch'])\""; }
isint() { [[ "${1:-x}" =~ ^[0-9]+$ ]]; }
P0=$(ssh -n -o BatchMode=yes xbabe5 "cat $R/PID.node0")
ARGS=$(ssh -n -o BatchMode=yes xbabe5 "ps -o args= -p $P0")
case "$ARGS" in *torch.distributed.run*"--node_rank 0"*"/$RUN "*) ;; *) log "REFUSED: PID $P0 is not $RUN's node-0 torchrun"; exit 1;; esac
ssh -n -o BatchMode=yes xbabe5 "kill -TERM $P0" && log "SIGTERM node 0 torchrun $P0"
wait_all || { log "ABORT: a node did not stop"; exit 1; }
read -r S E W B < <(pos)
isint "$S" && isint "$E" && isint "$W" && isint "$B" || { log "ABORT: bad position '$S $E $W $B'"; exit 1; }
NB=$(ssh -n -o BatchMode=yes xbabe5 "cd ~/jsr/.agent/worktrees/trainer-s2/python/ultra && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$PWD JSR_DATA=~/jsr-data taskset -c 56-59 ~/jsr-data/venvs/train/bin/python ~/h53/window_len.py $R $E $W")
isint "$NB" && [ "$NB" -gt 0 ] || { log "ABORT: bad window length '$NB'"; exit 1; }
log "stop checkpoint step $S at ($E, $W, $B); window ($E, $W) has $NB global batches at $RUN's shape"
if [ "$B" -gt 0 ] && [ "$B" -lt "$NB" ]; then
  for spec in $NODES; do IFS=: read -r h s n <<< "$spec"
    ssh -n -o BatchMode=yes "$h" "$ENVS MAX_STEPS=$((NB - B)) bash ~/${s}_${SUF}.sh 2>&1 | tail -1" | sed "s/^/$s: /" | tee -a $LOG & sleep 1
  done; wait
  sleep 20
  wait_all || { log "ABORT: a node did not finish the window"; exit 1; }
  read -r S2 E2 W2 B2 < <(pos)
  log "boundary checkpoint step $S2 at ($E2, $W2, $B2)"
  [ "$E2" = "$E" ] && [ "$W2" = "$W" ] && [ "$B2" = "$NB" ] && [ "$S2" = "$((S + NB - B))" ] || { log "NOT AT BOUNDARY"; exit 1; }
  S=$S2 B=$B2
fi
L=$(ssh -n -o BatchMode=yes xbabe5 "cat $R/LATEST"); H=$(ssh -n -o BatchMode=yes xbabe5 "sha256sum $R/$L.pt | cut -c1-64")
[ "$H" = "$(ssh -n -o BatchMode=yes xbabe5 "cut -c1-64 $R/$L.sha256")" ] || { log "ABORT: checkpoint hash mismatch"; exit 1; }
log "BOUNDARY OK $R/$L.pt $H step $S ($E, $W, $B)"
