#!/usr/bin/env bash
# trainer-lane: at T_cut = max(14:00Z, gate (b) PASS 13:24:51Z) = 14:00:00Z: pause S2b at its first exact window boundary
# (boundary.sh), then RC1 from that cut checkpoint (rc1_launch.sh), then the endpoint watcher. Each step aborts on failure.
set -u
D=/mnt/xbabe3-external/jsr/trainer-lane/s2; ST=$D/state; LOG=$ST/rc1_at_cut.log
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
until [ "$(date -u +%s)" -ge "$(date -u -d '2026-11-23 14:00:00' +%s)" ]; do sleep 2; done
log "T_cut reached; pausing S2b at a window boundary"
OUT=$(bash $D/boundary.sh H53_S2b_medium_s7919 s2b "$(cat $ST/s2b_env.txt)" 2>&1); echo "$OUT" | sed -E 's/[0-9]+-[0-9]+-[0-9]+-[0-9]+/<host>/g' >> $LOG
L=$(echo "$OUT" | grep 'BOUNDARY OK' | tail -1)
[ -n "$L" ] || { log "ABORT: no BOUNDARY OK"; exit 1; }
C=$(echo "$L" | awk '{for (i=1;i<=NF;i++) if ($i=="OK") {print $(i+1); exit}}'); H=$(echo "$L" | awk '{for (i=1;i<=NF;i++) if ($i=="OK") {print $(i+2); exit}}')
log "cut = $C $H"
bash $D/rc1_launch.sh "$C" "$H" >> $LOG 2>&1 || { log "ABORT: rc1_launch failed"; exit 1; }
grep -q 'RC1 fired' $ST/rc1_launch.log || { log "ABORT: RC1 not fired"; exit 1; }
cd $D && setsid nohup bash rc1_end_watch.sh > $ST/rc1_end_watch.out 2>&1 < /dev/null &
log "RC1 fired; endpoint watcher started (pid $!)"
