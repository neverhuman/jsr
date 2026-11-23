#!/usr/bin/env bash
# h45-wins: start a detached job on xbabe1 in its own session and record its PID (= PGID = SID).
#   x1_launch.sh <name> <script path relative to /mnt/amphora-backup/h45-wins> [args...]
N=$1; shift
timeout 20 ssh -o BatchMode=yes -n xbabe1 "cd /mnt/amphora-backup/h45-wins && (setsid bash $* > logs/$N.log 2>&1 < /dev/null & echo \$! > logs/$N.pid) > /dev/null 2>&1 < /dev/null"
sleep 1
pid=$(timeout 20 ssh -o BatchMode=yes -n xbabe1 "cat /mnt/amphora-backup/h45-wins/logs/$N.pid")
echo "$(date -u +%FT%TZ) xbabe1 $N pid=$pid (setsid: pgid=sid=pid) :: $*" | tee -a /mnt/xbabe3-external/jsr/h45-wins/logs/pids.txt
