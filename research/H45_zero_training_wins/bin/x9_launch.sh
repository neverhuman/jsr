#!/usr/bin/env bash
# h45-wins: start a detached job on xbabe9 in its own session and record its PID (= PGID = SID).
N=$1; shift
timeout 20 ssh -o BatchMode=yes -n xbabe9 "cd /home/ubuntu/jsr-data/h45-wins && (setsid bash $* > logs/$N.log 2>&1 < /dev/null & echo \$! > logs/$N.pid) > /dev/null 2>&1 < /dev/null"
sleep 1
pid=$(timeout 20 ssh -o BatchMode=yes -n xbabe9 "cat /home/ubuntu/jsr-data/h45-wins/logs/$N.pid")
echo "$(date -u +%FT%TZ) xbabe9 $N pid=$pid (setsid: pgid=sid=pid) :: $*" | tee -a /mnt/xbabe3-external/jsr/h45-wins/logs/pids.txt
