#!/usr/bin/env bash
# H52 pipeline snapshot (read-only).
S=${1:-small}; D=/dev/shm/h52/raw/librilight.$S; L=/mnt/xbabe3-external/jsr/librilight/logs
c() { ls $D 2>/dev/null | grep -c "$1"; }
echo "$(date -u +%H:%M:%SZ) H0[$S]: ready $(c '\.ready$') labeled $(c '\.labeled$') uploaded $(c '\.uploaded$') parquet $(c '\.parquet$') stud.done $(c 'student.done$') staged $(du -sh /dev/shm/h52 2>/dev/null | cut -f1)"
tail -1 $L/pass2_$S.log | cut -c1-170
tail -1 $L/uploader_$S.log | cut -c1-120
for h in xbabe2 xbabe1; do timeout 20 ssh -o BatchMode=yes $h "grep -E 'busy=|done ' ~/jsr-data/h52-student/logs/student_$S.log | tail -1" 2>/dev/null | cut -c1-150 | sed "s/^/$h: /"; done
timeout 20 ssh -o BatchMode=yes xbabe5 "grep -E 'busy=|done ' ~/jsr-data/librilight/student/logs/student_$S.log | tail -1; tail -1 ~/jsr-data/librilight/student/logs/guard.log; ls ~/jsr-data/librilight/shards/raw/librilight.$S/ | grep -c '\.ready$'" 2>/dev/null | cut -c1-170 | sed "s/^/x5: /"
