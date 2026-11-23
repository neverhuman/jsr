#!/usr/bin/env bash
# H52: relay the small from-source student-hyp files (cloud hosts cannot reach each other) via H0's
# /dev/shm to xbabe5 next to the shards. Append-only jsonl; rsync moves only the new blocks.
set -u
R=/dev/shm/h52/relay; mkdir -p $R
while :; do
  for h in xbabe6 xbabe7 xbabe8 xbabe9; do
    mkdir -p $R/$h
    timeout 300 rsync -a "$h:jsr-data/librilight/student/student_src_*.jsonl" $R/$h/ 2>/dev/null
  done
  for s in small medium; do
    ls $R/*/student_src_${s}_r*.jsonl >/dev/null 2>&1 || continue
    timeout 300 ssh -o BatchMode=yes xbabe5 "mkdir -p ~/jsr-data/librilight/shards/raw/librilight.$s/student_src"
    timeout 600 rsync -a $R/*/student_src_${s}_r*.jsonl xbabe5:jsr-data/librilight/shards/raw/librilight.$s/student_src/ \
      && echo "$(date -u +%H:%M:%SZ) relayed $s: $(cat $R/*/student_src_${s}_r*.jsonl | wc -l) lines"
  done
  sleep 120
done
