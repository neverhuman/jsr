#!/usr/bin/env bash
# H52 relay v3 (also catch-up files): from-source student-hyp files from cloud hosts and xbabe2, via H0 /dev/shm, to xbabe5.
set -u
R=/dev/shm/h52/relay; mkdir -p $R
SRC=("xbabe6:jsr-data/librilight/student" "xbabe7:jsr-data/librilight/student" "xbabe8:jsr-data/librilight/student"
     "xbabe9:jsr-data/librilight/student" "xbabe2:jsr-data/h52-student")
while :; do
  for s in "${SRC[@]}"; do
    h=${s%%:*}; mkdir -p $R/$h
    timeout 300 rsync -a "$s/student_src_*.jsonl" $R/$h/ 2>/dev/null
  done
  for sub in small medium; do
    ls $R/*/student_src_${sub}_*.jsonl >/dev/null 2>&1 || continue
    timeout 300 ssh -o BatchMode=yes xbabe5 "mkdir -p ~/jsr-data/librilight/shards/raw/librilight.$sub/student_src"
    for h in $(ls $R); do
      ls $R/$h/student_src_${sub}_*.jsonl >/dev/null 2>&1 || continue
      # keep host in the name so two hosts with the same rem never overwrite each other
      for f in $R/$h/student_src_${sub}_*.jsonl; do
        timeout 600 rsync -a "$f" "xbabe5:jsr-data/librilight/shards/raw/librilight.$sub/student_src/${h}_$(basename $f)"
      done
    done
    echo "$(date -u +%H:%M:%SZ) relayed $sub: $(cat $R/*/student_src_${sub}_*.jsonl | wc -l) lines"
  done
  sleep 120
done
