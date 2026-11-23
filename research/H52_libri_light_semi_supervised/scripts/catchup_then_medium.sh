#!/usr/bin/env bash
# H52: small catch-up (listed members) first, then this host's medium share; one runner at a time so
# the guard-written cap file bounds the engines. usage: catchup_then_medium.sh <tag> <rem> <max> <cpu_cap>
set -u
TAG=$1; R=$2; MAX=$3; CAP=$4
L=/home/ubuntu/jsr-data/librilight; S=$L/student; PY=$L/venv/bin/python
cd $S
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) catch-up small ($(wc -l < $S/catchup_small.txt) members)"
nice -n 19 $PY -I bin/ll_student_src.py --index $L/meta/small --skip $L/meta/small/skip_small.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/small.tar --subset small --members-file $S/catchup_small.txt \
  --engine $S/bin/ultra-engine-eval --loader $S/rt --weights $S/model --work /dev/shm/h52-src-catchup \
  --out $S/student_src_small_catchup_$TAG.jsonl --max-procs $MAX --start-procs 8 --max-file $S/src_max --min-avail-gb 40 --cpu-cap $CAP
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) catch-up done; medium r$R"
exec nice -n 19 $PY -I bin/ll_student_src.py --index $L/meta/medium --skip $L/meta/medium/skip_medium.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/medium.tar --subset medium --mod 4 --rem $R \
  --engine $S/bin/ultra-engine-eval --loader $S/rt --weights $S/model --work /dev/shm/h52-src-medium \
  --out $S/student_src_medium_r$R.jsonl --max-procs $MAX --start-procs 4 --max-file $S/src_max --min-avail-gb 40 --cpu-cap $CAP
