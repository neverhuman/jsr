#!/usr/bin/env bash
# H52: after this host's current medium runner ends, run an extra member list (same cap file / guard).
# usage: chain_after.sh <members_file> <out_tag> <max> <cpu_cap>
set -u
MF=$1; TAG=$2; MAX=$3; CAP=$4
L=/home/ubuntu/jsr-data/librilight; S=$L/student
while pgrep -f "^$L/venv/bin/python -I bin/ll_student_src.py --index $L/meta/medium" > /dev/null; do sleep 30; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) previous medium run ended; starting $TAG"
cd $S
exec nice -n 19 $L/venv/bin/python -I bin/ll_student_src.py --index $L/meta/medium --skip $L/meta/medium/skip_medium.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/medium.tar --subset medium --members-file $MF \
  --engine $S/bin/ultra-engine-eval --loader $S/rt --weights $S/model --work /dev/shm/h52-src-medium-$TAG \
  --out $S/student_src_medium_$TAG.jsonl --max-procs $MAX --start-procs 4 --max-file $S/src_max --min-avail-gb 40 --cpu-cap $CAP
