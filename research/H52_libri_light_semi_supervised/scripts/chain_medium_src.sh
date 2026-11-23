#!/usr/bin/env bash
# H52: after this host's small from-source run ends, start the medium one (same guard-controlled cap file).
# usage: chain_medium_src.sh <rem> <max_procs> <cpu_cap>
set -u
R=$1; MAX=$2; CAP=$3
L=/home/ubuntu/jsr-data/librilight; S=$L/student; M=$L/meta/medium
while pgrep -f "^$L/venv/bin/python -I bin/ll_student_src.py --index $L/meta/small" > /dev/null; do sleep 30; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) small run ended; starting medium r$R"
cd $S
exec nice -n 19 $L/venv/bin/python -I bin/ll_student_src.py --index $M --skip $M/skip_medium.tsv \
  --source https://dl.fbaipublicfiles.com/librilight/data/medium.tar --subset medium --mod 4 --rem $R \
  --engine $S/bin/ultra-engine-eval --loader $S/rt --weights $S/model --work /dev/shm/h52-src-medium \
  --out $S/student_src_medium_r$R.jsonl --max-procs $MAX --start-procs 4 --max-file $S/src_max --min-avail-gb 40 --cpu-cap $CAP
