#!/usr/bin/env bash
# trainer-lane: launch receipts for a world-6 run (x5 x2, x7 x2, x8, x9): per node commit / tree / init / GPU count / torchrun
# PID / NCCL transport; from x5's train.jsonl the start + ddp records and the first train record. Host names are not printed.
#   receipts.sh <run_name>
set -u
RUN=$1; R=/home/ubuntu/jsr-data/runs/train/$RUN
for spec in "xbabe5 x5 0" "xbabe7 x7 1" "xbabe8 x8 2" "xbabe9 x9 3"; do set -- $spec
  ssh -n -o BatchMode=yes $1 "python3 -c \"
import json; j = json.load(open('$R/launch.node$3.json'))
print('| $3 | $2 GPU %s | %s | %s… | %s… | %s |' % (j['gpus'], j['commit'][:7], j['tree_sha256'][:8], j['init_sha256'][:8], open('$R/PID.node$3').read().strip()))
\"; tr '\n' ' ' < $R/nccl_transport.node$3.txt 2>/dev/null | sed 's/^/  nccl: /'; echo"
done
ssh -n -o BatchMode=yes xbabe5 "grep -m1 '\"kind\": \"start\"' $R/train.jsonl | cut -c1-300; grep -m1 '\"kind\": \"ddp\"' $R/train.jsonl | cut -c1-400; grep -m1 '\"kind\": \"train\"' $R/train.jsonl | python3 -c \"
import json, sys; r = json.loads(sys.stdin.read())
print('first train record: step', r['step'], 'pos', r['pos'], 'loss %.4f lr %.3e audio-h/h %.0f data_wait %.2f%% peak %s GB' % (r['loss'], r['lr'], r['audio_h_per_h'], r['data_wait_pct'], r['gpu_peak_gb']))\""
