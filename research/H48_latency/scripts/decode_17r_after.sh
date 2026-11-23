#!/usr/bin/env bash
# x9: 17-r's deferred full-dev decodes, alone on the card once H48c-a's decode set is finished (capped cells).
set -u
O=$HOME/jsr-data/evals/burst/h41; R=$HOME/jsr-data/runs/train/H41_chunk_left256_s4321
until [ -e "$O/H48c_a_s1234.DONE" ] || [ -e "$O/H48c_a_s1234.FAILED" ]; do sleep 20; done
sleep 10
until flock --nonblock "$HOME/jsr-data/gpu-leases/gpu-0.lock" true; do sleep 10; done
[ "$(sha256sum "$R/model-00008000.pt" | cut -c1-64)" = "$(cut -c1-64 "$R/model-00008000.sha256")" ] || { echo "REFUSED: checkpoint hash"; exit 1; }
bash "$HOME/fulldev.sh" "$R" 0 28-55
