#!/usr/bin/env bash
# h45-wins: second batch of read-only copies (A1L final; H41 seed-4321 replicates for the bundle seed check).
P=/mnt/xbabe3-external/jsr/h45-wins/bin/pull.sh
while kill -0 3420994 2>/dev/null; do sleep 10; done
$P xbabe9 /home/ubuntu/jsr-data/runs/train/A1L_dp0.005/model-00008000.pt A1L-model-00008000.pt
$P xbabe5 /home/ubuntu/jsr-data/runs/train/H41_chunk_predctx4_s4321/model-00008000.pt H41_chunk_predctx4_s4321-model-00008000.pt
$P xbabe5 /home/ubuntu/jsr-data/runs/train/H41_chunk_specaug_s4321/model-00008000.pt H41_chunk_specaug_s4321-model-00008000.pt
echo "$(date -u +%FT%TZ) MORE-DONE"
