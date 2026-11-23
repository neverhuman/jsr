#!/usr/bin/env bash
# Two-way sync of a sweep's rows and hyps between H0 (data drive) and x6, every 90 s, so each
# host's workers skip layers the other finished. Each side writes only its own row files.
tag=$1; H0=/mnt/xbabe3-external/jsr/h51-pq/results/engine_sens/$tag
while true; do
  rsync -a "xbabe2:h51-pq/results/engine_sens/$tag/" "$H0/" --exclude '*h0.jsonl' 2>/dev/null
  rsync -a "$H0/" "xbabe2:h51-pq/results/engine_sens/$tag/" --include 'rows.*h0.jsonl' --include 'hyps/' --include 'hyps/**' --exclude '*' 2>/dev/null
  sleep 90
done
