#!/usr/bin/env bash
# Sample mean CPU (5 s window, all processes) and GPU utilisation on every xbabe host; one JSON line
# per host per round to stdout and to the data-drive log. Owner rule: keep every host under 90%.
set -uo pipefail
LOG=${FLEET_UTIL_LOG:-/mnt/xbabe3-external/jsr/logs/fleet_util.jsonl}
HOSTS=${FLEET_HOSTS:-"local xbabe0 xbabe1 xbabe2"}
probe='read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e)); gs=0; gn=0;
for _ in 1 2 3 4 5; do v=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1);
  if [ -n "$v" ]; then gs=$((gs+v)); gn=$((gn+1)); fi; sleep 1; done;
read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e));
cpu=$(( 1000 * ( (t2-t1) - (i2-i1) ) / (t2-t1) ));
if [ "$gn" -gt 0 ]; then gpu=$((gs/gn)); else gpu=none; fi;
echo "$cpu $gpu"'
while true; do
  for h in $HOSTS; do
    if [ "$h" = local ]; then out=$(bash -c "$probe"); name=$(hostname); else
      out=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 -o LogLevel=ERROR "$h" "bash -c '$probe'" 2>/dev/null); name=$h; fi
    read -r cpu gpu <<<"${out:-x x}"
    line=$(printf '{"t":"%s","host":"%s","cpu_pct":%s,"gpu_pct":%s}' "$(date -u +%FT%TZ)" "$name" \
      "$( [ "$cpu" = x ] && echo null || echo "scale=1; $cpu/10" | bc)" "$( [[ "$gpu" =~ ^[0-9]+$ ]] && echo "$gpu" || echo null)")
    echo "$line" >> "$LOG"
    echo "$line"
  done
  sleep "${FLEET_UTIL_PERIOD:-60}"
done
