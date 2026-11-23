#!/usr/bin/env bash
# GDR value-correctness check: x5 GPU${G5:-1} (rank 0) + x9 GPU0 (rank 1), exact-sum verify on every size.
# Run from H0. Holds the GPU0 lease on both hosts for the whole sweep (<= 4 min); refuses if either GPU0 is busy.
# Records the negotiated transport per config (NCCL_DEBUG=INFO, INIT,NET) next to the bandwidth rows.
set -uo pipefail
G5=${G5:-1}
TS=$(date -u +%H%M%S)v
OUT=/mnt/xbabe3-external/jsr/h50-fleet/nccl/xhost_$TS.jsonl
LOGD=/mnt/xbabe3-external/jsr/h50-fleet/nccl/xhost_$TS; mkdir -p $LOGD
X5=<xbabe5-private>
PYB='~/jsr-data/venvs/train/bin/python ~/jsr-data/h50-fleet/bin/nccl_bench.py --sizes-mb 12.5,25,116,232 --dtype bf16 --iters 12 --warmup 3 --verify'
for h in xbabe5 xbabe9; do
  gi=0; [[ $h == xbabe5 ]] && gi=$G5
  busy=$(ssh $h "nvidia-smi -i $gi --query-compute-apps=pid --format=csv,noheader")
  [[ -z $busy ]] || { echo "$h GPU0 busy: $busy"; exit 1; }
done
hold() { ssh $1 "cd ~/jsr-data/h50-fleet && (nohup flock -n ~/jsr-data/gpu-leases/gpu-$2.lock -c 'echo \$\$ > logs/lease_gpu$2.pid; sleep 300' > /dev/null 2>&1 < /dev/null &) ; sleep 0.5; cat logs/lease_gpu$2.pid 2>/dev/null"; }
L5=$(hold xbabe5 $G5); L9=$(hold xbabe9 0)
[[ -n $L5 && -n $L9 ]] || { echo "could not take both GPU0 leases ($L5/$L9): ABORT"; for spec in "xbabe5:$L5" "xbabe9:$L9"; do h=${spec%%:*}; p=${spec#*:}; [[ -n $p ]] && ssh $h "pkill -P $p -x sleep; kill $p"; done; exit 1; }
cfg() { # tag env...
  local tag=$1; shift
  local port=$((29700 + RANDOM % 200))
  local envs="$* NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET MASTER_ADDR=$X5 MASTER_PORT=$port WORLD_SIZE=2 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0"
  local py=$PYB; [[ $tag == *-fp32 ]] && py=${PYB/--dtype bf16/--dtype fp32}
  ssh xbabe9 "env $envs CUDA_VISIBLE_DEVICES=0 RANK=1 timeout 45 $py --tag $tag 2>&1 | grep -E 'NCCL INFO (NET|Channel|Connected|Using|comm)' | head -40" > $LOGD/$tag.x9.log &
  ssh xbabe5 "env $envs CUDA_VISIBLE_DEVICES=$G5 RANK=0 timeout 45 $py --tag $tag 2>&1" > $LOGD/$tag.x5.log
  wait
  grep '^{' $LOGD/$tag.x5.log >> $OUT
  local via=$(grep -o 'via NET/[A-Za-z_0-9]*' $LOGD/$tag.x5.log | sort | uniq -c | tr '\n' ' ')
  echo "$tag: $via"
}
IB="NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3"
cfg v-default
cfg v-gdr NCCL_NET_GDR_LEVEL=SYS
cfg v-gdr-fp32 NCCL_NET_GDR_LEVEL=SYS
# release the leases: kill each holder shell's sleep child (by parent PID), then the shell
for spec in "xbabe5:$L5" "xbabe9:$L9"; do h=${spec%%:*}; p=${spec#*:}
  [[ -n $p ]] && ssh $h "pkill -P $p -x sleep; kill $p 2>/dev/null; rm -f ~/jsr-data/h50-fleet/logs/lease_gpu*.pid"
done
echo "$OUT"
