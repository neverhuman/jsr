#!/usr/bin/env bash
# H53 v3 trunk / segment / release-cooldown / capacity-branch launcher: this node's share of ONE torchrun job,
# synchronous FP32 NCCL DDP (production per codex-lead 06:03Z), tuned NCCL profile frozen by h50-fleet 06:50Z.
# Draft by h53-trunk for trainer-lane to adopt as train/ops/h53_trunk.sh (trainer-lane launches; h53 launches nothing).
#
#   h53_trunk.sh <node_rank> <nnodes> <gpu_list> <cpu_list> <run_name> <master_ip> <set_file> [key=value ...]
#
# Required env:
#   PIN_TREE_SHA256  python/ultra *.py content sha256 (LC_ALL=C sorted file list) of the pinned commit
#   PIN_COMMIT       the pinned commit (HEAD of $TREE must equal it; python/ultra must be clean)
#   INIT, INIT_SHA256  init checkpoint / grown seed and its digest (also checked against its .sha256 marker)
# Optional env:
#   TREE (default ~/jsr/.agent/worktrees/trainer)   SPLITS (default: the 960 h)   MAX_STEPS (segment length)
#   NCCL_PROFILE=tuned (default) | a1env (comparator only)   MASTER_PORT (default 29753)
#   REQUIRE_CLEAN=1 (default) | 0 for A1L's tree 99bce400 (d5f231e + its uncommitted local diff; S1, Amendment 1)
#   FOREIGN_MEM_MAX_PCT (default 15): refuse a GPU that already holds more memory than this (the co-tenant project / other jobs)
# The trainer resumes from <run_dir>/LATEST when it exists (same command restarts a stopped node). Run dirs are
# host-local: a node that joins at a restart (x7 / x8 when the co-tenant project frees them) needs rank 0's LATEST + checkpoint copied
# (sha256-checked) into its own run dir first, or the start-state check refuses (A1's 20:17Z restart).
# Elastic join / leave keeps the global batch (ddp_global_batches) so the optimisation does not change; it is a
# restart at a 500-step checkpoint, recorded in the chat with the new world size.
# GPU cap: the trainer's synchronised step-boundary throttle (ddp_util_target in the set file); never SIGSTOP a rank.
set -uo pipefail
if [ "$#" -lt 7 ]; then sed -n '2,20p' "$0"; exit 2; fi
NR=$1 NN=$2 GPUS=$3 CPUS=$4 NAME=$5 MASTER=$6 SETF=$7
shift 7
W=${TREE:-$HOME/jsr/.agent/worktrees/trainer}; D=$HOME/jsr-data; R=$D/runs/train/$NAME
mkdir -p "$R"
LOG="$R/launch.node$NR.log"
die() { echo "$(date -u +%FT%TZ) REFUSED: $*" | tee -a "$LOG" >&2; exit 1; }

# 1. code identity: pinned commit, clean tree, LC_ALL=C content hash (same rule as h41 / h48 / h50 launchers)
[ "$(git -C "$W" rev-parse HEAD)" = "${PIN_COMMIT:?set PIN_COMMIT}" ] || die "HEAD of $W != PIN_COMMIT"
# A1L's tree (S1 under Amendment 1) is d5f231e + an uncommitted local diff: run it with REQUIRE_CLEAN=0. Its identity is the
# LC_ALL=C .py content hash 99bce400... (the convention of every launcher pinned to that tree; the non-.py ops scripts in the
# local diff differ in bytes between hosts and are not used here). Trees from S2 on (#80's commit) must be clean.
if [ "${REQUIRE_CLEAN:-1}" = 1 ]; then
  [ -z "$(git -C "$W" status --porcelain python/ultra)" ] || die "python/ultra has local changes (REQUIRE_CLEAN=1)"
fi
GOT=$(cd "$W" && git ls-files -co --exclude-standard python/ultra | grep -E '\.py$' | LC_ALL=C sort | xargs sha256sum \
      | sha256sum | cut -c1-64)
[ "$GOT" = "${PIN_TREE_SHA256:?set PIN_TREE_SHA256}" ] || die "python/ultra content sha256 $GOT != $PIN_TREE_SHA256"

# 2. init digest (file and its completion marker)
[ -f "${INIT:?set INIT}" ] || die "no init file $INIT"
IGOT=$(sha256sum "$INIT" | cut -c1-64)
[ "$IGOT" = "${INIT_SHA256:?set INIT_SHA256}" ] || die "init digest $IGOT != $INIT_SHA256"
[ "$(tr -d ' \n' < "${INIT%.pt}.sha256")" = "$IGOT" ] || die "init .sha256 marker disagrees"

# 3. co-tenancy: every chosen GPU must be (nearly) empty before we start; we never touch other processes
MAXF=${FOREIGN_MEM_MAX_PCT:-15}
for g in ${GPUS//,/ }; do
  pct=$(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' '{printf "%d", 100 * $1 / $2}')
  [ "$pct" -lt "$MAXF" ] || die "GPU $g already holds ${pct}% memory (another job / project): ask fleet-sched"
done

# 4. NCCL transport profile (frozen; drop NCCL_IB_* / socket knobs, inert on this fleet per h50-fleet 06:50Z)
export NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
case "${NCCL_PROFILE:-tuned}" in
  tuned) export NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3 NCCL_NET_GDR_LEVEL=SYS ;;
  a1env) export NCCL_IB_DISABLE=1 ;;
  *) die "NCCL_PROFILE must be tuned or a1env" ;;
esac

# 5. config: the set file (comments stripped, $D expanded), then command-line overrides (later keys win)
[ -f "$SETF" ] || die "no set file $SETF"
SETS=$(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$SETF" | sed "s|\$D|$D|g" | tr '\n' ' ')
SPLITS=${SPLITS:-train-clean-100 train-clean-360 train-other-500}
n=$(awk -F, '{print NF}' <<< "$GPUS")
export JSR_DATA=$D CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4

# 6. launch record (host, GPU UUIDs, NCCL env, code / init / config digests) before the job starts
{
  echo "{\"time\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\", \"node_rank\": $NR, \"nnodes\": $NN,"
  echo " \"gpus\": \"$GPUS\", \"gpu_uuids\": \"$(nvidia-smi -i "$GPUS" --query-gpu=uuid --format=csv,noheader | tr '\n' ' ')\","
  echo " \"commit\": \"$PIN_COMMIT\", \"tree_sha256\": \"$GOT\", \"require_clean\": \"${REQUIRE_CLEAN:-1}\", \"local_diff_sha256\": \"$(git -C "$W" diff HEAD | sha256sum | cut -c1-64)\", \"init\": \"$INIT\", \"init_sha256\": \"$IGOT\","
  echo " \"set_file\": \"$SETF\", \"set_sha256\": \"$(sha256sum "$SETF" | cut -c1-64)\", \"overrides\": \"$*\","
  echo " \"splits\": \"$SPLITS\", \"nccl\": \"$(env | grep '^NCCL_' | LC_ALL=C sort | tr '\n' ' ')\"}"
} > "$R/launch.node$NR.json"
echo "$(date -u +%T) node $NR/$NN gpus $GPUS tree $GOT init $IGOT profile ${NCCL_PROFILE:-tuned}" | tee -a "$LOG"

cd "$W/python/ultra" || die "no $W/python/ultra"
# shellcheck disable=SC2086
setsid nohup taskset -c "$CPUS" nice -n 10 "$D/venvs/train/bin/python" -m torch.distributed.run \
  --nnodes "$NN" --node_rank "$NR" --nproc_per_node "$n" --master_addr "$MASTER" --master_port "${MASTER_PORT:-29753}" \
  -m train.trainer --run-dir "$R" --preset S --splits $SPLITS --device cuda --threads 4 --init-from "$INIT" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} --set $SETS "$@" >> "$R/stdout.node$NR.log" 2>&1 < /dev/null &
pid=$!
sleep 2
echo "$pid" > "$R/PID.node$NR"; ps -o pgid= -p "$pid" | tr -d ' ' > "$R/PGID.node$NR"
echo "host=$(hostname) torchrun_pid=$pid nproc=$n run_dir=$R" | tee -a "$LOG"

# 7. transport capture (h50-fleet's check): after NCCL init, record what NCCL actually negotiated. A fallback is
#    a different configuration and gets its own receipt; it is reported, not silently accepted.
( sleep 180
  grep -hoE 'via NET/[A-Za-z0-9_]+(/[0-9]+)?(/GDRDMA)?|via SHM(/direct)?(/direct)?|GDR [01]' "$R/stdout.node$NR.log" \
    | LC_ALL=C sort | uniq -c > "$R/nccl_transport.node$NR.txt"
  if [ "$n" -gt 1 ] && ! grep -q "via SHM" "$R/nccl_transport.node$NR.txt"; then
    echo "$(date -u +%T) WARNING: no SHM transport seen between local ranks on node $NR (see nccl_transport)" | tee -a "$LOG"
  fi
  if [ "${NCCL_PROFILE:-tuned}" = tuned ] && [ "$NN" -gt 1 ] && ! grep -q GDRDMA "$R/nccl_transport.node$NR.txt"; then
    echo "$(date -u +%T) WARNING: tuned profile but no GDRDMA negotiated on node $NR (see nccl_transport)" | tee -a "$LOG"
  fi ) > /dev/null 2>&1 &
