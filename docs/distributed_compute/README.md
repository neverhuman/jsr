# Distributed compute rebuild and retirement

Use this runbook to rebuild the JSR training pool or recover a run from the
surviving `xbabe3` storage. The retirement decision for `xbabe5` through `xbabe9`
depends on both workload clearance and verified off-node custody for JSR and
co-tenant projects. The operator performs the shutdown.

## Fleet layout

| Host | Hardware | Historical six-worker layout |
| --- | --- | --- |
| `xbabe5` | 2 × A100 PCIe 40 GB | Ranks 0–1, rendezvous and global reporting |
| `xbabe7` | 2 × A100 PCIe 40 GB | Ranks 2–3 |
| `xbabe8` | 1 × A100 PCIe 40 GB | Rank 4 |
| `xbabe9` | 1 × A100 PCIe 40 GB | Rank 5 |
| `xbabe6` | 1 × A10 | Separate optimizer and labeling worker |
| `xbabe3` | Surviving H0 host | Durable archive and score collection |

Reuse the rank map saved with the run when recovering it. A smaller run can
have a different roster. H58 seed 7920 used three ranks across `xbabe5` and
`xbabe8`. Each host used its local data root for NVMe staging, rank logs,
checkpoints, and the per-GPU lease directory. Global rank 0 emitted the training
and evaluation summaries. Each node's local rank 0 could save a checkpoint,
and every rank wrote its configuration receipt. Gather the receipts and
rank-specific failure logs centrally with the rank-0 summaries.
[Trainer operations](../../python/ultra/train/ops/README.md) ·
[H53 trunk amendments](../../research/H53_v3_trunk_and_capacity/AMENDMENTS.md)

## Stage and launch synchronous DDP

1. Provision compatible NVIDIA drivers, CUDA, PyTorch, NCCL, and trainer
   dependencies. Pin the source commit, any local patch, and the environment
   on every worker. Record the GPU UUID and allowed CPU mask for each rank.
2. Set `JSR_DATA` to the local NVMe root on every worker. Stage the ordered
   corpus manifest and shards, both tokenizers, the verified seed or full
   checkpoint, and any registered label map or teacher cache required by the
   recipe. Run `sha256sum -c` from the staging root against the input manifest.
   A trainer `.sha256` sidecar contains a digest alone. Pair it with the
   `.pt` filename before passing it to `sha256sum -c`, and compare to the
   registered digest as well. Confirm shard ordering and dataset identity on
   every worker. For a sealed recipe, use the registered `--preflight-only`
   launch and retain the rank configs and identity receipt.
3. Resolve the rendezvous address on the private network. Verify the same
   interface routes to the peers, reserve the GPUs, check for other leases
   and compute processes, and allocate a free rendezvous port. Use the A1
   profile that ran for roughly 49,000 steps, with `NCCL_IB_DISABLE=1`, the
   private interface (`eno1` on the historical cloud pool), and default NCCL
   shared memory. Leave the experimental copy-engine variables unset.
   [H50 fleet record](../../research/H50_fleet_training/README.md)

For a two-host fresh start with one GPU on each host, set `MASTER_ADDR` and
`MASTER_PORT` on both, using host A's private address. Set the local absolute
paths `JSR_DATA` and `SEED`, the registered `SEED_SHA256`, and the allocated
`CPU_MASK` separately on each worker. Run from the pinned repo root.
The example uses GPU 0 and the `S` recipe. Substitute the registered trainer
arguments for a production run.

```bash
export MASTER_ADDR MASTER_PORT JSR_DATA WORLD_SIZE=2
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eno1
unset NCCL_SHM_USE_CUDA_MEMCPY NCCL_SHM_MEMCPY_MODE

# Set RANK_OFFSET=0 on host A and RANK_OFFSET=1 on host B.
python/ultra/train/ops/launch_ddp_ranks.sh \
  "$JSR_DATA/runs/train/two-host-example" "$RANK_OFFSET" 0 "$CPU_MASK" -- \
  --preset S --splits train-clean-100 --device cuda \
  --init-from "$SEED" --init-sha256 "$SEED_SHA256" \
  --set ddp_global_batches=true ddp_util_target=0.83
```

For the historic six-GPU layout, all launchers use `WORLD_SIZE=6`. Pass offsets
`0`, `2`, `4`, `5` with `GPU_LIST` and colon-separated CPU masks in local
order. Keep each supervisor active. A failed worker stops its local group.
Inspect every supervisor and stop the other groups cleanly before restarting
the fleet. Confirm the startup record reports rank agreement and archive
`config.json`, `config.rankN.json`, and the host/GPU/CPU map with the run.

### Bounded parity check on real audio

Use the same verified seed, tokenizer pair, ordered local corpus, and precision
for all arms. On one reserved GPU, set `WORLD_SIZE=1` and `RANK=0`. Then run the
same command concurrently on two reserved hosts with `WORLD_SIZE=2` and
`RANK=0` or `1`, using the rendezvous and NCCL settings above. Keep the physical
`GPU` selection and `CPU_MASK` local. Use a distinct receipt path per arm and
rank.

```bash
mkdir -p "$JSR_DATA/gpu-leases"
export WORLD_SIZE RANK LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
set -o pipefail
(
  cd python/ultra
  flock -n "$JSR_DATA/gpu-leases/gpu-$GPU.lock" \
    timeout --kill-after=30s 5m taskset -c "$CPU_MASK" ../../.venv/bin/python \
    -m train.ops.ddp_probe --seed "$SEED" --split train-clean-100 \
    --batch-size 32 --max-seconds 20 --steps 8 --warmup 2 --threads 2 \
    --device cuda --duty 0.85
) | tee "$RECEIPT"
```

Check that rank 0 emitted a `ddp_probe` receipt. Require matching seed,
initial-model and utterance-order digests across arms. The probe checks
disjoint input coverage, finite updates, and identical final model digests
among ranks of a single arm. Compare gradient norms across arms against the
registered tolerance. Reduction order can change floating-point results
across world sizes. Archive the receipts, the topology, and any failure logs.
[Probe implementation](../../python/ultra/train/ops/ddp_probe.py)

### Same-run restart

1. Choose the last common completed **full** checkpoint step. A `LATEST`
   file contains the `ckpt-########` stem. Verify the corresponding `.pt`
   against its adjacent digest and the source inventory. Restore each host's
   verified payload to its replacement run directory if the registered run
   stored node-local state. Node copies of one step can have different hashes.
   Compare their saved step and sampler cursor and keep the original rank
   layout, data identity, tokenizer, optimizer, and trainer recipe.
2. Restore each `LATEST` only after its payload and marker are in place. Load
   a representative checkpoint on CPU with the matching trainer and inspect
   the model, optimizer, scheduler, RNG and sampler state. Preserve any pause
   or divergence receipt and investigate it before clearing a stop marker.
3. Re-launch the registered command in the recovered run directory with
   `--init-from` and `--init-sha256` omitted. Confirm the expected `resume`
   step and position in rank 0's log, DDP startup agreement, and advancement
   of the next batches without repeating or skipping exposure. Qualify any
   world-size or sampler change before allocating a full run. A `model-*.pt`
   weights export can seed a fresh run. Full checkpoints carry the optimizer
   and sampling state needed for a continuation.
   [Checkpoint format](../../python/ultra/train/checkpoint.py) ·
   [Trainer restart](../../python/ultra/train/trainer.py)

## DiLoCo on slower links

DiLoCo is experimental. Match a synchronous DDP control before using islands
for a production claim. Run the HMAC-authenticated hub on loopback, typically
`127.0.0.1:8768`, using a mode-`0600` token file provisioned separately. Give
the workers the same token through their local credential store. From the
pinned repo root on the hub, set `ROSTER` to the ordered comma-separated
worker IDs and `ISLAND_COUNT` to their count. Start a fresh cohort with the
verified anchor and the registered outer optimizer settings.

```bash
PYTHONPATH=python/ultra .venv/bin/python -m train.diloco_http \
  --cohort "$COHORT" --host 127.0.0.1 --port 8768 --work-dir "$HUB_DIR" \
  --init-anchor "$ANCHOR" --auth-token-file "$TOKEN_FILE" \
  --roster "$ROSTER" --quorum-k "$ISLAND_COUNT" \
  --outer-lr 0.7 --outer-momentum 0.9
```

Forward only loopback traffic through SSH. From a worker with access to the
hub host, use a local forward.

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8768:127.0.0.1:8768 "$HUB_HOST"
```

Where the hub initiates the session, as in the cloud fleet, use a reverse
forward from the hub to each worker instead.

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:8768:127.0.0.1:8768 "$WORKER_HOST"
```

Check `/status` through the tunnel before starting workers. Pin the ordered
roster, each island's data rank and world, the starting anchor, outer optimizer
settings, and `H` local steps per round. Use full-roster quorum, disjoint shards,
and a total step count divisible by `H`. H50's cross-site launcher required
`H >= 500`. Reconfirm that the selected trainer revision consumes the registered
shard assignments. Collect hub commit metadata, anchor and outer optimizer
state, and every island's post-anchor checkpoint.
[H50 protocol](../../research/H50_fleet_training/DILOCO.md) ·
[Launcher](../../python/ultra/train/ops/launch_v3_diloco.py)

After a fault, inspect the durable hub round and anchor digest. Restore every
island's checkpoint from the most recent whole round for which all islands
match the hub's round and anchor. Verify each island's shard identity and
cursor. Stage the chosen round's `outer_meta_rN.json`, `outer_state_rN.safetensors`
and `anchor_rN.safetensors` in a recovery work directory. The hub's `--resume`
loads the newest intact round in that directory. Check its loaded anchor,
then restart the fixed roster together. If only the hub committed a newer
round, recover the earlier common round and replay the abandoned local steps.
Count exposure only from verified whole-roster commits.
[Resume checks](../../python/ultra/train/diloco_d1.py)

| Observation | Operational implication |
| --- | --- |
| `SHM/CE/CE` deadlocked DDP initialization on P2P-disabled PCIe A100s on 2026-11-23. It stalled in CUDA module loading during the initial state checksum. | Keep default NCCL SHM. Treat copy-engine settings as experimental and require an init broadcast witness on the intended topology before use. |
| H50 D1's triggering worker hit a 60 s `POST /delta` timeout at rounds 14 and 9 while the hub synchronously persisted about 1.135 GB on a saturated disk. The hub had committed each round. | Compare hub and island state after any timeout. A timed-out request can correspond to a published anchor. Keep hub storage headroom and qualify the transport under the intended load. |

[NCCL field measurements](../../research/H50_fleet_training/NETWORK_AND_NCCL.md) ·
[DiLoCo incident record](../ultra/releases/v3.0.0.md)

## Archive and shutdown checks

The JSR retirement archive on `xbabe3` is
`/mnt/xbabe3-external/jsr/archive/xbabe5-xbabe9-retirement/`. Verify that the
external drive is mounted before writing. See `STATUS.md` in the archive for
the last recorded host and custody checks. Record the source host and path,
commit or run ID, SHA-256, and durable destination for every retained artifact.
For any reused copy, read and hash the destination bytes. Keep local Git refs
and detached heads, staged and unstaged binary diffs, untracked source, and
launch scripts. Keep rank configs, score and gate records, selected or
registered checkpoints, promoted exports, and the latest recoverable full
checkpoints for interrupted runs. Start with protected branch points and H58
seed 7920 at step 8,767. A partial step-8,000 file in the earlier archive
cannot recover that run. Credentials belong in the project's credential store.

Verify bundle refs and detached tips against the saved head inventory. Fetch
the bundle into a disposable verification repository under the active repo's
`target/`, then verify the objects and replay representative dirty trees from
their recorded commits. Apply the staged patch to the index and worktree with
`git apply --index`, then apply the unstaged patch to the worktree with plain
`git apply`. Extract the untracked source and compare the restored index and
status against the captures. Verify destination hashes and load a representative
full checkpoint on the surviving host. Recheck the
[v3.2.0 closeout index](../../research/v3.2.0-closeout.md) and the H57 PL3
score manifest before migration.

The H57 PL3 manifest is relative to its harvest directory. From the repo
root, verify the two existing indexes as follows.

```bash
survivor=/mnt/xbabe3-external/jsr
index="$PWD/research/v3.2.0-closeout.SHA256SUMS"
(cd "$survivor" && sha256sum -c "$index")
(cd "$survivor/runtime-lane/h57_rc2cool_pl3_harvest" && sha256sum -c SHA256SUMS)
```

Confirm co-tenant code and key results by digest in each project's durable
storage or obtain an owner custody confirmation. Capture unique material in
that project's storage and retain the verification receipt.

Immediately before clearing each host, record one fresh UTC timestamp and
check the following.

- Query GPU utilization, memory and compute PIDs with `nvidia-smi`. Review
  process state, CPU counters, and recent log writes across a short interval
  to distinguish inactive workers from advancing jobs.
- Inspect running containers, user and system services and timers, queued
  jobs, and user and root cron. Resolve keepers, disk watchers, and orphaned
  workers with their owners.
- Check for held `gpu-*.lock` leases under each active data root. Test for
  lock ownership with `flock -s -n` on existing lock files. Consult the
  current fleet scheduler and co-tenant reservation records. The historical
  `$survivor/fleet-sched/queue.json` is a useful inventory reference. Check
  its generation time and obtain a current reservation view for clearance.

Resolve active jobs and missing custody first. Issue a per-host green light
only when both checks pass.
