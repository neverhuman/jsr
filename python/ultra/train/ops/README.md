# Training operations scripts

These are the scripts that ran the 2026-11 training campaign. They are committed here so the review can audit them. The copies that actually ran live on each host's data drive:
- H0 (xbabe3): `/mnt/xbabe3-external/jsr/runs/train/`
- xbabe1: `/mnt/amphora-backup/jsr-data/`
- xbabe4: `~/jsr-train/`

The scripts use absolute host paths, and each one starts with a comment saying what it does.

Two guard modules ship with the package itself and are not copied here:
- `../launch_guarded.sh`
- `../mac_guard.py` (and `ultra/fleet_guard.py`)

| Area | Scripts |
|---|---|
| Watching runs | `divergence_watch.py` (external loss/WER divergence stop for runs that predate the in-trainer guard), `sync_and_post.py` (pulls a remote run's log, posts evals), `dev_curve_remote.py`, `runs/D2_S_960h/dev_curve.*`, `qkv_norms.py` (per-block qkv norms from checkpoints) |
| Finishing runs | `finish_D2.sh`, `finish_remote.sh` (full dev-clean/dev-other scoring, copy plus sha256 check to H0), `r4t_finish.sh`, `runs/R4T_tc100/ref_eval.sh` |
| Checkpoint selection | `select_finalist.sh`, `select_D2MR.sh`, `post_selection.py` |
| Launch chains | `runs/D2_S_960h/launch_when_ready.sh`, `launch_R4T960.sh`, `xbabe1/abl.sh` (qkv ablations), `xbabe1/d3mrc.sh` and `sync_d3mrc.sh` (D3MRc), `xbabe4/d4m_relaunch.sh` |
| Reporting | `runs/D1d/post_evals.py` |

## Sharing a checkpoint between hosts

Select weights with the same unsealed DEV workload, decoder, and normalizer. Pin the
source run config and completed checkpoint. From `python/ultra`, prepare the seed:

```sh
python -m train.ops.prepare_seed \
  --source /data/source/model-00070000.pt \
  --source-config /data/source/config.json \
  --target-config /data/target/config.json \
  --out /data/fleet_stage/candidate
```

The seed has weights, both tokenizer digests, and source provenance. Exact-to-pruned
transfer initializes the pruned loss's auxiliary heads and preserves the recognizer.
Copy the `.pt` and `.sha256` files together, verify the hash on the receiving host,
and check the two local tokenizer files against the published digests. Start a new
guarded run directory with `--init-from` and the reviewed target settings. Evaluate
the seed and its first checkpoints on the same DEV workload before allocating more GPUs.

For a same-run migration, carry a verified full checkpoint and `LATEST` into the
destination run directory instead. Resume with the same model, tokenizer, data,
and sampler settings to restore the optimizer, schedule, and position.

## Data-parallel validation and launch

`train.ops.ddp_probe` runs a bounded check with real audio and writes its receipt
to stdout. Run it from `python/ultra` with the same pinned seed, global batch
size, split, step count, precision, and duty fraction for each comparison. The
receipt records the utterance order and final synchronized model digests.

`launch_ddp_ranks.sh` starts explicit global ranks for the distributed trainer.
This permits different GPU counts per host. For six workers on hosts with
2, 2, 1, and 1 GPUs, use rank offsets 0, 2, 4, and 5 with the same `WORLD_SIZE=6`,
`MASTER_ADDR`, and `MASTER_PORT`. Supply the GPUs and a CPU mask for each local
rank in the same order. Use the fleet private network for rendezvous and NCCL.
The supervisor checks that its GPUs are idle and stops its local group on a
worker failure. Keep the supervisor active on each host and monitor every rank.
Both launchers hold the same per-GPU lease under the data root while training.
Set `JSR_DATA` to the same local root for both launchers. Pass the physical GPU
index to `launch_guarded.sh`. It exposes that GPU as `cuda:0` to the trainer.
The rank launcher requires the distributed trainer from [PR #10][trainer-pr].

Check the full checkpoint and corpus digests on every host before a production
resume. Validate the sampler cursor when changing the worker count. Use the
trainer's step-boundary throttle for synchronized workers.

## Preflighting a label-only comparison

Use `seal_data_identity=true` and `label_map_only=1` in the registered trainer
recipe. Pin the map with `label_map_sha256` and the raw seed with
`--init-from PATH --init-sha256 SHA256`. Run the same recipe and rank layout
with `--preflight-only` before the first optimizer step. Collect each arm's
`config.json` and `config.rankN.json` files from the hosts that wrote them.

From `python/ultra`, run:

```sh
../../.venv/bin/python -m train.data_identity /path/to/arm-a /path/to/arm-b
```

The comparison checks the raw seed, trainer source, effective optimizer and
sampler settings, ordered shard and segment identities, and matched map UIDs.
Archive the comparison JSON with the launch registration. The sealed checkpoint
carries per-rank RNG, sampler cursor, and active helper state. Resume on the
registered rank layout with the same recipe and omit `--init-from`.

For a frozen Libriheavy confirmation roster, run
`python -m tools.plbt.audit_roster --help` for the ID-only speaker and book
directory check against the accepted TRAIN map. The check records access and
consumes the TRAIN `id` and `src` columns. Keep the H52 metadata checks for
canonical book identities and the registered TRAIN-only human-label audit.

[trainer-pr]: https://github.com/neverhuman/jsr/pull/10
