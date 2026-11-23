# H50 D0/D1 Host-Pair Amendment: `xbabe1` (RTX 4090) + `xbabe2` (RTX 3090)

- **Amends:** `research/H50_fleet_training/DILOCO.md` (draft at `/mnt/xbabe3-external/jsr/h50-fleet/D0D1/DILOCO.md`)
- **Registered Endpoint (unchanged):** `h50-D1-vs-D0-24GB-islands-ni-ema-pooled-fulldev-c4`
  - Registry file: `/mnt/xbabe3-external/jsr/evals/eval-lane/h43/margins.json`
  - `entries_sha256`: `f28850c6e7239c6632ce9fe43a302c6a3bfe295b4fc3b9ebe480dbae4bce41d5`
  - Margin: **+0.087 pp** (lower is better), non-inferiority on EMA pooled full dev (5,567 utts), chunk 4 primary (chunks 8/16 reported), canonical scorer, paired speaker bootstrap over all seed pairings (`1234`, `4321`).
  - Decision rule: **PASS iff one-sided 95% upper bound of `D1 − D0 <= +0.087 pp`; FAIL iff lower `> +0.087 pp`; else INCONCLUSIVE.** Islands join `S2b`/`S3` **only on PASS**.
- **Pinned Code (`origin/main`):** `31ce5a63e1c8703531c7a8a8563da83904632a1c` (includes merged `#73` hardened safetensors+HMAC DiLoCoHub `d0314b7`, `#74`, and `#103` D1 items 1/3/7 `dc1761a`).
- **Scheduled Window (`@fleet-sched` 10:32:59Z, `@claude-lead` 10:31:27Z):** starts immediately after `medium` labeling releases `xbabe1` + `xbabe2` (`~10:51Z`). Zero overlap with A100 `S2` (`x5/x7/x8/x9`) and zero overlap with H0 `S2-BP1` gate-(a) evals.

## 1. Host-Pair & Local Disk Verification (Read-Only Probe `10:35Z`)

Replaces `(xbabe2 3090 + H0 3090)` because H0's root NVMe has only ~7 GB free and H0 runs `S2-BP1` gate-(a) evals. Loaders read strictly from fast local NVMe — **never** from `/mnt/xbabe3-external`.

| Host | GPU (24 GB) | Role | Free Local NVMe (`/`) | Extra Local NVMe | Local Data Needed (`57 GB`) | Python Deps (`~/jsr/.venv`) |
|---|---|---|---|---|---|---|
| `xbabe1` | RTX 4090 | `x1` (island 0) | **118 GB** | `/mnt/<local-nvme>` (**461 GB**) | Stage `57 GB` LS 960h + seed + bpe from `x2` (LAN only, zero A100 reads) | `torch 2.7.0`, `safetensors`, `aiohttp` ✓ |
| `xbabe2` | RTX 3090 | `x2` (island 1 + hub) | **90 GB** | `/mnt/fast-scratch` (**168 GB**) | `57 GB` LS 960h + seed + bpe **already cached locally** | `torch 2.7.0`, `safetensors`, `aiohttp` ✓ |

## 2. Common Recipe & Matched Exposure (Identical Across D0 and D1)

- **Init checkpoint:** `H41/model-00060000.pt` (sha256 `d12336677f3946d9f8359d4f44f79a8f80909d9734fa3c38ef79ce4c307600dc`), `tokenizer=zipformer-en-2023-06-26/bpe.model`.
- **Corpus (960h on-prem, zero A100 traffic):** LibriSpeech 960h (`train-clean-100`, `train-clean-360`, `train-other-500`, `57 GB`, no `extra_splits`) — byte-identical local NVMe shards on both `x1` and `x2` (zero A100/WAN copy, zero USB drive reads). Both D0 and D1 consume the exact same ordered 960h global batches.
- **Global batch & matched exposure:** `budget_s=650.0`, `max_batch=144` (proven 24 GB safe), `total_steps=4000` (`~722` unique audio-h per seed; 16 whole DiLoCo rounds at `H=250`).
- **Optimization & recipe:** `preset=S`, `optimizer=adamw`, `lr=0.0002476`, `warmup=200`, `min_lr_ratio=0.1212`, `clip=5.0`, `rnnt_loss=pruned`, `prune_range=5`, `delay_penalty=0.005`, `chunk_modes=4,8,16`, `stratify_chunks=false` (held equal across D0 & D1 per `codex-full-fleet` `09:46:48Z`), `speed_perturb=0.9,1.0,1.1`, `specaug_start=0`, `ema_decay=0.9995`, `qkv_cap_from=100`, `qkv_cap_ratio=1.5`, `qkv_guard_ratio=2.0`, `diverge_after=1000`, `loader=process`, `workers=8`, `window_groups=16`.
- **Seeds:** `1234` and `4321` (2 seeds per arm = 4 runs total).
- **Logging & saves:** `log_every=25`, `eval_every=0`, `eval_each_epoch=false`, `ckpt_every_steps=1000`, `weights_every=1000` (exact round boundaries for `H=250`).

## 3. Exact Arm Definitions

| Arm | Topology | Exposure & Batch Equivalence | Outer Step / EMA | Evaluated Artifact |
|---|---|---|---|---|
| **D0 (Control)** | Single-card synchronous exact global batch on `xbabe1` 4090 (`world=1`, `data_world=0`) | `4,000` steps at `budget_s=650.0, max_batch=144` — mathematically identical synchronous global gradient and exact ordered utterance exposure as 2-rank `ddp_global_batches=true`, without 1 GbE LAN NCCL distortion | N/A (`ema_decay=0.9995` per step) | Final step-4,000 `model-00004000.pt` EMA (`ema`) + raw reported |
| **D1 (Treatment)** | Two heterogeneous 24 GB islands: `xbabe1` 4090 (`data_rank=0, data_world=2`, worker `xbabe1-4090`) + `xbabe2` 3090 (`data_rank=1, data_world=2`, worker `xbabe2-3090`) | `apply_data_shard` (`#103`) gives each island its exact disjoint rank share of every global `650 s` batch; union over `{0, 1}` equals D0's exact global batch at every step `0..3999` (`H=250`, 16 rounds = `4,000` steps) | Exact-sync FP32 Outer Nesterov (`outer_lr=0.7, outer_momentum=0.9`, uniform `1/2` fixed roster order, trainable params only, bit-equal buffers); hub EMA `hub_ema_decay = 0.9995^250 = 0.882469` | Final round-16 hub EMA (`ema.*` in `outer_state_r16.safetensors` exported to `.pt`) + raw `anchor_r16.safetensors` reported |

### Transport, Security & Pre-Flight Protocol
- **Hub host:** `xbabe2`, bound strictly to `127.0.0.1:8768` (`--roster xbabe1-4090,xbabe2-3090`, `--auth-token-file` mode `0600`, HMAC-SHA256 + safetensors wire only, zero non-loopback sockets).
- **Tunnel:** `xbabe1` connects via `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8768:127.0.0.1:8768 xbabe2`.
- **Pre-flight smoke (`< 60 s`):** 2-round `H=2` (`max_steps=4`) dry/live check confirming HMAC auth, safetensors delta/anchor sha256, buffer bit-equality, disjoint UIDs, and post-anchor-only ckpt at step 4 before launching full `4,000`-step seeds.
- **Kill rules:** any non-finite delta/loss, any hub subset commit (protocol bug), any anchor train loss `> 20%` above D0 at the same step, or steady-window data wait `> 15%`.

## 4. Staged Launch Scripts

All scripts are staged under `/mnt/xbabe3-external/jsr/h50-fleet/D0D1/run_xb1_xb2/`:
- `common_d0d1_24g.set` — frozen S2-matched `TrainConfig --set` keys
- `stage_data_xb1_xb2.sh` — post-10:51Z local NVMe staging + sha256 check (`xbabe1` + `xbabe2`)
- `launch_hub_xb2.sh` — starts hardened DiLoCoHub on `xbabe2` `127.0.0.1:8768`
- `launch_d1_islands.sh` — opens loopback SSH tunnel and runs both D1 islands (`--dry-run` verified)
- `launch_d0_control.sh` — runs matched D0 synchronous control on `xbabe1` (`--dry-run` verified)
