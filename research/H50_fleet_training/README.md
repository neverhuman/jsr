# H50 — Fleet Training: Island-Side Exact-FP32 DiLoCo (`D1`) and Heterogeneous Expansion (`D2`)

**Status:** Hardened island-side DiLoCo launcher (`python/ultra/train/ops/launch_v3_diloco.py`) aligned with `@codex-wave0-review`, `@h50-fleet`, `@trainer-lane`, `@claude-lead`, and `@v3-integrator`.  
**Lane boundaries (deconflicted):**
- **Synchronous DDP (`D0`) & topology ladder (`1 / 2-local / 2-cross-host / 4 / 6`):** owned solely by `@trainer-lane` (`python/ultra/train/ops/h50_topo.sh` + `TOPOLOGY.md`). This launcher intentionally rejects `d0-sync` so there is never a second D0 launcher.
- **Measurement harness:** owned by `@h50-fleet` (`h50_measure.py`, `gpu_sampler.sh`, window protocol).
- **DiLoCo island launcher (`D1` / `D2`):** owned here (`launch_v3_diloco.py`); live GPU launches remain strictly under `@trainer-lane` authority, and `xbabe0` is strictly excluded (quiet timing host).

---

## 1. Cross-Host Pre-Flight & Live GPU-UUID Reservation Contract

Every command emitted by `launch_v3_diloco.py` enforces four pre-flight invariants aligned with `h50_topo.sh`:

1. **Clean git tree:** aborts (`exit 43`) if `git status --porcelain` is non-empty.
2. **Locale-invariant content tree hash (`LC_ALL=C sort`):** matches `@trainer-lane`'s exact cross-host hash command (aborts `exit 42` on mismatch):
   ```bash
   git ls-files -co --exclude-standard python/ultra | grep '\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum
   ```
3. **Artifact digests:** optional `--ckpt-sha256` (`exit 45`) and `--data-manifest` / `--data-sha256` (`exit 46`) verified via `sha256sum -c`.
4. **Fresh live GPU-UUID reservation probe (no stale static lease):** `--inventory-json` is **required** with a fresh `probed_at_utc` (`<= 600 s` old) and live `GPU-...` UUID per slot. Host IPv4 endpoints come only from the live probe, `--host-ips-file` / `H50_HOST_IPS_FILE` (untracked local JSON or hosts file), or `--host-ips` (no non-loopback IPv4 literals are hardcoded in the repository). Duplicate `(host, gpu)` slots are rejected (`ValueError`), `xbabe0` is strictly forbidden, `>= 18.0 GB` free VRAM + `<= 85%` VRAM cap (+ `CPU < 70%` on `xbabe6`) auto-skip `cotenant-gpu`-occupied GPUs, and each worker verifies `nvidia-smi --query-gpu=uuid` (`exit 44`) at start.

---

## 2. `D1` Exact Synchronous FP32 DiLoCo Protocol (`127.0.0.1:8768`)

`D1` runs fixed-roster, full-quorum (`quorum_k == N`), uniform-weighted **exact FP32** DiLoCo (`outer_lr=0.7`, `outer_momentum=0.9`) matched directly against `@trainer-lane`'s exact-FP32 `D0` baseline (`budget_s=750, max_batch=160, ddp_global_batches=true`, NO `ddp_bf16_comm`, NO `stratify_chunks`):

- **Pre-registered whole-round stages (`--d1-stage`):**
  - `protocol`: `H = 2`, `total_steps = 6` (`3` rounds) — local single-site only (`x5-g0,x5-g1`).
  - `smoke`: `H = 50`, `total_steps = 200` (`4` rounds) — local single-site only.
  - `matched`: `H = 500`, `total_steps = 2500` (`5` rounds; or any exact positive whole multiple of `H`).
  - `total_steps % H == 0` is strictly enforced (`ValueError` otherwise, preventing partial-round tails like `2,659`). Cross-site runs require **`H >= 500`** (`ValueError` below `500`) to keep WAN delta transfer `< 10%` of round wall time.
- **Shard-disjointness wiring (`DILOCO_DATA_RANK` / `DILOCO_DATA_WORLD`, `@h50-fleet` item 1):** every island worker exports `DILOCO_DATA_RANK`, `DILOCO_DATA_WORLD`, `DILOCO_AUTH_TOKEN_FILE`, and `DILOCO_AUTH_TOKEN` via environment variables (never unknown `--set` keys), matching PR `#73`'s `TrainerDiLoCoHook` and hub rank-coverage check. **Current status:** `data.py` on `main` does **not** yet shard batches by `DILOCO_DATA_RANK`/`DILOCO_DATA_WORLD`; matched-unique-exposure `D1`/`D2` runs remain gated until that data-loader hook lands. `eval_every` and `ckpt_every_steps` fall on exact multiples of `H`.
- **World-1 GPU ceiling enforcement:** single-GPU (`world=1`) island workers launch via `python/ultra/train/launch_guarded.sh` (`ultra.fleet_guard` with `GPU_TARGET=0.87 <= 0.88`, `CPU_TARGET=0.80` / `0.68` on `xbabe6`, plus per-GPU `flock` lease) so the `90%` GPU ceiling is always enforced; multi-GPU islands (`x5-ddp2`) use the trainer's synchronized `ddp_util_target=0.83`.
- **Loopback-only hub (`127.0.0.1:8768`) + auth + SSH reverse tunnels:** port `8766` is forbidden (reserved for the training dashboard). The hub binds `127.0.0.1:8768` with `--auth-token-file` (passing only CLI flags supported by PR `#73`'s `diloco_hub`), starts in the background (`setsid nohup ... &`), and is gated by a `/status` readiness probe before starting tunnels or workers. Remote hosts reach the hub via H0-initiated `ssh -f -N -R 127.0.0.1:8768:127.0.0.1:8768 ubuntu@<ipv4>` using validated IPv4 addresses only (never stale `xbabe*` hostnames). Every hub and worker command executes `mkdir -p` before log redirection.
- **Non-inferiority gate (`D1` vs. `D0`):** canonical `ultra-score` on H43 DEV confirmation half (`dev_split_v2.json`) and full DEV (`5,567` clips), paired speaker-cluster bootstrap (`10,000` replicates, `alpha = 0.05`), point estimate and upper 95% CI `<= +0.17 pp` across paired seeds.

---

## 3. `D2` Heterogeneous Fleet Expansion (`x5-ddp2` Single Island + Separate NCCL Arm)

Once `D1` passes non-inferiority, `D2` (`--mode d2-hetero`) combines eligible on-prem and cloud islands under the H0 `127.0.0.1:8768` hub (`H >= 500`, reverse tunnels over IPv4):

- **Single `2×A100` DDP island on `xbabe5` (`x5-ddp2`):** both free `xbabe5` A100s launch together as **one** `torch.distributed.run --nproc_per_node 2` island worker (never two independent `x5` workers), joined by eligible `xbabe1` (`4090`), `xbabe2` (`3090`), `xbabe3` (`3090`), and `xbabe6` (`A10`, gated on `>= 18 GB` free VRAM and `CPU < 70%`).
- **Intra-island NCCL profiles (`--nccl-profile a1env|tuned-gdr|tuned-ce`, default `a1env`):**
  - `a1env` (**default, production-safe**): standard default NCCL SHM (`NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eno1`, proven over 49k DDP steps; **never** sets `NCCL_SHM_USE_CUDA_MEMCPY=1` / `NCCL_SHM_MEMCPY_MODE=3`).
  - `tuned-gdr`: cross-host `NCCL_NET_GDR_LEVEL=SYS` (`3.4 → 6.44 GB/s` `NET/IBext_v10/1/GDRDMA`) + `NCCL_SOCKET_IFNAME=eno1` + `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`, **without** `SHM/CE/CE`.
  - `tuned-ce` (**experimental opt-in only — NOT production-qualified**): adds `CUDA_MODULE_LOADING=EAGER NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3`. On `2026-11-23T08:37–08:39Z`, `SHM/CE/CE` deadlocked `DDP.__init__` / `state_checksum` lazy `cuLibraryGetModule` on P2P-disabled PCIe A100s; per `@claude-lead` (`08:38:45Z`), `tuned-ce` stays opt-in until qualified with a DDP-init broadcast + `CUDA_MODULE_LOADING=EAGER` preflight.
- **Token-weighted equal-exposure tracking:** `rnnt_tokens` weighting is recorded in the launch plan metadata (and requires the hub/data-loader shard hook before heterogeneous qualification; `#73`'s hub CLI runs uniform full-roster sync without a `--weighting` flag).
- **Qualification gate (`D2` vs. `D0`):** **`>= 10%` lower wall-clock time to matched WER** at matched unique exposure, measured with `@h50-fleet`'s `h50_measure.py` (3 steady windows `>= 20` blocks / `>= 300 s`), with full **energy (`J / audio-h`)** and **S / D / I** breakdown checks and zero interference with `other-project` (`cotenant-gpu`) workloads (`VRAM <= 85%`, `GPU <= 90%`, `CPU <= 80%`).

---

## 4. Measured Network & Transport Receipts (`@h50-fleet`, `06:49Z` & `08:38Z` safety note)

| Link | Fabric / Path | Measured | Notes |
|---|---|---|---|
| `xbabe5` `g0 ↔ g1` | PCIe host-bridge SHM (`a1env`, default) | `3.0 GB/s` (`64.3 ms` exposed FP32 comm) | Production-safe default (proven over 49k DDP steps) |
| `xbabe5` `g0 ↔ g1` | SHM + `NCCL_SHM_USE_CUDA_MEMCPY=1 MODE=3` | `16.9 GB/s` synthetic only (`6.4 ms`) | **NOT prod-qualified** (`08:38Z` `DDP.__init__` lazy-load deadlock; opt-in `tuned-ce` + `CUDA_MODULE_LOADING=EAGER` preflight required) |
| `xbabe5 ↔ xbabe9` | Cross-host `NCCL_NET_GDR_LEVEL=SYS` (`eno1`) | **`6.44 GB/s`** (`NET/IBext_v10/1/GDRDMA`) | **1.8×** vs `3.4 GB/s`; `111 Gbit/s` `ib_write_bw` (`tuned-gdr`) |
| `xbabe3` (H0) ↔ Cloud | Site ↔ Cloud WAN (`ssh -N -R` to `:8768`) | `110–125` down / `235–250 Mbit/s` up (`28–29 ms`) | `~24 s` / `232 MiB` FP32 round (`H >= 500`) |

---

## 5. Launcher Usage (`--dry-run` Default)

```bash
# D1: pre-registered whole-round stages (protocol H=2 / smoke H=50 local; matched H=500 cross-site via :8768)
PYTHONPATH=python/ultra .venv/bin/python -m train.ops.launch_v3_diloco \
  --mode d1-diloco-fp32 --d1-stage protocol --roster x5-g0,x5-g1 --inventory-json live_inv.json --dry-run
PYTHONPATH=python/ultra .venv/bin/python -m train.ops.launch_v3_diloco \
  --mode d1-diloco-fp32 --d1-stage matched --roster x5-g0,x9-g0 --inventory-json live_inv.json --dry-run

# D2: heterogeneous fleet (single x5-ddp2 island + on-prem/x6; safe a1env default, opt-in tuned-gdr / experimental tuned-ce)
PYTHONPATH=python/ultra .venv/bin/python -m train.ops.launch_v3_diloco \
  --mode d2-hetero --inventory-json live_inv.json --dry-run
PYTHONPATH=python/ultra .venv/bin/python -m train.ops.launch_v3_diloco \
  --mode d2-hetero --nccl-profile tuned-gdr --inventory-json live_inv.json --dry-run
```
