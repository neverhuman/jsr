"""Tree-hash-pinned H50 island-side DiLoCo launcher for D1 (exact FP32) and D2 (hetero).

Scoped strictly to DiLoCo (`D1`/`D2`); synchronous DDP (`D0`) and the topology ladder live
solely in `@trainer-lane`'s `h50_topo.sh` (`@h50-fleet`'s `h50_measure.py`). Read-only under
`--dry-run` (default). Requires a fresh live GPU-UUID probe (`--inventory-json`), forbids
`xbabe0` and port `:8766`, enforces `>= 18 GB` free VRAM / `<= 85%` cap (`CPU < 70%` on
`xbabe6`), and pins `LC_ALL=C sort` content SHA-256 + clean git tree + artifact digests.
"""
from __future__ import annotations

import argparse, hashlib, ipaddress, json, os, re, shlex, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORBIDDEN_HOSTS, FORBIDDEN_HUB_PORTS = frozenset({"xbabe0", "x0"}), frozenset({8766})
DEFAULT_HUB_PORT, MIN_FREE_VRAM_GB, MAX_VRAM_USED_FRAC, DEFAULT_MAX_INVENTORY_AGE_S = 8768, 18.0, 0.85, 600
GLOBAL_BUDGET_S, GLOBAL_MAX_BATCH, DEFAULT_LOG_EVERY, CROSS_SITE_MIN_H = 750.0, 160, 25, 500
D1_STAGES: dict[str, tuple[int, int]] = {"protocol": (2, 6), "smoke": (50, 200), "matched": (500, 2500)}
DEFAULT_AUTH_TOKEN_FILE, DEFAULT_SEED_CKPT = "$HOME/jsr-data/secrets/diloco_hub.token", "$HOME/jsr-data/seeds/A1_seed_ema.pt"
DEFAULT_TOKENIZER = "$HOME/jsr-data/models/zipformer-en-2023-06-26/bpe.model"
CONTENT_HASH_PIPE = "git ls-files -co --exclude-standard python/ultra | grep '\\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum"
GPU_UUID_RE, SHA256_RE = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$"), re.compile(r"^[0-9a-f]{64}$")
BASE_TRAIN_FLAGS = (
    f"ddp_global_batches=true optimizer=adamw lr=0.001 warmup=1000 rnnt_pieces=500 ctc_pieces=500 "
    f"tokenizer={DEFAULT_TOKENIZER} clip=5.0 specaug_start=0 speed_perturb=0.9,1.0,1.1 rnnt_loss=pruned "
    "prune_range=5 prune_warmup=0 prune_hold=0 ema_decay=0.9995 chunk_modes=4,8,16 eval_chunks=8 "
    "eval_each_epoch=false dev_limit=400 loader=process workers=16 window_groups=96 ddp_util_target=0.83"
)
_D, _P, _R = "$HOME/jsr-data", "$HOME/jsr-data/venvs/train/bin/python", "$HOME/jsr"
HOST_PROFILES: dict[str, dict[str, Any]] = {
    h: {"short": s, "site": st, "tier": t, "cpus": c, "data_dir": d, "py": py, "repo_dir": r, "nccl_if": ni, "max_cpu_pct": mc}
    for h, s, st, t, c, d, py, r, ni, mc in (
        ("xbabe5", "x5", "cloud", "A100-40GB", ("0-15", "16-31"), _D, _P, _R, "eno1", None),
        ("xbabe7", "x7", "cloud", "A100-40GB", ("0-15", "16-31"), _D, _P, _R, "eno1", None),
        ("xbabe8", "x8", "cloud", "A100-40GB", ("0-15",), _D, _P, _R, "eno1", None),
        ("xbabe9", "x9", "cloud", "A100-40GB", ("0-15",), _D, _P, _R, "eno1", None),
        ("xbabe1", "x1", "onprem", "RTX-4090-24GB", ("0-15",), _D, _P, _R, "", None),
        ("xbabe2", "x2", "onprem", "RTX-3090-24GB", ("0-15",), _D, _P, _R, "", None),
        ("xbabe3", "x3", "onprem", "RTX-3090-24GB", ("0-15",), "/mnt/xbabe3-external/jsr", "/home/ubuntu/jsr/.venv/bin/python", "/home/ubuntu/jsr", "", None),
        ("xbabe6", "x6", "cloud", "A10-24GB", ("0-15",), _D, _P, _R, "", 70.0),
    )
}
SHORT_TO_HOST = {p["short"]: h for h, p in HOST_PROFILES.items()} | {"h0": "xbabe3"}


@dataclass(frozen=True)
class GpuSlot:
    host: str; gpu: int; name: str; total_mib: int; free_mib: int; gpu_uuid: str
    cpu_pct: float = 0.0; other_project: bool = False; ssh_ip: str = ""
    @property
    def slot_id(self) -> str: return f"{HOST_PROFILES.get(self.host, {}).get('short', self.host)}-g{self.gpu}"
    @property
    def free_gb(self) -> float: return round(self.free_mib / 1024.0, 2)


def validate_ipv4(raw_ip: str, label: str = "endpoint") -> str:
    try:
        return str(ipaddress.IPv4Address(raw_ip.strip()))
    except ipaddress.AddressValueError as exc:
        raise ValueError(f"{label} must be a valid IPv4 address (never a stale hostname): {raw_ip!r}") from exc


def normalize_slot_id(spec: str) -> tuple[str, int, str]:
    m = re.fullmatch(r"([a-z0-9_-]+?)(?:[:\-]g?)(\d+)", spec.strip().lower())
    if not m:
        raise ValueError(f"invalid GPU slot spec {spec!r}; expected e.g. 'x5-g0' or 'xbabe5:0'")
    host_tok, gpu = m.group(1), int(m.group(2))
    host = SHORT_TO_HOST.get(host_tok, host_tok)
    if host_tok in FORBIDDEN_HOSTS or host in FORBIDDEN_HOSTS:
        raise ValueError(f"host {host_tok!r} is forbidden (quiet timing host xbabe0)")
    if host not in HOST_PROFILES:
        raise ValueError(f"unsupported host {host!r} in slot {spec!r}")
    return host, gpu, f"{HOST_PROFILES[host]['short']}-g{gpu}"


def parse_roster(roster_spec: str) -> list[tuple[str, int, str]]:
    parsed, seen = [], set()
    for tok in (t for t in roster_spec.split(",") if t.strip()):
        host, gpu, sid = normalize_slot_id(tok)
        if (host, gpu) in seen:
            raise ValueError(f"duplicate GPU slot ({host}, {gpu}) in roster {roster_spec!r}")
        seen.add((host, gpu)); parsed.append((host, gpu, sid))
    return parsed


def compute_content_tree_hash(repo_root: str | Path) -> tuple[str, str, bool]:
    root = Path(repo_root)
    out = subprocess.check_output(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "python/ultra"], text=True, timeout=10)
    py_files = sorted((p for p in out.splitlines() if p.endswith(".py")), key=lambda s: s.encode("utf-8"))
    if not py_files:
        raise ValueError(f"no python/ultra/*.py files found under {root}")
    digest = hashlib.sha256()
    for rel in py_files:
        digest.update(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel}\n".encode("utf-8"))
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=5).strip()
    return digest.hexdigest(), commit, (dirty == "")


def verify_content_hash(actual_hash: str, expected_hash: str | None) -> str:
    if not SHA256_RE.fullmatch(actual_hash):
        raise ValueError(f"invalid content hash {actual_hash!r}")
    if expected_hash and actual_hash != expected_hash.strip().lower():
        raise ValueError(f"content tree-hash mismatch: actual {actual_hash} != expected {expected_hash.strip()}")
    return actual_hash


def _verify_sha(digest: str | None, label: str) -> str | None:
    if digest is None:
        return None
    norm = digest.strip().lower()
    if not SHA256_RE.fullmatch(norm):
        raise ValueError(f"invalid {label} SHA-256 digest: {digest!r}")
    return norm


def load_live_inventory(raw: dict[str, Any], *, max_age_s: int = DEFAULT_MAX_INVENTORY_AGE_S, now_utc: datetime | None = None) -> tuple[list[GpuSlot], str]:
    """Validate a fresh live GPU-UUID reservation probe receipt (reject stale/static snapshots)."""
    if not isinstance(raw, dict) or "probed_at_utc" not in raw or not isinstance(raw.get("gpus"), list):
        raise ValueError("live inventory JSON must be an object with 'probed_at_utc' and 'gpus' list")
    probed_str = str(raw["probed_at_utc"]).strip()
    probed = datetime.fromisoformat(probed_str.replace("Z", "+00:00"))
    if probed.tzinfo is None:
        raise ValueError(f"probed_at_utc must include timezone (UTC): {probed_str!r}")
    age_s = ((now_utc or datetime.now(timezone.utc)) - probed.astimezone(timezone.utc)).total_seconds()
    if age_s < -60 or age_s > max_age_s:
        raise ValueError(f"live GPU inventory probe is stale ({age_s:.0f}s > {max_age_s}s max age)")
    slots, seen = [], set()
    for it in raw["gpus"]:
        host, gpu, uuid = str(it["host"]).strip().lower(), int(it["gpu"]), str(it.get("gpu_uuid", "")).strip()
        if host in FORBIDDEN_HOSTS:
            raise ValueError(f"forbidden host {host!r} present in fleet inventory")
        if (host, gpu) in seen:
            raise ValueError(f"duplicate GPU slot ({host}, {gpu}) in live inventory")
        if not GPU_UUID_RE.fullmatch(uuid):
            raise ValueError(f"missing or invalid live gpu_uuid for {host}:{gpu}: {uuid!r}")
        seen.add((host, gpu))
        raw_ip = str(it.get("ssh_ip", "")).strip()
        slots.append(GpuSlot(host, gpu, str(it.get("name", "GPU")), int(it["total_mib"]), int(it["free_mib"]),
                             uuid, float(it.get("cpu_pct", 0.0)), bool(it.get("other_project", False)),
                             validate_ipv4(raw_ip, f"{host} ssh_ip") if raw_ip else ""))
    return slots, probed_str


def filter_available_gpus(slots: list[GpuSlot], *, min_free_vram_gb: float = MIN_FREE_VRAM_GB, max_cpu_pct_by_host: dict[str, float] | None = None) -> tuple[list[GpuSlot], list[dict[str, Any]]]:
    min_mib, cpu_caps, admitted, skipped = int(round(min_free_vram_gb * 1024.0)), {"xbabe6": 70.0, **(max_cpu_pct_by_host or {})}, [], []
    for s in slots:
        if s.host in FORBIDDEN_HOSTS:
            raise ValueError(f"forbidden host {s.host!r} cannot be scheduled")
        reasons, used_frac = [], (s.total_mib - s.free_mib) / max(1, s.total_mib)
        if s.free_mib < min_mib:
            reasons.append(f"free_vram={s.free_gb}GB < {min_free_vram_gb:.1f}GB ({'other_project occupied' if s.other_project else 'insufficient free VRAM'})")
        if used_frac > MAX_VRAM_USED_FRAC:
            reasons.append(f"vram_used={used_frac * 100:.1f}% > {MAX_VRAM_USED_FRAC * 100:.0f}% cap")
        if (cap := cpu_caps.get(s.host)) is not None and s.cpu_pct >= cap:
            reasons.append(f"cpu={s.cpu_pct:.1f}% >= {cap:.0f}% cap")
        (skipped.append({"slot": s.slot_id, "host": s.host, "gpu": s.gpu, "gpu_uuid": s.gpu_uuid, "free_gb": s.free_gb, "cpu_pct": s.cpu_pct, "other_project": s.other_project, "reasons": reasons}) if reasons else admitted.append(s))
    return admitted, skipped


def _preflight(repo: str, chash: str, ckpt: str | None = None, ckpt_sha: str | None = None,
               manifest: str | None = None, data_sha: str | None = None, uuids: list[tuple[int, str]] | None = None) -> str:
    parts = [f'[ -z "$(git -C {repo} status --porcelain)" ] || {{ echo "DIRTY_TREE" >&2; exit 43; }}',
             f'H=$(cd {repo} && {CONTENT_HASH_PIPE} | awk \'{{print $1}}\') && [ "$H" = "{chash}" ] || {{ echo "CONTENT_HASH_MISMATCH $H != {chash}" >&2; exit 42; }}']
    if ckpt and ckpt_sha:
        parts.append(f'echo "{ckpt_sha}  {ckpt}" | sha256sum -c - >/dev/null || {{ echo "CKPT_SHA_MISMATCH" >&2; exit 45; }}')
    if manifest and data_sha:
        parts.append(f'echo "{data_sha}  {manifest}" | sha256sum -c - >/dev/null || {{ echo "DATA_SHA_MISMATCH" >&2; exit 46; }}')
    parts.extend(f'nvidia-smi --query-gpu=uuid --format=csv,noheader -i {i} | grep -qx {shlex.quote(u)} || {{ echo "GPU_UUID_MISMATCH g{i}" >&2; exit 44; }}' for i, u in (uuids or []))
    return " && ".join(parts) + " && "


def _nccl_env(nccl_profile: str, nccl_if: str) -> tuple[str, dict[str, str]]:
    iface = nccl_if or "eno1"
    if nccl_profile == "a1env":
        return f"NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME={iface} ", {"NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": iface}
    if nccl_profile == "tuned-gdr":
        env = {"NCCL_NET_GDR_LEVEL": "SYS", "NCCL_SOCKET_IFNAME": iface, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
        return " ".join(f"{k}={v}" for k, v in env.items()) + " ", env
    if nccl_profile == "tuned-ce":  # experimental opt-in only (SHM/CE/CE requires EAGER + DDP-init preflight)
        env = {"CUDA_MODULE_LOADING": "EAGER", "NCCL_SHM_USE_CUDA_MEMCPY": "1", "NCCL_SHM_MEMCPY_MODE": "3", "NCCL_NET_GDR_LEVEL": "SYS", "NCCL_SOCKET_IFNAME": iface, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
        return " ".join(f"{k}={v}" for k, v in env.items()) + " ", env
    raise ValueError(f"unsupported nccl_profile {nccl_profile!r}; expected 'a1env', 'tuned-gdr', or 'tuned-ce'")


def load_host_ips(path: str | Path | None = None, inline: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    if path and (txt := Path(path).read_text().strip()):
        out.update({k.strip().lower(): str(v).strip() for k, v in json.loads(txt).items()} if txt.startswith("{")
                   else {n.lower(): parts[0] for l in txt.splitlines() for parts in (l.split("#", 1)[0].split(),) for n in parts[1:]})
    out.update({k.strip().lower(): v.strip() for p in (inline or "").split(",") if "=" in p for k, v in (p.split("=", 1),)})
    return {SHORT_TO_HOST.get(k, k): validate_ipv4(v, f"{k} ssh_ip") for k, v in out.items()}

def _ssh_ip(host: str, slots: list[GpuSlot], ips: dict[str, str]) -> str:
    if not (cand := ips.get(host) or next((s.ssh_ip for s in slots if s.host == host and s.ssh_ip), "")):
        raise ValueError(f"host {host!r} requires an explicit IPv4 ssh_ip (via inventory, --host-ips-file, or --host-ips)")
    return validate_ipv4(cand, f"{host} ssh_ip")


def build_launch_plan(mode: str, *, content_hash: str, commit_sha: str, clean_tree: bool, inventory: list[GpuSlot], probed_at_utc: str, run_name: str, seed: int = 1234, min_free_vram_gb: float = MIN_FREE_VRAM_GB, nccl_profile: str = "a1env", roster_spec: str | None = None, d1_stage: str = "matched", diloco_every: int | None = None, total_steps: int | None = None, hub_host: str | None = None, hub_port: int = DEFAULT_HUB_PORT, auth_token_file: str = DEFAULT_AUTH_TOKEN_FILE, outer_lr: float = 0.7, outer_momentum: float = 0.9, init_ckpt: str = DEFAULT_SEED_CKPT, ckpt_sha256: str | None = None, data_manifest: str | None = None, data_sha256: str | None = None, host_ips: dict[str, str] | None = None) -> dict[str, Any]:
    if mode not in ("d1-diloco-fp32", "d2-hetero"):
        raise ValueError(f"unsupported mode {mode!r}; D0 sync DDP is owned by h50_topo.sh (use d1-diloco-fp32 or d2-hetero)")
    if hub_port in FORBIDDEN_HUB_PORTS or hub_port < 1024:
        raise ValueError(f"hub_port {hub_port} is forbidden (dashboard uses :8766; use :8768)")
    if d1_stage not in D1_STAGES:
        raise ValueError(f"unsupported d1_stage {d1_stage!r}; expected {tuple(D1_STAGES)}")
    ckpt_sha, data_sha = _verify_sha(ckpt_sha256, "ckpt"), _verify_sha(data_sha256, "data")
    ips = {k: validate_ipv4(v, f"{k} ssh_ip") for k, v in (host_ips or {}).items()}
    admitted, skipped = filter_available_gpus(inventory, min_free_vram_gb=min_free_vram_gb)
    by_id, skip_by_id = {s.slot_id: s for s in admitted}, {s["slot"]: s for s in skipped}
    stage_h, stage_steps = D1_STAGES[d1_stage]
    eff_h = diloco_every if diloco_every is not None else (max(CROSS_SITE_MIN_H, stage_h) if mode == "d2-hetero" else stage_h)
    eff_steps = total_steps if total_steps is not None else (eff_h * 5 if mode == "d2-hetero" else stage_steps)
    if eff_h <= 0 or eff_steps <= 0 or eff_steps % eff_h != 0:
        raise ValueError(f"total_steps ({eff_steps}) must be an exact positive whole multiple of H ({eff_h})")
    islands: list[tuple[str, str, list[GpuSlot]]] = []
    if mode == "d1-diloco-fp32":
        req = parse_roster(roster_spec or "x5-g0,x5-g1")
        if len(req) < 2:
            raise ValueError("d1-diloco-fp32 requires at least 2 distinct GPU slots")
        for _h, _g, sid in req:
            if sid not in by_id:
                raise RuntimeError(f"roster slot {sid} blocked by pre-flight guard: " + "; ".join(skip_by_id.get(sid, {}).get("reasons", ["not in live inventory"])))
            islands.append((sid, by_id[sid].host, [by_id[sid]]))
        chost, weighting = ("xbabe5" if ({h for _, h, _ in islands} == {"xbabe5"} and hub_host is None) else (hub_host or "xbabe3")), "uniform"
    else:
        x5 = sorted((s for s in admitted if s.host == "xbabe5"), key=lambda s: s.gpu)
        if x5:
            islands.append(("x5-ddp2" if len(x5) >= 2 else x5[0].slot_id, "xbabe5", x5[:2]))
        for h in ("xbabe1", "xbabe2", "xbabe3", "xbabe6"):
            islands.extend((s.slot_id, s.host, [s]) for s in sorted((g for g in admitted if g.host == h), key=lambda g: g.gpu))
        if len(islands) < 2:
            raise RuntimeError("d2-hetero requires at least 2 eligible islands passing VRAM/CPU guards")
        chost, weighting = (hub_host or "xbabe3"), "rnnt_tokens"
    cross_site = len({HOST_PROFILES[h]["site"] for _, h, _ in islands} | {HOST_PROFILES[chost]["site"]}) > 1
    if cross_site and eff_h < CROSS_SITE_MIN_H:
        raise ValueError(f"cross-site DiLoCo requires H >= {CROSS_SITE_MIN_H} (got H={eff_h})")
    cp, roster_ids, chost_ip = HOST_PROFILES[chost], [i[0] for i in islands], _ssh_ip(chost, inventory, ips)
    hdir = f"{cp['data_dir']}/runs/diloco/{run_name}_hub"
    hub_cmd = (f"mkdir -p {hdir} && {_preflight(cp['repo_dir'], content_hash, init_ckpt, ckpt_sha, data_manifest, data_sha)}"
               f"PYTHONPATH={cp['repo_dir']}/python/ultra setsid nohup {cp['py']} -m train.diloco_hub --cohort {run_name} --init-anchor {init_ckpt} --work-dir {hdir} --host 127.0.0.1 --port {hub_port} --auth-token-file {auth_token_file} --roster {','.join(roster_ids)} --quorum-k {len(roster_ids)} --outer-lr {outer_lr} --outer-momentum {outer_momentum} >> {hdir}/hub.log 2>&1 < /dev/null & echo $! > {hdir}/HUB_PID && "
               f"for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:{hub_port}/status >/dev/null && break; sleep 1; done && curl -fsS http://127.0.0.1:{hub_port}/status >/dev/null")
    commands: list[dict[str, Any]] = [{"host": chost, "ssh_ip": chost_ip, "role": "diloco_hub", "command": hub_cmd}]
    tunnels = []
    for whost, wip in sorted({h: _ssh_ip(h, inventory, ips) for _, h, _ in islands if h != chost}.items()):
        tcmd = f"ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 127.0.0.1:{hub_port}:127.0.0.1:{hub_port} ubuntu@{wip}"
        tunnels.append({"from_host": chost, "to_host": whost, "to_ip": wip, "command": tcmd})
        commands.append({"host": chost, "ssh_ip": chost_ip, "role": f"tunnel:{whost}", "to_ip": wip, "command": tcmd})
    coordinator = {"host": chost, "bind": "127.0.0.1", "port": hub_port, "url": f"http://127.0.0.1:{hub_port}", "auth_token_file": auth_token_file, "roster": roster_ids, "quorum_k": len(roster_ids), "sync_mode": "exact_fp32_full_roster", "weighting": weighting, "diloco_dtype": "float32", "outer_lr": outer_lr, "outer_momentum": outer_momentum, "tunnels": tunnels}
    _, nccl_meta = _nccl_env(nccl_profile, HOST_PROFILES["xbabe5"]["nccl_if"])
    eval_e, workers = (eff_h if eff_steps <= 500 else min(eff_steps, max(eff_h, (4000 // eff_h) * eff_h))), []
    for idx, (iid, h, isl_slots) in enumerate(islands):
        p, rdir = HOST_PROFILES[h], f"{HOST_PROFILES[h]['data_dir']}/runs/train/{run_name}_{iid}"
        pre = _preflight(p["repo_dir"], content_hash, init_ckpt, ckpt_sha, data_manifest, data_sha, [(s.gpu, s.gpu_uuid) for s in isl_slots])
        gpus, cpus = ",".join(str(s.gpu) for s in isl_slots), ",".join(p["cpus"][min(s.gpu, len(p["cpus"]) - 1)] for s in isl_slots)
        sets = f"seed={seed} budget_s={GLOBAL_BUDGET_S:.1f} max_batch={GLOBAL_MAX_BATCH} total_steps={eff_steps} eval_every={eval_e} ckpt_every_steps={eval_e} log_every={min(DEFAULT_LOG_EVERY, eff_h)} {BASE_TRAIN_FLAGS} diloco_hub=http://127.0.0.1:{hub_port} diloco_worker={iid} diloco_every={eff_h} diloco_dtype=float32"
        diloco_env = f'DILOCO_DATA_RANK={idx} DILOCO_DATA_WORLD={len(islands)} DILOCO_AUTH_TOKEN_FILE={auth_token_file} DILOCO_AUTH_TOKEN="$(cat {auth_token_file})" '
        if len(isl_slots) > 1:
            nccl_prefix, _ = _nccl_env(nccl_profile, p["nccl_if"])
            sets += " ddp_native=true"
            runner = f"{p['py']} -m torch.distributed.run --nnodes 1 --node_rank 0 --nproc_per_node {len(isl_slots)} --master_addr 127.0.0.1 --master_port 29720 -m train.trainer"
            w_cmd = f"mkdir -p {rdir} && {pre}{nccl_prefix}{diloco_env}CUDA_VISIBLE_DEVICES={gpus} OMP_NUM_THREADS=4 JSR_DATA={p['data_dir']} PYTHONPATH={p['repo_dir']}/python/ultra setsid nohup taskset -c {cpus} nice -n 10 {runner} --run-dir {rdir} --preset S --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --init-from {init_ckpt} --set {sets} >> {rdir}/stdout.log 2>&1 < /dev/null &"
        else:
            cpu_tgt = "0.68" if p["max_cpu_pct"] else "0.80"
            w_cmd = f"mkdir -p {rdir} && {pre}{diloco_env}JSR_DATA={p['data_dir']} PY={p['py']} GPU_TARGET=0.87 CPU_TARGET={cpu_tgt} bash {p['repo_dir']}/python/ultra/train/launch_guarded.sh {rdir} {cpus} {gpus} -- --preset S --splits train-clean-100 train-clean-360 train-other-500 --device cuda --threads 4 --init-from {init_ckpt} --set {sets}"
        workers.append({"island": iid, "host": h, "gpus": [s.gpu for s in isl_slots], "nproc": len(isl_slots), "gpu_uuids": [s.gpu_uuid for s in isl_slots], "tier": p["tier"], "data_rank": idx, "data_world": len(islands), "diloco_every": eff_h, "rounds": eff_steps // eff_h, "steps": eff_steps, "diloco_dtype": "float32", "weighting": weighting})
        commands.append({"host": h, "ssh_ip": _ssh_ip(h, inventory, ips), "role": f"worker:{iid}", "command": w_cmd})
    plan: dict[str, Any] = {"schema": "ultra.h50.launch_plan/v2", "mode": mode, "run_name": run_name, "seed": seed, "content_hash": content_hash, "commit_sha": commit_sha, "clean_tree": clean_tree, "probed_at_utc": probed_at_utc, "min_free_vram_gb": min_free_vram_gb, "excluded_hosts": sorted(FORBIDDEN_HOSTS), "d1_stage": d1_stage if mode == "d1-diloco-fp32" else "hetero", "diloco_every": eff_h, "total_steps": eff_steps, "rounds": eff_steps // eff_h, "cross_site": cross_site, "nccl_profile": nccl_profile, "nccl_env": nccl_meta, "nccl_production_qualified": (nccl_profile != "tuned-ce"), "comm_dtype": "float32", "coordinator": coordinator, "workers": workers, "skipped_gpus": skipped, "commands": commands}
    if mode == "d2-hetero":
        plan["d2_gates"] = {"min_time_to_wer_reduction_pct": 10.0, "weighting": "rnnt_tokens", "equal_exposure": True, "require_energy_check": True, "require_sdi_check": True}
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for flag, kw in (("--mode", {"choices": ("d1-diloco-fp32", "d2-hetero"), "required": True}), ("--inventory-json", {"required": True}), ("--run-name", {}), ("--repo-root", {"default": str(Path(__file__).resolve().parents[4])}), ("--expected-content-hash", {}), ("--max-inventory-age-s", {"type": int, "default": DEFAULT_MAX_INVENTORY_AGE_S}), ("--nccl-profile", {"choices": ("a1env", "tuned-gdr", "tuned-ce"), "default": "a1env"}), ("--d1-stage", {"choices": tuple(D1_STAGES), "default": "matched"}), ("--roster", {}), ("--hub-host", {}), ("--hub-port", {"type": int, "default": DEFAULT_HUB_PORT}), ("--auth-token-file", {"default": DEFAULT_AUTH_TOKEN_FILE}), ("--seed", {"type": int, "default": 1234}), ("--diloco-every", {"type": int}), ("--total-steps", {"type": int}), ("--min-free-vram-gb", {"type": float, "default": MIN_FREE_VRAM_GB}), ("--outer-lr", {"type": float, "default": 0.7}), ("--outer-momentum", {"type": float, "default": 0.9}), ("--init-ckpt", {"default": DEFAULT_SEED_CKPT}), ("--ckpt-sha256", {}), ("--data-manifest", {}), ("--data-sha256", {}), ("--host-ips", {"default": os.environ.get("H50_HOST_IPS")}), ("--host-ips-file", {"default": os.environ.get("H50_HOST_IPS_FILE")}), ("--require-clean-tree", {"action": "store_true"}), ("--dry-run", {"action": "store_true"}), ("--execute", {"action": "store_true"}), ("--out", {})):
        ap.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    chash, commit, clean = compute_content_tree_hash(a.repo_root)
    verify_content_hash(chash, a.expected_content_hash)
    if (a.execute or a.require_clean_tree) and not clean:
        raise RuntimeError("refusing launch on dirty git tree; commit or stash changes first")
    slots, probed_at = load_live_inventory(json.loads(Path(a.inventory_json).read_text()), max_age_s=a.max_inventory_age_s)
    plan = build_launch_plan(a.mode, content_hash=chash, commit_sha=commit, clean_tree=clean, inventory=slots, probed_at_utc=probed_at, run_name=a.run_name or f"H50_{a.mode.replace('-', '_')}_s{a.seed}", seed=a.seed, min_free_vram_gb=a.min_free_vram_gb, nccl_profile=a.nccl_profile, roster_spec=a.roster, d1_stage=a.d1_stage, diloco_every=a.diloco_every, total_steps=a.total_steps, hub_host=a.hub_host, hub_port=a.hub_port, auth_token_file=a.auth_token_file, outer_lr=a.outer_lr, outer_momentum=a.outer_momentum, init_ckpt=a.init_ckpt, ckpt_sha256=a.ckpt_sha256, data_manifest=a.data_manifest, data_sha256=a.data_sha256, host_ips=load_host_ips(a.host_ips_file, a.host_ips))
    payload = json.dumps(plan, indent=2) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(payload)
    print(payload, end="")
    if a.dry_run or not a.execute:
        return 0
    for step in plan["commands"]:
        subprocess.check_call(["bash", "-lc", step["command"]] if step["host"] == "xbabe3" else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"ubuntu@{step['ssh_ip']}", shlex.join(["bash", "-lc", step["command"]])])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
