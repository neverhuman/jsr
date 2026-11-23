"""Per-GPU dual-card, 80-90% VRAM, and busy-CPU fleet utilization auditor (stdlib-only)."""

from __future__ import annotations

import argparse, csv, json, os, re, subprocess, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOCAL_HOST, MAC_HOST, SNAPSHOT_HOST = "xbabe3", "xbabe4", "xbabe0"
FLEET_ORDER = (SNAPSHOT_HOST, "xbabe1", "xbabe2", LOCAL_HOST, MAC_HOST, "xbabe5", "xbabe6", "xbabe7", "xbabe8", "xbabe9")
DEFAULT_SNAPSHOT = Path("/mnt/xbabe3-external/jsr/fleet-sched/dash_snapshot.json")
FALLBACK_SNAPSHOTS = (Path("/mnt/xbabe3-external/jsr/fleet-sched/state/latest.json"),)
QUEUE_FILE = Path("/mnt/xbabe3-external/jsr/fleet-sched/queue.json")
TIMING_LOCKS = tuple(Path(f"/mnt/xbabe3-external/jsr/{p}") for p in ("TIMING.lock", "bench/TIMING.lock", "opponents/TIMING.lock"))
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR")
OTHER_MARKERS = tuple(m.strip().lower() for m in os.environ.get("FLEET_OTHER_MARKERS", "cotenant,other-project").split(",") if m.strip())
JSR_MARKERS = ("jsr", "train.trainer", "train.evaluate", "train.mwer", "ultra", "h45_decode", "diloco", "r0_gpu_check")
MIN_CAP_WINDOW_S, UNATTRIBUTED_VRAM_MB = 30.0, 256.0
MAC_PROBE_CMD = (
    "idle=$(top -l 2 -n 0 -s 1 | awk '/CPU usage/ {v=$7} END {sub(\"%\",\"\",v); print v}');"
    "if [ -n \"$idle\" ]; then awk -v i=\"$idle\" 'BEGIN{printf \"CPU_BUSY %d\\n\", (100-i)*10}'; fi;"
    "echo \"NCPU $(sysctl -n hw.ncpu)\"; echo \"LOAD $(sysctl -n vm.loadavg)\";"
    "echo \"GPU $(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o 'Device Utilization %.=[0-9]*' | head -1)\""
)


def _linux_probe_cmd(window_s: int = 1) -> str:
    n = max(1, int(window_s))
    return (
        "read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e));"
        f"for _ in $(seq 1 {n}); do nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed 's/^/USAMP /'; sleep 1; done;"
        "read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e));"
        f"dt=$((t2-t1)); if [ \"$dt\" -gt 0 ]; then echo \"CPU_BUSY $((1000*(dt-(i2-i1))/dt))\"; fi; echo \"WINDOW_S {n}\";"
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/^/GPU /';"
        "nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | sed 's/^/APP /';"
        "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do ps -o pid=,args= -p $p 2>/dev/null | cut -c1-240 | sed 's/^/PS /'; done;"
        "for f in \"${JSR_DATA:-$HOME/jsr-data}\"/gpu-leases/gpu-*.lock; do [ -e \"$f\" ] || continue; if ! flock -s -n \"$f\" true 2>/dev/null; then echo \"LEASE $(basename \"$f\")\"; fi; done;"
        "echo \"LOAD $(cut -d' ' -f1 /proc/loadavg) NCPU $(nproc)\""
    )


def _util_band(pct: float) -> str:
    return "over_cap" if pct > 90.0 else "high" if pct > 88.0 else "target_80_85" if pct >= 78.0 else "under"


def _vram_band(pct: float | None, ceiling_pct: float = 90.0, memory_kind: str = "dedicated") -> str:
    return "shared" if (memory_kind == "shared" or pct is None) else ("over_cap" if pct > ceiling_pct else "target_80_90" if pct >= 80.0 else "under_target")


def _gpu_id(host: str, gpu: dict) -> str:
    return f"{host}:{gpu['index']}:{gpu['uuid']}" if gpu.get("uuid") else f"{host}:{gpu['index']}"


def classify_gpu(
    index: int, name: str, util_pct: float, vram_used_mb: float | None, vram_total_mb: float | None,
    other_project_vram_mb: float = 0.0, *, uuid: str = "", jsr_vram_mb: float = 0.0,
    jsr_pids: list[int] | None = None, other_pids: list[int] | None = None, lease_active: bool = False,
    co_tenant: bool = False, memory_kind: str = "dedicated", util_5min_mean: float | None = None,
    window_samples: list[float] | None = None, window_s: float = 0.0,
) -> dict:
    """Classify one GPU by PID/lease evidence, 80-90% VRAM (85% co-tenant cap), and bounded compute window."""
    util = max(0.0, float(util_pct))
    used = max(0.0, float(vram_used_mb)) if vram_used_mb is not None else None
    total = max(0.0, float(vram_total_mb)) if vram_total_mb is not None else None
    other_mb, jsr_mb = max(0.0, float(other_project_vram_mb)), max(0.0, float(jsr_vram_mb))
    shared = memory_kind == "shared" or (total is not None and total <= 0 and "mps" in str(name).lower())
    mem_kind = "shared" if shared else "dedicated"
    vram_pct = None if (shared or not total or used is None) else (100.0 * used / total)
    has_other = bool(co_tenant or other_mb > 0.0 or (other_pids and len(other_pids) > 0))
    has_jsr = bool(lease_active or jsr_mb > 0.0 or (jsr_pids and len(jsr_pids) > 0))
    ceiling_pct = 85.0 if has_other else 90.0
    vram_band = _vram_band(vram_pct, ceiling_pct=ceiling_pct, memory_kind=mem_kind)
    unattributed = not has_jsr and not has_other and jsr_pids is None and used is not None and used >= UNATTRIBUTED_VRAM_MB
    owner_class = "jsr_active" if (has_jsr or unattributed or (vram_band == "over_cap" and not has_other)) else ("other_project" if has_other else "jsr_idle")
    if util_5min_mean is not None:
        eval_util, cap_window_valid = max(0.0, float(util_5min_mean)), True
    elif window_samples and len(window_samples) >= 2 and window_s >= MIN_CAP_WINDOW_S:
        eval_util, cap_window_valid = sum(float(v) for v in window_samples) / len(window_samples), True
    else:
        eval_util, cap_window_valid = util, False
    return {"index": int(index), "uuid": str(uuid), "name": str(name).replace("NVIDIA ", "").replace("GeForce ", "").strip(), "util_pct": round(util, 1), "util_window_pct": round(eval_util, 1) if cap_window_valid else None, "cap_window_valid": cap_window_valid, "memory_kind": mem_kind, "vram_used_mb": round(used, 1) if (used is not None and not shared) else None, "vram_total_mb": round(total, 1) if (total is not None and not shared) else None, "vram_pct": round(vram_pct, 1) if vram_pct is not None else None, "vram_ceiling_pct": ceiling_pct, "vram_band": vram_band, "co_tenant": has_other, "lease_active": bool(lease_active), "owner_class": owner_class, "band": _util_band(eval_util)}


def _normalize_gpu(raw: dict) -> dict:
    if all(k in raw for k in ("owner_class", "band", "vram_band", "vram_ceiling_pct", "cap_window_valid")):
        return {**raw, "uuid": str(raw.get("uuid") or "")}
    return classify_gpu(raw.get("index", raw.get("idx", 0)), raw.get("name", "GPU"), raw.get("util_pct", raw.get("util", 0.0)) or 0.0, raw.get("vram_used_mb", raw.get("mem_used")), raw.get("vram_total_mb", raw.get("mem_total")), raw.get("other_project_vram_mb", 0.0) or 0.0, uuid=str(raw.get("uuid") or ""), jsr_vram_mb=raw.get("jsr_vram_mb", 0.0) or 0.0, jsr_pids=raw.get("jsr_pids"), other_pids=raw.get("other_pids"), lease_active=bool(raw.get("lease_active", False)), co_tenant=bool(raw.get("co_tenant", False)), memory_kind=str(raw.get("memory_kind") or "dedicated"), util_5min_mean=raw.get("util_5min_mean"), window_samples=raw.get("window_samples"), window_s=float(raw.get("window_s") or 0.0))


def audit_host(host_record: dict) -> dict:
    """Audit busy CPU headroom, per-GPU owner/UUID/VRAM ceiling, dual-GPU balance, and TIMING exemption."""
    host = str(host_record.get("host") or host_record.get("name") or "unknown")
    up, ncpu = bool(host_record.get("up", True)), max(0, int(host_record.get("ncpu") or 0))
    load1 = max(0.0, float(host_record.get("load1", host_record.get("load", 0.0)) or 0.0))
    raw_cpu = host_record.get("cpu_pct")
    cpu_pct = round(max(0.0, min(100.0, float(raw_cpu))), 1) if raw_cpu is not None else None
    cpu_band, exempt = (_util_band(cpu_pct) if cpu_pct is not None else "unknown"), bool(host_record.get("timing_lock", False))
    gpus = [_normalize_gpu(g) for g in (host_record.get("gpus") or [])]
    jsr_gpus = [g for g in gpus if g["owner_class"] != "other_project"]
    other_gpus = [_gpu_id(host, g) for g in gpus if g["owner_class"] == "other_project"]
    if exempt or not up:
        dual_imbalance, idle_gpus, idle_cores = False, [], 0
    else:
        dual_imbalance = len(jsr_gpus) >= 2 and any(g["owner_class"] == "jsr_idle" or g["util_pct"] < 50.0 for g in jsr_gpus) and any(g["util_pct"] >= 70.0 for g in jsr_gpus)
        idle_gpus = [_gpu_id(host, g) for g in jsr_gpus if g["owner_class"] == "jsr_idle" and g["vram_band"] != "over_cap"]
        idle_cores = None if cpu_pct is None else (0 if (ncpu <= 0 or cpu_pct >= 78.0) else max(0, int(0.85 * ncpu - (ncpu * cpu_pct / 100.0))))
    status = "timing_quiet_exempt" if exempt else "offline" if not up else "imbalanced" if dual_imbalance else "other_project" if (gpus and not jsr_gpus) else "ok"
    return {"host": host, "up": up, "status": status, "timing_quiet_exempt": exempt, "ncpu": ncpu, "load1": round(load1, 2), "cpu_pct": cpu_pct, "cpu_band": cpu_band, "idle_cpu_cores_to_85pct": idle_cores, "gpus": gpus, "dual_gpu_imbalance": dual_imbalance, "idle_jsr_gpus": idle_gpus, "other_project_gpus": other_gpus}


def audit_fleet(hosts: list[dict]) -> dict:
    """Aggregate host audits into fleet-wide GPU/VRAM/CPU KPIs and actionable recommendations."""
    audited, jsr_count, in_target = [audit_host(h) for h in hosts], 0, 0
    idle_gpus, imbalanced, other_gpus, over_cap, under_vram, exempt_hosts, recs = [], [], [], [], [], [], []
    for h in audited:
        other_gpus.extend(h["other_project_gpus"])
        if h["timing_quiet_exempt"]:
            exempt_hosts.append(h["host"])
            continue
        if not h["up"]:
            recs.append(f"OFFLINE {h['host']}: host unreachable during probe")
            continue
        if h["dual_gpu_imbalance"]:
            imbalanced.append(h["host"])
            recs.append(f"DUAL_GPU_IMBALANCE {h['host']}: schedule JSR workload on idle sibling GPU(s) {', '.join(h['idle_jsr_gpus']) or h['host']} to reach 78-88% target band")
        for g in h["gpus"]:
            gid = _gpu_id(h["host"], g)
            if g["vram_band"] == "over_cap":
                over_cap.append(gid)
                recs.append(f"OVER_CAP_VRAM {gid} ({g['vram_pct']}% > {g['vram_ceiling_pct']:.0f}% ceiling): reduce VRAM footprint or co-tenant load")
            if g["owner_class"] == "other_project":
                continue
            jsr_count += 1
            if g["band"] == "target_80_85":
                in_target += 1
            elif g["cap_window_valid"] and g["band"] == "over_cap":
                over_cap.append(gid)
                recs.append(f"OVER_CAP_GPU {gid} ({g['util_window_pct']}% window mean): throttle below 90% cap")
            if g["vram_band"] == "under_target":
                under_vram.append(gid)
                if g["owner_class"] == "jsr_active":
                    recs.append(f"UNDER_VRAM_GPU {gid} ({g['vram_pct']}% VRAM): pack batch/co-locate toward 80-{g['vram_ceiling_pct']:.0f}% VRAM target")
        for gid in h["idle_jsr_gpus"]:
            idle_gpus.append(gid)
            if not h["dual_gpu_imbalance"]:
                recs.append(f"IDLE_JSR_GPU {gid}: assign JSR training/eval job (target 78-88% util)")
        if (h["idle_cpu_cores_to_85pct"] or 0) >= 8:
            recs.append(f"IDLE_CPU {h['host']}: +{h['idle_cpu_cores_to_85pct']} cores free to 85% busy-CPU target (cpu={h['cpu_pct']}%, load={h['load1']}/{h['ncpu']})")
    return {"hosts": audited, "jsr_gpu_count": jsr_count, "jsr_gpus_in_target_band": in_target, "idle_jsr_gpus": sorted(idle_gpus), "under_vram_jsr_gpus": sorted(under_vram), "dual_gpu_imbalanced_hosts": sorted(imbalanced), "other_project_gpus": sorted(other_gpus), "over_cap_gpus": sorted(set(over_cap)), "timing_quiet_exempt_hosts": sorted(exempt_hosts), "total_idle_cpu_cores_to_85pct": sum(h["idle_cpu_cores_to_85pct"] for h in audited if h["idle_cpu_cores_to_85pct"] is not None), "recommendations": recs}


def _run_cmd(argv: list[str], timeout: float = 12.0) -> str | None:
    try:
        res = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=timeout, check=False)
        return res.stdout if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_linux_probe(host: str, text: str) -> dict:
    """Parse Linux /proc/stat busy-CPU, nvidia-smi UUID/apps/samples, leases, and loadavg."""
    gpus_raw, apps, cmds, samples, leases = {}, [], {}, {}, set()
    load1, ncpu, cpu_pct, window_s = 0.0, 0, None, 0.0
    for line in text.splitlines():
        if line.startswith("GPU "):
            row = next(csv.reader([line[4:]], skipinitialspace=True), [])
            if len(row) >= 6 and row[0].isdigit():
                gpus_raw[row[1]] = {"index": int(row[0]), "uuid": row[1], "name": row[2], "util_pct": float(row[3] or 0), "vram_used_mb": float(row[4] or 0), "vram_total_mb": float(row[5] or 0), "jsr_vram_mb": 0.0, "other_project_vram_mb": 0.0, "jsr_pids": [], "other_pids": []}
        elif line.startswith("USAMP "):
            row = next(csv.reader([line[6:]], skipinitialspace=True), [])
            if len(row) >= 2:
                try:
                    samples.setdefault(row[0], []).append(float(row[1]))
                except ValueError:
                    continue
        elif line.startswith("APP "):
            row = next(csv.reader([line[4:]], skipinitialspace=True), [])
            if len(row) >= 3 and row[0].isdigit():
                try:
                    apps.append((int(row[0]), float(row[1]), row[2]))
                except ValueError:
                    continue
        elif line.startswith("PS "):
            parts = line[3:].strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                cmds[int(parts[0])] = parts[1].lower()
        elif line.startswith("LEASE ") and (m := re.search(r"gpu-(\d+)\.lock", line)):
            leases.add(int(m.group(1)))
        elif line.startswith("CPU_BUSY ") and line[9:].strip().lstrip("-").isdigit():
            cpu_pct = round(max(0.0, min(100.0, int(line[9:].strip()) / 10.0)), 1)
        elif line.startswith("WINDOW_S ") and line[9:].strip().isdigit():
            window_s = float(int(line[9:].strip()))
        elif line.startswith("LOAD ") and (m := re.match(r"LOAD\s+([0-9.]+)\s+NCPU\s+(\d+)", line)):
            load1, ncpu = float(m.group(1)), int(m.group(2))
    for pid, used_mb, uuid in apps:
        if (gpu := gpus_raw.get(uuid)) is not None:
            cmd = cmds.get(pid, "")
            if any(t in cmd for t in JSR_MARKERS) and not any(t in cmd for t in OTHER_MARKERS):
                gpu["jsr_pids"].append(pid)
                gpu["jsr_vram_mb"] += used_mb
            else:
                gpu["other_pids"].append(pid)
                gpu["other_project_vram_mb"] += used_mb
    for uuid, gpu in gpus_raw.items():
        gpu.update(lease_active=gpu["index"] in leases, window_samples=samples.get(uuid, []), window_s=window_s)
    return {"host": host, "up": True, "ncpu": ncpu, "load1": load1, "cpu_pct": cpu_pct, "gpus": sorted(gpus_raw.values(), key=lambda g: g["index"])}


def parse_mac_probe(host: str, text: str) -> dict:
    """Parse macOS busy-CPU, sysctl, and ioreg output (unified memory labeled 'shared')."""
    ncpu, load1, cpu_pct, gpu_util = 0, 0.0, None, None
    for line in text.splitlines():
        if line.startswith("CPU_BUSY ") and line[9:].strip().lstrip("-").isdigit(): cpu_pct = round(max(0.0, min(100.0, int(line[9:].strip()) / 10.0)), 1)
        elif line.startswith("NCPU ") and line[5:].strip().isdigit(): ncpu = int(line[5:].strip())
        elif line.startswith("LOAD ") and (m := re.search(r"([0-9]+\.[0-9]+)", line[5:])): load1 = float(m.group(1))
        elif line.startswith("GPU ") and (m := re.search(r"=\s*(\d+)", line[4:])): gpu_util = float(m.group(1))
    gpus = [{"index": 0, "name": "Apple MPS", "util_pct": gpu_util, "vram_used_mb": None, "vram_total_mb": None, "memory_kind": "shared"}] if gpu_util is not None else []
    return {"host": host, "up": True, "ncpu": ncpu, "load1": load1, "cpu_pct": cpu_pct, "gpus": gpus}


def read_xbabe0_snapshot(snapshot_path: Path | None = None) -> dict:
    """Read xbabe0 status ONLY from local snapshot files or local dashboard API (never SSH)."""
    explicit, data = snapshot_path is not None, None
    for p in ((snapshot_path,) if explicit else (DEFAULT_SNAPSHOT, *FALLBACK_SNAPSHOTS)):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if data is None and not explicit:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8765/api/snapshot", timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except (OSError, urllib.error.URLError, ValueError):
            data = {}
    node = (data or {}).get("nodes", {}).get(SNAPSHOT_HOST) or (data or {}).get("h", {}).get(SNAPSHOT_HOST) or (data if (data or {}).get("host") == SNAPSHOT_HOST else {})
    lock_active = bool(node.get("timing_lock") or (data or {}).get("timing_lock") or any(p.exists() for p in TIMING_LOCKS))
    if not explicit and not lock_active:
        try:
            lock_active = "TIMING" in str((json.loads(QUEUE_FILE.read_text(encoding="utf-8")).get("rules") or {}).get(SNAPSHOT_HOST, ""))
        except (OSError, ValueError):
            pass
    gpus = [g if isinstance(g, dict) else {"index": g[0], "name": "GPU", "util_pct": g[1], "vram_used_mb": g[2], "vram_total_mb": g[3]} for g in (node.get("gpus") if isinstance(node.get("gpus"), list) else [])]
    return {"host": SNAPSHOT_HOST, "up": bool(node.get("up", bool(node))), "ncpu": int(node.get("ncpu") or 0), "load1": float(node.get("load1", node.get("load", 0.0)) or 0.0), "cpu_pct": node.get("cpu_pct", node.get("cpu")), "gpus": gpus, "timing_lock": lock_active}


def probe_fleet(snapshot_path: Path | None = None, window_s: int = 1) -> list[dict]:
    """Probe all fleet hosts concurrently (local H0, SSH Linux/Mac, snapshot-only xbabe0)."""
    linux_cmd, timeout = _linux_probe_cmd(window_s), max(12.0, float(window_s) + 12.0)
    def _probe(host: str) -> dict:
        if host == SNAPSHOT_HOST:
            return read_xbabe0_snapshot(snapshot_path)
        argv = ["bash", "-c", linux_cmd] if host == LOCAL_HOST else ["ssh", *SSH_OPTS, host, MAC_PROBE_CMD if host == MAC_HOST else linux_cmd]
        out = _run_cmd(argv, timeout=timeout)
        if not out:
            return {"host": host, "up": False, "ncpu": (os.cpu_count() or 0) if host == LOCAL_HOST else 0, "load1": 0.0, "cpu_pct": None, "gpus": []}
        return parse_mac_probe(host, out) if host == MAC_HOST else parse_linux_probe(host, out)
    with ThreadPoolExecutor(max_workers=len(FLEET_ORDER)) as pool:
        return list(pool.map(_probe, FLEET_ORDER))


def format_report(report: dict) -> str:
    """Render a compact human-readable fleet utilization summary."""
    lines = [
        f"JSR GPUs: {report['jsr_gpu_count']} (target 78-88%: {report['jsr_gpus_in_target_band']}, idle: {len(report['idle_jsr_gpus'])}, under-VRAM<80%: {len(report['under_vram_jsr_gpus'])}, over-cap: {len(report['over_cap_gpus'])}, other_project: {len(report['other_project_gpus'])})",
        f"Dual-GPU imbalanced: {report['dual_gpu_imbalanced_hosts'] or 'none'} | Idle CPU cores to 85%: {report['total_idle_cpu_cores_to_85pct']}",
    ]
    for h in report["hosts"]:
        cpu_s = f"{h['cpu_pct']:5.1f}%" if h["cpu_pct"] is not None else "  unk "
        idle_s = f"+{h['idle_cpu_cores_to_85pct']}c" if h["idle_cpu_cores_to_85pct"] is not None else "?c"
        gs = ", ".join(f"g{g['index']}({g['uuid'][:12] or 'no-uuid'})={g['util_pct']:.0f}%/{'shared' if (g['memory_kind'] == 'shared' or g['vram_pct'] is None) else str(round(g['vram_pct'])) + '%'}[{g['owner_class']}:{g['band']}|vram:{g['vram_band']}]" for g in h["gpus"]) or "no-gpu"
        lines.append(f"  {h['host']:7s} [{h['status']:19s}] cpu={cpu_s} load={h['load1']:.2f}/{h['ncpu']} ({idle_s}) | {gs}")
    return "\n".join(lines + (["Recommendations:", *(f"  - {r}" for r in report["recommendations"])] if report["recommendations"] else []))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="collect one live fleet utilization audit and exit")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON instead of text summary")
    ap.add_argument("--window-s", type=int, default=1, help="bounded sampling window in seconds (>=30 validates compute cap window)")
    ap.add_argument("--snapshot-path", type=Path, default=None, help="override xbabe0 local snapshot JSON path")
    args = ap.parse_args(argv)
    if not args.once:
        ap.error("pass --once to run a single read-only audit")
    report = audit_fleet(probe_fleet(snapshot_path=args.snapshot_path, window_s=args.window_s))
    print(json.dumps(report, indent=2) if args.as_json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
