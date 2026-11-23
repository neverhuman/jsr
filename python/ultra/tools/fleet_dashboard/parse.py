"""Pure parsing and aggregation for the fleet dashboard (:8765), backed by dashboard_events."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from tools.dashboard_events import (
    DEFAULT_MEASUREMENT_DOMAIN,
    MonotonicEventStore,
    SCHEMA_VERSION,
    aggregate_run_groups,
    annotate_host_hardware,
    attempt_rank as _attempt_rank,
    extract_artifact as _extract_artifact,
    extract_cohort as _extract_cohort,
    extract_sdi as _extract_sdi,
    infer_measurement_domain as _infer_measurement_domain,
    is_valid_schema,
    normalize_gate as _norm_gate,
    normalize_stale_state,
    read_v3_scorecard,
)

# Train history is decimated so a client snapshot stays small (fast first paint).
MAX_TRAIN_POINTS = 1500
STALE_SECONDS = 300
# A live trainer whose log has been silent this long is a real stall (evals can take a few minutes).
STALL_SECONDS = 600
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def parse_timing_lock(text: str) -> dict | None:
    """Parse `TIMING_LOCK <held|free> [owner...]` and `TIMING_QUIET <state...>` probe lines."""
    status, owner, quiet = None, "", None
    for line in text.splitlines():
        if line.startswith("TIMING_LOCK ") and len(p := line.split(None, 2)) >= 2 and p[1] in ("held", "free"):
            status, owner = p[1], p[2].strip() if len(p) > 2 else ""
        elif line.startswith("TIMING_QUIET "):
            quiet = line[len("TIMING_QUIET "):].strip() or None
    return None if status is None and quiet is None else {"status": status or "free", "owner": owner, "quiet": quiet}


def parse_other_load(text: str, gpus: list[dict], live_runs: set[str], cpu_pct: float | None = None) -> dict:
    """Detect non-JSR ('other project') GPU/CPU load from probe lines or unowned host utilization."""
    gmb, cpct, labels = 0.0, 0.0, []
    for line in text.splitlines():
        if line.startswith(("OTHER_GPU ", "OTHER_CPU ")):
            p = line.split(None, 2)
            try:
                val = float(p[1])
                gmb, cpct = (gmb + val, cpct) if line.startswith("OTHER_GPU ") else (gmb, cpct + val)
            except (IndexError, ValueError):
                pass
            if len(p) > 2 and p[2].strip():
                labels.append(p[2].strip()[:32])
    gbusy, cbusy = gmb >= 256.0, cpct >= 15.0
    if not live_runs:
        for g in gpus:
            u = float(g.get("mem_used") or g.get("memory_used_mib") or 0.0)
            t = float(g.get("mem_total") or g.get("memory_total_mib") or 0.0)
            if float(g.get("util") or g.get("util_pct") or 0.0) >= 15.0 or (t > 0 and 100.0 * u / t >= 10.0):
                gbusy, gmb = True, max(gmb, u)
        if cpu_pct is not None and cpu_pct >= 25.0:
            cbusy, cpct = True, max(cpct, float(cpu_pct))
    if not gbusy and not cbusy:
        return {}
    uniq = sorted(set(labels))
    return {"gpu": gbusy, "cpu": cbusy, "gpu_mem_mb": round(gmb, 1), "cpu_pct": round(cpct, 1),
            "label": f"other project ({', '.join(uniq[:2])})" if uniq else "other project"}


def parse_diloco_status(raw: dict) -> dict | None:
    """Normalise one DiLoCoHub `/status` or `diloco_status.json` payload."""
    if not isinstance(raw, dict) or not raw.get("cohort"):
        return None
    workers = [{
        "worker_id": wid, "host": w.get("host", "?"), "base_round": w.get("base_round", 0),
        "staleness": int(w.get("staleness") or 0), "audio_hours_total": round(float(w.get("audio_hours_total") or 0.0), 2),
        "audio_h_per_h": round(float(w.get("audio_h_per_h") or 0.0), 1), "delta_norm": w.get("delta_norm"),
        "cosine": w.get("cosine"), "status": w.get("status", "unknown"), "last_seen": w.get("last_seen"),
    } for wid, w in sorted((raw.get("workers") or {}).items())]
    recent = raw.get("recent_rounds") or []
    last_r = recent[-1] if recent else None
    mode = raw.get("mode") or ("sync_roster" if raw.get("expected_roster") else "async_quorum")
    dtype = raw.get("delta_dtype") or ("float32" if mode == "sync_roster" else "bfloat16")
    tag = "FP32" if dtype == "float32" else "BF16"
    mlabel = raw.get("mode_label") or (f"Sync-{tag} Roster" if mode == "sync_roster" else f"Async-{tag} Quorum")
    norms = [float(w["delta_norm"]) for w in workers if w.get("delta_norm") is not None]
    mnorm = raw.get("mean_delta_norm")
    if mnorm is None:
        mnorm = last_r.get("mean_delta_norm") if last_r and last_r.get("mean_delta_norm") is not None else (
            round(sum(norms) / len(norms), 4) if norms else None)
    mstale = raw.get("max_staleness") if raw.get("max_staleness") is not None else max((w["staleness"] for w in workers), default=0)
    quar = sorted(set((list(last_r.get("quarantined") or []) if last_r else []) + list(raw.get("quarantined_workers") or [])
                      + [w["worker_id"] for w in workers if w["status"] == "quarantined"]))
    return {"schema": SCHEMA_VERSION, "cohort": str(raw["cohort"]), "mode": mode, "delta_dtype": dtype, "mode_label": mlabel,
            "expected_roster": raw.get("expected_roster"), "round": int(raw.get("round", 0)), "quorum_k": int(raw.get("quorum_k", 1)),
            "pending_count": int(raw.get("pending_count", 0)), "worker_count": len(workers),
            "total_audio_h": round(float(raw.get("total_audio_h") or 0.0), 2), "effective_audio_h_per_h": round(float(raw.get("effective_audio_h_per_h") or 0.0), 1),
            "mean_delta_norm": mnorm, "max_staleness": int(mstale), "mean_cos": last_r.get("mean_cos") if last_r else None,
            "quarantined": quar, "workers": workers, "recent_rounds": recent[-10:], "updated_at": raw.get("updated_at")}


def parse_live_runs(text: str) -> set[str]:
    """Run names from probe lines `RUN <name>` (the basename of each live trainer's --run-dir)."""
    out = set()
    for line in text.splitlines():
        p = line.split()
        if len(p) == 2 and p[0] == "RUN" and _RUN_NAME.match(p[1]):
            out.add(p[1])
    return out


def classify_cohort(name: str) -> str:
    """Separate production/warm/v3 runs from 960h scratch and 100h ablations."""
    if re.search(r"^(D2_S_|D2[c-z]_|L1w|V3_|DILOCO_|OMEGA_|A1$|A1_|F1$|F1_)", name) or "warm" in name.lower():
        return "prod"
    if re.search(r"^(H13[bcd]|H23|ABL_|D0_|D1)", name):
        return "ablation"
    return "scratch"


def parse_line(line: str) -> dict | None:
    """Parse one train.jsonl line into a compact record, or None if invalid/unknown schema."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(rec, dict) or not is_valid_schema(rec):
        return None
    kind = rec.get("kind")
    if kind == "train":
        out = {
            "k": "t", "step": rec.get("step"), "ep": (rec.get("pos") or {}).get("epoch"),
            "loss": rec.get("loss"), "rnnt": rec.get("rnnt"), "ctc": rec.get("ctc"),
            "lr": rec.get("lr"), "gn": rec.get("grad_norm"), "ahh": rec.get("audio_h_per_h"),
            "audio_s": rec.get("audio_s"), "pad_pct": rec.get("pad_pct"),
            "wait_pct": rec.get("data_wait_pct"), "qkv": rec.get("max_qkv_norm"),
            "gsnr": rec.get("gsnr"), "diloco_round": rec.get("diloco_round"), "time": rec.get("time"),
        }
        for key in ("schema", "run_id", "attempt_id", "rank", "world_size", "seq", "planned_exposure_ahh",
                    "planned_audio_h_per_h", "rank_local_ahh", "rank_audio_h_per_h", "cohort",
                    "diloco_roster", "diloco_mode", "measurement_domain"):
            if rec.get(key) is not None:
                out[key] = rec[key]
        art = _extract_artifact(rec)
        if art:
            out["artifact"] = art
        return out
    if kind == "eval":
        out = {"k": "e", "step": rec.get("step"), "ep": rec.get("epoch"),
               "split": rec.get("split"), "wer": rec.get("wer"), "time": rec.get("time")}
        if rec.get("ema_wer") is not None:
            out["ema_wer"] = rec.get("ema_wer")
        sdi = _extract_sdi(rec)
        if sdi is not None:
            out["sdi"] = sdi
        gate = _norm_gate(rec.get("wave_gate") or rec.get("gate"))
        if gate is not None:
            out["gate"] = gate
        dom = _infer_measurement_domain(rec, sdi)
        if dom is not None:
            out["measurement_domain"] = dom
        for key in ("schema", "run_id", "attempt_id", "rank", "world_size", "seq", "cohort"):
            if rec.get(key) is not None:
                out[key] = rec[key]
        art = _extract_artifact(rec)
        if art:
            out["artifact"] = art
        return out
    if kind in ("diverged", "start", "resume", "done", "finished"):
        out = {"k": kind[:3], "kind": kind, "time": rec.get("time"), "step": rec.get("step"),
               "msg": str(rec.get("error", ""))[:160]}
        for key in ("schema", "run_id", "attempt_id", "rank", "world_size", "seq", "cohort",
                    "diloco_roster", "diloco_mode", "measurement_domain", "wave_gate", "gate"):
            if rec.get(key) is not None:
                out[key] = rec[key]
        art = _extract_artifact(rec)
        if art:
            out["artifact"] = art
        return out
    return None


@dataclass
class RunState:
    """Incrementally-updated state of one training run (monotonic by run_id/attempt_id/rank/seq)."""

    name: str
    host: str = "?"
    train: list = field(default_factory=list)  # [step, loss, ahh, time, lr, rnnt]
    evals: list = field(default_factory=list)  # [step, split, wer, time, ema_wer]
    last: dict = field(default_factory=dict)
    audio_h_total: float = 0.0
    status: str = "running"
    stride: int = 1
    total_steps: int | None = None
    live_hosts: set = field(default_factory=set)
    live_since: float = 0.0
    run_id: str | None = None
    attempt_id: str | int | None = None
    rank: int = 0
    world_size: int | None = None
    cohort_override: str | None = None
    cohort_meta: dict | None = None
    diloco_roster: list[str] | None = None
    diloco_mode: str | None = None
    artifact: dict | None = None
    measurement_domain: str | None = None
    last_sdi: dict | None = None
    wave_gate: str | None = None
    planned_exposure_ahh: float | None = None
    rank_local_ahh: dict[int, float] = field(default_factory=dict)
    _n: int = 0
    _recent: list = field(default_factory=list)
    _store: MonotonicEventStore = field(default_factory=MonotonicEventStore)

    def _reset_attempt_buffers(self) -> None:
        self.train.clear()
        self.evals.clear()
        self.last = {}
        self.audio_h_total = 0.0
        self.status = "running"
        self.stride = 1
        self.wave_gate = None
        self.planned_exposure_ahh = None
        self.rank_local_ahh.clear()
        self._n = 0
        self._recent.clear()

    def _absorb_meta(self, r: dict, ev: dict) -> None:
        if r.get("run_id") is not None:
            self.run_id = str(r["run_id"])
        if ev.get("world_size") is not None:
            self.world_size = ev["world_size"]
        raw_coh = r.get("cohort")
        if isinstance(raw_coh, str) and raw_coh.strip():
            self.cohort_override = raw_coh.strip()
        elif isinstance(raw_coh, dict) and raw_coh.get("name"):
            self.cohort_override = str(raw_coh["name"])
        if ev.get("cohort"):
            self.cohort_meta = {**(self.cohort_meta or {}), **ev["cohort"]}
            if ev["cohort"].get("roster"):
                self.diloco_roster = list(ev["cohort"]["roster"])
            if ev["cohort"].get("mode"):
                self.diloco_mode = str(ev["cohort"]["mode"])
        if ev.get("measurement_domain"):
            self.measurement_domain = ev["measurement_domain"]
        if ev.get("artifact"):
            self.artifact = {**(self.artifact or {}), **ev["artifact"]}

    def add(self, r: dict | None) -> bool:
        if not isinstance(r, dict):
            return False
        cur = self._store.cursor(self.name)
        prev_att = cur.attempt_id
        accepted, _, ev = self._store.ingest(r, default_run_id=self.run_id or self.name, stream_key=self.name)
        if not accepted:
            return False
        if prev_att is not None and cur.attempt_id != prev_att:
            self._reset_attempt_buffers()
        self.attempt_id = cur.attempt_id
        self._absorb_meta(r, ev)
        k = r.get("k") or str(r.get("kind") or "")[:3]
        if k == "sta":
            self._reset_attempt_buffers()
            return True
        if k == "res" and r.get("step") is not None:
            step = int(r["step"])
            self.train = [row for row in self.train if (row[0] or 0) <= step]
            self.evals = [row for row in self.evals if (row[0] or 0) <= step]
            return True
        if k == "t":
            rank = r.get("rank")
            if cur.rank_local_ahh:
                self.rank_local_ahh = dict(cur.rank_local_ahh)
            if rank is not None and int(rank) > 0:
                return True  # never overwrite rank-0 planned exposure or double-count follower rank
            if cur.planned_exposure_ahh is not None:
                self.planned_exposure_ahh = cur.planned_exposure_ahh
            self.last = r
            if r.get("step") is not None and r.get("time") is not None:
                self._recent.append((r["step"], r["time"]))
                if len(self._recent) > 40:
                    self._recent = self._recent[-40:]
            self.audio_h_total += float(r.get("audio_s") or 0.0) / 3600.0
            self._n += 1
            if self._n % self.stride == 0:
                self.train.append([r["step"], r["loss"], r["ahh"], r["time"], r["lr"], r.get("rnnt")])
                if len(self.train) > MAX_TRAIN_POINTS:
                    self.train = self.train[::2]
                    self.stride *= 2
        elif k == "e":
            self.evals.append([r["step"], r["split"], r["wer"], r["time"], r.get("ema_wer")])
            if ev.get("sdi") is not None:
                self.last_sdi = ev["sdi"]
            if ev.get("measurement_domain") is not None:
                self.measurement_domain = ev["measurement_domain"]
            if ev.get("wave_gate") is not None:
                self.wave_gate = ev["wave_gate"]
        elif k == "div":
            self.status = "diverged"
            if self.wave_gate is None:
                self.wave_gate = "FAIL"
        elif k in ("don", "fin"):
            self.status = "done"
            if ev.get("wave_gate") is not None:
                self.wave_gate = ev["wave_gate"]
        return True

    def best_wer(self, split: str = "dev-clean") -> list | None:
        rows = [e for e in self.evals if e[1] == split and e[2] is not None]
        return min(rows, key=lambda e: e[2]) if rows else None

    def best_ema_wer(self, split: str = "dev-clean") -> list | None:
        rows = [e for e in self.evals if e[1] == split and len(e) > 4 and e[4] is not None]
        return min(rows, key=lambda e: e[4]) if rows else None

    def set_live(self, node: str, live: bool, now: float) -> None:
        if live:
            if not self.live_hosts:
                self.live_since = now
            self.live_hosts.add(node)
        else:
            self.live_hosts.discard(node)

    def canonical(self, now: float, inventory_complete: bool = False) -> dict:
        s = self.summary(now, inventory_complete=inventory_complete)
        return {
            "schema": SCHEMA_VERSION, "run_id": s["run_id"], "attempt_id": s["attempt_id"],
            "rank": s["rank"], "world_size": s["world_size"], "seq": s["seq"], "step": s["step"],
            "sdi": s["sdi"], "measurement_domain": s["measurement_domain"], "artifact": s["artifact"],
            "cohort": s["cohort_meta"] or {"name": s["cohort"], "diloco_round": s["diloco_round"],
                                           "roster": s["diloco_roster"], "mode": self.diloco_mode},
            "planned_exposure_ahh": s["planned_exposure_ahh"], "rank_local_ahh": s["rank_local_ahh"],
            "wave_gate": s["wave_gate"], "stale_state": s["stale_state"], "rejected_events": s["rejected_events"],
        }

    def summary(self, now: float, inventory_complete: bool = False) -> dict:
        """inventory_complete: every node answered its latest probe, so 'no live process' is authoritative."""
        lt = self.last.get("time") or 0
        best, best_ema = self.best_wer(), self.best_ema_wer()
        last_e = next((e for e in reversed(self.evals) if e[1] == "dev-clean"), None)
        last_o = next((e for e in reversed(self.evals) if e[1] == "dev-other"), None)
        last_ema = next((e for e in reversed(self.evals) if e[1] == "dev-clean" and len(e) > 4 and e[4] is not None), None)
        age = (now - lt) if lt else None
        if self.status in ("diverged", "done"):
            status = self.status
        elif self.live_hosts:
            fresh_start = not lt or self.live_since > lt
            if fresh_start and now - self.live_since <= STALL_SECONDS:
                status = "starting"
            else:
                status = "stalled" if (age is None or age > STALL_SECONDS) else "running"
        elif inventory_complete:
            status = "stopped"
        else:
            status = "running" if (age is not None and age <= STALE_SECONDS) else "stopped"
        rate = None
        if len(self._recent) >= 2:
            (s0, t0), (s1, t1) = self._recent[0], self._recent[-1]
            if t1 > t0 and s1 > s0:
                rate = (s1 - s0) / (t1 - t0) * 3600.0
        step = self.last.get("step")
        eta_h = (self.total_steps - step) / rate if (rate and self.total_steps and step is not None and self.total_steps > step) else None
        gate = self.wave_gate or ("FAIL" if status == "diverged" else ("RUNNING" if status in ("running", "starting", "stalled") else None))
        cur = self._store.cursor(self.name)
        planned = self.planned_exposure_ahh if self.planned_exposure_ahh is not None else self.last.get("planned_audio_h_per_h", self.last.get("ahh"))
        return {
            "schema": SCHEMA_VERSION, "name": self.name, "run_id": self.run_id or self.name,
            "attempt_id": self.attempt_id, "rank": self.rank, "world_size": self.world_size,
            "seq": cur.seq_by_rank.get(self.rank),
            "cohort": self.cohort_override or classify_cohort(self.name),
            "cohort_meta": self.cohort_meta,
            "total_steps": self.total_steps, "steps_per_h": round(rate) if rate else None,
            "eta_h": round(eta_h, 1) if eta_h is not None else None,
            "progress": round(100.0 * step / self.total_steps, 1) if (self.total_steps and step is not None) else None,
            "host": ",".join(sorted(self.live_hosts)) if self.live_hosts else self.host,
            "hosts": sorted(self.live_hosts), "step": step, "ep": self.last.get("ep"),
            "loss": self.last.get("loss"), "rnnt": self.last.get("rnnt"), "lr": self.last.get("lr"),
            "gn": self.last.get("gn"), "pad_pct": self.last.get("pad_pct"), "wait_pct": self.last.get("wait_pct"),
            "qkv": self.last.get("qkv"), "gsnr": self.last.get("gsnr"),
            "diloco_round": self.last.get("diloco_round"), "diloco_roster": self.diloco_roster,
            "diloco_mode": self.diloco_mode,
            "ahh": planned if planned is not None else self.last.get("ahh"),
            "planned_ahh": planned, "planned_exposure_ahh": planned,
            "rank_local_ahh": dict(self.rank_local_ahh) if self.rank_local_ahh else None,
            "audio_h_total": round(self.audio_h_total, 1),
            "best": best, "best_ema": best_ema, "last_clean": last_e, "last_ema": last_ema, "last_other": last_o,
            "sdi": self.last_sdi, "ref_words": self.last_sdi.get("ref_words") if self.last_sdi else None,
            "wave_gate": gate, "measurement_domain": self.measurement_domain, "artifact": self.artifact,
            "age": age, "status": status, "stale_state": normalize_stale_state(status),
            "rejected_events": cur.rejected_count,
        }


def global_summary(runs: list[dict], nodes: dict, *, scorecard: dict | None = None) -> dict:
    """Fleet-wide aggregates for the header KPIs (deduplicating same-group multi-rank entries)."""
    running = [r for r in runs if r.get("status") == "running" and (r.get("planned_exposure_ahh") or r.get("planned_ahh") or r.get("ahh"))]
    grouped = aggregate_run_groups(running)
    gpus = [g for n in nodes.values() for g in (n.get("gpus") or [])]
    mem_used = sum(g["mem_used"] for g in gpus)
    mem_total = sum(g["mem_total"] for g in gpus) or 1
    cpu_threads = sum(n.get("ncpu") or 0 for n in nodes.values())
    cpu_busy = sum((n.get("cpu_pct") or 0) / 100.0 * (n.get("ncpu") or 0) for n in nodes.values())
    wers, prod_wers = [], []
    for r in runs:
        w = r["best_ema"][4] if r.get("best_ema") else (r["best"][2] if r.get("best") else None)
        if w is not None:
            wers.append(w)
            if r.get("cohort") == "prod":
                prod_wers.append(w)
    imbalanced = sorted(k for k, n in nodes.items() if n.get("dual_gpu_imbalance"))
    return {
        "schema": SCHEMA_VERSION,
        "audio_h_per_h": grouped["planned_exposure_ahh"],
        "planned_exposure_ahh": grouped["planned_exposure_ahh"],
        "live_runs": grouped["live_runs"], "runs": len(runs),
        "gpus": len(gpus), "gpus_busy": sum(1 for g in gpus if g["util"] >= 50),
        "gpu_util": round(sum(g["util"] for g in gpus) / len(gpus), 1) if gpus else 0,
        "gpu_mem_pct": round(100.0 * mem_used / mem_total, 1),
        "gpu_mem_used_gb": round(mem_used / 1024, 1), "gpu_mem_total_gb": round(mem_total / 1024, 1),
        "cpu_pct": round(100.0 * cpu_busy / cpu_threads, 1) if cpu_threads else 0,
        "cpu_threads": cpu_threads,
        "best_wer": min(wers) if wers else None, "best_prod_wer": min(prod_wers) if prod_wers else None,
        "nodes_up": sum(1 for n in nodes.values() if n.get("up")), "nodes": len(nodes),
        "stalled": sorted(r["name"] for r in runs if r["status"] == "stalled"),
        "imbalanced_hosts": imbalanced, "v3_scorecard": scorecard,
        "idle_nodes": sorted(
            k for k, n in nodes.items()
            if n.get("up") and n.get("gpus") and not n.get("mac") and not n.get("runs")
            and not (n.get("other_load") or {}).get("gpu")
        ),
    }


def summarize_rank(rec: dict, now: float) -> dict:
    """Compact per-rank DDP heartbeat (`rank<R>.jsonl`, written by every rank each log interval)."""
    iv = rec.get("interval_s") or 0
    ahh = (rec.get("audio_s") or 0) / iv if iv else None
    return {
        "run": rec.get("run"), "rank": rec.get("rank"), "world": rec.get("world"), "gpu": rec.get("gpu"),
        "step": rec.get("step"), "loss": rec.get("loss"), "ahh": round(ahh, 1) if ahh is not None else None,
        "compute_s": rec.get("compute_s"), "comm_s": rec.get("comm_s"), "wait_s": rec.get("wait_s"),
        "sleep_s": rec.get("sleep_s"), "pad_pct": rec.get("pad_pct"),
        "age": (now - rec["time"]) if rec.get("time") else None,
    }


def parse_nvidia(text: str) -> list[dict]:
    """Parse `nvidia-smi` CSV lines into GPU dicts (preserving distinct per-GPU indices)."""
    out = []
    for row in text.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue

        def num(v: str) -> float:
            try:
                return float(v)
            except ValueError:
                return 0.0

        idx = int(parts[0])
        out.append({
            "idx": idx, "index": idx, "name": parts[1].replace("NVIDIA ", "").replace("GeForce ", ""),
            "util": num(parts[2]), "util_pct": num(parts[2]), "mem_used": num(parts[3]), "mem_total": num(parts[4]),
            "power": num(parts[5]) if len(parts) > 5 else 0.0, "temp": num(parts[6]) if len(parts) > 6 else 0.0,
        })
    return out


def annotate_node(info: dict) -> dict:
    """Annotate a fleet node dict with per-GPU target bands, dual_gpu_imbalance, and cpu_load_pct."""
    hw = annotate_host_hardware(
        info.get("gpus") or [], cpu_load=info.get("load"), cpu_count=info.get("ncpu"),
        cpu_pct=info.get("cpu_pct"), other_load=info.get("other_load"), timing_lock=info.get("timing_lock"),
    )
    return {**info, "gpus": hw["gpus"], "dual_gpu_imbalance": hw["dual_gpu_imbalance"],
            "cpu_load_pct": hw["cpu_load_pct"], "host_band": hw["host_band"]}


__all__ = [
    "DEFAULT_MEASUREMENT_DOMAIN", "MAX_TRAIN_POINTS", "SCHEMA_VERSION", "STALE_SECONDS", "STALL_SECONDS",
    "RunState", "annotate_node", "classify_cohort", "global_summary", "parse_diloco_status", "parse_line",
    "parse_live_runs", "parse_nvidia", "parse_other_load", "parse_timing_lock", "read_v3_scorecard", "summarize_rank",
]
