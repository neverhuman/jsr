"""Shared versioned event adapter (ultra.dashboard.event/v1) for :8765 and :8766 dashboards."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from math import isfinite
import os
from pathlib import Path
import re

SCHEMA_VERSION = "ultra.dashboard.event/v1"
DEFAULT_MEASUREMENT_DOMAIN = "ultra-en-wer-norm/v1"
LEGACY_MEASUREMENT_DOMAIN = "legacy"
GATE_STATES = frozenset({"PASS", "FAIL", "INCONCLUSIVE", "RUNNING"})
STALE_STATES = frozenset({"live", "stale", "offline"})
DEFAULT_SCORECARD_PATH = Path("/mnt/xbabe3-external/jsr/v3-scoreboard/scoreboard.json")
DEFAULT_DILOCO_HUB = "http://127.0.0.1:8768"
_TERMINAL = {"diverged": "diverged", "div": "diverged", "done": "done", "don": "done", "finished": "done", "fin": "done"}

def _ffloat(v: object) -> float | None: return float(v) if type(v) in (int, float) and isfinite(v) else None
def _int_opt(v: object) -> int | None: return int(v) if type(v) is int else None
def is_valid_schema(raw: dict) -> bool:
    s = raw.get("schema") if raw.get("schema") is not None else raw.get("schema_version")
    return s is None or s == SCHEMA_VERSION
def attempt_rank(val: str | int | None) -> tuple[int, float | str]:
    if val is None: return (0, 0)
    if isinstance(val, (int, float)) and not isinstance(val, bool): return (1, float(val))
    t = str(val).strip()
    if t.isdigit(): return (1, float(t))
    m = re.search(r"(\d+)$", t)
    return (2, f"{t[:m.start()]}{int(m.group(1)):012d}") if m else (2, t)
def normalize_gate(val: object) -> str | None:
    up = val.strip().upper() if isinstance(val, str) else ""
    return up if up in GATE_STATES else None
def normalize_stale_state(state: str | None) -> str:
    return "live" if state in ("live", "running", "starting") else ("stale" if state in ("stale", "stalled", "idle") else "offline")
def extract_sdi(rec: dict) -> dict | None:
    for src in (rec.get("sdi"), rec, rec.get("legacy_sdi")):
        if not isinstance(src, dict): continue
        s = src.get("s", src.get("S", src.get("sub", src.get("subs", src.get("substitutions")))))
        d = src.get("d", src.get("D", src.get("del", src.get("dels", src.get("deletions")))))
        i = src.get("i", src.get("I", src.get("ins", src.get("insertions"))))
        if s is None or d is None or i is None: continue
        out = {"s": int(s), "d": int(d), "i": int(i)}
        ref = src.get("ref_words", rec.get("ref_words", src.get("words", src.get("ref_word_count", src.get("n_words")))))
        if ref is not None: out["ref_words"] = int(ref)
        return out
    return None
def extract_artifact(rec: dict) -> dict | None:
    nest = rec.get("artifact") if isinstance(rec.get("artifact"), dict) else {}
    pairs = (("ckpt_sha256", nest.get("ckpt_sha256") or rec.get("ckpt_sha256") or rec.get("checkpoint_sha256") or rec.get("artifact_sha256")),
             ("git_commit", nest.get("git_commit") or rec.get("git_commit") or rec.get("commit")),
             ("git_tree", nest.get("git_tree") or rec.get("git_tree") or rec.get("tree_sha256")))
    return {k: str(v).strip() for k, v in pairs if v} or None
def extract_cohort(rec: dict) -> dict | None:
    raw, out = rec.get("cohort"), {}
    if isinstance(raw, dict): out.update({k: raw[k] for k in ("name", "diloco_round", "roster", "mode") if raw.get(k) is not None})
    elif isinstance(raw, str) and raw.strip(): out["name"] = raw.strip()
    if rec.get("diloco_round") is not None: out["diloco_round"] = int(rec["diloco_round"])
    if isinstance(roster := rec.get("diloco_roster") or rec.get("roster"), list) and roster: out["roster"] = [str(x) for x in roster]
    if isinstance(mode := rec.get("diloco_mode") or rec.get("mode"), str) and mode.strip(): out["mode"] = mode.strip()
    return out or None
def infer_measurement_domain(raw: dict, sdi: dict | None) -> str | None:
    if isinstance(dom := raw.get("measurement_domain") or raw.get("normaliser_id"), str) and dom.strip(): return dom.strip()
    if sdi is None: return None
    legacy = (isinstance(raw.get("legacy_sdi"), dict) or any(k in raw for k in ("sub", "del", "subs", "dels", "substitutions", "deletions"))
              or (raw.get("schema") != SCHEMA_VERSION and raw.get("schema_version") != SCHEMA_VERSION))
    return LEGACY_MEASUREMENT_DOMAIN if legacy else DEFAULT_MEASUREMENT_DOMAIN

def normalize_event(raw: dict, *, default_run_id: str | None = None) -> dict:
    ddp = raw.get("ddp") if isinstance(raw.get("ddp"), dict) else {}
    rank = raw.get("rank") if raw.get("rank") is not None else ddp.get("rank")
    world = raw.get("world_size") if raw.get("world_size") is not None else ddp.get("world_size")
    sdi = extract_sdi(raw)
    planned = raw.get("planned_exposure_ahh")
    if planned is None: planned = raw.get("planned_audio_h_per_h", raw.get("audio_h_per_h", raw.get("ahh")))
    rl = raw.get("rank_local_ahh") if raw.get("rank_local_ahh") is not None else raw.get("rank_audio_h_per_h")
    return {
        "schema": SCHEMA_VERSION, "kind": str(raw.get("kind") or raw.get("k") or "train"),
        "run_id": str(raw.get("run_id") or default_run_id or raw.get("name") or raw.get("id") or "default"),
        "attempt_id": raw.get("attempt_id"), "rank": _int_opt(rank), "world_size": _int_opt(world),
        "seq": _int_opt(raw.get("seq")), "step": _int_opt(raw.get("step")), "time": _ffloat(raw.get("time")),
        "sdi": sdi, "measurement_domain": infer_measurement_domain(raw, sdi), "artifact": extract_artifact(raw),
        "cohort": extract_cohort(raw), "planned_exposure_ahh": _ffloat(planned), "rank_local_ahh": _ffloat(rl),
        "wave_gate": normalize_gate(raw.get("wave_gate") or raw.get("gate")),
        "stale_state": normalize_stale_state(raw.get("stale_state") or raw.get("status")),
    }

@dataclass
class RunCursor:
    run_id: str
    attempt_id: str | int | None = None
    rank: int = 0
    world_size: int | None = None
    seq_by_rank: dict[int, int] = field(default_factory=dict)
    max_train_step_by_rank: dict[int, int] = field(default_factory=dict)
    latest_event: dict | None = None
    train_events: list[dict] = field(default_factory=list)
    eval_events: list[dict] = field(default_factory=list)
    planned_exposure_ahh: float | None = None
    rank_local_ahh: dict[int, float] = field(default_factory=dict)
    artifact: dict | None = None
    cohort: dict | None = None
    sdi: dict | None = None
    measurement_domain: str | None = None
    wave_gate: str | None = None
    status: str = "running"
    quarantined: deque = field(default_factory=lambda: deque(maxlen=64))
    rejected_count: int = 0

    def reset_attempt(self, attempt_id: str | int | None) -> None:
        self.attempt_id, self.latest_event, self.planned_exposure_ahh, self.status, self.wave_gate = attempt_id, None, None, "running", None
        for box in (self.seq_by_rank, self.max_train_step_by_rank, self.train_events, self.eval_events, self.rank_local_ahh):
            box.clear()

class MonotonicEventStore:
    """Monotonic event store keyed by (run_id, attempt_id, rank, seq)."""
    def __init__(self) -> None:
        self.runs: dict[str, RunCursor] = {}

    def cursor(self, key: str) -> RunCursor:
        return self.runs.setdefault(str(key), RunCursor(run_id=str(key)))

    def _reject(self, cur: RunCursor, reason: str, raw: dict) -> tuple[bool, str, dict]:
        cur.rejected_count += 1
        cur.quarantined.append({"reason": reason, "schema": raw.get("schema") or raw.get("schema_version"),
                                "attempt_id": raw.get("attempt_id"), "rank": raw.get("rank"), "seq": raw.get("seq"), "step": raw.get("step")})
        return False, reason, {}

    def ingest(self, raw: dict, *, default_run_id: str | None = None, stream_key: str | None = None) -> tuple[bool, str | None, dict]:
        if not isinstance(raw, dict): return False, "invalid_payload", {}
        cur = self.cursor(str(stream_key or raw.get("run_id") or default_run_id or raw.get("name") or raw.get("id") or "default"))
        if not is_valid_schema(raw): return self._reject(cur, "unknown_schema", raw)
        ev = normalize_event(raw, default_run_id=default_run_id or cur.run_id)
        kind, att, rank, seq, step, advancing = ev["kind"], ev["attempt_id"], int(ev["rank"] or 0), ev["seq"], ev["step"], False
        if att is not None and cur.attempt_id is not None:
            cur_k, new_k = attempt_rank(cur.attempt_id), attempt_rank(att)
            if new_k < cur_k: return self._reject(cur, "older_attempt", raw)
            advancing = new_k > cur_k
        if not advancing:
            if seq is not None and (p := cur.seq_by_rank.get(rank)) is not None and seq <= p:
                return self._reject(cur, "duplicate_seq" if seq == p else "stale_seq", raw)
            if seq is None and kind in ("train", "t") and step is not None and step < cur.max_train_step_by_rank.get(rank, -1):
                return self._reject(cur, "out_of_order_step", raw)
        if advancing: cur.reset_attempt(att)
        elif att is not None and cur.attempt_id is None: cur.attempt_id = att
        if raw.get("run_id") is not None: cur.run_id = str(raw["run_id"])
        if seq is not None: cur.seq_by_rank[rank] = seq
        if kind in ("start", "sta"):
            cur.reset_attempt(cur.attempt_id)
            if seq is not None: cur.seq_by_rank[rank] = seq
            return self._absorb(cur, ev)
        if kind == "resume" and step is not None:
            cur.train_events = [r for r in cur.train_events if (r.get("step") or 0) <= step]
            cur.eval_events = [r for r in cur.eval_events if (r.get("step") or 0) <= step]
            cur.max_train_step_by_rank = {r: min(s, step) for r, s in cur.max_train_step_by_rank.items()}
            return self._absorb(cur, ev)
        self._absorb(cur, ev)
        if kind in _TERMINAL:
            cur.status = _TERMINAL[kind]
            if cur.status == "diverged" and cur.wave_gate is None: cur.wave_gate = "FAIL"
            return True, None, ev
        if kind in ("train", "t"):
            if step is not None: cur.max_train_step_by_rank[rank] = max(step, cur.max_train_step_by_rank.get(rank, step))
            if ev["rank_local_ahh"] is not None: cur.rank_local_ahh[rank] = round(ev["rank_local_ahh"], 1)
            elif ev["rank"] is not None and rank > 0 and ev["planned_exposure_ahh"] is not None: cur.rank_local_ahh[rank] = round(ev["planned_exposure_ahh"], 1)
            if ev["rank"] is None or rank == 0:
                if ev["planned_exposure_ahh"] is not None: cur.planned_exposure_ahh = ev["planned_exposure_ahh"]
                cur.latest_event, cur.train_events = ev, (cur.train_events + [ev])[-1500:]
        elif kind in ("eval", "e"):
            cur.eval_events = (cur.eval_events + [ev])[-1500:]
        return True, None, ev

    @staticmethod
    def _absorb(cur: RunCursor, ev: dict) -> tuple[bool, None, dict]:
        if ev["world_size"] is not None: cur.world_size = ev["world_size"]
        if ev["artifact"]: cur.artifact = {**(cur.artifact or {}), **ev["artifact"]}
        if ev["cohort"]: cur.cohort = {**(cur.cohort or {}), **ev["cohort"]}
        if ev["sdi"]: cur.sdi = ev["sdi"]
        if ev["measurement_domain"]: cur.measurement_domain = ev["measurement_domain"]
        if ev["wave_gate"]: cur.wave_gate = ev["wave_gate"]
        return True, None, ev

    def canonical_view(self, key: str, *, stale_state: str = "live") -> dict | None:
        cur = self.runs.get(str(key))
        if cur is None: return None
        dom = cur.measurement_domain or (DEFAULT_MEASUREMENT_DOMAIN if cur.sdi else LEGACY_MEASUREMENT_DOMAIN)
        return {"schema": SCHEMA_VERSION, "run_id": cur.run_id, "attempt_id": cur.attempt_id, "rank": cur.rank,
                "world_size": cur.world_size, "seq": cur.seq_by_rank.get(cur.rank), "step": (cur.latest_event or {}).get("step"),
                "sdi": cur.sdi, "measurement_domain": dom, "artifact": cur.artifact, "cohort": cur.cohort,
                "planned_exposure_ahh": cur.planned_exposure_ahh,
                "rank_local_ahh": dict(cur.rank_local_ahh) if cur.rank_local_ahh else None,
                "wave_gate": cur.wave_gate, "stale_state": normalize_stale_state(stale_state), "rejected_events": cur.rejected_count}

def run_group_key(run: dict, *, host: str | None = None) -> str:
    ddp = run.get("ddp") if isinstance(run.get("ddp"), dict) else None
    if ddp and ddp.get("group"): return f"ddp:{ddp['group']}"
    if run.get("run_id") and run.get("_shared_group", True): return f"run:{run['run_id']}"
    name = str(run.get("name") or run.get("id") or "")
    if run.get("rank") is not None and re.search(r"([._-]rank\d+)$", name): return f"name:{re.sub(r'([._-]rank\d+)$', '', name)}"
    return f"solo:{f'{host}:' if host else ''}{run.get('id') or name}"
def aggregate_run_groups(runs: list[dict]) -> dict:
    groups, rank_local = {}, {}
    for run in runs:
        gk = run_group_key(run, host=run.get("_host"))
        ddp = run.get("ddp") if isinstance(run.get("ddp"), dict) else None
        rank = run.get("rank") if run.get("rank") is not None else (ddp.get("rank") if ddp else None)
        rate = next((v for k in ("planned_exposure_ahh", "planned_ahh", "ahh", "audio_per_second") if (v := _ffloat(run.get(k))) is not None), None)
        if rank is not None and rate is not None:
            rl = run.get("rank_local_ahh")
            lv = _ffloat(rl.get(rank)) if isinstance(rl, dict) else _ffloat(rl)
            rank_local.setdefault(gk, {})[int(rank)] = round(lv if lv is not None else rate, 1)
        prev = groups.get(gk)
        if prev is None or (rank == 0 and prev.get("_rank") != 0):
            groups[gk] = {**run, "_rank": rank, "_rate": rate}
    rates = [g["_rate"] for g in groups.values() if g["_rate"] is not None]
    return {"groups": groups, "live_runs": len(groups), "planned_exposure_ahh": round(sum(rates), 1) if rates else 0.0,
            "audio_per_second": sum(rates) if rates else None, "rank_local_ahh": rank_local}
def gpu_target_band(util: float | None, *, other_load: dict | None = None, timing_lock: dict | None = None) -> str:
    if timing_lock and timing_lock.get("status") == "held": return "QUIET TIMING"
    if other_load and other_load.get("gpu"): return "OTHER PROJECT"
    if util is None or util < 70.0: return "UNDER <70%"
    return "OVER >90%" if util > 90.0 else ("TARGET 80-85%" if 80.0 <= util <= 85.0 else "NEAR TARGET")
def annotate_host_hardware(gpus: list[dict], *, cpu_load: float | None = None, cpu_count: int | None = None,
                           cpu_pct: float | None = None, other_load: dict | None = None, timing_lock: dict | None = None) -> dict:
    annotated, utils = [], []
    for pos, g in enumerate(gpus or []):
        idx = g.get("index") if g.get("index") is not None else g.get("idx", pos)
        util = _ffloat(g.get("util_pct") if g.get("util_pct") is not None else g.get("util"))
        if util is not None: utils.append(util)
        annotated.append({**g, "index": int(idx), "target_band": gpu_target_band(util, other_load=other_load, timing_lock=timing_lock)})
    load_pct = round(100.0 * float(cpu_load) / float(cpu_count), 1) if (cpu_load is not None and cpu_count and cpu_count > 0) else _ffloat(cpu_pct)
    hband = ("QUIET TIMING" if timing_lock and timing_lock.get("status") == "held" else
             "OTHER PROJECT" if other_load and (other_load.get("gpu") or other_load.get("cpu")) else
             gpu_target_band(sum(utils) / len(utils)) if utils else None)
    return {"gpus": annotated, "dual_gpu_imbalance": len(utils) >= 2 and (max(utils) - min(utils) >= 25.0),
            "cpu_load_pct": load_pct, "host_band": hband}
def read_timing_status(opponents_dir: Path | str | None = None) -> dict | None:
    base = Path(opponents_dir) if opponents_dir is not None else (Path.home() / "jsr-data" / "opponents")
    if not base.is_dir(): return None
    ldir, qfile, status, owner, quiet = base / "TIMING.lock", base / "TIMING.quiet", "free", "", None
    if ldir.is_dir():
        status, of = "held", ldir / "owner"
        try: owner = of.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip() if of.is_file() else "locked"
        except (OSError, IndexError): owner = "locked"
    elif ldir.is_file(): status, owner = "held", "locked"
    if qfile.is_file():
        try: quiet = (qfile.read_text(encoding="utf-8", errors="replace").splitlines() or [None])[0]
        except OSError: quiet = None
    return {"status": status, "owner": owner, "quiet": quiet.strip() if isinstance(quiet, str) else None}
def read_v3_scorecard(path: Path | str | None = None) -> dict | None:
    target = Path(path) if path is not None else Path(os.environ.get("JSR_V3_SCOREBOARD") or os.environ.get("JSR_V3_SCORECARD") or str(DEFAULT_SCORECARD_PATH))
    if not target.is_file(): return None
    try: raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError): return None
    return raw if isinstance(raw, dict) else None
