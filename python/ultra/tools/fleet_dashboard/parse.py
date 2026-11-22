"""Pure parsing and aggregation for the fleet dashboard (no I/O; unit-tested)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Train history is decimated so a client snapshot stays small (fast first paint).
MAX_TRAIN_POINTS = 1500
STALE_SECONDS = 300
# A live trainer whose log has been silent this long is a real stall (evals can take a few minutes).
STALL_SECONDS = 600
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def parse_live_runs(text: str) -> set[str]:
    """Run names from probe lines `RUN <name>` (the basename of each live trainer's --run-dir)."""
    out = set()
    for line in text.splitlines():
        p = line.split()
        if len(p) == 2 and p[0] == "RUN" and _RUN_NAME.match(p[1]):
            out.add(p[1])
    return out


def classify_cohort(name: str) -> str:
    """Separate production/warm runs from 960h scratch and 100h ablations."""
    if re.search(r"^(D2_S_|D2[c-z]_|L1w)", name) or "warm" in name.lower():
        return "prod"
    if re.search(r"^(H13[bcd]|H23|ABL_|D0_|D1)", name):
        return "ablation"
    return "scratch"


def parse_line(line: str) -> dict | None:
    """Parse one train.jsonl line into a compact record, or None if it is not useful."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    kind = rec.get("kind")
    if kind == "train":
        return {
            "k": "t",
            "step": rec.get("step"),
            "ep": (rec.get("pos") or {}).get("epoch"),
            "loss": rec.get("loss"),
            "rnnt": rec.get("rnnt"),
            "ctc": rec.get("ctc"),
            "lr": rec.get("lr"),
            "gn": rec.get("grad_norm"),
            "ahh": rec.get("audio_h_per_h"),
            "audio_s": rec.get("audio_s"),
            "pad_pct": rec.get("pad_pct"),
            "wait_pct": rec.get("data_wait_pct"),
            "qkv": rec.get("max_qkv_norm"),
            "time": rec.get("time"),
        }
    if kind == "eval":
        out = {
            "k": "e",
            "step": rec.get("step"),
            "ep": rec.get("epoch"),
            "split": rec.get("split"),
            "wer": rec.get("wer"),
            "time": rec.get("time"),
        }
        if rec.get("ema_wer") is not None:
            out["ema_wer"] = rec.get("ema_wer")
        return out
    if kind in ("diverged", "start", "done", "finished"):
        return {"k": kind[0:3], "time": rec.get("time"), "step": rec.get("step"), "msg": str(rec.get("error", ""))[:160]}
    return None


@dataclass
class RunState:
    """Incrementally-updated state of one training run."""

    name: str
    host: str = "?"
    train: list = field(default_factory=list)  # [step, loss, ahh, time, lr, rnnt]
    evals: list = field(default_factory=list)  # [step, split, wer, time, ema_wer]
    last: dict = field(default_factory=dict)
    audio_h_total: float = 0.0
    status: str = "running"
    stride: int = 1
    total_steps: int | None = None
    live_hosts: set = field(default_factory=set)  # nodes with a live trainer process for this run (DDP: several)
    live_since: float = 0.0  # when a live process was first seen after none (restart / new launch)
    _n: int = 0
    _recent: list = field(default_factory=list)  # [(step, time)] for the rate / ETA

    def add(self, r: dict) -> None:
        k = r["k"]
        if k == "t":
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
        elif k == "div":
            self.status = "diverged"
        elif k in ("don", "fin"):
            self.status = "done"

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

    def summary(self, now: float, inventory_complete: bool = False) -> dict:
        """inventory_complete: every node answered its latest probe, so 'no live process' is authoritative."""
        lt = self.last.get("time") or 0
        best = self.best_wer()
        best_ema = self.best_ema_wer()
        last_e = next((e for e in reversed(self.evals) if e[1] == "dev-clean"), None)
        last_o = next((e for e in reversed(self.evals) if e[1] == "dev-other"), None)
        last_ema = next((e for e in reversed(self.evals) if e[1] == "dev-clean" and len(e) > 4 and e[4] is not None), None)
        age = (now - lt) if lt else None
        if self.status in ("diverged", "done"):
            status = self.status
        elif self.live_hosts:  # a process is alive
            fresh_start = not lt or self.live_since > lt  # no train record since this launch yet
            if fresh_start and now - self.live_since <= STALL_SECONDS:
                status = "starting"
            else:  # silent log = stalled
                status = "stalled" if (age is None or age > STALL_SECONDS) else "running"
        elif inventory_complete:  # every node reported and none runs it
            status = "stopped"
        else:  # some node unreachable: trust a fresh log
            status = "running" if (age is not None and age <= STALE_SECONDS) else "stopped"
        rate = None
        if len(self._recent) >= 2:
            (s0, t0), (s1, t1) = self._recent[0], self._recent[-1]
            if t1 > t0 and s1 > s0:
                rate = (s1 - s0) / (t1 - t0) * 3600.0
        step = self.last.get("step")
        eta_h = None
        if rate and self.total_steps and step is not None and self.total_steps > step:
            eta_h = (self.total_steps - step) / rate
        return {
            "name": self.name,
            "cohort": classify_cohort(self.name),
            "total_steps": self.total_steps,
            "steps_per_h": round(rate) if rate else None,
            "eta_h": round(eta_h, 1) if eta_h is not None else None,
            "progress": round(100.0 * step / self.total_steps, 1) if (self.total_steps and step is not None) else None,
            "host": ",".join(sorted(self.live_hosts)) if self.live_hosts else self.host,
            "hosts": sorted(self.live_hosts),
            "step": step,
            "ep": self.last.get("ep"),
            "loss": self.last.get("loss"),
            "rnnt": self.last.get("rnnt"),
            "lr": self.last.get("lr"),
            "gn": self.last.get("gn"),
            "pad_pct": self.last.get("pad_pct"),
            "wait_pct": self.last.get("wait_pct"),
            "qkv": self.last.get("qkv"),
            "ahh": self.last.get("ahh"),
            "audio_h_total": round(self.audio_h_total, 1),
            "best": best,
            "best_ema": best_ema,
            "last_clean": last_e,
            "last_ema": last_ema,
            "last_other": last_o,
            "age": age,
            "status": status,
        }


def global_summary(runs: list[dict], nodes: dict) -> dict:
    """Fleet-wide aggregates for the header KPIs."""
    live = [r for r in runs if r["status"] == "running" and r.get("ahh")]
    gpus = [g for n in nodes.values() for g in (n.get("gpus") or [])]
    mem_used = sum(g["mem_used"] for g in gpus)
    mem_total = sum(g["mem_total"] for g in gpus) or 1
    cpu_threads = sum(n.get("ncpu") or 0 for n in nodes.values())
    cpu_busy = sum((n.get("cpu_pct") or 0) / 100.0 * (n.get("ncpu") or 0) for n in nodes.values())
    wers = []
    prod_wers = []
    for r in runs:
        w = r["best_ema"][4] if r.get("best_ema") else (r["best"][2] if r.get("best") else None)
        if w is not None:
            wers.append(w)
            if r.get("cohort") == "prod":
                prod_wers.append(w)
    return {
        "audio_h_per_h": round(sum(r["ahh"] for r in live), 1),
        "live_runs": len(live),
        "runs": len(runs),
        "gpus": len(gpus),
        "gpus_busy": sum(1 for g in gpus if g["util"] >= 50),
        "gpu_util": round(sum(g["util"] for g in gpus) / len(gpus), 1) if gpus else 0,
        "gpu_mem_pct": round(100.0 * mem_used / mem_total, 1),
        "gpu_mem_used_gb": round(mem_used / 1024, 1),
        "gpu_mem_total_gb": round(mem_total / 1024, 1),
        "cpu_pct": round(100.0 * cpu_busy / cpu_threads, 1) if cpu_threads else 0,
        "cpu_threads": cpu_threads,
        "best_wer": min(wers) if wers else None,
        "best_prod_wer": min(prod_wers) if prod_wers else None,
        "nodes_up": sum(1 for n in nodes.values() if n.get("up")),
        "nodes": len(nodes),
        "stalled": sorted(r["name"] for r in runs if r["status"] == "stalled"),
        # GPU nodes that are up with no live trainer process: idle capacity
        "idle_nodes": sorted(k for k, n in nodes.items()
                             if n.get("up") and n.get("gpus") and not n.get("mac") and not n.get("runs")),
    }


def parse_nvidia(text: str) -> list[dict]:
    """Parse `nvidia-smi` CSV lines into GPU dicts."""
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

        out.append({
            "idx": int(parts[0]),
            "name": parts[1].replace("NVIDIA ", "").replace("GeForce ", ""),
            "util": num(parts[2]),
            "mem_used": num(parts[3]),
            "mem_total": num(parts[4]),
            "power": num(parts[5]) if len(parts) > 5 else 0.0,
            "temp": num(parts[6]) if len(parts) > 6 else 0.0,
        })
    return out
