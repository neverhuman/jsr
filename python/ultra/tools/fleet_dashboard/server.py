"""aiohttp server: tails train.jsonl logs, polls nodes over persistent ssh, pushes WebSocket deltas."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

from tools.dashboard_events import DEFAULT_DILOCO_HUB, DEFAULT_SCORECARD_PATH, SCHEMA_VERSION
from tools.fleet_dashboard.parse import (RunState, annotate_node, global_summary, parse_diloco_status, parse_line,
                                          parse_live_runs, parse_nvidia, parse_other_load,
                                          parse_timing_lock, read_v3_scorecard, summarize_rank)
from tools.fleet_dashboard.remote_tail import tail_remote_node

RUN_ROOT = Path(os.environ.get("FLEET_RUN_ROOT", "/mnt/xbabe3-external/jsr/runs/train"))
DILOCO_HUB = os.environ.get("FLEET_DILOCO_HUB", DEFAULT_DILOCO_HUB)
SCORECARD_PATH = Path(os.environ.get("JSR_V3_SCOREBOARD") or os.environ.get("JSR_V3_SCORECARD") or str(DEFAULT_SCORECARD_PATH))
STATIC = Path(__file__).parent / "static"
LINUX_NODES = os.environ.get("FLEET_NODES", "xbabe3 xbabe0 xbabe1 xbabe2 xbabe5 xbabe6 xbabe7 xbabe8 xbabe9").split()
MAC_NODES = os.environ.get("FLEET_MAC_NODES", "xbabe4").split()
LOCAL_NODE = os.environ.get("FLEET_LOCAL_NODE", "xbabe3")
NODE_PERIOD = float(os.environ.get("FLEET_NODE_PERIOD", "3"))
LOG_PERIOD = float(os.environ.get("FLEET_LOG_PERIOD", "2"))
ACTIVE_DAYS = float(os.environ.get("FLEET_ACTIVE_DAYS", "1"))
# Stopped/finished runs drop out of the table after this many hours (history stays in the charts).
HIDE_AFTER_S = float(os.environ.get("FLEET_HIDE_AFTER_H", "2")) * 3600

# Run name -> node (override with FLEET_HOST_MAP='{"RUN": "xbabeN"}').
HOST_RULES = [
    (r"^D2c", "xbabe5"), (r"^D2_", "xbabe3"), (r"^D3MR", "xbabe1"), (r"^H13a", "xbabe2"),
    (r"^D4M", "xbabe4"), (r"^H13[bc]", "xbabe5"), (r"^H13d", "xbabe6"), (r"^H23", "xbabe6"),
    (r"^L1w", "xbabe8"), (r"^L1", "xbabe8"), (r"^L2", "xbabe9"),
    (r"^L3", "xbabe7"), (r"^H22", "xbabe7"), (r"^R4T960", "xbabe2"), (r"^D2MR", "xbabe2"),
]
HOST_OVERRIDE = json.loads(os.environ.get("FLEET_HOST_MAP", "{}"))

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=10m",
            "-o", f"ControlPath={Path.home()}/.ssh/cm-dash-%r@%h:%p"]

LINUX_PROBE = r"""read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e)); sleep 1
read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e))
echo "CPU $(( 1000*((t2-t1)-(i2-i1))/(t2-t1) ))"; echo "NCPU $(nproc)"
free -m | awk '/Mem/{print "MEM",$3,$2}'; awk '{print "LOAD",$1}' /proc/loadavg
LDIR="$HOME/jsr-data/opponents/TIMING.lock"
if [ -f "$LDIR/owner" ]; then echo "TIMING_LOCK held $(head -n1 "$LDIR/owner")"; elif [ -d "$LDIR" ] || [ -f "$LDIR" ]; then echo "TIMING_LOCK held locked"; elif [ -d "$HOME/jsr-data/opponents" ]; then echo "TIMING_LOCK free"; fi
QFILE="$HOME/jsr-data/opponents/TIMING.quiet"
if [ -f "$QFILE" ]; then echo "TIMING_QUIET $(head -n1 "$QFILE")"; fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r p m; do p=${p// /}; m=${m// /}; [ -n "$p" ] || continue; c=$(ps -o args= -p "$p" 2>/dev/null || true); echo "$c" | grep -qE 'train[.](trainer|mwer|evaluate)|ultra-|jsr' || echo "OTHER_GPU ${m:-0} $(ps -o comm= -p "$p" 2>/dev/null || echo other)"; done || true
ps -eo pcpu=,comm=,args= | awk '$1+0 >= 15 && $0 !~ /train[.](trainer|mwer|evaluate)|ultra-|jsr/ {print "OTHER_CPU",$1,$2}' | head -n 4
ps -eo args | grep -E -- '-m train[.](trainer|mwer) ' | grep -oE -- '--(run-dir|out) [^ ]+' | sed -E 's#/+$##; s#/[^/]*[.]pt$##; s#.*/#RUN #'; echo GPUS
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true"""

MAC_PROBE = r"""idle=$(top -l 2 -n 0 -s 1 | awk '/CPU usage/ {v=$7} END {sub("%","",v); print v}')
echo "CPU $(awk -v i="$idle" 'BEGIN{printf "%d", (100-i)*10}')"; echo "NCPU $(sysctl -n hw.ncpu)"
echo "LOAD $(sysctl -n vm.loadavg | awk '{print $2}')"
ps -axo command | grep -E -- '-m train[.](trainer|mwer) ' | grep -oE -- '--(run-dir|out) [^ ]+' | sed -E 's#/+$##; s#/[^/]*[.]pt$##; s#.*/#RUN #' 
g=$(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o 'Device Utilization %.=[0-9]*' | head -1 | grep -o '[0-9]*$')
echo GPUS; echo "0, M3 Ultra (MPS), ${g:-0}, 0, 0, 0, 0" """


def host_for(run: str) -> str:
    if run in HOST_OVERRIDE:
        return HOST_OVERRIDE[run]
    for pat, host in HOST_RULES:
        if re.search(pat, run):
            return host
    return "?"


class Fleet:
    def __init__(self, *, scorecard_path: Path | None = None, diloco_hub: str | None = None, run_root: Path | None = None) -> None:
        self.runs: dict[str, RunState] = {}
        self.offsets: dict[str, int] = {}
        self.pending: dict[str, dict] = {}
        self.nodes: dict[str, dict] = {n: {"name": n, "up": False} for n in LINUX_NODES + MAC_NODES}
        self.diloco: list[dict] = []
        self.v3_scorecard: dict | None = None
        self.scorecard_path = scorecard_path or SCORECARD_PATH
        self.diloco_hub = diloco_hub if diloco_hub is not None else DILOCO_HUB
        self.run_root = run_root or RUN_ROOT
        self.clients: set[web.WebSocketResponse] = set()
        self.probed: set[str] = set()
        self.dirty = True

    def scan_diloco(self) -> None:
        sc = read_v3_scorecard(self.scorecard_path)
        if sc != self.v3_scorecard:
            self.v3_scorecard = sc
            self.dirty = True
        by_cohort: dict[str, dict] = {}
        try:
            dirs = sorted(d for d in self.run_root.iterdir() if d.is_dir())
        except OSError:
            dirs = []
        for d in dirs:
            st_file = d / "diloco_status.json"
            if not st_file.is_file():
                continue
            try:
                parsed = parse_diloco_status(json.loads(st_file.read_text()))
            except (OSError, ValueError):
                continue
            if parsed is not None:
                by_cohort[parsed["cohort"]] = parsed
        if by_cohort:
            cohorts = [by_cohort[k] for k in sorted(by_cohort)]
            if cohorts != self.diloco:
                self.diloco = cohorts
                self.dirty = True

    async def poll_diloco_hub(self, session: aiohttp.ClientSession | None = None) -> None:
        if not self.diloco_hub:
            return
        url = self.diloco_hub.rstrip("/") + "/status"
        close_after = session is None
        client = session or aiohttp.ClientSession()
        try:
            async with client.get(url, timeout=aiohttp.ClientTimeout(total=1.5)) as resp:
                if resp.status == 200:
                    parsed = parse_diloco_status(await resp.json())
                    if parsed is not None:
                        by_cohort = {c["cohort"]: c for c in self.diloco}
                        by_cohort[parsed["cohort"]] = parsed
                        merged = [by_cohort[k] for k in sorted(by_cohort)]
                        if merged != self.diloco:
                            self.diloco = merged
                            self.dirty = True
        except (OSError, ValueError, asyncio.TimeoutError, aiohttp.ClientError):
            pass
        finally:
            if close_after:
                await client.close()

    # ---------- training logs ----------
    def scan_logs(self) -> None:
        now = time.time()
        try:
            dirs = [d for d in RUN_ROOT.iterdir() if d.is_dir()]
        except OSError:
            return
        for d in dirs:
            log = d / "train.jsonl"
            try:
                st = log.stat()
            except OSError:
                continue
            name = d.name.removesuffix("_copy")
            if name not in self.runs and now - st.st_mtime > ACTIVE_DAYS * 86400:
                continue
            if name in self.runs and d.name.endswith("_copy") and (RUN_ROOT / name).exists():
                continue
            off = self.offsets.get(str(log), 0)
            if st.st_size < off:  # rotated or rewritten
                off = 0
                self.runs.pop(name, None)
            if st.st_size == off:
                continue
            if name not in self.runs:
                total = None
                try:
                    cfg = json.loads((d / "config.json").read_text())
                    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
                    total = tr.get("total_steps") or cfg.get("total_steps") or tr.get("max_steps") or cfg.get("max_steps")
                except (OSError, ValueError):
                    pass
                self.runs[name] = RunState(name=name, host=host_for(name), total_steps=int(total) if total else None)
            rs = self.runs[name]
            for gname in ("wave_gate.json", "gate.json"):
                gfile = d / gname
                if gfile.is_file():
                    try:
                        gd = json.loads(gfile.read_text())
                        gv = str(gd.get("wave_gate") or gd.get("verdict") or gd.get("status") or "").strip().upper()
                        if gv in ("PASS", "FAIL", "INCONCLUSIVE", "RUNNING"):
                            rs.wave_gate = gv
                    except (OSError, ValueError):
                        pass
                    break
            with log.open("rb") as fh:
                fh.seek(off)
                chunk = fh.read()
            # keep a partial last line for the next read
            cut = chunk.rfind(b"\n") + 1
            self.offsets[str(log)] = off + cut
            pend = self.pending.setdefault(name, {"train": [], "evals": []})
            for raw in chunk[:cut].decode("utf-8", "replace").splitlines():
                r = parse_line(raw)
                if r is None:
                    continue
                before_t, before_e = len(rs.train), len(rs.evals)
                rs.add(r)
                if len(rs.train) > before_t:
                    pend["train"].append(rs.train[-1])
                if len(rs.evals) > before_e:
                    pend["evals"].append(rs.evals[-1])
            self.dirty = True

    async def log_loop(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.to_thread(self.scan_logs)
                await asyncio.to_thread(self.scan_diloco)
                await self.poll_diloco_hub(session)
                await asyncio.sleep(LOG_PERIOD)

    # ---------- nodes ----------
    async def probe(self, node: str) -> None:
        mac = node in MAC_NODES
        script = MAC_PROBE if mac else LINUX_PROBE
        if node == LOCAL_NODE:
            cmd = ["bash", "-c", script]
        else:
            cmd = ["ssh", *SSH_OPTS, node, "bash -s"]
        t0 = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(
                proc.communicate(None if node == LOCAL_NODE else script.encode()), timeout=12)
        except (asyncio.TimeoutError, OSError):
            self.nodes[node].update(up=False, err="timeout")
            self.dirty = True
            return
        text = out.decode("utf-8", "replace")
        info: dict = {"name": node, "up": True, "rtt_ms": int((time.time() - t0) * 1000), "mac": mac}
        head, _, gpu_txt = text.partition("GPUS")
        for line in head.splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "CPU" and len(p) > 1:
                info["cpu_pct"] = int(p[1]) / 10.0
            elif p[0] == "NCPU" and len(p) > 1:
                info["ncpu"] = int(p[1])
            elif p[0] == "MEM" and len(p) > 2:
                info["ram_used_gb"], info["ram_total_gb"] = round(int(p[1]) / 1024, 1), round(int(p[2]) / 1024, 1)
            elif p[0] == "LOAD" and len(p) > 1:
                info["load"] = float(p[1])
        info["gpus"] = parse_nvidia(gpu_txt)
        live = parse_live_runs(head)
        tlock = parse_timing_lock(head)
        if tlock is not None:
            info["timing_lock"] = tlock
        other = parse_other_load(head, info["gpus"], live, info.get("cpu_pct"))
        if other:
            info["other_load"] = other
        for name, rs in self.runs.items():  # where each run's processes actually are (DDP spans nodes)
            # on a node's first probe, a live run was not *seen* starting: leave live_since unknown (0)
            rs.set_live(node, name in live, time.time() if node in self.probed else 0.0)
        info["runs"] = sorted(live)
        self.probed.add(node)
        info["t"] = time.time()
        self.nodes[node] = annotate_node(info)
        self.dirty = True

    def _ingest_lines(self, name: str, host: str, total: int | None, lines: list[str]) -> None:
        if name not in self.runs:
            self.runs[name] = RunState(name=name, host=host, total_steps=int(total) if total else None)
        rs = self.runs[name]
        if host != "?":
            rs.host = host
        if total and not rs.total_steps:
            rs.total_steps = int(total)
        pend = self.pending.setdefault(name, {"train": [], "evals": []})
        for raw in lines:
            r = parse_line(raw)
            if not r:
                continue
            bt, be = len(rs.train), len(rs.evals)
            rs.add(r)
            if len(rs.train) > bt:
                pend["train"].append(rs.train[-1])
            if len(rs.evals) > be:
                pend["evals"].append(rs.evals[-1])
        self.dirty = True

    async def node_loop(self, node: str) -> None:
        while True:
            await self.probe(node)
            if node != LOCAL_NODE and self.nodes.get(node, {}).get("up"):
                ranks: list[dict] = []
                for item in await tail_remote_node(node, SSH_OPTS, self.offsets):
                    if "ranks" in item:
                        ranks = [summarize_rank(r, time.time()) for r in item["ranks"]]
                        continue
                    key = f"{node}:{item['path']}"
                    self.offsets[key] = int(item["off"])
                    if item["lines"]:
                        self._ingest_lines(item["run"], node, item.get("total"), item["lines"])
                self.nodes[node]["ranks"] = sorted(ranks, key=lambda r: (str(r["run"]), r["rank"] or 0))
            await asyncio.sleep(NODE_PERIOD)

    # ---------- websocket ----------
    def summaries(self) -> list[dict]:
        now = time.time()
        rank = {"prod": 0, "scratch": 1, "ablation": 2}
        complete = all(n.get("up") for n in self.nodes.values())
        sums = [r.summary(now, inventory_complete=complete) for r in self.runs.values()]
        sums = [s for s in sums if s["status"] in ("running", "starting", "stalled") or (s["age"] or 1e12) < HIDE_AFTER_S]
        return sorted(
            sums,
            key=lambda s: (s["status"] != "running", rank.get(s.get("cohort", "scratch"), 1), s["name"]),
        )

    def snapshot(self) -> dict:
        now = time.time()
        sums = self.summaries()
        complete = all(n.get("up") for n in self.nodes.values())
        nodes = {k: annotate_node(v) if v.get("up") else v for k, v in self.nodes.items()}
        return {
            "schema": SCHEMA_VERSION, "type": "snapshot", "t": now,
            "runs": {n: {"train": r.train, "evals": r.evals} for n, r in self.runs.items()},
            "summaries": sums,
            "canonical_runs": [r.canonical(now, inventory_complete=complete) for r in self.runs.values()],
            "nodes": nodes, "diloco": self.diloco, "v3_scorecard": self.v3_scorecard,
            "global": global_summary(sums, nodes, scorecard=self.v3_scorecard),
        }

    async def broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if not self.dirty or not self.clients:
                continue
            self.dirty = False
            now = time.time()
            sums = self.summaries()
            complete = all(n.get("up") for n in self.nodes.values())
            nodes = {k: annotate_node(v) if v.get("up") else v for k, v in self.nodes.items()}
            msg = json.dumps({
                "schema": SCHEMA_VERSION, "type": "delta", "t": now,
                "points": {n: p for n, p in self.pending.items() if p["train"] or p["evals"]},
                "summaries": sums,
                "canonical_runs": [r.canonical(now, inventory_complete=complete) for r in self.runs.values()],
                "nodes": nodes, "diloco": self.diloco, "v3_scorecard": self.v3_scorecard,
                "global": global_summary(sums, nodes, scorecard=self.v3_scorecard),
            }, separators=(",", ":"))
            self.pending = {}
            dead = []
            for ws in self.clients:
                try:
                    await ws.send_str(msg)
                except (ConnectionError, RuntimeError):
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


async def index(_req: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def ws_handler(req: web.Request) -> web.WebSocketResponse:
    fleet: Fleet = req.app["fleet"]
    ws = web.WebSocketResponse(heartbeat=20, compress=True)
    await ws.prepare(req)
    await ws.send_str(json.dumps(fleet.snapshot(), separators=(",", ":")))
    fleet.clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
    finally:
        fleet.clients.discard(ws)
    return ws


async def start_background(app: web.Application) -> None:
    fleet: Fleet = app["fleet"]
    await asyncio.to_thread(fleet.scan_logs)
    await asyncio.to_thread(fleet.scan_diloco)
    app["tasks"] = [asyncio.create_task(fleet.log_loop()), asyncio.create_task(fleet.broadcast_loop())]
    app["tasks"] += [asyncio.create_task(fleet.node_loop(n)) for n in fleet.nodes]


async def stop_background(app: web.Application) -> None:
    for t in app["tasks"]:
        t.cancel()


async def api_snapshot(req: web.Request) -> web.Response:
    return web.json_response(req.app["fleet"].snapshot())


async def api_v3_scoreboard(req: web.Request) -> web.Response:
    sc = read_v3_scorecard(req.app["fleet"].scorecard_path)
    if sc is None:
        return web.json_response({"error": "v3 scoreboard unavailable"}, status=503,
                                 headers={"Cache-Control": "no-cache"})
    return web.json_response(sc, headers={"Cache-Control": "no-cache"})


def make_app(*, fleet: Fleet | None = None, start_background_tasks: bool = True) -> web.Application:
    app = web.Application()
    app["fleet"] = fleet or Fleet()
    app.router.add_get("/", index)
    app.router.add_get("/api/snapshot", api_snapshot)
    app.router.add_get("/v3/scoreboard.json", api_v3_scoreboard)
    app.router.add_get("/ws", ws_handler)
    if start_background_tasks:
        app.on_startup.append(start_background)
        app.on_cleanup.append(stop_background)
    return app


def main() -> None:
    host = os.environ.get("FLEET_BIND", "127.0.0.1")
    port = int(os.environ.get("FLEET_PORT", "8765"))
    print(f"fleet dashboard on http://{host}:{port}  (ssh -L {port}:127.0.0.1:{port} <host>)", flush=True)
    web.run_app(make_app(), host=host, port=port, print=None)
