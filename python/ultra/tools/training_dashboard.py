"""Serve the read-only training dashboard (:8766) with unified ULTRA v3 event & fleet panels."""

import argparse
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from tools.dashboard_events import (
    DEFAULT_DILOCO_HUB,
    DEFAULT_SCORECARD_PATH,
    MonotonicEventStore,
    SCHEMA_VERSION,
    attempt_rank,
    read_v3_scorecard,
)
from tools.fleet_dashboard.parse import parse_diloco_status as normalize_diloco_status
from tools.training_dashboard_metrics import attach_shared_runs, fleet_summary, host_view, probe_view

ROOT = Path(__file__).resolve().parents[3]
HTML = ROOT / "training-dashboard.html"
PROBE = Path(__file__).with_name("training_dashboard_probe.py")
DEFAULT_HOSTS = tuple(f"xbabe{i}" for i in range(10))
SSH = ("ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2")
FRAME_LIMIT = 4 << 20
V3_OVERLAY_HTML = """
<style id="ultra-v3-css">
.v3-wrap{max-width:94rem;margin:0 auto 2rem;padding:0 clamp(1rem,3vw,2.75rem);display:grid;gap:1rem}
.v3-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.75rem}
.v3-card{background:#171e2a;border:1px solid #303b4c;border-radius:.75rem;padding:.9rem 1rem}
.v3-badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.75rem;font-weight:600;margin-right:.35rem}
.v3-ok{background:#143a2e;color:#65e0b5}.v3-warn{background:#3f2d14;color:#f8bb76}.v3-bad{background:#421b20;color:#ffa29b}.v3-muted{background:#232c3b;color:#aab8cb}
.v3-bar{display:grid;grid-template-columns:56px 1fr 118px;gap:.45rem;align-items:center;font-size:.78rem;margin:.25rem 0}
.v3-track{height:7px;background:#0d131e;border-radius:4px;position:relative;overflow:hidden}
.v3-band{position:absolute;left:80%;width:5%;top:0;bottom:0;background:rgba(101,224,181,.32)}
.v3-fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.v3-table{width:100%;border-collapse:collapse;font-size:.82rem}.v3-table th,.v3-table td{padding:.38rem .5rem;border-bottom:1px solid #263143;text-align:right;font-family:var(--numbers)}.v3-table th:first-child,.v3-table td:first-child{text-align:left}
</style>
<section class="v3-wrap" id="ultra-v3-panels" aria-label="ULTRA v3 Unified Fleet & Scorecard">
  <article class="v3-card" id="v3-scorecard-card"><h3>ULTRA v3 Best-Model Scorecard (WER · FTL Both Clocks · ×RT · Always-On Watts)</h3><div id="v3-scorecard-body" class="secondary" style="margin-top:.45rem">Waiting for @v3-scoreboard/scoreboard.json…</div></article>
  <article class="v3-card" id="v3-diloco-card"><h3>Live Cross-Node DiLoCo Hub (127.0.0.1:8768)</h3><div id="v3-diloco-body" class="secondary" style="margin-top:.45rem">Waiting for DiLoCo cohorts…</div></article>
  <article class="v3-card" id="v3-fleet-card"><h3>Full Fleet Hardware (xbabe0–xbabe9 · Per-GPU 80–85% Target Band · CPU load/nproc · TIMING.lock · Other Project)</h3><div id="v3-fleet-grid" class="v3-grid" style="margin-top:.55rem"></div></article>
</section>
<script id="ultra-v3-js">
(()=>{const $=id=>document.getElementById(id),esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const bandCls=b=>b==="TARGET 80-85%"?"v3-ok":b==="OVER >90%"?"v3-bad":(b==="OTHER PROJECT"||b==="QUIET TIMING")?"v3-muted":"v3-warn";
const fillCol=b=>b==="TARGET 80-85%"?"#65e0b5":b==="OVER >90%"?"#ffa29b":b==="OTHER PROJECT"?"#6f7d95":"#f8bb76";
function renderV3(snap){if(!snap)return;const sc=snap.v3_scorecard||(snap.fleet&&snap.fleet.v3_scorecard);
  const rows=(sc&&(sc.rows||sc.models||sc.progression))||[];
  $("v3-scorecard-body").innerHTML=rows.length?`<table class="v3-table"><thead><tr><th>Model</th><th>WER</th><th>FTL (Audio)</th><th>FTL (Wall)</th><th>×RT</th><th>Watts</th><th>Evidence</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.name||r.id||r.row||"?")}</td><td>${esc(r.wer??r.pooled_wer??"–")}</td><td>${esc(r.ftl_audio_ms??r.ftl_end_p50_ms??"–")}</td><td>${esc(r.ftl_wall_ms??r.ftl_paced_ms??"–")}</td><td>${esc(r.x_real_time??r.xrt??"–")}</td><td>${esc(r.always_on_watts??r.watts_per_stream_core_register??r.watts??"–")}</td><td>${esc(r.evidence||r.status||"–")}</td></tr>`).join("")}</tbody></table>`:(sc&&sc.release_gate?`Release gate active: ≤${esc(sc.release_gate.max_edits)} edits / ${esc(sc.release_gate.ref_words)} ref words`:"No /mnt/xbabe3-external/jsr/v3-scoreboard/scoreboard.json yet (graceful standby).");
  const dl=snap.diloco||(snap.fleet&&snap.fleet.diloco)||[];
  $("v3-diloco-body").innerHTML=dl.length?dl.map(c=>`<div style="margin:.35rem 0"><span class="v3-badge ${c.mode==="sync_roster"?"v3-ok":"v3-warn"}">${esc(c.mode_label||c.mode)}</span> <b>${esc(c.cohort)}</b> · Round #${esc(c.round)} · workers ${esc(c.worker_count)} (pending ${esc(c.pending_count)}/${esc(c.quorum_k)}) · eff ${esc(c.effective_audio_h_per_h)} ah/h · cos ${esc(c.mean_cos??"–")}</div>`).join(""):"No active DiLoCo cohort on 127.0.0.1:8768 yet.";
  const hs=Array.isArray(snap.hosts)?snap.hosts:[];
  $("v3-fleet-grid").innerHTML=hs.map(h=>{const badges=[];
    if(h.other_load&&(h.other_load.gpu||h.other_load.cpu))badges.push(`<span class="v3-badge v3-muted">${esc(h.other_load.label||"OTHER PROJECT")}</span>`);
    if(h.timing_lock)badges.push(`<span class="v3-badge ${h.timing_lock.status==="held"?"v3-bad":"v3-ok"}">TIMING.lock ${h.timing_lock.status==="held"?"HELD":"FREE"}${h.timing_lock.owner?": "+esc(h.timing_lock.owner):""}</span>`);
    if(h.timing_lock&&h.timing_lock.quiet)badges.push(`<span class="v3-badge v3-muted">${esc(h.timing_lock.quiet)}</span>`);
    if(h.dual_gpu_imbalance)badges.push(`<span class="v3-badge v3-warn">DUAL-GPU IMBALANCE</span>`);
    const cpuP=h.cpu_load_pct!=null?h.cpu_load_pct:h.cpu_pct;
    const cpuBar=`<div class="v3-bar"><span>CPU</span><div class="v3-track"><div class="v3-fill" style="width:${Math.min(100,Math.max(0,cpuP||0))}%;background:#88c6fd"></div></div><span class="numeric">${cpuP==null?"–":cpuP.toFixed(0)+"%"} (${esc(h.cpu_load??"–")}/${esc(h.cpu_count??"–")})</span></div>`;
    const gpuBars=(h.gpus||[]).map(g=>{const u=g.util_pct??0,b=g.target_band||"UNDER <70%";return `<div class="v3-bar"><span>GPU${esc(g.index)}</span><div class="v3-track"><div class="v3-band"></div><div class="v3-fill" style="width:${Math.min(100,Math.max(0,u))}%;background:${fillCol(b)}"></div></div><span><b class="numeric">${u.toFixed(0)}%</b> <span class="v3-badge ${bandCls(b)}">${esc(b)}</span></span></div>`;}).join("")||`<div class="secondary">No GPU</div>`;
    return `<div class="v3-card" style="background:#1c2533"><div style="display:flex;justify-content:space-between"><b>${esc(h.host)}</b><span class="secondary">${esc(h.state)}</span></div><div style="margin:.3rem 0">${badges.join("")}</div>${cpuBar}${gpuBars}</div>`;}).join("");}
async function poll(){try{const r=await fetch("/api/snapshot",{cache:"no-store"});if(r.ok)renderV3(await r.json());}catch{}}
poll();setInterval(poll,3000);})();
</script>
"""


def wire(message):
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def render_dashboard_html() -> str:
    raw = HTML.read_text(encoding="utf-8")
    return raw.replace("</body>", f"{V3_OVERLAY_HTML}\n</body>", 1) if "</body>" in raw else raw + V3_OVERLAY_HTML


class Fleet:
    def __init__(self, hosts, *, scorecard_path=None, diloco_hub=None, run_root=None):
        self.readings = {host: (None, None, "") for host in hosts}
        self.hosts = {}
        self.revisions = dict.fromkeys(hosts, 0)
        self.history = deque(maxlen=1440)
        self.waiters = set()
        self.clients = set()
        self.event_store = MonotonicEventStore()
        self._run_cache: dict[tuple[str, str], dict] = {}
        self.scorecard_path = scorecard_path if scorecard_path is not None else DEFAULT_SCORECARD_PATH
        self.diloco_hub = diloco_hub if diloco_hub is not None else os.environ.get("JSR_DILOCO_HUB", DEFAULT_DILOCO_HUB)
        self.run_root = Path(run_root) if run_root is not None else Path(os.environ.get("FLEET_RUN_ROOT", "/mnt/xbabe3-external/jsr/runs/train"))
        self.v3_scorecard = None
        self.diloco: list[dict] = []
        self.pulse()

    def _merge_run_monotonic(self, host: str, run: dict) -> dict:
        cache_key = (host, run["id"])
        stream_key = f"{host}:{run['run_id'] or run['id']}"
        cur = self.event_store.cursor(stream_key)
        prev_run = self._run_cache.get(cache_key)
        att, rank, seq, step = run.get("attempt_id"), int(run.get("rank") or 0), run.get("seq"), run.get("step")
        if prev_run is not None:
            if att is not None and cur.attempt_id is not None and attempt_rank(att) < attempt_rank(cur.attempt_id):
                cur.rejected_count += 1
                return {**prev_run, "rejected_events": prev_run.get("rejected_events", 0) + 1}
            same_att = att is None or cur.attempt_id is None or attempt_rank(att) == attempt_rank(cur.attempt_id)
            if same_att:
                prev_seq = cur.seq_by_rank.get(rank)
                if seq is not None and prev_seq is not None and (seq < prev_seq or (seq == prev_seq and step != prev_run.get("step"))):
                    cur.rejected_count += 1
                    return {**prev_run, "rejected_events": prev_run.get("rejected_events", 0) + 1}
                prev_step = cur.max_train_step_by_rank.get(rank)
                if seq is None and step is not None and prev_step is not None and step < prev_step and run.get("train"):
                    cur.rejected_count += 1
                    return {**prev_run, "rejected_events": prev_run.get("rejected_events", 0) + 1}
        if att is not None:
            if cur.attempt_id is not None and attempt_rank(att) > attempt_rank(cur.attempt_id):
                cur.reset_attempt(att)
            elif cur.attempt_id is None:
                cur.attempt_id = att
        if seq is not None:
            cur.seq_by_rank[rank] = max(seq, cur.seq_by_rank.get(rank, seq))
        if step is not None and run.get("train"):
            cur.max_train_step_by_rank[rank] = max(int(step), cur.max_train_step_by_rank.get(rank, int(step)))
        merged = {**run, "rejected_events": max(run.get("rejected_events", 0), cur.rejected_count)}
        self._run_cache[cache_key] = merged
        return merged

    def feed(self, host, payload):
        view = probe_view(payload)
        view["runs"] = [self._merge_run_monotonic(host, run) for run in view["runs"]]
        self.readings[host] = (view, time.time(), "")

    def lost(self, host, reason):
        payload, seen, _ = self.readings[host]
        self.readings[host] = (payload, seen, reason)

    def refresh_scorecard_and_files(self) -> None:
        self.v3_scorecard = read_v3_scorecard(self.scorecard_path)
        file_cohorts = {}
        try:
            dirs = sorted(d for d in self.run_root.iterdir() if d.is_dir()) if self.run_root.is_dir() else []
        except OSError:
            dirs = []
        for d in dirs:
            st_file = d / "diloco_status.json"
            if not st_file.is_file():
                continue
            try:
                parsed = normalize_diloco_status(json.loads(st_file.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            if parsed is not None:
                file_cohorts[parsed["cohort"]] = parsed
        if file_cohorts:
            existing = {c["cohort"]: c for c in self.diloco}
            existing.update(file_cohorts)
            self.diloco = [existing[k] for k in sorted(existing)]

    async def poll_v3(self, session: aiohttp.ClientSession | None = None) -> None:
        self.refresh_scorecard_and_files()
        if not self.diloco_hub:
            return
        url = self.diloco_hub.rstrip("/") + "/status"
        close_after = session is None
        client = session or aiohttp.ClientSession()
        try:
            async with client.get(url, timeout=aiohttp.ClientTimeout(total=1.5)) as resp:
                if resp.status == 200:
                    parsed = normalize_diloco_status(await resp.json())
                    if parsed is not None:
                        by_cohort = {c["cohort"]: c for c in self.diloco}
                        by_cohort[parsed["cohort"]] = parsed
                        self.diloco = [by_cohort[k] for k in sorted(by_cohort)]
        except (OSError, ValueError, asyncio.TimeoutError, aiohttp.ClientError):
            pass
        finally:
            if close_after:
                await client.close()

    def pulse(self):
        self.at = now = time.time()
        views = {host: host_view(host, reading, seen, detail, now)
                 for host, (reading, seen, detail) in self.readings.items()}
        attach_shared_runs(views)
        for host, view in views.items():
            if view != self.hosts.get(host):
                self.revisions[host] += 1
                self.hosts[host] = view
        self.summary = fleet_summary(self.hosts.values(), self.history, now,
                                     scorecard=self.v3_scorecard, diloco=self.diloco)
        for waiter in self.waiters:
            waiter.set()

    def snapshot_payload(self):
        return {
            "schema": SCHEMA_VERSION, "type": "snapshot", "at": self.at,
            "hosts": list(self.hosts.values()), "fleet": self.summary,
            "v3_scorecard": self.v3_scorecard, "diloco": self.diloco,
        }

    async def clock(self):
        async with aiohttp.ClientSession() as session:
            ticks = 0
            while True:
                if ticks % 3 == 0:
                    await self.poll_v3(session)
                self.pulse()
                ticks += 1
                await asyncio.sleep(1)

    async def publish(self, ws, waiter):
        async def send(kind, extra, summary):
            frame = wire(dict(schema=SCHEMA_VERSION, type=kind, at=self.at, fleet=summary,
                              v3_scorecard=self.v3_scorecard, diloco=self.diloco, **extra))
            await asyncio.wait_for(ws.send_str(frame), timeout=5)

        revisions = self.revisions.copy()
        await send("snapshot", {"hosts": list(self.hosts.values())}, self.summary)
        while not ws.closed:
            await waiter.wait()
            waiter.clear()
            changed = [host for host in revisions if self.revisions[host] != revisions[host]]
            if not changed:
                await send("tick", {}, self.summary)
            for index, host in enumerate(changed):
                revisions[host] = self.revisions[host]
                summary = self.summary if index == len(changed) - 1 else {
                    key: value for key, value in self.summary.items() if key != "history"
                }
                await send("update", {"host": self.hosts[host]}, summary)


async def read_samples(stream, fleet, host, timeout=None):
    while line := await asyncio.wait_for(stream.readline(), timeout):
        try:
            fleet.feed(host, json.loads(line))
        except (ValueError, TypeError, OverflowError):
            continue


async def reap(process):
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 2)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


async def collect(host, fleet, args):
    local = host == args.local_host
    local_args = [part for path in args.run_dir for part in ("--run-dir", path)]
    command = [sys.executable, "-u", str(PROBE), *local_args] if local else [*SSH, host, "python3", "-u", "-"]
    source = PROBE.read_bytes() if not local else b""
    attempt = 0
    while True:
        process = None
        last_sample = fleet.readings[host][1]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE if source else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=FRAME_LIMIT,
            )
            if source:
                process.stdin.write(source)
                await asyncio.wait_for(process.stdin.drain(), 5)
                process.stdin.close()
            await read_samples(process.stdout, fleet, host, timeout=15)
            code = await asyncio.wait_for(process.wait(), 2)
            fleet.lost(host, f"Collector exited ({code})")
        except asyncio.TimeoutError:
            fleet.lost(host, "Collector timed out")
        except (OSError, ValueError):
            fleet.lost(host, "Collector connection failed")
        finally:
            if process is not None:
                await reap(process)
        attempt = 0 if fleet.readings[host][1] != last_sample else min(5, attempt + 1)
        await asyncio.sleep(2 ** attempt)


async def collect_pipe(host, fleet):
    loop = asyncio.get_running_loop()
    stream = asyncio.StreamReader(limit=FRAME_LIMIT)
    transport = None
    try:
        transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stream), sys.stdin.buffer)
        await read_samples(stream, fleet, host)
        fleet.lost(host, "Input pipe closed")
    except (OSError, ValueError):
        fleet.lost(host, "Input pipe unavailable")
    finally:
        if transport:
            transport.close()


def allowed_origin(request, extra):
    origin = request.headers.get("Origin")
    if origin is None or origin in extra:
        return True
    url = urlsplit(origin)
    return (url.scheme in ("http", "https") and url.netloc == request.host
            and url.hostname in ("localhost", "127.0.0.1", "::1"))


async def handle_websocket(request, fleet, extra_origins):
    try:
        allowed = allowed_origin(request, extra_origins)
    except ValueError:
        allowed = False
    if not allowed:
        raise web.HTTPForbidden(text="Use the served dashboard or an explicitly allowed origin.")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1024, compress=True)
    await ws.prepare(request)
    waiter = asyncio.Event()
    fleet.clients.add(ws)
    fleet.waiters.add(waiter)
    sender = asyncio.create_task(fleet.publish(ws, waiter))

    async def receive():
        async for message in ws:
            if message.type in (web.WSMsgType.TEXT, web.WSMsgType.BINARY):
                await ws.close(code=1008, message=b"Dashboard stream is read-only")
                return

    receiver = asyncio.create_task(receive())
    try:
        await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
    finally:
        fleet.waiters.discard(waiter)
        fleet.clients.discard(ws)
        for task in (sender, receiver):
            task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)
        await ws.close()
    return ws


def application(args, *, start_collectors=True, run_root=None):
    app = web.Application()
    fleet = Fleet(args.hosts, scorecard_path=getattr(args, "scorecard_path", None),
                  diloco_hub=getattr(args, "diloco_hub", None), run_root=run_root)
    app["fleet"] = fleet

    async def index(_request):
        return web.Response(text=render_dashboard_html(), content_type="text/html",
                            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    async def api_snapshot(_request):
        return web.json_response(fleet.snapshot_payload(), headers={"Cache-Control": "no-cache"})

    async def websocket(request):
        return await handle_websocket(request, fleet, args.allow_origin)

    async def background(_app):
        tasks = [asyncio.create_task(fleet.clock())]
        if start_collectors:
            for host in args.hosts:
                work = collect_pipe(host, fleet) if host == args.stdin else collect(host, fleet, args)
                tasks.append(asyncio.create_task(work))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_clients(_app):
        await asyncio.gather(*(ws.close(code=1001) for ws in list(fleet.clients)), return_exceptions=True)

    app.router.add_get("/", index)
    app.router.add_get("/training-dashboard.html", index)
    app.router.add_get("/api/snapshot", api_snapshot)
    app.router.add_get("/ws", websocket)
    app.cleanup_ctx.append(background)
    app.on_shutdown.append(close_clients)
    return app


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS, metavar="HOST")
    parser.add_argument("--local-host", default=socket.gethostname().split(".")[0])
    parser.add_argument("--run-dir", action="append", default=[], help="also inspect this run directory on the local host")
    parser.add_argument("--stdin", metavar="HOST", help="read this host's probe JSON lines from a pipe")
    parser.add_argument("--allow-origin", action="append", default=[], help="exact additional WebSocket origin")
    parser.add_argument("--scorecard-path", default=None, help="path to @v3-scoreboard scoreboard.json")
    parser.add_argument("--diloco-hub", default=os.environ.get("JSR_DILOCO_HUB", DEFAULT_DILOCO_HUB),
                        help="DiLoCoHub HTTP URL (default http://127.0.0.1:8768)")
    args = parser.parse_args(argv)
    if not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", host) for host in args.hosts):
        parser.error("--hosts expects SSH aliases or hostnames")
    args.hosts = tuple(dict.fromkeys(args.hosts))
    if args.stdin and args.stdin not in args.hosts:
        parser.error("--stdin must name one of --hosts")
    return args


def main():
    args = arguments()
    print(f"Training dashboard on http://{args.bind}:{args.port} ({len(args.hosts)} nodes)", flush=True)
    web.run_app(application(args), host=args.bind, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
