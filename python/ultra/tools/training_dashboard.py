"""Serve the read-only training dashboard on a local WebSocket."""

import argparse
import asyncio
from collections import deque
import json
from pathlib import Path
import re
import socket
import sys
import time
from urllib.parse import urlsplit

from aiohttp import web

from tools.training_dashboard_metrics import fleet_summary, host_view, probe_view

ROOT = Path(__file__).resolve().parents[3]
HTML = ROOT / "training-dashboard.html"
PROBE = Path(__file__).with_name("training_dashboard_probe.py")
DEFAULT_HOSTS = tuple(f"xbabe{i}" for i in range(10))
SSH = ("ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2")
FRAME_LIMIT = 1 << 20


def wire(message):
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


class Fleet:
    def __init__(self, hosts):
        self.readings = {host: (None, None, "") for host in hosts}
        self.hosts = {}
        self.revisions = dict.fromkeys(hosts, 0)
        self.history = deque(maxlen=240)
        self.waiters = set()
        self.clients = set()
        self.pulse()

    def feed(self, host, payload):
        self.readings[host] = (probe_view(payload), time.time(), "")

    def lost(self, host, reason):
        payload, seen, _ = self.readings[host]
        self.readings[host] = (payload, seen, reason)

    def pulse(self):
        self.at = now = time.time()
        for host, (reading, seen, detail) in self.readings.items():
            view = host_view(host, reading, seen, detail, now)
            if view != self.hosts.get(host):
                self.revisions[host] += 1
                self.hosts[host] = view
        self.summary = fleet_summary(self.hosts.values(), self.history, now)
        for waiter in self.waiters:
            waiter.set()

    async def clock(self):
        while True:
            self.pulse()
            await asyncio.sleep(1)

    async def publish(self, ws, waiter):
        async def send(kind, extra, summary):
            frame = wire(dict(type=kind, at=self.at, fleet=summary, **extra))
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


def application(args):
    app = web.Application()
    fleet = Fleet(args.hosts)

    async def index(_request):
        return web.FileResponse(HTML, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    async def websocket(request):
        return await handle_websocket(request, fleet, args.allow_origin)

    async def background(_app):
        tasks = [asyncio.create_task(fleet.clock())]
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
