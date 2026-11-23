#!/usr/bin/env python3
"""Minimal iperf3-style TCP throughput + RTT probe (stdlib only; iperf3 is not installed on the fleet).

  server: tcpbench.py server --bind 0.0.0.0 --port 5299 --life 90
  client: tcpbench.py client --host <xbabe5-private> --port 5299 --streams 8 --seconds 8 --dir send
          tcpbench.py client ... --dir recv        (server sends to the client: the reverse path)

The server exits after --life seconds (no leftover listeners). Each client stream opens one TCP
connection, sends a 16-byte header (direction, seconds), then moves 4 MiB buffers as fast as it
can for the duration. Prints one JSON line: aggregate and per-stream Gbit/s, connect() times
(TCP handshake = 1 RTT), and the kernel's smoothed RTT / cwnd from TCP_INFO.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time

BUF = 4 << 20


def tcp_info(sock: socket.socket) -> dict:
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
        # struct tcp_info: 8 x u8, then u32 fields; rtt at u32 index 15, rttvar 16, snd_cwnd 18 (after 8 bytes)
        u32 = struct.unpack_from("<24I", raw, 8)
        return {"rtt_us": u32[15], "rttvar_us": u32[16], "snd_cwnd": u32[18], "retrans_total": u32[23]}
    except OSError:
        return {}


def serve(bind: str, port: int, life: float) -> None:
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((bind, port))
    ls.listen(64)
    ls.settimeout(1.0)
    end = time.time() + life

    def handle(c: socket.socket) -> None:
        try:
            hdr = b""
            while len(hdr) < 16:
                chunk = c.recv(16 - len(hdr))
                if not chunk:
                    return
                hdr += chunk
            mode, secs = hdr[:1], struct.unpack("<d", hdr[8:16])[0]
            buf = bytearray(BUF)
            mv = memoryview(buf)
            if mode == b"S":  # client sends: drain
                while c.recv_into(mv):
                    pass
            else:  # client receives: send for secs
                t_end = time.time() + secs
                while time.time() < t_end:
                    c.sendall(mv)
                c.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        finally:
            c.close()

    while time.time() < end:
        try:
            c, _ = ls.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(c,), daemon=True).start()
    ls.close()


def client(host: str, port: int, streams: int, seconds: float, direction: str) -> dict:
    results: list[dict] = [None] * streams  # type: ignore[list-item]
    barrier = threading.Barrier(streams)

    def one(i: int) -> None:
        t0 = time.perf_counter()
        s = socket.create_connection((host, port), timeout=10)
        conn_ms = (time.perf_counter() - t0) * 1e3
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(seconds + 30)
        s.sendall((b"S" if direction == "send" else b"R") + b"\0" * 7 + struct.pack("<d", seconds))
        buf = bytearray(BUF)
        mv = memoryview(buf)
        barrier.wait()
        n, t_start = 0, time.perf_counter()
        info = {}
        try:
            if direction == "send":
                t_end = t_start + seconds
                while time.perf_counter() < t_end:
                    s.sendall(mv)
                    n += BUF
                info = tcp_info(s)
                s.shutdown(socket.SHUT_WR)
                while s.recv(65536):
                    pass
            else:
                while True:
                    k = s.recv_into(mv)
                    if not k:
                        break
                    n += k
                info = tcp_info(s)
        finally:
            dt = time.perf_counter() - t_start
            s.close()
        results[i] = {"bytes": n, "s": dt, "gbps": n * 8 / dt / 1e9, "connect_ms": round(conn_ms, 3), **info}

    th = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    agg = sum(r["bytes"] for r in results) * 8 / max(r["s"] for r in results) / 1e9
    return {"host": host, "port": port, "dir": direction, "streams": streams, "seconds": seconds,
            "gbps": round(agg, 3), "per_stream_gbps": [round(r["gbps"], 3) for r in results],
            "connect_ms": [r["connect_ms"] for r in results],
            "rtt_us": [r.get("rtt_us") for r in results], "snd_cwnd": [r.get("snd_cwnd") for r in results],
            "retrans": [r.get("retrans_total") for r in results]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["server", "client"])
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=5299)
    ap.add_argument("--life", type=float, default=90.0)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--dir", default="send", choices=["send", "recv"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode == "server":
        serve(a.bind, a.port, a.life)
    else:
        r = client(a.host, a.port, a.streams, a.seconds, a.dir)
        r["tag"] = a.tag
        r["src"] = socket.gethostname()
        r["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
