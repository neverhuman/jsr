#!/usr/bin/env bash
# H52: stop this host's large from-source labeller (Lsrc-* in <base>/logs/pids.jsonl) by recorded PID / PGID, each
# command line checked first. usage (on the host): large_src_down.sh <base>
B=$1
python3 - "$B" <<'PY'
import json, os, signal, sys, time
B = sys.argv[1]
recs = {}
for line in open(os.path.join(B, "logs", "pids.jsonl")):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("name", "").startswith("Lsrc-"):
        recs[r["name"]] = r
def cmd(p):
    try:
        return open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except OSError:
        return ""
def mine(r):
    c = cmd(r["pid"])
    return (f"lab_gpu.py serve --feats {B}/L-feats" in c) or (f"lab_front.py serve --inbox {B}/L-src" in c)
for n, r in recs.items():
    g = r.get("guard_pid")
    if g and f"fleet_guard.py --pgid {r['pgid']} " in cmd(g):
        os.kill(g, signal.SIGTERM); print("TERM guard", g, n)
for n, r in recs.items():
    if mine(r):
        try:
            os.killpg(r["pgid"], signal.SIGCONT); os.killpg(r["pgid"], signal.SIGTERM); print("TERM pgid", r["pgid"], n)
        except ProcessLookupError:
            pass
time.sleep(6)
for n, r in recs.items():
    if mine(r):
        try:
            os.killpg(r["pgid"], signal.SIGKILL); print("KILL pgid", r["pgid"], n)
        except ProcessLookupError:
            pass
PY
