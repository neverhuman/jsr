#!/usr/bin/env python3
"""H52 large: keep every card on NEW rows (runs on H0, every 2 min).
1. Coverage both ways: from-source units already fully covered by x6 per-shard labels are marked done in the tier-A
   driver's state (it never claims them); raw shards fully covered by published from-source units are claimed on x6
   as 'covered-by-from-source' (the per-shard driver skips them).
2. When the tier-A from-source driver exits (every unit done or covered), start the tier-B driver (same workers /
   card manager), labels to x6 librilight.large/r0src_tierB/.
3. When x6 has no open tier-A shard left and segmentation is done, stop its per-shard labeller (recorded PIDs) and
   start a from-source labeller there (worker x6S in the shared workers file)."""
import datetime as dt
import json
import os
import subprocess
import time

LL = "/home/ubuntu/.h52-labeler/ll"
SA = "/home/ubuntu/.h52-labeler/large-src-state"
SB = "/home/ubuntu/.h52-labeler/large-srcB-state"
WFILE = f"{LL}/large/wL-src.json"
LOG = f"{LL}/large/chain.log"
def driver_a_pid():
    try:
        return int(open(os.path.join(SA, "driver.pid")).read().strip())
    except (OSError, ValueError):
        return 1731837
B_CMD = ["/usr/bin/python3", "/home/ubuntu/.h52-labeler/code-next/lab_drive_units_v2.py", "--units",
         f"{LL}/large/units-large-tierB-src.json", "--workers", WFILE, "--dest",
         "xbabe6:/home/ubuntu/jsr-data/librilight/shards/raw/librilight.large/r0src_tierB", "--state", SB, "--poll", "10"]
X6_PER_SHARD = {"driver": 3175036, "procs": ["xL-cons", "xL-cons2", "xL-prod", "xL-prod2"]}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{now()} {m}\n")


def ssh6(cmd, timeout=180):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "xbabe6", cmd], capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def alive(pid, needle):
    try:
        return needle in open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except OSError:
        return False


def post(summary, lines):
    body = f"{LL}/large/body_chain.md"
    with open(body, "w") as f:
        f.write("@claude-lead @fleet-sched\n" + "".join(f"- {x}\n" for x in lines))
    subprocess.run([f"{LL}/chat_post.sh", summary, body], capture_output=True, timeout=180)


def mark_done(units):
    n = 0
    for u in units:
        d = os.path.join(SA, "claims", u)
        try:
            os.mkdir(d)
        except FileExistsError:
            continue
        with open(os.path.join(d, "done.json"), "w") as f:
            json.dump({"skipped": "covered by x6 per-shard labels", "time": now()}, f)
        n += 1
    return n


X6_SWITCH = r'''set -e
B=$HOME/jsr-data/h52-labeler
python3 - <<'PY'
import json, os, signal, time
B = os.path.expanduser("~/jsr-data/h52-labeler")
recs = {}
for line in open(os.path.join(B, "logs", "pids.jsonl")):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("name") in ("xL-cons", "xL-cons2", "xL-prod", "xL-prod2"):
        recs[r["name"]] = r
def cmd(p):
    try:
        return open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except OSError:
        return ""
d = 3175036
if "lab_drive_large.py --contract" in cmd(d):
    os.kill(d, signal.SIGTERM); print("TERM driver", d)
for n, r in recs.items():
    g = r.get("guard_pid")
    if g and f"fleet_guard.py --pgid {r['pgid']} " in cmd(g):
        os.kill(g, signal.SIGTERM)
    if f"{B}/L-feats" in cmd(r["pid"]) or f"{B}/L-src" in cmd(r["pid"]):
        os.killpg(r["pgid"], signal.SIGCONT); os.killpg(r["pgid"], signal.SIGTERM); print("TERM", n, r["pgid"])
time.sleep(8)
PY
mkdir -p $B/h52bin-large && cp ~/jsr-data/librilight/bin/ll_pass2.py ~/jsr-data/librilight/bin/ll_segment.py $B/h52bin-large/
find $B/L-src $B/L-stage $B/L-feats $B/L-out -maxdepth 1 -type f ! -name '*.log.jsonl' -delete
bash $B/large_src_up.sh $B 2 8 2
'''


def main():
    started_b = os.path.exists(os.path.join(SB, "drive_units.log.jsonl"))
    switched = "x6S" in json.load(open(WFILE))
    log("chain start")
    while True:
        try:
            rc, out, err = ssh6("python3 ~/jsr-data/h52-labeler/large_coverage.py --apply-x6")
            cov = json.loads(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else None
        except Exception as e:  # noqa: BLE001
            cov = None
            log(f"coverage failed: {e!r}")
        if cov:
            n = mark_done(cov["cover_units"])
            if n or cov["applied"]:
                log(f"coverage: {n} from-source units marked covered by x6; {cov['applied']} x6 shards claimed covered-by-from-source; "
                    f"open x6 shards {cov['open_shards']}, pass2_done {cov['pass2_done']}")
            if not switched and cov["pass2_done"] and cov["open_shards"] == 0 and cov.get("inflight_x6", 1) == 0:
                rc, out, err = ssh6(X6_SWITCH, timeout=400)
                recs = []
                for line in out.splitlines():
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
                ok = rc == 0 and sum(1 for r in recs if r.get("name", "").startswith("Lsrc-cons")) >= 2
                log(f"x6 switch rc {rc} ok {ok}: {out.strip()[-400:]} {err.strip()[-300:]}")
                if ok:
                    w = json.load(open(WFILE))
                    R = "/home/ubuntu/jsr-data/h52-labeler"
                    w["x6S"] = {"kind": "ship", "front_host": "xbabe6", "front_inbox": f"{R}/L-src", "front_stage": f"{R}/L-stage",
                                "host": "xbabe6", "dest_host": "local", "feats": f"{R}/L-feats", "out": f"{R}/L-out", "depth": 10,
                                "weight": 0.7, "paused": False}
                    tmp = WFILE + ".tmp"
                    json.dump(w, open(tmp, "w"), indent=1)
                    os.replace(tmp, WFILE)
                    pids = {r["name"]: r["pid"] for r in recs if r.get("name", "").startswith("Lsrc-")}
                    ssh6("echo '" + " ".join(str(p) for p in pids.values()) + "' > ~/jsr-data/h52-labeler/large_stat.pids")
                    post("H52 large: x6 switched from per-shard to from-source labelling (tier-A shards all labelled or covered)",
                         [f"x6 A10: every tier-A raw shard is labelled or covered by from-source units; its per-shard labeller is "
                          f"stopped by PID and a from-source labeller runs there now ({sorted(pids.values())})."])
                    switched = True
        if not started_b and not alive(driver_a_pid(), "lab_drive_units"):
            os.makedirs(SB, exist_ok=True)
            pb = subprocess.Popen(B_CMD, stdout=open("/home/ubuntu/.h52-labeler/logs/drive_large_srcB.log", "a"),
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            open(os.path.join(SB, "driver.pid"), "w").write(str(pb.pid))
            started_b = True
            log("tier-A from-source driver finished; tier-B driver started")
            post("H52 large: tier A from-source labelling complete; tier-B labelling started (x1 / x2 / H0)",
                 ["Every tier-A unit is labelled or covered by x6's per-shard labels. The from-source cards move to tier B "
                  "(18,943 files, 842 units); labels land in x6 `librilight.large/r0src_tierB/`. Card windows unchanged."])
        time.sleep(120)


if __name__ == "__main__":
    main()
