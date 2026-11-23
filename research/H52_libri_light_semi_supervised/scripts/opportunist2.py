#!/usr/bin/env python3
"""H52 large tier A: card manager for the FROM-SOURCE labellers on H0 / x1 / x2 (v2, lead 13:2xZ). Runs on H0.

Start rules (unchanged from v1):
  * H0 starts after eval-lane's gate-(b) stage-1 result post (13:00Z or later) and 5 min of GPU idle.
  * x1 / x2: lead override (12:21Z) = start as soon as the card has no other compute process (1 min check).
  * Any card YIELDS at once when a foreign compute process appears on it (stop by recorded PID, worker paused) and may
    restart after 20 min of idle.
Fixed eval-lane windows (lead 13:2xZ): the card is UNLOADED (worker paused, servers / fronts stopped by PID, so <= 1.5 GB
is left) for the whole window and nothing starts inside it; it resumes at the window's end, or earlier if eval-lane
posts that window's reading:
  * H0: 13:48Z -> 14:10Z (S2b-BP2 gate (a), reported only) and 14:45Z -> 15:20Z (RC1 A6 look, release-critical).
  * x1 / x2: 14:45Z -> 15:20Z ONLY if H0 is not clear (> 1.5 GB used) at 14:45Z (fallback for the RC1 A6 look).
On start it ADOPTS labellers that are already running (the latest Lsrc-* records in each host's pids.jsonl).
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

LL = "/home/ubuntu/.h52-labeler/ll"
WFILE = f"{LL}/large/wL-src.json"
CHAT = "/home/ubuntu/jsr/AGENT_CHAT.md"
LOG = f"{LL}/large/opportunist.log"
HOSTS = {
    "h0L": {"ssh": None, "base": "/home/ubuntu/.h52-labeler", "up": f"{LL}/large/large_src_up.sh",
            "down": f"{LL}/large/large_src_down.sh", "ns": 2, "fw": 14, "nf": 2, "mode": "chat", "label": "H0 3090"},
    "x1L": {"ssh": "xbabe1", "base": "/home/ubuntu/jsr-data/h52-labeler", "up": "~/jsr-data/h52-labeler/large_src_up.sh",
            "down": "~/jsr-data/h52-labeler/large_src_down.sh", "ns": 2, "fw": 12, "nf": 2, "mode": "now", "label": "x1 4090"},
    "x2L": {"ssh": "xbabe2", "base": "/home/ubuntu/jsr-data/h52-labeler", "up": "~/jsr-data/h52-labeler/large_src_up.sh",
            "down": "~/jsr-data/h52-labeler/large_src_down.sh", "ns": 2, "fw": 10, "nf": 2, "mode": "now", "label": "x2 3090"},
}
# (start HH:MM, end HH:MM, early-resume header regexes (all must match an eval-lane header posted after the start), what)
VERDICT = r"(?i:\b(CONTINUE|STOP|PASS(?:ES|ED)?|FAIL(?:S|ED)?|INCONCLUSIVE|NON-INFERIOR|READING|RESULT)\b|free now)"
BP_READY = re.compile(r"\| trainer-lane \|.*\b((?:S2[a-z]?|S3)-BP(\d+))\s+READY\b")
EVENT_MAX_S = 1800  # an event window never lasts longer than 30 min (eval-lane's reading normally ends it)
H0_WINDOWS = [("13:48", "14:10", [r"(?i:\bBP2\b)", r"(?i:gate)[\s-]*\(a\)", VERDICT], "S2b-BP2 gate (a)"),
              ("14:45", "15:20", [r"(?i:\bRC1\b)", r"(?i:\bA6\b)", VERDICT], "RC1 A6 look")]
FALLBACK = ("14:45", "15:20", [r"(?i:\bRC1\b)", r"(?i:\bA6\b)", VERDICT], "RC1 A6 look (fallback: H0 not clear)")
IDLE_MB = 1500
CHECK_S = 30
SIG_RE = (re.compile(r"\| eval-lane \|"), re.compile(r"gate[\s-]*\(?b\)?", re.I), re.compile(r"stage[\s-]*1", re.I),
          re.compile(r"\b(PASS|FAIL|INCONCLUSIVE)\b"))


def now():
    return dt.datetime.now(dt.timezone.utc)


def hm():
    return now().strftime("%H:%M")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now():%Y-%m-%dT%H:%M:%SZ} {msg}\n")


def run(h, cmd, timeout=120):
    args = ["bash", "-c", cmd] if h["ssh"] is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", h["ssh"], cmd]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def gpu(h):
    """({pid: pgid} of compute processes on the card, memory used MiB) or None."""
    rc, out, _ = run(h, "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
                        "echo \"P $p $(ps -o pgid= -p $p | tr -d ' ')\"; done; echo MEM; "
                        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=60)
    if rc != 0 or "MEM" not in out:
        return None
    a, b = out.split("MEM")
    pids = {}
    for line in a.splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] == "P" and f[1].isdigit():
            pids[int(f[1])] = int(f[2]) if len(f) > 2 and f[2].isdigit() else -1
    return pids, int(b.split()[0])


ADOPT = r'''python3 - "%s" <<'PY'
import json, os, sys
B = sys.argv[1]; recs = {}
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
out = [r for r in recs.values() if f"{B}/L-feats" in cmd(r["pid"]) or f"{B}/L-src" in cmd(r["pid"])]
print(json.dumps(out))
PY'''


def adopt(h):
    for _ in range(12):  # never adopt while an up / down script is mid-flight on the host (13:28Z swap race)
        rc, out, _ = run(h, "ps -eo args | grep -E 'large_src_(up|down)\\.sh' | grep -v grep | wc -l", timeout=60)
        if rc == 0 and out.strip() == "0":
            break
        time.sleep(10)
    rc, out, _ = run(h, ADOPT % h["base"], timeout=60)
    try:
        recs = json.loads(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else []
    except (ValueError, IndexError):
        recs = []
    return ({r["pid"] for r in recs if r["name"].startswith("Lsrc-cons")}, {r["pgid"] for r in recs})


def post(summary, lines):
    body = f"{LL}/large/body_opp.md"
    with open(body, "w") as f:
        f.write("@claude-lead @eval-lane @fleet-sched\n" + "".join(f"- {x}\n" for x in lines))
    subprocess.run([f"{LL}/chat_post.sh", summary, body], capture_output=True, timeout=180)


def set_paused(name, paused):
    w = json.load(open(WFILE))
    w[name]["paused"] = paused
    tmp = WFILE + ".tmp"
    json.dump(w, open(tmp, "w"), indent=1)
    os.replace(tmp, WFILE)


class Chat:
    """Incremental reader of AGENT_CHAT headers."""

    def __init__(self):
        self.off = max(0, os.path.getsize(CHAT) - 600_000)
        self.heads = []  # (HH:MM:SS, header line)

    def poll(self):
        with open(CHAT, "rb") as f:
            f.seek(self.off)
            data = f.read().decode("utf-8", "replace")
            self.off = f.tell()
        for line in data.splitlines():
            if line.startswith("## 2026-11-23T"):
                self.heads.append((line[14:22], line))
        self.heads = self.heads[-3000:]

    def find(self, after_hm, regexes, who=r"\| eval-lane \|"):
        rs = [re.compile(who)] + [re.compile(x) for x in regexes]  # case rules are inside each pattern
        for t, line in self.heads:
            if t[:5] >= after_hm and all(r.search(line) for r in rs):
                return line[:170]
        return None

    def gate_b(self):
        for t, line in self.heads:
            if t[:5] >= "13:00" and all(r.search(line) for r in SIG_RE):
                return line[:170]
        return None


def stop(name, h, why):
    set_paused(name, True)
    rc, out, err = run(h, f"bash {h['down']} {h['base']}", timeout=180)
    log(f"{name} STOP ({why}): {out.strip()} {err.strip()[:200]}")


def main():
    open(f"{LL}/large/opportunist.pid", "w").write(str(os.getpid()))
    h0_now = "--h0-now" in sys.argv
    st = {k: {"state": "waiting", "idle": 0, "need": 10 if v["mode"] == "chat" else 2, "mine": set(), "mine_pg": set(),
              "fail_t": 0.0, "t_start": 0.0, "gone": 0, "win": None} for k, v in HOSTS.items()}
    for name, h in HOSTS.items():
        mine, mine_pg = adopt(h)
        if mine:
            st[name].update(state="running", mine=mine, mine_pg=mine_pg, t_start=time.time() - 600)
            log(f"{name} ADOPTED running servers {sorted(mine)} (pgids {sorted(mine_pg)})")
    if h0_now and st["h0L"]["state"] == "waiting":
        st["h0L"]["need"] = 2  # lead 13:3xZ: resume H0 now (1-min idle check), still unloading at 13:48Z
    chat = Chat()
    fallback_on = False
    fallback_done = False
    resumed = set()  # windows ended early by eval-lane's reading: (host, start)
    t_boot = (now() - dt.timedelta(minutes=30)).strftime("%H:%M:%S")  # a restart still honours a BP window opened <30 min ago
    events = {}  # BPn -> [open HH:MM:SS, closed?]
    log("opportunist v2 start (eval-lane windows: H0 13:48-14:10, 14:45-15:20; x1/x2 14:45-15:20 if H0 not clear)")
    while True:
        chat.poll()
        t = hm()
        # which window (if any) applies to each host now
        active = {}
        for ts, line in chat.heads:
            m = BP_READY.search(line)
            if m and ts > t_boot and m.group(1) not in events:
                events[m.group(1)] = [ts, False]
                log(f"event window OPEN for {m.group(1)} gate (a) (trainer-lane {ts}Z)")
        for bp, ev in events.items():
            if ev[1]:
                continue
            n = re.search(r"BP(\d+)", bp).group(1)
            started = dt.datetime.strptime(ev[0], "%H:%M:%S").replace(year=now().year, month=now().month, day=now().day,
                                                                     tzinfo=dt.timezone.utc)
            if chat.find(ev[0][:5], [rf"(?i:\bBP{n}\b)", r"(?i:gate)[\s-]*\(a\)", VERDICT]) or \
                    (now() - started).total_seconds() > EVENT_MAX_S:
                ev[1] = True
                log(f"event window CLOSED for {bp} (eval-lane reading or 30-min cap)")
            else:
                active["h0L"] = (ev[0][:5], "reading", f"{bp} gate (a)")
        for (a, b, rx, what) in H0_WINDOWS:
            if a <= t < b and ("h0L", a) not in resumed:
                if chat.find(a, rx):
                    resumed.add(("h0L", a)); log(f"h0L window {a}-{b} ended early: eval-lane posted ({what})")
                else:
                    active["h0L"] = (a, b, what)
        if not fallback_done and not fallback_on and FALLBACK[0] <= t < FALLBACK[1] and st["h0L"]["state"] == "window":
            # once, right after H0's own unload: is H0 still held by anything that started BEFORE the window opened?
            # (eval-lane's own job starts after 14:45Z, so it never triggers this)
            secs = (now() - now().replace(hour=int(FALLBACK[0][:2]), minute=int(FALLBACK[0][3:]), second=0,
                                          microsecond=0)).total_seconds()
            rc, out, _ = run(HOSTS["h0L"], "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | "
                                           "while IFS=', ' read p m; do echo \"$p $m $(ps -o etimes= -p $p | tr -d ' ')\"; done",
                             timeout=60)
            old_holders = [f[0] for f in (ln.split() for ln in out.splitlines())
                           if len(f) == 3 and f[1].isdigit() and f[2].isdigit() and int(f[2]) > secs + 5 and int(f[1]) > 1000]
            fallback_done = not old_holders
            if old_holders:
                fallback_on = True
                log(f"FALLBACK: H0 not clear at {t} (held by pre-window PIDs {old_holders}); unloading x1 / x2 until {FALLBACK[1]}")
            else:
                log(f"H0 clear at {t} for {FALLBACK[3]}; no x1 / x2 fallback needed")
        if fallback_on:
            if True:
                if t >= FALLBACK[1] or chat.find(FALLBACK[0], FALLBACK[2]):
                    fallback_on = False
                    fallback_done = True
                    log("FALLBACK window over")
                else:
                    active["x1L"] = active["x2L"] = FALLBACK[:2] + (FALLBACK[3],)
        for name, h in HOSTS.items():
            s = st[name]
            win = active.get(name)
            if win:
                if s["win"] != win[0]:
                    s["win"] = win[0]
                    if s["state"] == "running":
                        stop(name, h, f"eval-lane window {win[0]}-{win[1]}")
                        post(f"H52 large: {h['label']} labeller UNLOADED for eval-lane's window {win[0]}-{win[1]}Z ({win[2]})",
                             [f"{h['label']}: my R0 servers and fronts are stopped by PID; the card is free for eval-lane until "
                              f"{win[1]}Z, or until eval-lane posts its reading."])
                    s.update(state="window", mine=set(), mine_pg=set(), idle=0)
                continue
            if s["state"] == "window":  # window over: resume after a short idle check
                s.update(state="waiting", idle=0, need=2, win=None)
                log(f"{name} window over; resuming after a 1-min idle check")
            try:
                g = gpu(h)
            except Exception as e:  # noqa: BLE001
                log(f"{name} gpu query failed: {e!r}")
                continue
            if g is None:
                continue
            pids, mem = g
            if s["state"] == "running":
                foreign = {p for p, pg in pids.items() if p not in s["mine"] and pg not in s["mine_pg"]}
                if foreign:
                    stop(name, h, f"foreign GPU process {sorted(foreign)}")
                    post(f"H52 large: {h['label']} labeller YIELDED (foreign GPU process appeared)",
                         [f"{h['label']}: a non-H52 compute process appeared; my servers and fronts there are stopped by PID "
                          f"and the worker is paused. Restart only after 20 min of idle."])
                    s.update(state="waiting", idle=0, need=40, mine=set(), mine_pg=set())
                elif not (s["mine"] & set(pids)) and time.time() - s["t_start"] > 180:
                    s["gone"] += 1
                    if s["gone"] >= 3:
                        stop(name, h, "servers gone from the GPU for 3 checks")
                        s.update(state="waiting", idle=0, need=40, mine=set(), mine_pg=set(), gone=0)
                else:
                    s["gone"] = 0
                continue
            s["idle"] = s["idle"] + 1 if (not pids and mem < IDLE_MB) else 0
            if h["mode"] == "chat":
                gate = bool(chat.gate_b()) or t >= "13:45"
                if not chat.gate_b() and t >= "13:45":
                    s["need"] = max(s["need"], 40)
            else:
                gate = True
            # never start inside the last 10 min before one of this card's windows
            soon = [a for (a, b, rx, w) in (H0_WINDOWS if name == "h0L" else []) if t < a and
                    (dt.datetime.strptime(a, "%H:%M") - dt.datetime.strptime(t, "%H:%M")).seconds < 600]
            if not gate or soon or s["idle"] < s["need"] or time.time() - s["fail_t"] < 600:
                continue
            rc, out, err = run(h, f"bash {h['up']} {h['base']} {h['ns']} {h['fw']} {h['nf']}", timeout=300)
            recs = []
            for line in out.splitlines():
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
            mine = {r["pid"] for r in recs if r.get("name", "").startswith("Lsrc-cons")}
            mine_pg = {r["pgid"] for r in recs if r.get("name", "").startswith("Lsrc-")}
            if rc != 0 or len(mine) < h["ns"]:
                log(f"{name} start FAILED rc {rc}: {out.strip()[-300:]} {err.strip()[-300:]}")
                run(h, f"bash {h['down']} {h['base']}", timeout=180)
                s["fail_t"] = time.time()
                continue
            time.sleep(20)
            s.update(state="running", mine=mine, mine_pg=mine_pg, idle=0, t_start=time.time(), gone=0)
            set_paused(name, False)
            log(f"{name} STARTED servers {sorted(mine)}; all {[(r['name'], r['pid']) for r in recs]}")
            post(f"H52 large: from-source labeller STARTED on {h['label']}",
                 [f"{h['label']}: {h['ns']} R0 servers (v1.1 only, frame-KD on) + {h['nf']} from-source front(s); labels land in "
                  f"x6 `librilight.large/r0src/`, joined by id. It yields at once to any other compute process on this card "
                  f"and unloads for eval-lane's fixed windows."])
        time.sleep(CHECK_S)


if __name__ == "__main__":
    main()
