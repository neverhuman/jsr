"""Trainer-first guard for one process group of background work (h51-pq sweeps on a training host).

Every PERIOD s: host CPU busy fraction over the period (/proc/stat) and the trainer's latest
data_wait_pct (last line carrying it in train.jsonl, ignored if the file is older than 180 s).
Stop the group (SIGSTOP) when data_wait > DW_HI or cpu > CPU_HI; resume (SIGCONT) when
data_wait < DW_LO and cpu < CPU_LO. Always resumes the group on exit. JSON lines log.

usage: python3 h51_guard.py <pgid> <train.jsonl> <log> [--dw-hi 5 --dw-lo 4 --cpu-hi 0.85 --cpu-lo 0.75]
"""
import json, os, signal, sys, time

pgid, tj, log = int(sys.argv[1]), sys.argv[2], sys.argv[3]
args = sys.argv[4:]
def opt(name, default):
    return float(args[args.index(name) + 1]) if name in args else default
DW_HI, DW_LO = opt("--dw-hi", 5.0), opt("--dw-lo", 4.0)
CPU_HI, CPU_LO = opt("--cpu-hi", 0.85), opt("--cpu-lo", 0.75)
PERIOD = opt("--period", 15.0)
stopped = False


def cpu_sample():
    with open("/proc/stat") as fh:
        v = list(map(int, fh.readline().split()[1:]))
    return sum(v), v[3] + v[4]


def data_wait():
    try:
        if time.time() - os.path.getmtime(tj) > 180:
            return None
        with open(tj, "rb") as fh:
            fh.seek(0, 2)
            n = fh.tell()
            fh.seek(max(0, n - 20000))
            lines = fh.read().decode(errors="ignore").splitlines()
        for l in reversed(lines):
            try:
                d = json.loads(l)
            except Exception:
                continue
            if "data_wait_pct" in d:
                return float(d["data_wait_pct"])
    except OSError:
        return None
    return None


def resume(*_):
    try:
        os.killpg(pgid, signal.SIGCONT)
    except ProcessLookupError:
        pass
    sys.exit(0)


signal.signal(signal.SIGTERM, resume)
signal.signal(signal.SIGINT, resume)
t0, i0 = cpu_sample()
while True:
    time.sleep(PERIOD)
    t1, i1 = cpu_sample()
    cpu = 1.0 - (i1 - i0) / max(1, t1 - t0)
    t0, i0 = t1, i1
    dw = data_wait()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        break
    hi = (dw is not None and dw > DW_HI) or cpu > CPU_HI
    lo = (dw is None or dw < DW_LO) and cpu < CPU_LO
    act = None
    if hi and not stopped:
        os.killpg(pgid, signal.SIGSTOP); stopped = True; act = "stop"
    elif lo and stopped:
        os.killpg(pgid, signal.SIGCONT); stopped = False; act = "cont"
    with open(log, "a") as fh:
        fh.write(json.dumps({"t": round(time.time(), 1), "cpu": round(cpu, 3), "data_wait": dw,
                             "stopped": stopped, "act": act}) + "\n")
resume()
