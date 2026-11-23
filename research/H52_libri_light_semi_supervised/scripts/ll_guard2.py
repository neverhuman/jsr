"""H52 host guard v2: auto-discovers the live trainers and protects them (lead 08:57Z rule).

Every --discover seconds it scans /proc for running trainers (`-m train.trainer` or `train/trainer.py`),
reads each one's `--run-dir`, and watches `<run-dir>/train.jsonl`. Several trainers -> all are watched.
Every --interval seconds it decides the cap for this host's background work (written to --throttle-file):
  * a trainer is running but has no step record yet  -> cap 0 (worst case);
  * any trainer's median data_wait_pct over its last 5 step records > --max-wait-pct -> cap halves at once;
  * any trainer below 0.90 x its own baseline rate (median of its step records 3-12) on 2 checks -> halves;
  * every trainer healthy for 6 checks in a row -> cap + 2 (up to --max);
  * no trainer running -> normal: the cap ramps to --max (+2 per 3 healthy checks).
Every decision is logged with the trainers it saw.

usage: python -I ll_guard2.py --throttle-file F --max 16 [--start 0] [--interval 30] [--discover 60] [--log L]
"""
import argparse
import json
import os
import statistics
import time


def log(msg, f):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def live_trainers():
    """{run_dir: [pids]} for running trainer processes (ranks of one run share a run dir)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                args = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if not args:
            continue
        is_trainer = ("train.trainer" in args and "-m" in args) or any(a.endswith("train/trainer.py") for a in args)
        if not is_trainer:
            continue
        rd = None
        if "--run-dir" in args:
            i = args.index("--run-dir")
            rd = args[i + 1] if i + 1 < len(args) else None
        else:
            rd = next((a.split("=", 1)[1] for a in args if a.startswith("--run-dir=")), None)
        if rd:
            out.setdefault(os.path.abspath(os.path.expanduser(rd)), []).append(int(pid))
    return out


def step_records(run_dir, keep=60):
    p = os.path.join(run_dir, "train.jsonl")
    recs = []
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 400_000))
            tail = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return [], None
    for line in tail:
        if '"kind": "train"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs.append(r)
    head = []
    try:  # first records for the baseline (records 3-12)
        with open(p) as fh:
            for line in fh:
                if '"kind": "train"' in line:
                    try:
                        head.append(json.loads(line))
                    except ValueError:
                        pass
                    if len(head) >= 12:
                        break
    except OSError:
        pass
    return recs[-keep:], head


def rate(r):
    w = r.get("wall_s") or 0
    return r["audio_s"] / w if w > 0 and "audio_s" in r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throttle-file", required=True)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--start", type=int, default=-1, help="initial cap (default: 0 if a trainer runs, else --max)")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--discover", type=float, default=60)
    ap.add_argument("--max-wait-pct", type=float, default=5.0)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    lf = open(a.log or a.throttle_file + ".log", "a")
    trainers = live_trainers()
    cap = a.start if a.start >= 0 else (0 if trainers else a.max)
    open(a.throttle_file, "w").write(f"{cap}\n")
    log(f"guard2 start cap={cap} max={a.max} trainers={ {os.path.basename(k): v for k, v in trainers.items()} }", lf)
    last_disc, low, good = time.time(), 0, 0
    last_status = 0.0
    while True:
        time.sleep(a.interval)
        if time.time() - last_disc >= a.discover:
            new = live_trainers()
            if set(new) != set(trainers):
                log(f"trainers now { {os.path.basename(k): v for k, v in new.items()} }", lf)
            trainers, last_disc = new, time.time()
        detail, state = {}, "ok"
        for rd in trainers:
            name = os.path.basename(rd)
            recs, head = step_records(rd)
            if not recs:
                detail[name] = "no step record yet"
                state = "worst"
                continue
            waits = [float(r["data_wait_pct"]) for r in recs[-5:] if "data_wait_pct" in r]
            if waits:
                w = statistics.median(waits)
                detail[name + ":wait%"] = round(w, 2)
                if w > a.max_wait_pct:
                    state = "bad_wait" if state != "worst" else state
            base_rates = [x for x in (rate(r) for r in (head or [])[2:12]) if x]
            cur_rates = [x for x in (rate(r) for r in recs[-5:]) if x]
            if len(base_rates) >= 5 and cur_rates:
                ratio = statistics.median(cur_rates) / statistics.median(base_rates)
                detail[name] = round(ratio, 3)
                if ratio < 0.90 and state == "ok":
                    state = "low_rate"
        prev = cap
        if state == "worst":
            cap, low, good = 0, 0, 0
        elif state == "bad_wait":
            cap, low, good = cap // 2, 0, 0
        elif state == "low_rate":
            low += 1
            good = 0
            if low >= 2:
                cap, low = cap // 2, 0
        else:
            low = 0
            good += 1
            need = 6 if trainers else 3
            if good >= need and cap < a.max:
                cap, good = min(a.max, cap + 2), 0
        if cap != prev:
            open(a.throttle_file, "w").write(f"{cap}\n")
            log(f"cap {prev} -> {cap} ({state}) {detail}", lf)
        elif time.time() - last_status > 300:
            last_status = time.time()
            log(f"status cap={cap} ({state}) {detail}", lf)


if __name__ == "__main__":
    main()
