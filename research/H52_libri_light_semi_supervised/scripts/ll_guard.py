"""H52 throughput guard for CPU work on a host with our trainers (lead's 05:00Z rule).

Every interval it reads each active run's train.jsonl (modified in the last 10 min), computes
throughput = audio_s / wall_s per record, and compares the median of the last 5 records with the
run's baseline: the median of its last 10 records before the guard started (or, for a run that
starts later, of its records 3-12). If any run is below 0.90 x baseline on 2 consecutive checks, the
cap in the throttle file is halved (down to 0 = pause); after 6 consecutive checks with every run
>= 0.95 x baseline it rises by 2 up to --max. Every decision is logged.

usage: python -I ll_guard.py --runs '<glob of train.jsonl>' --throttle-file F --max 16 [--interval 30]
"""
import argparse
import glob
import json
import os
import statistics
import time


def log(msg, f):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def waits(path):
    out = []
    try:
        for line in open(path):
            if '"kind": "train"' in line and "data_wait_pct" in line:
                try:
                    out.append(float(json.loads(line)["data_wait_pct"]))
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return out


def rates(path):
    out = []
    try:
        for line in open(path):
            if '"kind": "train"' not in line:
                continue
            try:
                r = json.loads(line)
                if r.get("wall_s", 0) > 0:
                    out.append(r["audio_s"] / r["wall_s"])
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--throttle-file", required=True)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--start", type=int, default=0, help="initial cap (default: --max)")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--log", default="")
    ap.add_argument("--max-wait-pct", type=float, default=5.0)
    a = ap.parse_args()
    lf = open(a.log or a.throttle_file + ".log", "a")
    cap = a.start or a.max
    open(a.throttle_file, "w").write(f"{cap}\n")
    base, seen_at_start = {}, {}
    for p in glob.glob(os.path.expanduser(a.runs)):
        if time.time() - os.path.getmtime(p) < 600:
            r = rates(p)
            seen_at_start[p] = len(r)
            if len(r) >= 5:
                base[p] = statistics.median(r[-10:])
    log(f"guard start cap={cap} baselines={ {os.path.basename(os.path.dirname(k)): round(v, 1) for k, v in base.items()} }", lf)
    low, good = 0, 0
    while True:
        time.sleep(a.interval)
        worst, detail = 1e9, {}
        for p in glob.glob(os.path.expanduser(a.runs)):
            if time.time() - os.path.getmtime(p) > 600:
                continue
            r = rates(p)
            name = os.path.basename(os.path.dirname(p))
            if p not in base:
                if len(r) >= 12:
                    base[p] = statistics.median(r[2:12])
                    log(f"baseline {name} = {base[p]:.1f} (records 3-12; run started under the guard)", lf)
                continue
            if len(r) < 5:
                continue
            cur = statistics.median(r[-5:])
            ratio = cur / base[p]
            detail[name] = round(ratio, 3)
            worst = min(worst, ratio)
            w = waits(p)
            if len(w) >= 5:
                wcur = statistics.median(w[-5:])
                detail[name + ":wait%"] = round(wcur, 2)
                if wcur > a.max_wait_pct:  # trainer-lane 06:55Z: back off above ~5% data wait
                    worst = min(worst, 0.0)
        if worst < 0.90:
            low += 1
            good = 0
            if low >= 2 and cap > 0:
                cap = cap // 2
                open(a.throttle_file, "w").write(f"{cap}\n")
                log(f"THROTTLE DOWN cap={cap} ratios={detail}", lf)
                low = 0
        elif worst >= 0.95 or worst == 1e9:
            good += 1
            low = 0
            if good >= 6 and cap < a.max:
                cap = min(a.max, cap + 2)
                open(a.throttle_file, "w").write(f"{cap}\n")
                log(f"throttle up cap={cap} ratios={detail}", lf)
                good = 0
        else:
            low = good = 0
        if int(time.time()) % 300 < a.interval:
            log(f"status cap={cap} ratios={detail}", lf)


if __name__ == "__main__":
    main()
