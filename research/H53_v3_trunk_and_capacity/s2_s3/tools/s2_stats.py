#!/usr/bin/env python3
"""Window stats from a trainer train.jsonl on stdin.
usage: ssh x5 cat .../train.jsonl | s2_stats.py <t0_iso> <t1_iso> <world> [w4_per_gpu]
Prints total and per-GPU audio-h/h (sum audio_s / sum wall_s over records whose time is in [t0, t1]),
logged data wait (wall-weighted), per-rank data wait share, util, VRAM peak, cumulative audio-h at the end."""
import json, sys, datetime as dt
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
t0, t1, world = ts(sys.argv[1]), ts(sys.argv[2]), int(sys.argv[3])
w4 = float(sys.argv[4]) if len(sys.argv) > 4 else 2272.0
cum = 0.0; win = []; last = None
for l in sys.stdin:
    if '"kind": "train"' not in l: continue
    r = json.loads(l); cum += r["audio_s"]
    if t0 <= r["time"] <= t1: win.append(r); last = (r, cum)
if not win: print("no records in window"); sys.exit(1)
a = sum(r["audio_s"] for r in win); w = sum(r["wall_s"] for r in win)
dw = sum(r["data_wait_pct"] * r["wall_s"] for r in win) / w
n = len(win[0].get("rank_step_s", [])) or world
rdw = [sum(r["rank_data_wait_s"][i] for r in win) / sum(r["rank_step_s"][i] for r in win) * 100 for i in range(n)]
r, c = last
fmt = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%SZ")
print(f"window {fmt(win[0]['time'])}..{fmt(r['time'])} steps {win[0]['step']}..{r['step']} records {len(win)}")
print(f"total {a/w:.0f} audio-h/h; per GPU {a/w/world:.0f} = {100*a/w/world/w4:.1f}% of {w4:.0f}")
print(f"logged data wait (wall-weighted) {dw:.2f}%; per-rank data wait % {[round(x,2) for x in rdw]}")
print(f"last: step {r['step']} loss {r['loss']:.4f} lr {r['lr']:.3e} util5 {r.get('rank_gpu_util_5min')} peak {r['gpu_peak_gb']} GB")
print(f"cumulative logged audio at last record: {c/3600:.0f} audio-h ({c:.4g} s)")
