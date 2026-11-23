#!/usr/bin/env python3
"""H50 measurement: steady-window throughput, per-rank efficiency, comm / data / throttle shares,
GPU energy and time-to-WER from a trainer run dir (stdlib only).

  h50_measure.py RUN_DIR [--ref RUN_DIR_1GPU] [--gpu-csv FILE ...] [--target-wer X] [--gate 1.6]

Inputs: RUN_DIR/train.jsonl (rank 0: kind=train blocks with audio_s, time, rank_* lists; kind=eval);
optional GPU sampler CSVs written by gpu_sampler.sh (nvidia-smi -lms 1000:
timestamp,index,uuid,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu).

Steady window (pre-registered): a run of consecutive eligible log blocks, >= MIN_BLOCKS blocks and
>= MIN_S seconds, cut greedily from the start of each eligible run. A block (prev.step, step] is
eligible when step > WARM_STEPS, its start is >= WARM_S after the first block, no eval record and
no resume/start record falls inside it, and it holds no checkpoint step (multiple of ckpt_every).
The first three windows are reported; throughput = sum(audio_s) / (t_end - t_start).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
import sys

WARM_STEPS, WARM_S, MIN_BLOCKS, MIN_S = 200, 300.0, 20, 300.0


def load(run_dir: str) -> tuple[list[dict], list[dict], dict]:
    recs = []
    with open(f"{run_dir}/train.jsonl") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cfg = {}
    try:
        with open(f"{run_dir}/config.json") as fh:
            cfg = json.load(fh)
    except OSError:
        pass
    train = [r for r in recs if r.get("kind") == "train"]
    other = [r for r in recs if r.get("kind") in ("eval", "resume", "start", "ddp", "diloco_sync")]
    return train, other, cfg


def blocks(train: list[dict], other: list[dict], ckpt_every: int) -> list[dict]:
    out = []
    marks = [(r.get("time", 0.0), r.get("kind")) for r in other if r.get("kind") in ("eval", "resume", "start")]
    t_first = train[0]["time"] if train else 0.0
    for prev, cur in zip(train, train[1:]):
        if cur["step"] <= prev["step"]:  # restart: never bridge it
            continue
        t0, t1 = prev["time"], cur["time"]
        has_mark = any(t0 < t <= t1 for t, _ in marks)
        has_ckpt = ckpt_every > 0 and (cur["step"] // ckpt_every) > (prev["step"] // ckpt_every)
        ok = cur["step"] > WARM_STEPS and t0 - t_first >= WARM_S and not has_mark and not has_ckpt
        out.append({"s0": prev["step"], "s1": cur["step"], "t0": t0, "t1": t1, "audio_s": cur.get("audio_s", 0.0),
                    "ok": ok, "rec": cur})
    return out


def windows(bl: list[dict], n_max: int = 3) -> list[dict]:
    wins, cur = [], []
    for b in bl + [None]:
        if b is not None and b["ok"] and (not cur or b["s0"] == cur[-1]["s1"]):
            cur.append(b)
            if len(cur) >= MIN_BLOCKS and cur[-1]["t1"] - cur[0]["t0"] >= MIN_S:
                wins.append(cur)
                cur = []
            continue
        cur = [b] if (b is not None and b["ok"]) else []
    return [summarize(w) for w in wins[:n_max]]


def _mean_list(w: list[dict], key: str) -> list[float] | None:
    rows = [b["rec"][key] for b in w if key in b["rec"]]
    if not rows:
        return None
    return [round(sum(r[k] for r in rows), 3) for k in range(len(rows[0]))]


def summarize(w: list[dict]) -> dict:
    t0, t1 = w[0]["t0"], w[-1]["t1"]
    audio = sum(b["audio_s"] for b in w)
    steps = w[-1]["s1"] - w[0]["s0"]
    wall = t1 - t0
    step_s = _mean_list(w, "rank_step_s")
    fwd = _mean_list(w, "rank_fwd_s")
    bwd = _mean_list(w, "rank_bwd_comm_s")
    wait = _mean_list(w, "rank_data_wait_s")
    sleep = _mean_list(w, "rank_throttle_sleep_s")
    out = {"t0": t0, "t1": t1, "t0_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "steps": steps, "blocks": len(w), "wall_s": round(wall, 1), "audio_h": round(audio / 3600, 2),
           "audio_h_per_h": round(audio / wall, 1), "step_s": round(wall / max(1, steps), 4)}
    if step_s:
        out["ranks"] = len(step_s)
        out["rank_step_s_per_step"] = [round(v / steps, 4) for v in step_s]
        out["outside_step_share"] = round(1 - st.mean(step_s) / wall, 4)  # data wait, logging, throttle sleep
    if fwd:
        out["rank_fwd_s_per_step"] = [round(v / steps, 4) for v in fwd]
        out["straggler_index"] = round(max(fwd) / st.median(fwd), 3)  # compute-only spread across ranks
    if bwd and step_s:
        out["bwd_comm_share"] = round(st.mean(bwd) / st.mean(step_s), 4)
    if wait:
        out["data_wait_share"] = round(st.mean(wait) / wall, 4)
    if sleep:
        out["throttle_sleep_share"] = round(st.mean(sleep) / wall, 4)
    utils = [b["rec"].get("rank_gpu_util_5min") for b in w if b["rec"].get("rank_gpu_util_5min")]
    if utils:
        out["gpu_util_5min_mean"] = round(st.mean(st.mean(u) for u in utils if min(u) >= 0), 3)
    return out


def gpu_energy(csv_paths: list[str], t0: float, t1: float, expect_gpus: int) -> dict:
    """Joules per GPU (trapezoid on power.draw) inside [t0, t1] from gpu_sampler.sh CSVs. Energy is
    reported only when exactly `expect_gpus` GPU UUIDs each cover >= 95% of the window at ~1 Hz and
    no gap exceeds 5 s; otherwise it is marked unavailable (never 0)."""
    per: dict[str, list[tuple[float, float]]] = {}
    for p in csv_paths:
        with open(p) as fh:
            for row in csv.reader(fh):
                if len(row) < 4:
                    continue
                try:
                    ts = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(
                        tzinfo=dt.timezone.utc).timestamp()
                    watts = float(row[3])
                except ValueError:
                    continue
                if t0 <= ts <= t1:
                    per.setdefault(f"{p}:{row[2].strip()}", []).append((ts, watts))
    joules, problems = {}, []
    for k, pts in per.items():
        pts.sort()
        span = pts[-1][0] - pts[0][0] if len(pts) > 1 else 0.0
        gap = max((b[0] - a[0] for a, b in zip(pts, pts[1:])), default=float("inf"))
        if span < 0.95 * (t1 - t0) or gap > 5.0:
            problems.append(f"{k}: covers {span:.0f} of {t1 - t0:.0f} s, max gap {gap:.1f} s")
        joules[k] = round(sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:])), 1)
    if len(per) != expect_gpus:
        problems.append(f"{len(per)} GPU UUIDs in the samples, {expect_gpus} expected")
    if problems:
        return {"energy": "unavailable", "energy_problems": problems}
    return {"gpu_joules": joules, "total_J": round(sum(joules.values()), 1)}


def time_to_wer(wer_file: str, train: list[dict], target: float, identity: dict) -> dict:
    """Wall time until the first checkpoint whose CANONICAL score reaches `target`. Rows come from an
    external JSONL ({step, ckpt_sha256, cohort, scorer, weights, wer}) written by the pinned scorer;
    every row used must match `identity` exactly (cohort, scorer, weights) and carry a checkpoint
    hash, and mixed identities are refused. The trainer's in-run dev evals are never used here.
    Time = when training reached that step (first train block with step >= it)."""
    rows = []
    with open(wer_file) as fh:
        for line in fh:
            r = json.loads(line)
            if all(r.get(k) == v for k, v in identity.items()):
                if not r.get("ckpt_sha256"):
                    return {"verdict": "REFUSED", "reason": f"step {r.get('step')} has no ckpt_sha256"}
                rows.append(r)
    if not rows:
        return {"verdict": "REFUSED", "reason": f"no canonical rows match {identity}"}
    t_start = train[0]["time"] if train else None
    for r in sorted(rows, key=lambda r: r["step"]):
        if r["wer"] <= target:
            reach = next((b["time"] for b in train if b["step"] >= r["step"]), None)
            return {"step": r["step"], "wer": r["wer"], "ckpt_sha256": r["ckpt_sha256"], "identity": identity,
                    "wall_s_to_step": round(reach - t_start, 1) if reach and t_start else None}
    return {"verdict": "NOT_REACHED", "best": min(r["wer"] for r in rows), "identity": identity}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", help="1-GPU reference run dir (per-rank efficiency, gate ratio)")
    ap.add_argument("--gpu-csv", nargs="*", default=[])
    ap.add_argument("--target-wer", type=float)
    ap.add_argument("--wer-file", help="canonical scores JSONL (step, ckpt_sha256, cohort, scorer, weights, wer)")
    ap.add_argument("--cohort")
    ap.add_argument("--scorer")
    ap.add_argument("--weights", choices=["raw", "ema"])
    ap.add_argument("--expect-gpus", type=int, help="GPU UUIDs the energy samples must cover (default: world)")
    ap.add_argument("--gate", type=float, help="required throughput ratio vs --ref (e.g. 1.6, 2.6)")
    a = ap.parse_args()
    train, other, cfg = load(a.run_dir)
    tcfg = cfg.get("train") or {}
    ckpt_every = int(tcfg.get("ckpt_every_steps", 0) or 0)
    if tcfg.get("weights_every"):  # eval-only snapshots are async saves too: the coarser grid of the two
        ckpt_every = min(v for v in (ckpt_every, int(tcfg["weights_every"])) if v > 0)
    world = int(cfg.get("world") or (len(train[-1].get("rank_step_s", [])) if train else 1) or 1)
    wins = windows(blocks(train, other, ckpt_every))
    for w in wins:
        if a.gpu_csv:
            w.update(gpu_energy(a.gpu_csv, w["t0"], w["t1"], a.expect_gpus or world))
            if "total_J" in w and w["audio_h"] > 0:
                w["J_per_audio_h"] = round(w["total_J"] / w["audio_h"], 1)
    rep = {"run_dir": a.run_dir, "world": world, "n_windows": len(wins), "windows": wins,
           "protocol": {"warm_steps": WARM_STEPS, "warm_s": WARM_S, "min_blocks": MIN_BLOCKS, "min_s": MIN_S}}
    if wins:
        thr = [w["audio_h_per_h"] for w in wins]
        rep["audio_h_per_h_median"] = st.median(thr)
        rep["audio_h_per_h_range"] = [min(thr), max(thr)]
    if a.ref and wins:
        rt, ro, rc = load(a.ref)
        rw = windows(blocks(rt, ro, int((rc.get("train") or {}).get("ckpt_every_steps", 0) or 0)))
        rep["ref_n_windows"] = len(rw)
        if rw:
            ref = st.median(w["audio_h_per_h"] for w in rw)
            ratio = [w["audio_h_per_h"] / ref for w in wins]
            rep["ref_audio_h_per_h_median"] = ref
            rep["speedup_median"] = round(st.median(ratio), 3)
            rep["speedup_range"] = [round(min(ratio), 3), round(max(ratio), 3)]
            rep["per_rank_efficiency"] = round(st.median(ratio) / world, 3)
            if a.gate:
                lo, hi = min(ratio), max(ratio)
                if len(wins) < 3 or len(rw) < 3:  # a verdict needs three steady windows on both sides
                    verdict = "INSUFFICIENT_WINDOWS"
                else:
                    verdict = "PASS" if lo >= a.gate else ("FAIL" if hi < a.gate else "INCONCLUSIVE")
                rep["gate"] = {"required": a.gate, "verdict": verdict,
                               "windows": [len(wins), len(rw)]}
    if a.target_wer is not None:
        if not (a.wer_file and a.cohort and a.scorer and a.weights):
            rep["time_to_wer"] = {"verdict": "REFUSED", "reason": "needs --wer-file --cohort --scorer --weights"}
        else:
            rep["time_to_wer"] = time_to_wer(a.wer_file, train, a.target_wer,
                                             {"cohort": a.cohort, "scorer": a.scorer, "weights": a.weights})
    if len(wins) < 3:
        rep["warning"] = f"only {len(wins)} steady window(s); the protocol needs 3"
    json.dump(rep, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
