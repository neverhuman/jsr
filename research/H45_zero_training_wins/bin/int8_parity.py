#!/usr/bin/env python3
"""h45-wins: score PR #76's ultra-engine-commit full-dev outputs (provisional + committed) with the pinned ultra-score.

  int8_parity.py <name> <outdir> <engine output .jsonl> [...]   (all 5,567 dev utterances must be covered)
Refs, speakers and durations come from eval-lane's B0 INT8 full-dev score input (same 5,567 utterances). Paired
comparisons: provisional vs eval-lane's B0 INT8 (cold-wav path, 8.272), committed vs the PyTorch chunk-16 row (7.703).
"""
import json, os, subprocess, sys
H = "/mnt/xbabe3-external/jsr/h45-wins"
REF = "/mnt/xbabe3-external/jsr/evals/eval-lane/export/A1-49000-ema/fulldev-int8/vs-pytorch.engine.score-input.jsonl"
PT16 = "/mnt/xbabe3-external/jsr/evals/eval-lane/curve/A1-49000.ema.c16.ms8.score-input.jsonl"
name, out = sys.argv[1:3]
inputs = sys.argv[3:]  # any number of engine output files; truncated lines (a parked run) are skipped
ref = {json.loads(l)["id"]: json.loads(l) for l in open(REF)}
eng = {}
dup_rows = []
for p in inputs:
    for l in open(p):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("summary"):
            continue
        if r["uid"] in eng:
            o = eng[r["uid"]]
            same = all(o[k] == r[k] for k in ("provisional", "committed", "revised"))
            dup_rows.append({"uid": r["uid"], "same_text": same})
            if not same:
                raise SystemExit(f"duplicate uid {r['uid']} with different TEXT across runs")
            continue
        eng[r["uid"]] = r
assert set(eng) == set(ref), (len(eng), len(ref))
os.makedirs(out, exist_ok=True)
paths = {}
for field in ("provisional", "committed"):
    p = os.path.join(out, f"{name}.{field}.score-input.jsonl")
    with open(p, "w") as fh:
        for uid, r in ref.items():
            fh.write(json.dumps(dict(r, hyp=eng[uid][field])) + "\n")
    paths[field] = p
rev = sum(bool(r["revised"]) for r in eng.values()) / len(eng)
def cmp(b, a):
    o = subprocess.run([sys.executable, f"{H}/bin/h45_score.py", "rows", b, "--vs", a, "--replicates", "10000"],
                       capture_output=True, text=True, check=True, env=dict(os.environ, H45_TMP="/dev/shm/h45-wins/tmp"))
    return json.loads(o.stdout)
p = cmp(paths["provisional"], REF)
c = cmp(paths["committed"], PT16)
pc = cmp(paths["committed"], paths["provisional"])
res = {"name": name, "utterances": len(eng), "rows_seen_twice": len(dup_rows),
       "rows_seen_twice_identical_text": sum(d["same_text"] for d in dup_rows), "revised_pct": round(100 * rev, 2),
       "provisional_wer": p["b"]["pooled"]["wer"], "evallane_int8_wer": p["a"]["pooled"]["wer"],
       "prov_minus_evallane_int8": [p["delta_pp"], p["delta_lower_pp"], p["delta_upper_pp"]],
       "committed_wer": c["b"]["pooled"]["wer"], "pytorch_c16_wer": c["a"]["pooled"]["wer"],
       "committed_minus_pytorch_c16": [c["delta_pp"], c["delta_lower_pp"], c["delta_upper_pp"]],
       "committed_minus_provisional": [pc["delta_pp"], pc["delta_lower_pp"], pc["delta_upper_pp"]],
       "commit_ms_p50_p95": None}
cms = sorted(r["commit_wall_ms"] for r in eng.values())
res["commit_ms_p50_p95"] = [round(cms[len(cms) // 2], 1), round(cms[int(0.95 * (len(cms) - 1))], 1)]
res["commit_cpu_s_per_audio_s"] = round(sum(cms) / 1000 / sum(r["audio_s"] for r in eng.values()), 4)
json.dump(res, open(os.path.join(out, f"{name}.parity.json"), "w"), indent=1)
print(json.dumps(res))
