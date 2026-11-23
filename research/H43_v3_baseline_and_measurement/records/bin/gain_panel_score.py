#!/usr/bin/env python3
"""eval-lane: B0 gain-stress panel (lead 08:53Z item 2). For each input-gain level, canonical WER (ultra-score) and first-word error
(first reference word not correct in the final transcript, canonical normaliser + frozen aligner), each paired against 0 dB
(speaker-cluster bootstrap). Decision aid fixed by the lead: degradation > 2x the chunk-4 four-run spread (0.174 pp) at -20 / -30 dB.
  gain_panel_score.py OUTDIR LEVEL:utts.jsonl:predictions.jsonl [LEVEL:...]   (predictions keyed by the staged id; utts.jsonl gives uid/ref)"""
import importlib.util, json, random, subprocess, sys, os
spec = importlib.util.spec_from_file_location("h43l", "/mnt/xbabe3-external/jsr/evals/eval-lane/bin/h43_latency.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SC = "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score"
first_json = lambda b: json.JSONDecoder().raw_decode(b.decode())[0]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
levels = {}
for arg in sys.argv[2:]:
    lv, utts, pred = arg.split(":")
    U = [json.loads(l) for l in open(utts)]
    P = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(pred)}
    rows = []
    for u in U:
        key = u.get("staged_id", u["uid"])
        if key not in P:
            sys.exit(f"level {lv}: missing prediction for {key}")
        rows.append({"id": u["uid"], "speaker": u["speaker"], "dataset": u["dataset"], "ref": u["ref"], "hyp": P[key], "duration_s": u.get("samples", 0) / 16000})
    p = f"{out}/B0-gain{lv}.score-input.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fw = {}
    for r in rows:
        ref, hyp = m.normalise_words(r["ref"]), m.normalise_words(r["hyp"])
        ops = m.align(ref, hyp); ok = any(op == "C" and i == 0 for op, i, j in ops)
        fw[r["id"]] = (r["dataset"], r["speaker"], 0 if ok else 1)
    levels[lv] = (p, fw)
base = levels["0"]
summary = {"label": "exploratory", "levels": {}}
for lv, (p, fw) in levels.items():
    res = first_json(subprocess.run([SC, base[0], p, "--seed", "20261122", "--replicates", "10000", "--alpha", "0.05"], capture_output=True, check=True).stdout)
    json.dump(res, open(f"{out}/B0-gain{lv}-vs-gain0.score.json", "w"), indent=1)
    c = res["comparison"]["pooled_delta"]; s = res["system_b"]
    # first-word error, paired speaker bootstrap vs 0 dB
    ids = sorted(fw); clusters = {}
    for i in ids:
        clusters.setdefault((fw[i][0], fw[i][1]), []).append(i)
    point = sum(fw[i][2] for i in ids) / len(ids); d0 = sum(fw[i][2] - base[1][i][2] for i in ids) / len(ids)
    rng = random.Random(20261122); keys = list(clusters); ds = []
    for _ in range(2000):
        sel = [i for k in (rng.choice(keys) for _ in keys) for i in clusters[k]]
        ds.append(sum(fw[i][2] - base[1][i][2] for i in sel) / len(sel))
    ds.sort()
    summary["levels"][lv] = {"wer_pct": {x: round(100 * s["datasets"][x]["wer"], 3) for x in ("dev-clean", "dev-other")} | {"pooled": round(100 * s["pooled_wer"], 3)},
                             "delta_vs_0dB_pp": round(100 * c["delta"], 4), "one_sided_95": [round(100 * c["lower"], 4), round(100 * c["upper"], 4)],
                             "first_word_error_pct": round(100 * point, 2), "first_word_error_delta_vs_0dB_pp": round(100 * d0, 2),
                             "first_word_error_delta_95": [round(100 * ds[int(0.025 * len(ds))], 2), round(100 * ds[int(0.975 * len(ds)) - 1], 2)]}
    L = summary["levels"][lv]
    print(f"{lv:>4} dB: WER {L['wer_pct']['pooled']} (clean {L['wer_pct']['dev-clean']}, other {L['wer_pct']['dev-other']}) | vs 0 dB {L['delta_vs_0dB_pp']:+.3f} [{L['one_sided_95'][0]:+.3f}, {L['one_sided_95'][1]:+.3f}] | first-word error {L['first_word_error_pct']}% (vs 0 dB {L['first_word_error_delta_vs_0dB_pp']:+.2f} [{L['first_word_error_delta_95'][0]:+.2f}, {L['first_word_error_delta_95'][1]:+.2f}] pp)")
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
