#!/usr/bin/env python3
"""h45-wins amendment 2 table: commit-pass final at chunk 16 by beam width, without / with M5 + LODR.

  amend2_report.py <dumps dir> <name> <select.json> [--subset confirmation|selection|full]

Rows: final greedy c16 (eval-lane dump), FastBeam finals b4 / b8 / b16 without LM and with the S-chosen LM weights (+ T4).
Per row: final-text WER (canonical ultra-score) and the paired delta vs the greedy c16 final; revision burden, segments
revised, fixes / breaks and first-word error for two provisional streams (greedy c4, beam-4 c4: eval-lane's dumps);
final-pass CPU from the INT8 engine runs (results/amend2/cpu, 200-utterance hash panel, chunk 16, xbabe2, not quiet).
"""
import json, os, subprocess, sys

H = "/mnt/xbabe3-external/jsr/h45-wins"
E = "/mnt/xbabe3-external/jsr/evals/eval-lane/curve"
PROV = {"greedy c4": (f"{E}/A1-49000.ema.c4.ms8", "hyp"), "beam-4 c4": (f"{E}/A1-49000.ema.c4.ms8.b4", "beam_hyp")}
G16 = f"{E}/A1-49000.ema.c16.ms8"
CPU = f"{H}/results/amend2/cpu"
ENV = dict(os.environ, H45_TMP="/dev/shm/h45-wins/tmp")


def run(cmd):
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, env=ENV).stdout.strip().splitlines()[-1])


def main():
    d, name, sel = sys.argv[1:4]
    subset = sys.argv[sys.argv.index("--subset") + 1] if "--subset" in sys.argv else "confirmation"
    uids = None if subset == "full" else f"{H}/prereg/uids_{subset}.txt"
    pick = json.load(open(sel))["pick"]
    rows = [("greedy", G16, "c16_greedy")]
    for b in (4, 8, 16):
        p = pick.get(f"b{b}")
        if not p:
            continue
        rows.append((f"beam {b}", f"{d}/{name}.{p['nolm']}", f"c16_b{b}"))
        rows.append((f"beam {b} + LM ({p['lm'].split('_', 1)[1]})", f"{d}/{name}.{p['lm']}", f"c16_b{b}_lm"))
    if os.path.exists(f"{d}/{name}.b4_l0.60_r0.45_T4.dev-clean.jsonl"):
        rows.append(("beam 4 + LM (T4 0.60 / 0.45)", f"{d}/{name}.b4_l0.60_r0.45_T4", "c16_b4_lm"))
    cpu = {}
    for f in os.listdir(CPU):
        s = json.load(open(os.path.join(CPU, f)))
        s = s.get("summary", s)
        cpu[f.split(".")[0]] = s["cpu_s_per_audio_s"]
    base_cpu = cpu.get("c16_greedy")
    out = []
    for label, prefix, ck in rows:
        sc = ["--uids", uids] if uids else []
        w = run([sys.executable, f"{H}/bin/h45_score.py", "row", prefix, "--vs", G16, "--replicates", "10000"] + sc)
        r = {"final": label, "wer": w["b"]["pooled"]["wer"], "delta_vs_greedy_c16": w["delta_pp"],
             "bounds": [w["delta_lower_pp"], w["delta_upper_pp"]], "edits": w["b"]["pooled"]["edits"],
             "cpu_s_per_audio_s": cpu.get(ck), "cpu_rel_greedy": (cpu.get(ck) / base_cpu) if (cpu.get(ck) and base_cpu) else None,
             "cpu_s_per_7s_segment": (7.03 * cpu[ck]) if cpu.get(ck) else None, "prov": {}}
        for pl, (pp, pf) in PROV.items():
            args = [sys.executable, f"{H}/bin/h45_commit_probe.py", "--prov", pp, "--prov-field", pf, "--final", prefix,
                    "--name", f"amend2.{subset}.{label.replace(' ', '_').replace('/', '-')}.{pl.replace(' ', '_')}",
                    "--out", f"{H}/results/amend2/probe"]
            if uids:
                args += ["--uids", uids]
            s = run(args)
            r["prov"][pl] = {"wer_prov": s["wer_prov"], "segments_changed_pct": s["segments_changed_pct"],
                             "rev_per_100w": s["revision_word_edits_per_100_words"], "fix": s["fixes_per_100_words"],
                             "brk": s["breaks_per_100_words"], "first_word_err_final": s["first_word_err_pct"]["final"],
                             "first_word_err_prov": s["first_word_err_pct"]["prov"]}
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open(f"{H}/results/amend2/table.{subset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
