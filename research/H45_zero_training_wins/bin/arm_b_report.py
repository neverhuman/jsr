#!/usr/bin/env python3
"""h45-wins Arm B report: B1 gate (M12 vs trained arm 12, same GPU) and the B2 bundle table vs the chunk arm.

  arm_b_report.py <h45 dir>      (expects out/b1/*.c{4,8,16}.ms8.dev-*.jsonl and out/b2/...)
Pooled WER from the pinned ultra-score; paired deltas with the speaker-cluster bootstrap (10,000, seed 20261122).
"""
import json, os, subprocess, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/xbabe3-external/jsr/h45-wins"
SC = os.path.join(D, "bin", "h45_score.py")
SPREAD = 0.087


def score(prefix, vs=None, uids=None):
    cmd = [sys.executable, SC, "row", prefix]
    if vs:
        cmd += ["--vs", vs, "--replicates", "10000"]
    if uids:
        cmd += ["--uids", uids]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def hyps(prefix):
    out = {}
    for s in ("dev-clean", "dev-other"):
        for line in open(f"{prefix}.{s}.jsonl"):
            r = json.loads(line)
            out[r["uid"]] = r["hyp"]
    return out


def have(prefix):
    return all(os.path.exists(f"{prefix}.{s}.jsonl") for s in ("dev-clean", "dev-other"))


lines = ["## B1: M12 = arm10 + arm11 - control (EMA) vs trained arm 12 (same GPU, full dev, pooled)", "",
         "| chunk | M12 | arm 12 | arm 12b | M12 - arm12 [95% one-sided] | arm12b - arm12 | gate |pM12-arm12| <= 0.087 | identical hyps M12/arm12 | arm12b/arm12 |",
         "|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
gate_all = True
for c in (4, 8, 16):
    p = lambda t: f"{D}/out/b1/{t}.c{c}.ms8"
    if not (have(p("M12")) and have(p("arm12")) and have(p("arm12b"))):
        lines.append(f"| {c} | (missing) |"); gate_all = False
        continue
    m = score(p("M12"), vs=p("arm12"))
    b = score(p("arm12b"), vs=p("arm12"))
    hm, ha, hb = hyps(p("M12")), hyps(p("arm12")), hyps(p("arm12b"))
    agree_m = sum(hm[u] == ha[u] for u in ha) / len(ha)
    agree_b = sum(hb[u] == ha[u] for u in ha) / len(ha)
    ok = abs(m["delta_pp"]) <= SPREAD
    gate_all &= ok
    lines.append(f"| {c} | {m['b']['pooled']['wer']:.3f} | {m['a']['pooled']['wer']:.3f} | {b['b']['pooled']['wer']:.3f} | "
                 f"{m['delta_pp']:+.3f} [{m['delta_lower_pp']:+.3f}, {m['delta_upper_pp']:+.3f}] | {b['delta_pp']:+.3f} | "
                 f"{'PASS' if ok else 'FAIL'} | {100 * agree_m:.1f}% | {100 * agree_b:.1f}% |")
lines += ["", f"**B1 verdict: {'PASS' if gate_all else 'FAIL'}** (pre-registered: |M12 - arm 12| <= 0.087 pp pooled at each of chunks 4, 8, 16).", ""]
extra = []
for t in ("arm10", "arm11", "control", "M12raw"):
    row = []
    for c in (4, 8, 16):
        pr = f"{D}/out/b1/{t}.c{c}.ms8"
        row.append(f"{score(pr)['pooled']['wer']:.3f}" if have(pr) else "-")
    extra.append(f"| {t} | " + " | ".join(row) + " |")
if extra:
    lines += ["| other rows (same GPU) | c4 | c8 | c16 |", "|---|---:|---:|---:|"] + extra + [""]

lines += ["## B2: bundles onto the chunk arm (EMA, same GPU, full dev, pooled; delta vs chunk arm)", "",
          "| model | c4 | c8 | c16 | Δc4 [95%] | Δc8 | Δc16 |", "|---|---:|---:|---:|---:|---:|---:|"]
single = {}
for t in ("chunk", "arm16", "arm17", "arm18", "arm7", "B2-full", "B2+7", "B2-16", "B2-17", "B2-18", "chunk_s4321", "B2-full-s4321"):
    base = "chunk_s4321" if t == "B2-full-s4321" else "chunk"
    cells, deltas = [], []
    for c in (4, 8, 16):
        pr, pb = f"{D}/out/b2/{t}.c{c}.ms8", f"{D}/out/b2/{base}.c{c}.ms8"
        if not (have(pr) and have(pb)):
            cells.append("-"); deltas.append("-"); continue
        if t == base:
            cells.append(f"{score(pr)['pooled']['wer']:.3f}"); deltas.append("0"); continue
        r = score(pr, vs=pb)
        cells.append(f"{r['b']['pooled']['wer']:.3f}")
        single[(t, c)] = r["delta_pp"]
        deltas.append(f"{r['delta_pp']:+.3f}" + (f" [{r['delta_lower_pp']:+.3f}, {r['delta_upper_pp']:+.3f}]" if c == 4 else ""))
    lines.append(f"| {t}{' (vs chunk_s4321)' if base != 'chunk' else ''} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
lines += ["", "| additivity | c4 | c8 | c16 |", "|---|---:|---:|---:|"]
for name, parts in (("B2-full", ("arm16", "arm17", "arm18")), ("B2-16", ("arm17", "arm18")), ("B2-17", ("arm16", "arm18")),
                    ("B2-18", ("arm16", "arm17"))):
    row = []
    for c in (4, 8, 16):
        if all((p, c) in single for p in parts) and (name, c) in single:
            pred = sum(single[(p, c)] for p in parts)
            row.append(f"obs {single[(name, c)]:+.3f} / sum {pred:+.3f}")
        else:
            row.append("-")
    lines.append(f"| {name} | " + " | ".join(row) + " |")
print("\n".join(lines))
