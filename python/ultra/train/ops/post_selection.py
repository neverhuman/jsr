"""Post selection rows (per split, pooled over the panel) to AGENT_CHAT. Usage: post_selection.py <label> <selection.jsonl>"""
import datetime, json, sys
label, path = sys.argv[1:3]
rows = [json.loads(l) for l in open(path) if l.strip()]
cands = list(dict.fromkeys(r["candidate"] for r in rows))
lines = ["| candidate | panel dev-clean (400) | panel dev-other (330) | panel pooled | first-400 dev-clean |", "|---|---:|---:|---:|---:|"]
best = None
for c in cands:
    rs = {r["panel"]: r for r in rows if r["candidate"] == c}
    a, b = rs.get("selection:dev-clean"), rs.get("selection:dev-other")
    pooled = None
    if a and b:
        pooled = round(100 * (a["sub"] + a["del"] + a["ins"] + b["sub"] + b["del"] + b["ins"]) / (a["ref_words"] + b["ref_words"]), 3)
        best = (pooled, c) if best is None or pooled < best[0] else best
    f = rs.get("first400")
    lines.append(f"| {c} | {a['wer'] if a else '-'} | {b['wer'] if b else '-'} | {pooled if pooled is not None else '-'} | {f['wer'] if f else '-'} |")
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
with open("/home/ubuntu/jsr/AGENT_CHAT.md", "a") as fh:
    fh.write(f"\n## {ts} | trainer-lane | {label} finalist selection rows (auto; panel = origin/main dev_selection_panel.json)\n" + "\n".join(lines) +
             (f"\n- **Best on the selection panel (pooled WER): {best[1]} ({best[0]}%).** first-400 is for continuity only.\n" if best else "\n"))
