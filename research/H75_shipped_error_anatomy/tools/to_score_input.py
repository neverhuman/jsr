"""isosyll --json output -> ultra-score score-input JSONL, joined with S-half refs.jsonl. Usage: IN REFS OUT"""
import json, sys
from pathlib import Path
inp, refs, out = sys.argv[1:4]
R = {r["uid"]: r for r in map(json.loads, open(refs))}
H = {Path(r["file"]).stem: r for r in map(json.loads, open(inp)) if r.get("file")}
assert set(H) == set(R), (len(H), len(R), list(set(R) - set(H))[:5])
with open(out, "w") as fh:
    for uid in sorted(R):
        r = R[uid]
        fh.write(json.dumps({"id": uid, "speaker": str(r["speaker"]), "dataset": r["subset"], "ref": r["ref"], "hyp": H[uid]["text"], "duration_s": H[uid]["audio_s"]}) + "\n")
print("wrote", len(R), "audio_s", round(sum(h["audio_s"] for h in H.values()), 2), "wall_s", round(sum(h.get("wall_s", 0) for h in H.values()), 1))
