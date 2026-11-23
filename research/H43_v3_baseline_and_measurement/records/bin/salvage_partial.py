#!/usr/bin/env python3
"""eval-lane: split interrupted full-dev paced traces into (a) COMPLETE utterances (a final=true text event exists; every event of
an utterance is kept) and (b) the remainder ids to re-run. The cut utterance of each shard is dropped entirely.
  salvage_partial.py <trace dir> <index.json> <out complete.trace.jsonl> <out remainder.ids>"""
import json, sys, glob
d, index, out_trace, out_ids = sys.argv[1:5]
events, final = {}, set()
bad = 0
for p in sorted(glob.glob(f"{d}/shard-*.trace.jsonl")):
    for line in open(p):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:          # a line cut by the termination
            bad += 1; continue
        events.setdefault(e["id"], []).append(line if line.endswith("\n") else line + "\n")
        if e.get("kind") == "text" and e.get("final"):
            final.add(e["id"])
ids = [r["id"] for r in json.load(open(index))]
with open(out_trace, "w") as f:
    for i in ids:
        if i in final:
            f.writelines(events[i])
rem = [i for i in ids if i not in final]
open(out_ids, "w").write("\n".join(rem) + "\n")
print(json.dumps({"dir": d, "complete": len(final), "remainder": len(rem), "cut_utterances_dropped": len(set(events) - final), "malformed_lines": bad}))
