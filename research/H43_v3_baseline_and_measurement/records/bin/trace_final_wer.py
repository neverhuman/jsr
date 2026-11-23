#!/usr/bin/env python3
"""eval-lane: canonical full-dev WER of the ENGINE path from paced traces (the final text event of each utterance), so every latency
row has its own engine WER (e.g. B0 at 80 ms, which was never decoded in PyTorch).   trace_final_wer.py TRACE_DIR OUT_PREFIX"""
import json, glob, subprocess, sys
d, out = sys.argv[1:3]; E = "/mnt/xbabe3-external/jsr/evals/eval-lane"; SC = "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score"
fin = {}
for p in glob.glob(f"{d}/*.trace.jsonl"):
    for line in open(p):
        e = json.loads(line)
        if e.get("kind") == "text" and e.get("final"):
            fin[e["id"]] = e.get("text") or ""
ix = json.load(open(f"{E}/export/fulldev-wav.index.json"))
missing = [r["id"] for r in ix if r["id"] not in fin]
if missing:
    sys.exit(f"{len(missing)} ids without a final event, e.g. {missing[:3]}")
with open(out + ".score-input.jsonl", "w") as f:
    for r in ix:
        f.write(json.dumps({"id": r["uid"], "speaker": r["speaker"], "dataset": r["dataset"], "ref": r["ref"], "hyp": fin[r["id"]], "duration_s": r["duration_s"]}) + "\n")
res = json.JSONDecoder().raw_decode(subprocess.run([SC, out + ".score-input.jsonl", "--replicates", "200"], capture_output=True, check=True).stdout.decode())[0]
json.dump(res, open(out + ".score.json", "w"), indent=1); s = res["system_a"]
print(f"{out.rsplit('/', 1)[-1]}: engine WER from trace finals {100*s['datasets']['dev-clean']['wer']:.3f} / {100*s['datasets']['dev-other']['wer']:.3f} / pooled {100*s['pooled_wer']:.3f}")
