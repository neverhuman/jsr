"""RC2 exact-isa: per-utterance transcript identity of each witness run vs the confirmatory look's Zen 2 output,
and token identity across witness runs of the same policy. usage: python3 -I rc2_compare.py E_ROOT LOOK_MODES_DIR OUT"""
import hashlib, json, sys
from pathlib import Path
E, LOOK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
index = json.loads((E / "panel" / "index.json").read_bytes())
IDS = [r["uid"] for r in index]
assert len(IDS) == 730
POLICY = {"c4": "live_c4_greedy", "c4lm": "live_c4_lm", "c8lm": "live_c8_lm"}

def look(mode):
    rows = [json.loads(x) for x in (LOOK / f"{mode}.panel730.score-input.jsonl").read_text().splitlines() if x.strip()]
    return {r["id"]: r["hyp"] for r in rows}

res = {"schema": "exact-isa.rc2-compare/v1", "runs": []}
tokens = {}
for host_dir in sorted((E / "runs-rc2").iterdir()):
    if not host_dir.is_dir():
        continue
    for run in sorted(host_dir.iterdir()):
        rec = json.loads((run / "receipt.json").read_bytes()) if (run / "receipt.json").exists() else None
        if rec is None or rec["exit_code"] != 0:
            res["runs"].append({"host": host_dir.name, "run": run.name, "status": "missing or failed"}); continue
        if run.name.startswith("eval-"):
            j = json.loads((run / "eval" / "engine_eval.json").read_bytes())
            rows = j["rows"]; assert len(rows) == 730
            pol = POLICY[run.name.split("-")[2]]
            hyp = {IDS[r["index"]]: r["hyp"] for r in rows}
            tok = hashlib.sha256(json.dumps([r["tokens"] for r in sorted(rows, key=lambda r: r["index"])]).encode()).hexdigest()
            tokens.setdefault(pol, {})[f"{host_dir.name}/{run.name}"] = tok
            meta = {"isa": j["isa"], "streams": j["streams"], "errors": j["errors"], "chunk_beam_lm": [j.get("max_chunks"), j["beam"], bool(j["lm"])]}
        else:
            pol = "final_m5"
            rows = [json.loads(x) for x in (run / "commit.jsonl").read_text().splitlines() if x.strip()]
            rows = [r for r in rows if not r.get("summary")]
            assert len(rows) == 730
            hyp = {IDS[i]: r["committed"] for i, r in enumerate(rows)}
            meta = {}
        ref = look(pol)
        diff = [i for i in IDS if hyp[i] != ref[i]]
        res["runs"].append({"host": host_dir.name, "run": run.name, "policy": pol, "cpu_model": rec["cpu_model"],
                            "binary_sha256": rec["binary"]["sha256"], **meta,
                            "text_mismatches_vs_look": len(diff), "first_mismatches": diff[:5]})
res["token_stream_sha256_by_policy"] = tokens
res["token_identical_within_policy"] = {p: len(set(v.values())) == 1 for p, v in tokens.items()}
OUT.write_text(json.dumps(res, indent=1) + "\n")
for r in res["runs"]:
    print(r["host"], r["run"], r.get("policy"), r.get("isa"), r.get("streams"), "mismatch", r.get("text_mismatches_vs_look"), r.get("first_mismatches"))
print(res["token_identical_within_policy"])
