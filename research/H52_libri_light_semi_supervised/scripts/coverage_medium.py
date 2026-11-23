"""H52: medium coverage under the label-integrity policy (rows with an accepted R0 label whose src/t0/t1 match)."""
import glob, json, os, sys
import pyarrow.parquet as pq
raw, policy = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
pol = json.load(open(policy)) if policy else None
ldir = os.path.join(raw, "r0src")
lab, rank, units = {}, {}, {"code": 0, "witness": 0, "rejected": 0}
for fn in sorted(os.listdir(ldir)):
    if not fn.endswith(".labeled"):
        continue
    mk = json.load(open(os.path.join(ldir, fn)))
    unit, code = mk.get("unit", fn[:-8]), mk.get("code_sha256", "")
    if pol is None:
        pr = 1
    elif code in pol.get("accept_code", []):
        pr, units["code"] = 2, units["code"] + 1
    elif unit in set(pol.get("accept_units_v1", [])):
        pr, units["witness"] = 1, units["witness"] + 1
    else:
        units["rejected"] += 1
        continue
    for line in open(os.path.join(ldir, mk.get("labels", fn[:-8] + ".r0labels.jsonl"))):
        if '"header"' in line[:20]:
            continue
        r = json.loads(line)
        if "id" in r and pr >= rank.get(r["id"], 0):
            lab[r["id"]] = (r.get("src"), r.get("t0"), r.get("t1")); rank[r["id"]] = pr
src = {}
for f in glob.glob(os.path.join(raw, "student_src", "*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "id" in r:
            src[r["id"]] = (r["t0"], r["t1"])
n = nl = ns = 0
for p in sorted(glob.glob(os.path.join(raw, "*.parquet"))):
    t = pq.read_table(p, columns=["id", "t0", "t1", "src"])
    for i, t0, t1, s in zip(*(t.column(c).to_pylist() for c in ("id", "t0", "t1", "src"))):
        n += 1
        nl += lab.get(i) == (s, t0, t1)
        ns += src.get(i) == (t0, t1)
print(json.dumps({"rows": n, "r0_labelled": nl, "r0_frac": round(nl / max(1, n), 4), "student": ns,
                  "student_frac": round(ns / max(1, n), 4), "units": units}))
