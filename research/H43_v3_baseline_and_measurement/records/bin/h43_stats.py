#!/usr/bin/env python3
"""eval-lane H43 statistics tooling on top of the CANONICAL scorer (ultra-score; no WER is recomputed here).

Commands
  margins register --file REG.json --endpoint NAME --margin PP --direction lower_is_better [--note TEXT]
        Append a non-inferiority / superiority margin to a registry BEFORE any arm result exists. Each entry stores its UTC
        time and the registry's running sha256; `test` refuses an endpoint whose margin was registered after every compared result file (base and arm) existed.
  test  --a BASE.score-input.jsonl[,BASE2...] --b ARM.score-input.jsonl[,ARM2...] --kind superiority|noninferiority
        [--margin PP | --registry REG.json --endpoint NAME] [--alpha 0.05] [--subset dev_split_v2.json:confirmation]
        One comparison. With two files on a side it is the seed-pair reading: d = mean(arm seeds) - mean(base seeds), and the
        bootstrap runs ultra-score on a set where each utterance appears once per (arm seed, base seed) pairing, ids
        suffixed, speaker kept, so clusters resample together. The one-sided p-value is read off ultra-score's own bootstrap
        by bisection on --alpha (the alpha at which its one-sided bound crosses the null), with its fixed seed and replicates.
  holm  --family FAMILY.json [--alpha 0.05]
        Up to 4 predeclared comparisons (each a `test` spec); Holm step-down on their p-values at familywise alpha.
Decisions (per comparison; d in percentage points, negative = arm better):
  superiority     PASS if the one-sided (1-alpha') upper bound < 0, where alpha' is the Holm-adjusted level; else FAIL if the
                  point estimate >= 0; else INCONCLUSIVE.
  noninferiority  PASS if the upper bound <= +margin; FAIL if the LOWER bound > +margin (clearly worse); else INCONCLUSIVE.
Exploratory label always attached. The 2x A/A-spread replay rule stays a separate, necessary repo rule (not a test).
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, tempfile

SC = os.environ.get("ULTRA_SCORE", "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score")
TMP = os.environ.get("H43_TMP", "/dev/shm/eval-lane/h43-tmp")   # never H0's root disk


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def subset_ids(spec):
    if not spec:
        return None
    path, half = spec.rsplit(":", 1)
    return set(json.load(open(path))["utterances"][half])


def paired_files(base, arm, keep):
    os.makedirs(TMP, exist_ok=True)
    pairs = [(b, a) for a in arm for b in base]
    fd_a, pa = tempfile.mkstemp(dir=TMP, suffix=".arm.jsonl"); fd_b, pb = tempfile.mkstemp(dir=TMP, suffix=".base.jsonl")
    with os.fdopen(fd_a, "w") as fa, os.fdopen(fd_b, "w") as fb:
        for k, (b, a) in enumerate(pairs):
            for src, f in ((a, fa), (b, fb)):
                for line in open(src):
                    r = json.loads(line)
                    if keep is not None and r["id"] not in keep:
                        continue
                    if len(pairs) > 1:
                        r["id"] = f"{r['id']}#{k}"
                    f.write(json.dumps(r) + "\n")
    return pb, pa


def run(pb, pa, alpha):
    out = subprocess.run([SC, pb, pa, "--seed", "20261122", "--replicates", "10000", "--alpha", repr(alpha)], capture_output=True, check=True).stdout
    return json.loads(out)


def pvalue(pb, pa, null_pp):
    """One-sided p for H0: delta >= null (pp) vs H1: delta < null, by bisection of ultra-score's alpha."""
    f = lambda al: 100 * run(pb, pa, al)["comparison"]["pooled_delta"]["upper"]
    lo, hi = 1e-4, 0.499          # ultra-score accepts alpha in (0, 0.5)
    if f(lo) < null_pp:
        return lo
    if f(hi) >= null_pp:
        return 1.0 if 100 * run(pb, pa, 0.05)["comparison"]["pooled_delta"]["delta"] >= null_pp else 0.5
    for _ in range(14):
        mid = (lo * hi) ** 0.5
        if f(mid) < null_pp:
            hi = mid
        else:
            lo = mid
    return hi


def margin_from_registry(reg, endpoint, result_files):
    entries = [e for e in json.load(open(reg))["entries"] if e["endpoint"] == endpoint]
    if not entries:
        sys.exit(f"no margin registered for endpoint {endpoint}")
    e = entries[0]
    t_reg = datetime.datetime.fromisoformat(e["registered_utc"].replace("Z", "+00:00")).timestamp()
    if result_files and all(os.path.getmtime(p) < t_reg for p in result_files):
        sys.exit(f"margin for {endpoint} registered at {e['registered_utc']} after EVERY compared result already existed -> refused")
    return e["margin_pp"], e


def decide(kind, d, lo, hi, margin):
    if kind == "superiority":
        return "PASS" if hi < 0 else ("FAIL" if d >= 0 else "INCONCLUSIVE")
    return "PASS" if hi <= margin else ("FAIL" if lo > margin else "INCONCLUSIVE")


def one_test(spec, alpha_eff=0.05, with_p=True):
    base = spec["a"]; arm = spec["b"]
    keep = subset_ids(spec.get("subset"))
    margin = spec.get("margin", 0.0); reg = None
    if spec.get("registry"):
        margin, reg = margin_from_registry(spec["registry"], spec["endpoint"], base + arm)   # valid if any compared result is newer than the margin
    pb, pa = paired_files(base, arm, keep)
    try:
        r = run(pb, pa, alpha_eff)["comparison"]["pooled_delta"]
        d, lo, hi = 100 * r["delta"], 100 * r["lower"], 100 * r["upper"]
        null = 0.0 if spec["kind"] == "superiority" else margin
        p = pvalue(pb, pa, null) if with_p else None
    finally:
        os.remove(pb); os.remove(pa)
    return {"name": spec.get("name", ""), "kind": spec["kind"], "base": base, "arm": arm, "subset": spec.get("subset"), "margin_pp": margin,
            "margin_registry_entry": reg, "delta_pp": round(d, 4), "alpha": alpha_eff, "lower_pp": round(lo, 4), "upper_pp": round(hi, 4),
            "p_one_sided": p, "decision": decide(spec["kind"], d, lo, hi, margin), "inputs_sha256": {p_: sha(p_) for p_ in base + arm}}


def cmd_margins(a):
    reg = json.load(open(a.file)) if os.path.exists(a.file) else {"schema": "ultra.h43.margins/v1", "entries": []}
    if any(e["endpoint"] == a.endpoint for e in reg["entries"]):
        sys.exit(f"endpoint {a.endpoint} already registered; margins are never edited after registration")
    e = {"endpoint": a.endpoint, "margin_pp": a.margin, "direction": a.direction, "note": a.note,
         "registered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    reg["entries"].append(e)
    reg["entries_sha256"] = hashlib.sha256(json.dumps(reg["entries"], sort_keys=True).encode()).hexdigest()
    json.dump(reg, open(a.file, "w"), indent=1)
    print(json.dumps(e))


def cmd_test(a):
    spec = {"a": a.a.split(","), "b": a.b.split(","), "kind": a.kind, "margin": a.margin or 0.0, "registry": a.registry, "endpoint": a.endpoint,
            "subset": a.subset, "name": a.name}
    res = one_test(spec, a.alpha)
    res["label"] = "exploratory"; res["scorer_sha256"] = sha(SC)
    print(json.dumps(res, indent=1))


def cmd_holm(a):
    fam = json.load(open(a.family))
    tests = fam["comparisons"]
    if not 1 <= len(tests) <= 4:
        sys.exit("a family holds 1 to 4 predeclared comparisons")
    first = [one_test(t, a.alpha, with_p=True) for t in tests]
    order = sorted(range(len(first)), key=lambda i: first[i]["p_one_sided"])
    m = len(first); out = [None] * m; stop = False
    for rank, i in enumerate(order):
        alpha_i = a.alpha / (m - rank)
        if stop:
            r = dict(first[i]); r["holm_alpha"] = alpha_i; r["holm_reject_null"] = False
            r["decision"] = "INCONCLUSIVE" if r["decision"] != "FAIL" else "FAIL"
            out[i] = r; continue
        r = one_test(tests[i], alpha_i, with_p=False); r["p_one_sided"] = first[i]["p_one_sided"]
        r["holm_alpha"] = alpha_i; r["holm_reject_null"] = r["p_one_sided"] <= alpha_i
        if not r["holm_reject_null"]:
            stop = True
            if r["decision"] == "PASS":
                r["decision"] = "INCONCLUSIVE"
        out[i] = r
    rep = {"schema": "ultra.h43.holm/v1", "label": "exploratory", "family": a.family, "family_sha256": sha(a.family), "familywise_alpha": a.alpha,
           "scorer_sha256": sha(SC), "results": out}
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("margins"); m.add_argument("action", choices=["register"]); m.add_argument("--file", required=True); m.add_argument("--endpoint", required=True)
    m.add_argument("--margin", type=float, required=True); m.add_argument("--direction", default="lower_is_better"); m.add_argument("--note", default="")
    t = sub.add_parser("test"); t.add_argument("--a", required=True); t.add_argument("--b", required=True); t.add_argument("--kind", choices=["superiority", "noninferiority"], required=True)
    t.add_argument("--margin", type=float); t.add_argument("--registry"); t.add_argument("--endpoint"); t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--subset"); t.add_argument("--name", default="")
    h = sub.add_parser("holm"); h.add_argument("--family", required=True); h.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    {"margins": cmd_margins, "test": cmd_test, "holm": cmd_holm}[a.cmd](a)


if __name__ == "__main__":
    main()
