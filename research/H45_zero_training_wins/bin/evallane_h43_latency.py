#!/usr/bin/env python3
"""eval-lane H43 latency metrics on word-lag traces, against locked MFA word times (dev only), on either clock.

Clocks (all in the DESIGN / audio-sample clock unless stated):
  end clock    t - b_j   (b_j = MFA end of reference word j)
  onset clock  t - a_j   (a_j = MFA start of reference word j)
Metrics per system (each with a speaker-cluster bootstrap 95% interval, strata = dev subset):
  ftl            first text with >= 1 word (any word, right or wrong)          - word 0, both clocks; diagnostic
  fcw            first CORRECT word: time from which the reference's first word is shown, aligned, and never revised
                 again, minus word 0 (both clocks); coverage = share of utterances whose word 0 is correct in the end
  first_cq[D]    share of ALL utterances whose word 0 is stably correct by b_0 + D (end) or a_0 + D (onset)
  cq_prov[D]     share of ALL timed reference words correct (in the final transcript) and first shown at their
                 position by b_j + D (provisional: later flicker allowed, but the word must be right at the end)
  cq_stable[D]   the same with the stable time (never revised afterwards)
  stable_lag     stable time - b_j over reference words that are correct in the end
  revision       words retracted or changed between consecutive snapshots, per 100 final words; share of utterances
                 with any revision; mean time a retracted word was on screen (flicker, ms). A partial last word growing into a
                 longer word (subword display) is counted separately as a completion, not as a revision
  finalization   final event audio time - b_last (includes trailing silence; diagnostic)
  paced          the same ftl / fcw on the trace host's paced clock (emitted_ns), when the trace has it; NOT quiet
Words: the canonical normaliser ultra-en-wer-norm/v1 (a faithful port, checked with --parity against ultra-score counts) and
its frozen aligner tie-break (diagonal, deletion, insertion). MFA tokens are expanded by the same rules; expanded tokens
share their word's interval. Reference words without an MFA time are left out of the timed metrics (counted in coverage).
Usage:
  h43_latency.py score --trace-glob 'DIR/shard-*.trace.jsonl' --index fulldev-wav.index.json --mfa DIR --name NAME --out PREFIX
  h43_latency.py score --trace-glob '...compact.trace.jsonl' --clips100 timestamps_100_mfa.json --mfa DIR --name NAME --out PREFIX
  h43_latency.py compare A.utts.jsonl B.utts.jsonl --out PREFIX        (paired bootstrap of B - A on shared utterances)
  h43_latency.py parity SCORE-INPUT.jsonl                               (normaliser/aligner port vs ultra-score counts)
"""
import argparse, glob, hashlib, json, os, random, statistics as st, sys

NORMALISER_ID = "ultra-en-wer-norm/v1"
APOS = ["’", "‘", "ʼ", "`", "＇"]
WHOLE = [("can't", "can not"), ("cannot", "can not"), ("won't", "will not"), ("shan't", "shall not"), ("let's", "let us"), ("it's", "it is"),
         ("that's", "that is"), ("what's", "what is"), ("there's", "there is"), ("here's", "here is"), ("he's", "he is"), ("she's", "she is"),
         ("who's", "who is"), ("where's", "where is"), ("how's", "how is"), ("y'all", "you all")]
SUFFIX = [("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'m", " am"), ("'d", " would")]
WHOLE_D = dict(WHOLE)


def rule_table():
    s = NORMALISER_ID + "\nsteps:lowercase;apostrophe-variants;non-alnum-to-space-keep-intra-word-apostrophe;whole-word;suffix;collapse-ws\napostrophes:"
    s += "".join(f"U+{ord(c):04X}," for c in APOS) + "\nwhole:" + "".join(f"{a}={b};" for a, b in WHOLE)
    s += "\nsuffix:" + "".join(f"{a}={b};" for a, b in SUFFIX) + "\n"
    return s


def expand(tok):
    if tok in WHOLE_D:
        return WHOLE_D[tok].split(" ")
    for suf, rep in SUFFIX:
        if tok.endswith(suf) and len(tok) > len(suf):
            return [tok[: -len(suf)]] + rep.split()
    return [tok]


def normalise_words(text):
    chars = []
    for ch in text or "":
        for c in ch.lower():          # per character, like Rust char::to_lowercase (no contextual final sigma)
            chars.append("'" if c in APOS else c)
    out = []
    for i, c in enumerate(chars):
        if c.isalnum():
            out.append(c)
        elif c == "'":
            out.append(c if (i > 0 and chars[i - 1].isalnum() and i + 1 < len(chars) and chars[i + 1].isalnum()) else " ")
        else:
            out.append(" ")
    words = []
    for tok in "".join(out).split():
        words.extend(expand(tok))
    return words


def align(ref, hyp):
    """Frozen ultra-eval alignment (tie-break: diagonal, then deletion, then insertion). Returns ops: (op, i_ref, j_hyp)."""
    n, m = len(ref), len(hyp)
    dirs = [[0] * (m + 1) for _ in range(n + 1)]
    prev = list(range(m + 1))
    for j in range(1, m + 1):
        dirs[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        dirs[i][0] = 1
        r = ref[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (r != hyp[j - 1]); dele = prev[j] + 1; ins = cur[j - 1] + 1
            if sub <= dele and sub <= ins:
                cur[j], dirs[i][j] = sub, 0
            elif dele <= ins:
                cur[j], dirs[i][j] = dele, 1
            else:
                cur[j], dirs[i][j] = ins, 2
        prev = cur
    ops = []; i, j = n, m
    while i > 0 or j > 0:
        d = dirs[i][j]
        if i > 0 and j > 0 and d == 0:
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and (d == 1 or j == 0):
            ops.append(("D", i - 1, None)); i -= 1
        else:
            ops.append(("I", None, j - 1)); j -= 1
    return ops[::-1]


def load_mfa(mfa_dir):
    t = {}
    for split in ("dev-clean", "dev-other"):            # dev files only, opened by name; test files are never read
        for line in open(os.path.join(mfa_dir, f"{split}.jsonl")):
            r = json.loads(line)
            toks = []
            for w in r["words"]:
                for e in normalise_words(w["w"]):
                    toks.append((e, w["start_s"], w["end_s"]))
            t[r["id"]] = toks
    return t


def load_traces(paths):
    by = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("kind") == "packet":
                continue
            if e.get("kind") == "text":
                uid, a, wall = e["id"], e["through_sample"] / 16000.0, e.get("emitted_ns", None)
                wall = None if wall is None else wall / 1e9
            elif "id" in e:
                uid, a, wall = e["id"], float(e["audio_s"]), None
            else:
                uid, a, wall = e["input"].rsplit("/", 1)[-1].rsplit(".", 1)[0], float(e["audio_s"]), None
            by.setdefault(uid, []).append((a, wall if wall is not None else -1.0, normalise_words(e.get("text") or ""), bool(e.get("final", False))))
    for k in by:
        by[k].sort(key=lambda x: (x[0], x[1]))
    return by


def utt_metrics(ev, ref, times):
    """ev: [(audio_t, wall_t, words, final)], ref: canonical words, times: [(a, b) or None] per ref word."""
    seen = False; evs = []
    for a, w, words, fin in ev:                      # NeMo-style empty end markers after the first text are ignored
        if words:
            seen = True
        if words or not seen:
            evs.append((a, w, words))
    out = {"n_ref": len(ref), "n_timed": sum(1 for x in times if x is not None)}
    if not evs or not any(w for _, _, w in evs):
        out.update(final=[], any_text=False)
        return out
    final = evs[-1][2]; n = len(final)
    # stable time per final position: earliest snapshot after which positions 0..i never change again
    ok, stab = [True] * n, [None] * n
    for a, w, words in reversed(evs):
        k = 0
        while k < n and k < len(words) and words[k] == final[k]:
            k += 1
        for i in range(n):
            if ok[i]:
                if i < k:
                    stab[i] = a
                else:
                    ok[i] = False
    prov = [None] * n                                 # first time the final word is shown at its position
    for a, w, words in evs:
        for i in range(min(n, len(words))):
            if prov[i] is None and words[i] == final[i]:
                prov[i] = a
    # revision burden
    revised = 0; completed = 0; flicker = []; shown_at = []; prev = []
    for a, w, words in evs:
        k = 0
        while k < len(prev) and k < len(words) and prev[k] == words[k]:
            k += 1
        if k < len(prev):
            # the last shown word growing into a longer word (a partial word completed) is a completion, not a revision
            if k == len(prev) - 1 and k < len(words) and words[k].startswith(prev[k]):
                completed += 1
                shown_at = shown_at[:k] + [shown_at[k]] + [a] * (len(words) - k - 1)
                prev = words
                continue
            revised += len(prev) - k
            flicker.extend(a - shown_at[p] for p in range(k, len(prev)))
        shown_at = shown_at[:k] + [a] * (len(words) - k)
        prev = words
    ops = align(ref, final)
    match = {i: j for op, i, j in ops if op == "C"}
    first = next(((a, w) for a, w, words in evs if words), None)
    out.update(any_text=True, final=final, revised=revised, completed=completed, flicker=flicker, n_final=n,
               edits=sum(1 for op, _, _ in ops if op != "C"),
               ftl_t=first[0], ftl_wall=first[1], last_t=evs[-1][0],
               word=[(j, (stab[match[j]] if j in match else None), (prov[match[j]] if j in match else None)) for j in range(len(ref))])
    return out


def summ(x):
    if not x:
        return None
    s = sorted(x)
    return {"n": len(s), "mean_ms": round(1000 * st.mean(s), 1), "p50_ms": round(1000 * s[len(s) // 2], 1), "p95_ms": round(1000 * s[int(len(s) * 0.95)], 1)}


DS = (150, 300, 500)
FDS_END = (150, 250, 300, 500, 1000)
FDS_ON = (300, 500, 1000)


def per_utt_rows(by, refs, mfa):
    rows = []
    for uid_trace, info in refs.items():
        ref = normalise_words(info["ref"]); mt = mfa.get(info["uid"], [])
        mw = [w for w, _, _ in mt]
        times = [None] * len(ref)
        if mw == ref:
            times = [(a, b) for _, a, b in mt]
        else:                                        # transfer MFA times onto the canonical reference by alignment
            for op, i, j in align(ref, mw):
                if op == "C":
                    times[i] = (mt[j][1], mt[j][2])
        u = utt_metrics(by.get(uid_trace, []), ref, times)
        r = {"id": uid_trace, "uid": info["uid"], "speaker": info["speaker"], "dataset": info["dataset"], "n_ref": len(ref), "n_timed": u["n_timed"],
             "any_text": u["any_text"]}
        if u["any_text"]:
            w0 = times[0] if times else None
            r.update(revised=u["revised"], completed=u["completed"], n_final=u["n_final"], flicker_s=round(sum(u["flicker"]), 4), n_flicker=len(u["flicker"]), edits=u["edits"])
            if w0:
                r["ftl_end"] = u["ftl_t"] - w0[1]; r["ftl_onset"] = u["ftl_t"] - w0[0]
                if u["ftl_wall"] >= 0:
                    r["ftl_paced_end"] = u["ftl_wall"] - w0[1]
            j0 = u["word"][0] if u["word"] else None
            if j0 and j0[1] is not None and w0:
                r["fcw_end"] = j0[1] - w0[1]; r["fcw_onset"] = j0[1] - w0[0]
            r["first_word_correct"] = bool(j0 and j0[1] is not None)
            cq_p = {D: 0 for D in DS}; cq_s = {D: 0 for D in DS}; lags = []
            for j, s_t, p_t in u["word"]:
                if times[j] is None or s_t is None:
                    continue
                b = times[j][1]; lags.append(s_t - b)
                for D in DS:
                    cq_s[D] += (s_t - b) * 1000 <= D; cq_p[D] += (p_t - b) * 1000 <= D
            r["cq_stable"] = cq_s; r["cq_prov"] = cq_p; r["stable_lags"] = [round(x, 4) for x in lags]
            last_b = next((times[j][1] for j in range(len(ref) - 1, -1, -1) if times[j]), None)
            if last_b is not None:
                r["finalization"] = u["last_t"] - last_b
        rows.append(r)
    return rows


def aggregate(rows):
    U = len(rows); timed = sum(r["n_timed"] for r in rows)
    get = lambda k: [r[k] for r in rows if k in r]
    lags = [x for r in rows for x in r.get("stable_lags", [])]
    fcw_cov = sum(1 for r in rows if r.get("first_word_correct"))
    first_cq_end = {D: sum(1 for r in rows if "fcw_end" in r and r["fcw_end"] * 1000 <= D) / U for D in FDS_END}
    first_cq_on = {D: sum(1 for r in rows if "fcw_onset" in r and r["fcw_onset"] * 1000 <= D) / U for D in FDS_ON}
    cq_s = {D: sum(r.get("cq_stable", {}).get(D, 0) for r in rows) / timed for D in DS}
    cq_p = {D: sum(r.get("cq_prov", {}).get(D, 0) for r in rows) / timed for D in DS}
    nfinal = sum(r.get("n_final", 0) for r in rows)
    return {"utterances": U, "ref_words": sum(r["n_ref"] for r in rows), "timed_ref_words": timed,
            "utterances_with_text": sum(1 for r in rows if r["any_text"]),
            "ftl_end": summ(get("ftl_end")), "ftl_onset": summ(get("ftl_onset")), "ftl_paced_end_trace_host": summ(get("ftl_paced_end")),
            "fcw_coverage": round(fcw_cov / U, 4), "fcw_end": summ(get("fcw_end")), "fcw_onset": summ(get("fcw_onset")),
            "first_cq_end": {str(D): round(v, 4) for D, v in first_cq_end.items()}, "first_cq_onset": {str(D): round(v, 4) for D, v in first_cq_on.items()},
            "cq_stable": {str(D): round(v, 4) for D, v in cq_s.items()}, "cq_provisional": {str(D): round(v, 4) for D, v in cq_p.items()},
            "stable_lag": summ(lags),
            "revision": {"revised_words_per_100_final": round(100 * sum(r.get("revised", 0) for r in rows) / max(1, nfinal), 3),
                         "partial_word_completions_per_100_final": round(100 * sum(r.get("completed", 0) for r in rows) / max(1, nfinal), 3),
                         "utterances_with_any_revision": round(sum(1 for r in rows if r.get("revised", 0) > 0) / U, 4),
                         "mean_flicker_ms": round(1000 * sum(r.get("flicker_s", 0) for r in rows) / max(1, sum(r.get("n_flicker", 0) for r in rows)), 1)},
            "finalization": summ(get("finalization"))}


def scalar_stats(rows):
    """Utterance-level numerators/denominators for ratio metrics (bootstrap-friendly)."""
    out = {}
    for r in rows:
        d = {"U": 1, "T": r["n_timed"], "F": r.get("n_final", 0), "R": r.get("revised", 0), "fcw": int(bool(r.get("first_word_correct"))),
             "ftl_end_sum": r.get("ftl_end", 0.0), "ftl_end_n": int("ftl_end" in r), "fcw_end_sum": r.get("fcw_end", 0.0), "fcw_end_n": int("fcw_end" in r),
             "ftl_on_sum": r.get("ftl_onset", 0.0), "fcw_on_sum": r.get("fcw_onset", 0.0)}
        cs, cp = r.get("cq_stable", {}), r.get("cq_prov", {})          # int keys in memory, str keys after a JSON round trip
        for D in DS:
            d[f"cqs{D}"] = cs.get(D, cs.get(str(D), 0)); d[f"cqp{D}"] = cp.get(D, cp.get(str(D), 0))
        for D in FDS_END:
            d[f"fe{D}"] = int("fcw_end" in r and r["fcw_end"] * 1000 <= D)
        for D in FDS_ON:
            d[f"fo{D}"] = int("fcw_onset" in r and r["fcw_onset"] * 1000 <= D)
        q = {k: r[k] for k in ("ftl_end", "fcw_end", "fcw_onset", "ftl_onset") if k in r}
        out[r["id"]] = (r["dataset"], r["speaker"], d, q)
    return out


def ratio_metrics(tot):
    m = {"ftl_end_mean_ms": 1000 * tot["ftl_end_sum"] / max(1, tot["ftl_end_n"]), "ftl_onset_mean_ms": 1000 * tot["ftl_on_sum"] / max(1, tot["ftl_end_n"]),
         "fcw_coverage": tot["fcw"] / tot["U"], "fcw_end_mean_ms": 1000 * tot["fcw_end_sum"] / max(1, tot["fcw_end_n"]),
         "fcw_onset_mean_ms": 1000 * tot["fcw_on_sum"] / max(1, tot["fcw_end_n"]), "revised_per_100_final": 100 * tot["R"] / max(1, tot["F"])}
    for D in DS:
        m[f"cq_stable_{D}"] = tot[f"cqs{D}"] / max(1, tot["T"]); m[f"cq_prov_{D}"] = tot[f"cqp{D}"] / max(1, tot["T"])
    for D in FDS_END:
        m[f"first_cq_end_{D}"] = tot[f"fe{D}"] / tot["U"]
    for D in FDS_ON:
        m[f"first_cq_onset_{D}"] = tot[f"fo{D}"] / tot["U"]
    return m


def bootstrap(stats_a, stats_b=None, reps=2000, seed=20261122):
    """Speaker-cluster bootstrap (strata = dataset). One system: CIs of its metrics. Two: CIs of B - A on shared utterances."""
    ids = sorted(set(stats_a) & set(stats_b)) if stats_b else sorted(stats_a)
    clusters = {}
    for i in ids:
        ds, spk = stats_a[i][0], stats_a[i][1]
        clusters.setdefault(ds, {}).setdefault(spk, []).append(i)

    def totals(sel, src):
        t = {}
        for i in sel:
            for k, v in src[i][2].items():
                t[k] = t.get(k, 0) + v
        return t

    def quants(sel, src):
        m = {}
        for k in ("ftl_end", "fcw_end", "fcw_onset"):
            v = sorted(src[i][3][k] for i in sel if k in src[i][3])
            if v:
                m[f"{k}_p50_ms"] = 1000 * v[len(v) // 2]; m[f"{k}_p95_ms"] = 1000 * v[int(len(v) * 0.95)]
        return m

    def value(sel):
        ma = ratio_metrics(totals(sel, stats_a)); ma.update(quants(sel, stats_a))
        if stats_b is None:
            return ma
        mb = ratio_metrics(totals(sel, stats_b)); mb.update(quants(sel, stats_b))
        return {k: mb[k] - ma[k] for k in ma if k in mb}

    point = value(ids); rng = random.Random(seed); draws = {k: [] for k in point}
    for _ in range(reps):
        sel = []
        for ds, spk in clusters.items():
            keys = list(spk)
            for _ in keys:
                sel.extend(spk[rng.choice(keys)])
        v = value(sel)
        for k in point:
            draws[k].append(v.get(k, point[k]))
    ci = {}
    for k in point:
        d = sorted(draws[k]); ci[k] = {"point": round(point[k], 5), "lo95": round(d[int(0.025 * len(d))], 5), "hi95": round(d[int(0.975 * len(d)) - 1], 5)}
    return ci, len(ids)


def cmd_score(a):
    paths = sorted(glob.glob(a.trace_glob))
    by = load_traces(paths)
    if a.index:
        refs = {r["id"]: r for r in json.load(open(a.index))}
    else:
        refs = {}
        dsmap = {}
        for split in ("dev-clean", "dev-other"):
            for line in open(os.path.join(a.mfa, f"{split}.jsonl")):
                dsmap[json.loads(line)["id"]] = split
        for r in json.load(open(a.clips100)):
            refs[r["id"]] = {"id": r["id"], "uid": r["uid"], "speaker": r["uid"].split("-")[0], "dataset": dsmap[r["uid"]], "ref": " ".join(r["ref_words"])}
    refs = {k: v for k, v in refs.items() if k in by or not a.only_traced}
    mfa = load_mfa(a.mfa)
    rows = per_utt_rows(by, refs, mfa)
    agg = aggregate(rows)
    ci, n = bootstrap(scalar_stats(rows), reps=a.reps)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    rep = {"schema": "ultra.h43.latency/v1", "name": a.name, "label": "exploratory", "clock": "design (audio sample clock); paced numbers are from the trace host, not quiet",
           "normaliser_id": NORMALISER_ID, "normaliser_rule_table_sha256": hashlib.sha256(rule_table().encode()).hexdigest(),
           "mfa_sha256": {s: sha(os.path.join(a.mfa, f"{s}.jsonl")) for s in ("dev-clean", "dev-other")},
           "traces": {os.path.basename(p): sha(p) for p in paths}, "aggregate": agg, "bootstrap_95": ci, "bootstrap": {"replicates": a.reps, "seed": 20261122, "clusters": "speaker", "strata": "dataset"}}
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".utts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"name": a.name, "utterances": agg["utterances"], "ftl_end": agg["ftl_end"], "ftl_onset": agg["ftl_onset"], "fcw_coverage": agg["fcw_coverage"],
                      "fcw_end": agg["fcw_end"], "cq_stable": agg["cq_stable"], "cq_provisional": agg["cq_provisional"], "first_cq_end": agg["first_cq_end"],
                      "revision": agg["revision"], "rule_sha": rep["normaliser_rule_table_sha256"][:16]}))


def cmd_compare(a):
    load = lambda p: [json.loads(l) for l in open(p)]
    ra, rb = load(a.a), load(a.b)
    ci, n = bootstrap(scalar_stats(ra), scalar_stats(rb), reps=a.reps)
    json.dump({"schema": "ultra.h43.latency-compare/v1", "a": a.a, "b": a.b, "shared_utterances": n, "b_minus_a_95": ci}, open(a.out + ".json", "w"), indent=1)
    for k in ("ftl_end_mean_ms", "fcw_end_mean_ms", "fcw_end_p50_ms", "fcw_end_p95_ms", "fcw_onset_p95_ms", "fcw_coverage", "cq_stable_300", "cq_stable_500", "first_cq_end_300", "revised_per_100_final"):
        print(f"{k}: B - A = {ci[k]['point']:+.4f} [{ci[k]['lo95']:+.4f}, {ci[k]['hi95']:+.4f}]")


def cmd_parity(a):
    tot = {}
    for line in open(a.score_input):
        r = json.loads(line); ops = align(normalise_words(r["ref"]), normalise_words(r["hyp"]))
        t = tot.setdefault(r["dataset"], {"S": 0, "D": 0, "I": 0, "N": 0})
        for op, i, j in ops:
            if op in "SDI":
                t[op] += 1
        t["N"] += len(normalise_words(r["ref"]))
    print(json.dumps({"normaliser_rule_table_sha256": hashlib.sha256(rule_table().encode()).hexdigest(), "counts": tot}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--trace-glob", required=True); s.add_argument("--index"); s.add_argument("--clips100"); s.add_argument("--mfa", required=True)
    s.add_argument("--name", required=True); s.add_argument("--out", required=True); s.add_argument("--reps", type=int, default=2000); s.add_argument("--only-traced", action="store_true")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--out", required=True); c.add_argument("--reps", type=int, default=2000)
    p = sub.add_parser("parity"); p.add_argument("score_input")
    a = ap.parse_args()
    {"score": cmd_score, "compare": cmd_compare, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    main()
