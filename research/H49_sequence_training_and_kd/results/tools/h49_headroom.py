"""H49 oracle-headroom report (h49-seqkd): where the candidate unions could move B0's WER.

Reads candidate lists for TRAIN (kd-cache's pilot: `<seq_dir>/*_sp1.0.seq.jsonl` unions with R0 full-sum
scores, and the student N-best files) or DEV (h49_dev_nbest.py rows, optionally R0 DEV unions), and
reports, with CANONICAL word edits (`ultra-en-wer-norm/v1`, eval-lane's parity-checked port of the Rust
normaliser and its frozen aligner tie-break; rule-table sha256 checked against the Rust pin):

  systems     student 1-best; student beam-K top and oracle (K = 4 / 8 / 16 when present); R0 beam top and
              oracle; union oracle (R0 + student beam 8); the <= 12 quota union; the teacher's pick (argmax R0
              full-sum over the union) without and with the reference. Every oracle is computed BEFORE
              reference injection (owner plan 5.1); the reference only enters the teacher-pick-with-ref row.
  headroom    pp of pooled WER against the student 1-best, and against the registered 4-run spread.
  categories  every S / D / I of a system's selection is put in one frozen category (rules below); the
              headroom of a selection is reported per category (student errors minus selection errors).
  recall      share of the student 1-best's S / D reference positions at which some candidate of a set has
              the reference word aligned (the H43 ledger's first-correct-candidate notion), per category.
  slices      clean / other, reference length bins.

Category rules (frozen in the H49b pre-registration before any pilot row was read):
  S r->h: function   both words in FUNCTION_WORDS
          convention r and h equal after dropping apostrophes / hyphens, or a split / merge with the
                     adjacent D / I (r == h + h_next, or h == r + r_next)
          rare       r seen < 10 times in the TRAIN transcripts (canonical words)
          near       min CMU phone edit distance <= 1 over listed pronunciations (homophones included),
                     or a morphological variant (same stem after stripping s / es / ed / d / ing / er / est /
                     ly / 's), or, when either word is not in the dictionary, character edit distance <= 2
          acoustic   every other substitution
  D r:    function (r in FUNCTION_WORDS) / rare (r < 10 TRAIN counts) / content
  I h:    function (h in FUNCTION_WORDS) / content

  python h49_headroom.py dev --dev-rows DEV.jsonl [--r0-seq R0_DEV_SEQ.jsonl] --out PREFIX
  python h49_headroom.py train --seq-dir DIR --student-dir DIR[,DIR...] --out PREFIX
Outputs PREFIX.json, PREFIX.md, PREFIX.utts.jsonl (per-utterance errors per system) and, per system,
PREFIX.<system>.score-input.jsonl for the pinned ultra-score cross-check (`--ultra-score PATH`).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------- canonical normaliser (port)
NORMALISER_ID = "ultra-en-wer-norm/v1"
RULE_TABLE_PIN = "988c59acad873bf0a0c50bdcf434dde0b5125cf51c2cef887bfc63f4f89113e8"
APOS = ["’", "‘", "ʼ", "`", "＇"]
WHOLE = [("can't", "can not"), ("cannot", "can not"), ("won't", "will not"), ("shan't", "shall not"),
         ("let's", "let us"), ("it's", "it is"), ("that's", "that is"), ("what's", "what is"),
         ("there's", "there is"), ("here's", "here is"), ("he's", "he is"), ("she's", "she is"),
         ("who's", "who is"), ("where's", "where is"), ("how's", "how is"), ("y'all", "you all")]
SUFFIX = [("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'m", " am"), ("'d", " would")]
WHOLE_D = dict(WHOLE)


def rule_table() -> str:
    s = NORMALISER_ID + ("\nsteps:lowercase;apostrophe-variants;non-alnum-to-space-keep-intra-word-apostrophe;"
                         "whole-word;suffix;collapse-ws\napostrophes:")
    s += "".join(f"U+{ord(c):04X}," for c in APOS) + "\nwhole:" + "".join(f"{a}={b};" for a, b in WHOLE)
    s += "\nsuffix:" + "".join(f"{a}={b};" for a, b in SUFFIX) + "\n"
    return s


if hashlib.sha256(rule_table().encode()).hexdigest() != RULE_TABLE_PIN:
    raise SystemExit("normaliser rule table differs from the Rust pin")


def _expand(tok):
    if tok in WHOLE_D:
        return WHOLE_D[tok].split(" ")
    for suf, rep in SUFFIX:
        if tok.endswith(suf) and len(tok) > len(suf):
            return [tok[: -len(suf)]] + rep.split()
    return [tok]


def normalise_words(text: str) -> list[str]:
    chars = []
    for ch in text or "":
        for c in ch.lower():  # per character, like Rust char::to_lowercase
            chars.append("'" if c in APOS else c)
    out = []
    for i, c in enumerate(chars):
        if c.isalnum():
            out.append(c)
        elif c == "'":
            out.append(c if (0 < i < len(chars) - 1 and chars[i - 1].isalnum() and chars[i + 1].isalnum()) else " ")
        else:
            out.append(" ")
    words = []
    for tok in "".join(out).split():
        words.extend(_expand(tok))
    return words


def align(ref, hyp):
    """Frozen ultra-eval alignment (tie-break: diagonal, then deletion, then insertion): [(op, i, j)]."""
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
            sub = prev[j - 1] + (r != hyp[j - 1])
            dele = prev[j] + 1
            ins = cur[j - 1] + 1
            if sub <= dele and sub <= ins:
                cur[j], dirs[i][j] = sub, 0
            elif dele <= ins:
                cur[j], dirs[i][j] = dele, 1
            else:
                cur[j], dirs[i][j] = ins, 2
        prev = cur
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        d = dirs[i][j]
        if i > 0 and j > 0 and d == 0:
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (d == 1 or j == 0):
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    return ops[::-1]


try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def n_errors(ref, hyp) -> int:
        return _Lev.distance(ref, hyp)
except ImportError:  # pragma: no cover
    def n_errors(ref, hyp) -> int:
        return sum(op != "C" for op, _, _ in align(ref, hyp))

# ---------------------------------------------------------------- frozen category rules
FUNCTION_WORDS = frozenset("""
a an the and or but nor so yet if as because though although while than that whether
in on at to of by with from into onto upon over under about above after against along among around
before behind below beneath beside besides between beyond down during except inside near off out
outside since through throughout till toward towards until up within without via per unto
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its
itself we us our ours ourselves they them their theirs themselves this these those who whom whose
which what whatever whoever one ones
am is are was were be been being have has had having do does did will would shall should can could
may might must ought not no nor
there here then when where how why all any some each every both either neither such own other
another very just only also too now oh o ah well yes nay
""".split())
CMU = os.environ.get("H49_CMUDICT", "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict")
CMU_SHA = "9de99dd2a24b63c653c1c30ab39388d05185cae36d0875f15c319b4ad6dc43af"
TRAIN_TEXT = os.environ.get("H49_TRAIN_TEXT", "/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt")
RARE_BELOW = 10
_SUFFIXES = ("'s", "ing", "est", "es", "ed", "er", "ly", "s", "d")


def load_cmu(path: str = CMU) -> dict:
    pron = collections.defaultdict(list)
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.split()
        if len(parts) < 2:
            continue
        w = parts[0].lower()
        if w.endswith(")") and "(" in w:
            w = w[: w.rindex("(")]
        pron[w].append(tuple(parts[1:]))
    return dict(pron)


def load_train_freq(path: str = TRAIN_TEXT, cache: str | None = None) -> collections.Counter:
    if cache and os.path.exists(cache):
        return collections.Counter(json.load(open(cache)))
    c = collections.Counter()
    for line in open(path, encoding="utf-8"):
        c.update(normalise_words(line))
    if cache:
        with open(cache, "w") as fh:
            json.dump(c, fh)
    return c


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _char_dist(a: str, b: str) -> int:
    return n_errors(list(a), list(b))


class Categoriser:
    def __init__(self, cmu: dict, freq: collections.Counter):
        self.cmu, self.freq = cmu, freq
        self._pair: dict = {}

    def rare(self, w: str) -> bool:
        return self.freq.get(w, 0) < RARE_BELOW

    def sub(self, r: str, h: str) -> str:
        key = (r, h)
        got = self._pair.get(key)
        if got is not None:
            return got
        if r in FUNCTION_WORDS and h in FUNCTION_WORDS:
            cat = "function"
        elif r.replace("'", "").replace("-", "") == h.replace("'", "").replace("-", ""):
            cat = "convention"
        elif self.rare(r):
            cat = "rare"
        elif _stem(r) == _stem(h):
            cat = "near"
        elif r in self.cmu and h in self.cmu:
            d = min(n_errors(list(pr), list(ph)) for pr in self.cmu[r] for ph in self.cmu[h])
            cat = "near" if d <= 1 else "acoustic"
        else:
            cat = "near" if _char_dist(r, h) <= 2 else "acoustic"
        self._pair[key] = cat
        return cat

    def errors(self, ref: list[str], hyp: list[str]) -> list[tuple[str, str, int | None]]:
        """[(op, category, ref_index)] for every S / D / I of the canonical alignment."""
        ops = align(ref, hyp)
        out = []
        for k, (op, i, j) in enumerate(ops):
            if op == "C":
                continue
            if op == "S":
                r, h = ref[i], hyp[j]
                nxt = ops[k + 1] if k + 1 < len(ops) else None
                prv = ops[k - 1] if k > 0 else None
                merge = any(o is not None and o[0] == "D" and ref[o[1]] and h in (r + ref[o[1]], ref[o[1]] + r)
                            for o in (nxt, prv))
                split = any(o is not None and o[0] == "I" and r in (h + hyp[o[2]], hyp[o[2]] + h)
                            for o in (nxt, prv))
                cat = "convention" if (merge or split) else self.sub(r, h)
                out.append(("S", cat, i))
            elif op == "D":
                w = ref[i]
                cat = "function" if w in FUNCTION_WORDS else ("rare" if self.rare(w) else "content")
                out.append(("D", cat, i))
            else:
                w = hyp[j]
                out.append(("I", "function" if w in FUNCTION_WORDS else "content", None))
        return out


def matched_ref_positions(ref, hyp) -> set:
    return {i for op, i, _ in align(ref, hyp) if op == "C"}


# ---------------------------------------------------------------- loading
def _cond(split: str) -> str:
    return "other" if "other" in split else "clean"


def load_dev(rows_path: str, r0_seq: str | None, sp=None, r0_1best: str | None = None):
    """DEV rows from h49_dev_nbest.py; ids decoded with the student tokenizer (`sp`, sentencepiece).
    `r0_1best`: an existing R0 1-best score input (eval-lane's published INT8 competitor control), added
    as one candidate per utterance (source `r0_1best`, no teacher score): the 'teacher opportunity' of
    owner plan 5.1 with R0's own lookahead, without any new R0 decode on DEV."""
    r1 = {}
    if r0_1best:
        for line in open(r0_1best):
            d = json.loads(line)
            r1[d["id"]] = d["hyp"]
    r0 = {}
    if r0_seq:
        for line in open(r0_seq):
            d = json.loads(line)
            if not d.get("header"):
                r0[d["uid"]] = d
    utts = []
    for line in open(rows_path):
        d = json.loads(line)
        dec = (lambda ids: sp.decode([i - 1 for i in ids if i != 0])) if sp is not None else None
        u = {"uid": d["uid"], "split": d["split"], "cond": _cond(d["split"]), "speaker": str(d["speaker"]),
             "ref": d["ref"], "seconds": d.get("seconds"), "one_best": d["greedy"]["text"], "cands": {}}
        for bm, hyps in d["beams"].items():
            for rank, (ids, sc) in enumerate(hyps):
                _add(u, dec(ids), f"student_b{bm}", rank, None)
        if d["uid"] in r1:
            _add(u, r1[d["uid"]], "r0_1best", 0, None)
        if d["uid"] in r0:
            for c in r0[d["uid"]]["candidates"]:
                for s in c["sources"]:
                    if s == "ref":
                        continue
                    _add(u, c["text"], "r0" if s == "r0_beam" else s, c.get("r0_beam_rank", 0) if s == "r0_beam"
                         else c.get("rank", 0), c.get("full_sum"))
                _setscore(u, c["text"], c.get("full_sum"))
            u["ref_full_sum"] = next((c["full_sum"] for c in r0[d["uid"]]["candidates"] if "ref" in c["sources"]), None)
        utts.append(u)
    return utts


def _add(u, text, source, rank, full_sum):
    key = " ".join(normalise_words(text))
    c = u["cands"].setdefault(key, {"text": text, "words": key.split(), "ranks": {}, "full_sum": None})
    if source not in c["ranks"] or rank < c["ranks"][source]:
        c["ranks"][source] = rank
    if full_sum is not None:
        c["full_sum"] = full_sum if c["full_sum"] is None else max(c["full_sum"], full_sum)


def _setscore(u, text, full_sum):
    key = " ".join(normalise_words(text))
    if key in u["cands"] and full_sum is not None:
        c = u["cands"][key]
        # canonical-string aggregation: alternative tokenizations of one string are log-summed
        c["full_sum"] = full_sum if c["full_sum"] is None else float(
            max(c["full_sum"], full_sum) + math.log1p(math.exp(-abs(c["full_sum"] - full_sum))))


def load_train(seq_dir: str, student_dirs: list[str]):
    """kd-cache pilot unions (`*_sp1.0.seq.jsonl`) plus student N-best files (`*.student.jsonl`, any beam)."""
    utts = {}
    for p in sorted(glob.glob(os.path.join(seq_dir, "*_sp1.0.seq.jsonl"))):
        split = os.path.basename(p).split("_")[0]
        for line in open(p):
            d = json.loads(line)
            if d.get("header"):
                continue
            ref = next(c for c in d["candidates"] if "ref" in c["sources"])
            u = {"uid": d["uid"], "split": split, "cond": _cond(split), "speaker": d["uid"].split("-")[0],
                 "ref": ref["text"], "seconds": round(0.04 * d["n_frames"], 2), "one_best": None, "cands": {},
                 "ref_full_sum": ref["full_sum"]}
            seen_texts = {}
            for c in d["candidates"]:
                srcs = [s for s in c["sources"] if s != "ref"]
                for s in srcs:
                    if s == "r0_beam":
                        _add(u, c["text"], "r0", c.get("r0_beam_rank", 0), None)
                    elif not student_dirs:  # the union's own extra tags only when no ranked student lists
                        _add(u, c["text"], s, c.get("rank", 99), None)
                if srcs:
                    seen_texts[c["text"]] = c["full_sum"]
            u["_scores"] = {}
            for t, fs in seen_texts.items():  # teacher score per canonical string (tokenizations log-summed)
                key = " ".join(normalise_words(t))
                u["_scores"].setdefault(key, []).append(fs)
            u["_scores"] = {k: (max(v) + math.log(sum(math.exp(x - max(v)) for x in v))) for k, v in u["_scores"].items()}
            for k, cc in u["cands"].items():
                cc.pop("_fs", None)
                if k in u["_scores"]:
                    cc["full_sum"] = u["_scores"][k]
            utts[d["uid"]] = u
    for sd in student_dirs:
        beam_tag = None
        meta = os.path.join(sd, "STUDENT_META.json")
        if os.path.exists(meta):
            beam_tag = json.load(open(meta)).get("beam")
        for p in sorted(glob.glob(os.path.join(sd, "*.student.jsonl"))):
            for line in open(p):
                d = json.loads(line)
                u = utts.get(d["uid"])
                if u is None:
                    continue
                src = f"student_b{beam_tag}" if beam_tag else d.get("source", "student")
                _add(u, d["text"], src, d["rank"], u.get("_scores", {}).get(" ".join(normalise_words(d["text"]))))
                if d["rank"] == 0 and (u["one_best"] is None or beam_tag == 8):
                    u["one_best"] = d["text"]
    return [u for u in utts.values() if u["one_best"] is not None]


# ---------------------------------------------------------------- systems
def _pick_oracle(u, keys):
    """Fewest canonical errors; ties keep the student's own higher-ranked candidate, then R0's."""
    best = None
    for k in keys:
        c = u["cands"][k]
        e = c["err"]
        tie = (min((r for s, r in c["ranks"].items() if s.startswith("student")), default=99),
               c["ranks"].get("r0", 99))
        cand = (e, tie, k)
        if best is None or cand < best:
            best = cand
    return best[2] if best else None


def systems_for(u, quota=(6, 5)):
    """{system: canonical key} for one utterance. Oracles exclude the reference."""
    out = {"student_1best": " ".join(normalise_words(u["one_best"]))}
    if out["student_1best"] not in u["cands"]:
        _add(u, u["one_best"], "student_1best", 0, None)
        u["cands"][out["student_1best"]]["err"] = n_errors(u["ref_words"], out["student_1best"].split())
    by_src = collections.defaultdict(list)
    for k, c in u["cands"].items():
        for s, r in c["ranks"].items():
            by_src[s].append((r, k))
    for s in by_src:
        by_src[s].sort()
    for s, lst in by_src.items():
        if s == "student_1best":
            continue
        out[f"{s}_top"] = lst[0][1]
        out[f"{s}_oracle"] = _pick_oracle(u, [k for _, k in lst])
    st8 = [k for _, k in by_src.get("student_b8", by_src.get("student", []))]
    r1 = [k for _, k in by_src.get("r0_1best", [])]
    if r1 and st8:
        out["union_s8_r0_1best_oracle"] = _pick_oracle(u, list(dict.fromkeys(st8 + r1)))
    r0 = [k for _, k in by_src.get("r0", [])]
    if r0 and st8:
        union = list(dict.fromkeys(r0 + st8))
        out["union_oracle"] = _pick_oracle(u, union)
        # <= 12 per utterance: ref + R0 top-6 + student top-5, refilled from the next ranks after dedup
        q = list(dict.fromkeys(r0[:quota[0]] + st8[:quota[1]]))
        extra = [k for k in union if k not in q]
        while len(q) < sum(quota) and extra:
            q.append(extra.pop(0))
        u["quota_keys"] = q
        out["union12_oracle"] = _pick_oracle(u, q)
        scored = [k for k in union if u["cands"][k]["full_sum"] is not None]
        if scored:
            out["teacher_pick_union"] = max(scored, key=lambda k: u["cands"][k]["full_sum"])
            sq = [k for k in q if u["cands"][k]["full_sum"] is not None]
            if sq:
                out["teacher_pick_union12"] = max(sq, key=lambda k: u["cands"][k]["full_sum"])
    return out


def recall_sets(u):
    sets = {}
    for s in ("student_b4", "student_b8", "student_b16", "r0", "r0_1best"):
        ks = [k for k, c in u["cands"].items() if s in c["ranks"]]
        if ks:
            sets[s] = ks
    if "r0_1best" in sets and "student_b8" in sets:
        sets["student_b8+r0_1best"] = list(dict.fromkeys(sets["student_b8"] + sets["r0_1best"]))
    if "r0" in sets and "student_b8" in sets:
        sets["union"] = list(dict.fromkeys(sets["r0"] + sets["student_b8"]))
        if u.get("quota_keys"):
            sets["union12"] = u["quota_keys"]
    return sets


def _len_bin(n: int) -> str:
    return "1-4" if n < 5 else "5-9" if n < 10 else "10-19" if n < 20 else "20+"


def analyse(utts, cat: Categoriser, spread_pp: float):
    tot = collections.defaultdict(lambda: collections.Counter())
    rec = collections.defaultdict(lambda: collections.Counter())
    per_utt = []
    n_cands = collections.Counter()
    ref_first = ref_scored = 0
    for u in utts:
        u["ref_words"] = normalise_words(u["ref"])
        for c in u["cands"].values():
            c["err"] = n_errors(u["ref_words"], c["words"])
        sysk = systems_for(u)
        N = len(u["ref_words"])
        slices = ("all", u["cond"], "len:" + _len_bin(N))
        row = {"uid": u["uid"], "split": u["split"], "N": N}
        errs_cache = {}
        for sname, key in sysk.items():
            if key is None:
                continue
            if key not in errs_cache:
                errs_cache[key] = cat.errors(u["ref_words"], key.split())
            ev = errs_cache[key]
            row[sname] = len(ev)
            for sl in slices:
                t = tot[(sname, sl)]
                t["N"] += N
                t["utts"] += 1
                for op, c, _ in ev:
                    t[op] += 1
                    t[f"{op}:{c}"] += 1
                    t[f"cat:{c}"] += 1
        # recall of the 1-best's S / D positions
        base_ev = errs_cache[sysk["student_1best"]]
        miss = [(op, c, i) for op, c, i in base_ev if op in "SD"]
        if miss:
            sets = recall_sets(u)
            matched = {k: matched_ref_positions(u["ref_words"], k.split()) for s in sets.values() for k in s}
            for sname, ks in sets.items():
                for op, c, i in miss:
                    hit = any(i in matched[k] for k in ks)
                    for sl in slices:
                        r = rec[(sname, sl)]
                        r["positions"] += 1
                        r["hit"] += hit
                        r[f"positions:{c}"] += 1
                        r[f"hit:{c}"] += hit
        for s in ("student_b4", "student_b8", "student_b16", "r0"):
            n_cands[s] += sum(1 for c in u["cands"].values() if s in c["ranks"])
        if u.get("quota_keys") is not None:
            n_cands["union12"] += len(u["quota_keys"])
        ref_key = " ".join(u["ref_words"])
        alts = sorted([k for k, c in u["cands"].items() if c["full_sum"] is not None and k != ref_key],
                      key=lambda k: -u["cands"][k]["full_sum"])[:3]
        if alts:  # the H49b training selection (top-3 by teacher score): where do the retained strings come from
            n_cands["retained_alts"] += len(alts)
            n_cands["retained_student_only"] += sum("r0" not in u["cands"][k]["ranks"] for k in alts)
            n_cands["retained_r0_only"] += sum(not any(s.startswith("student") for s in u["cands"][k]["ranks"])
                                               for k in alts)
        if u.get("ref_full_sum") is not None:
            scored = [c["full_sum"] for c in u["cands"].values() if c["full_sum"] is not None]
            ref_scored += 1
            ref_first += int(not scored or u["ref_full_sum"] >= max(scored))
        per_utt.append(row)
    secs = sorted(float(u.get("seconds") or 0.0) for u in utts)
    spk = collections.Counter(u["speaker"] for u in utts)
    q = (lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None)
    sample = {"speakers": len(spk), "utts_per_speaker_p50_max": [q(sorted(spk.values()), 0.5), max(spk.values(), default=0)],
              "seconds_p10_p50_p90": [q(secs, 0.1), q(secs, 0.5), q(secs, 0.9)], "hours": round(sum(secs) / 3600, 2),
              "by_split": dict(collections.Counter(u["split"] for u in utts))}
    rep = {"utts": len(utts), "spread_pp": spread_pp, "sample": sample, "systems": {}, "recall": {},
           "candidates_per_utt": {k: round(v / max(1, len(utts)), 2) for k, v in n_cands.items()
                                  if not k.startswith("retained")}}
    if ref_scored:
        rep["teacher_ranks_reference_first"] = round(ref_first / ref_scored, 4)
    if n_cands.get("retained_alts"):
        rep["h49b_selection"] = {"retained_alts_per_utt": round(n_cands["retained_alts"] / max(1, len(utts)), 3),
                                 "student_only_share": round(n_cands["retained_student_only"] / n_cands["retained_alts"], 4),
                                 "r0_only_share": round(n_cands["retained_r0_only"] / n_cands["retained_alts"], 4)}
    base = {sl: tot[("student_1best", sl)] for sl in {k[1] for k in tot}}
    for (sname, sl), t in sorted(tot.items()):
        e = t["S"] + t["D"] + t["I"]
        b = base.get(sl)
        be = b["S"] + b["D"] + b["I"] if b else None
        cats = sorted({k.split(":", 1)[1] for k in t if k.startswith("cat:")} |
                      ({k.split(":", 1)[1] for k in b if k.startswith("cat:")} if b else set()))
        rep["systems"].setdefault(sname, {})[sl] = {
            "N": t["N"], "utts": t["utts"], "S": t["S"], "D": t["D"], "I": t["I"], "errors": e,
            "wer": round(100.0 * e / max(1, t["N"]), 3),
            "headroom_pp": round(100.0 * (be - e) / max(1, t["N"]), 3) if b else None,
            "headroom_spreads": round(100.0 * (be - e) / max(1, t["N"]) / spread_pp, 2) if b else None,
            "by_category": {c: {"errors": t[f"cat:{c}"], "student_1best": b[f"cat:{c}"] if b else None,
                                "fixed_net": (b[f"cat:{c}"] - t[f"cat:{c}"]) if b else None,
                                "headroom_pp": round(100.0 * (b[f"cat:{c}"] - t[f"cat:{c}"]) / max(1, t["N"]), 3)
                                if b else None} for c in cats},
            "by_op_category": {k: v for k, v in t.items() if ":" in k and not k.startswith("cat:")}}
    for (sname, sl), r in sorted(rec.items()):
        cats = sorted({k.split(":", 1)[1] for k in r if k.startswith("positions:")})
        rep["recall"].setdefault(sname, {})[sl] = {
            "positions": r["positions"], "recall": round(r["hit"] / max(1, r["positions"]), 4),
            "by_category": {c: {"positions": r[f"positions:{c}"],
                                "recall": round(r[f"hit:{c}"] / max(1, r[f"positions:{c}"]), 4)} for c in cats}}
    return rep, per_utt


def stop_rule(rep: dict) -> dict | None:
    """H49b stop rule (pre-registered): R0's added oracle headroom H_T = WER(student beam-8 oracle) -
    WER(union oracle), both before reference injection, must reach one spread; the teacher's pick over the
    union must not be worse than the student 1-best; writing conventions must be < 50% of H_T; the
    teacher's pick must not raise insertions or rare-word errors over the student 1-best."""
    S = rep["systems"]
    if "union_oracle" not in S or "student_b8_oracle" not in S or "student_1best" not in S:
        return None
    a, u, b = S["student_b8_oracle"]["all"], S["union_oracle"]["all"], S["student_1best"]["all"]
    N = max(1, b["N"])
    gain = a["errors"] - u["errors"]
    h_t = 100.0 * gain / N
    conv = (a["by_category"].get("convention", {}).get("errors", 0)
            - u["by_category"].get("convention", {}).get("errors", 0))
    tp = S.get("teacher_pick_union", {}).get("all")
    crit = {"H_T_pp": round(h_t, 3), "H_T_spreads": round(h_t / rep["spread_pp"], 2),
            "H_T_ge_one_spread": h_t >= rep["spread_pp"],
            "convention_share_of_H_T": round(conv / gain, 4) if gain > 0 else None,
            "convention_share_below_half": (conv / gain < 0.5) if gain > 0 else False}
    if tp is not None:
        crit.update({"teacher_pick_vs_1best_pp": round(100.0 * (tp["errors"] - b["errors"]) / N, 3),
                     "teacher_pick_not_worse": tp["errors"] <= b["errors"],
                     "teacher_pick_insertions_not_up": tp["I"] <= b["I"],
                     "teacher_pick_rare_not_up": tp["by_category"].get("rare", {}).get("errors", 0)
                     <= b["by_category"].get("rare", {}).get("errors", 0)})
    keys = [k for k in ("H_T_ge_one_spread", "convention_share_below_half", "teacher_pick_not_worse",
                        "teacher_pick_insertions_not_up", "teacher_pick_rare_not_up") if k in crit]
    crit["verdict"] = "PROCEED" if all(crit[k] for k in keys) else "STOP"
    return crit


def write_score_inputs(utts, prefix, names):
    paths = {}
    for sname in names:
        p = f"{prefix}.{sname}.score-input.jsonl"
        n = 0
        with open(p, "w") as fh:
            for u in utts:
                key = systems_for(u).get(sname)
                if key is None:
                    continue
                fh.write(json.dumps({"id": u["uid"], "speaker": u["speaker"], "dataset": u["split"],
                                     "ref": u["ref"], "hyp": u["cands"][key]["text"]}) + "\n")
                n += 1
        paths[sname] = (p, n)
    return paths


def markdown(rep, title):
    L = [f"# {title}", "", f"Utterances: {rep['utts']}. Canonical words ({NORMALISER_ID}, rule table {RULE_TABLE_PIN[:8]}...).",
         f"Sample: {json.dumps(rep.get('sample', {}))}",
         f"Spread unit: {rep['spread_pp']} pp. Oracles exclude the reference.", ""]
    L.append("| system | WER all | clean | other | headroom pp (all) | spreads |")
    L.append("|---|---:|---:|---:|---:|---:|")
    order = sorted(rep["systems"], key=lambda s: (s != "student_1best", s))
    for s in order:
        v = rep["systems"][s]
        a, c, o = v.get("all", {}), v.get("clean", {}), v.get("other", {})
        L.append(f"| {s} | {a.get('wer', '')} | {c.get('wer', '')} | {o.get('wer', '')} | "
                 f"{a.get('headroom_pp', '')} | {a.get('headroom_spreads', '')} |")
    L += ["", "Headroom by category (pp of pooled WER, student 1-best errors minus the selection's):", ""]
    cats = sorted({c for s in rep["systems"].values() for c in s.get("all", {}).get("by_category", {})})
    L.append("| system | " + " | ".join(cats) + " |")
    L.append("|---|" + "---:|" * len(cats))
    for s in order:
        bc = rep["systems"][s].get("all", {}).get("by_category", {})
        L.append(f"| {s} | " + " | ".join(str(bc.get(c, {}).get("headroom_pp", "")) for c in cats) + " |")
    if rep["recall"]:
        L += ["", "Recall of the student 1-best's S / D positions (some candidate has the reference word there):", ""]
        rcats = sorted({c for s in rep["recall"].values() for c in s.get("all", {}).get("by_category", {})})
        L.append("| set | all | " + " | ".join(rcats) + " |")
        L.append("|---|---:|" + "---:|" * len(rcats))
        for s, v in sorted(rep["recall"].items()):
            a = v.get("all", {})
            L.append(f"| {s} | {a.get('recall', '')} | " +
                     " | ".join(str(a.get("by_category", {}).get(c, {}).get("recall", "")) for c in rcats) + " |")
    if rep.get("h49b_selection"):
        L += ["", "H49b training selection (top-3 non-reference strings by R0 full-sum): " + json.dumps(rep["h49b_selection"])]
    if rep.get("teacher_ranks_reference_first") is not None:
        L += ["", f"Teacher ranks the reference first: {rep['teacher_ranks_reference_first']}"]
    if rep.get("stop_rule_h49b"):
        L += ["", "H49b stop rule (pre-registered): " + json.dumps(rep["stop_rule_h49b"])]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["dev", "train"])
    ap.add_argument("--dev-rows")
    ap.add_argument("--r0-seq")
    ap.add_argument("--r0-1best", help="DEV: an existing R0 1-best score input (published competitor control)")
    ap.add_argument("--seq-dir")
    ap.add_argument("--student-dir", default="")
    ap.add_argument("--tokenizer", default="/dev/shm/eval-lane/jsr_data/tokenizers/bpe.model")
    ap.add_argument("--freq-cache", default="/dev/shm/h49-seqkd/train_word_freq.json")
    ap.add_argument("--spread-pp", type=float, default=0.087, help="registered 4-run spread at chunk 4")
    ap.add_argument("--ultra-score", default="/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cmu_sha = hashlib.sha256(open(CMU, "rb").read()).hexdigest()
    cat = Categoriser(load_cmu(), load_train_freq(cache=a.freq_cache))
    if a.mode == "dev":
        import sentencepiece as spm

        utts = load_dev(a.dev_rows, a.r0_seq, spm.SentencePieceProcessor(model_file=a.tokenizer), a.r0_1best)
        title = "H49 DEV oracle headroom (analysis only)"
    else:
        utts = load_train(a.seq_dir, [d for d in a.student_dir.split(",") if d])
        title = "H49 TRAIN pilot oracle headroom"
    rep, per_utt = analyse(utts, cat, a.spread_pp)
    rep["stop_rule_h49b"] = stop_rule(rep)
    rep.update({"title": title, "normaliser": NORMALISER_ID, "rule_table_sha256": RULE_TABLE_PIN,
                "cmudict": CMU, "cmudict_sha256": cmu_sha, "cmudict_expected": cmu_sha == CMU_SHA,
                "rare_below_train_count": RARE_BELOW, "function_words_sha256":
                hashlib.sha256(" ".join(sorted(FUNCTION_WORDS)).encode()).hexdigest(),
                "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                "inputs": {k: v for k, v in vars(a).items() if k in ("dev_rows", "r0_seq", "r0_1best", "seq_dir", "student_dir")}})
    # cross-check every system's totals with the pinned Rust scorer
    if a.ultra_score and os.path.exists(a.ultra_score):
        rep["ultra_score_sha256"] = hashlib.sha256(open(a.ultra_score, "rb").read()).hexdigest()
        names = sorted(rep["systems"])
        for sname, (p, n) in write_score_inputs(utts, a.out, names).items():
            if n == 0:
                continue
            try:
                out = subprocess.run([a.ultra_score, p, "--replicates", "1000"], capture_output=True, check=True,
                                     text=True, timeout=900).stdout
                sc = json.loads(out)
                ds = sc["system_a"]["datasets"]
                S = sum(v["counts"]["substitutions"] for v in ds.values())
                D = sum(v["counts"]["deletions"] for v in ds.values())
                I = sum(v["counts"]["insertions"] for v in ds.values())
                mine = rep["systems"][sname]["all"]
                rep["systems"][sname]["all"]["ultra_score"] = {"S": S, "D": D, "I": I,
                                                               "agrees": (S, D, I) == (mine["S"], mine["D"], mine["I"])}
            except Exception as e:  # noqa: BLE001
                rep["systems"][sname]["all"]["ultra_score"] = {"error": str(e)[:300]}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(a.out + ".md", "w") as fh:
        fh.write(markdown(rep, title))
    with open(a.out + ".utts.jsonl", "w") as fh:
        for r in per_utt:
            fh.write(json.dumps(r) + "\n")
    print(markdown(rep, title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
