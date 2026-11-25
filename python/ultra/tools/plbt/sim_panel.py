"""plbt/v1 self-check on the committed 730-utterance dev panel (no audio, no model, seconds to run).

The panel file holds, per utterance, the LibriSpeech reference and R0's transcript. The check turns
the references back into "book text" (sentence case, punctuation, quotation marks, Mr./Dr., digits for
number words, hyphenated and fused compounds), places each utterance inside a longer stretch of book
(its neighbours before and after, in one cut or in separate cuts; or alone, as a row cut on the cut's
own boundaries), and asks `project` for the label from R0's transcript. The label is then scored against the reference, next to R0's own transcript.

It measures the algorithm, not Libriheavy: the simulated book text is the reference by construction.
The gold check on real Libriheavy text is V1 in research/H61_book_text_labels/README.md.

usage (from python/ultra): python -m tools.plbt.sim_panel ../../docs/paper/data/raw/energy/cold/zipformer-int8/round-3/scores.jsonl
"""
from __future__ import annotations

import json
import random
import sys

from tools.plbt.norm import book_words, label_words
from tools.plbt.project import Cut, project

NUM = {"TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
       "TEN": "10", "TWELVE": "12", "TWENTY": "20", "THIRTY": "30", "FORTY": "40", "FIFTY": "50",
       "HUNDRED": "100", "THOUSAND": "1,000"}
ABBR = {"MISTER": "Mr.", "MISSUS": "Mrs.", "DOCTOR": "Dr.", "SAINT": "St."}
FUSE = {("ANY", "ONE"): "anyone", ("EVERY", "ONE"): "everyone", ("SOME", "ONE"): "someone", ("FOR", "EVER"): "forever",
        ("TO", "DAY"): "to-day", ("TO", "MORROW"): "to-morrow", ("TO", "NIGHT"): "to-night", ("GOOD", "BYE"): "good-bye"}
PUNCT = [",", ",", ";", ".", "!", "?", ":", " --"]


def as_book(ref: list[str], rng: random.Random) -> str:
    """A reference as a book might print it."""
    out, i, cap, quote = [], 0, True, False
    while i < len(ref):
        w = ref[i]
        pair = (w, ref[i + 1]) if i + 1 < len(ref) else None
        if pair in FUSE and rng.random() < 0.8:
            tok, i = FUSE[pair], i + 2
        elif w in ABBR:
            tok, i = ABBR[w], i + 1
        elif w in NUM and rng.random() < 0.5:
            tok, i = NUM[w], i + 1
        else:
            tok, i = (w if w == "I" or w.startswith("I'") else w.lower()), i + 1
        if cap:
            tok, cap = tok[:1].upper() + tok[1:], False
        if not quote and rng.random() < 0.04:
            tok, quote = rng.choice(["“", "‘", "'"]) + tok, True
        if rng.random() < 0.12 and i < len(ref):
            p = rng.choice(PUNCT)
            tok, cap = tok + p, p in ".!?"
            if quote and rng.random() < 0.5:
                tok, quote = tok + rng.choice(["”", "’", "'"]), False
        out.append(tok)
    return " ".join(out) + rng.choice([".", ".", "!", "?", ".”"])


def fused(words: list[str]) -> list[str]:
    """Score ANY ONE and ANYONE alike: the simulated book fuses some pairs, which is the book's spelling,
    not an error of the label."""
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words) and (words[i], words[i + 1]) in FUSE and "-" not in FUSE[(words[i], words[i + 1])]:
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return out


def edits(ref: list[str], hyp: list[str]) -> int:
    ref, hyp = fused(ref), fused(hyp)
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def run(path: str, seed: int = 61, layout: str = "one_cut") -> dict:
    rows = [json.loads(line) for line in open(path)]
    rng = random.Random(seed)
    refs = [label_words(r["ref"]) for r in rows]
    books = [as_book(ref, rng) for ref in refs]
    n = len(rows)
    out = {"rows": n, "ref_words": 0, "r0_edits": 0, "kept": 0, "kept_ref_words": 0, "kept_r0_edits": 0,
           "kept_label_edits": 0, "kept_exact": 0, "kept_r0_exact": 0, "drops": {}, "worse": 0, "better": 0,
           "repairs": 0, "filled": 0, "edge_sub": 0}
    for k, r in enumerate(rows):
        hyp, ref = label_words(r["hyp"]), refs[k]
        before, after = books[(k - 1) % n], books[(k + 1) % n]
        dur = float(r["duration_s"])
        if layout == "one_cut":  # the row lies inside one long cut: no help from timing
            cuts = [Cut(-30.0, dur + 30.0, book_words(" ".join([before, books[k], after])))]
        elif layout == "own_cut":  # the row holds one cut exactly, with neighbours on both sides
            cuts = [Cut(-30.0, -0.4, book_words(before)), Cut(0.0, dur, book_words(books[k])),
                    Cut(dur + 0.4, dur + 30.0, book_words(after))]
        else:  # "whole": the row was cut on the cut's own boundaries (route B)
            cuts = [Cut(0.0, dur, book_words(books[k]))]
        res = project(hyp, cuts, -0.3, dur + 0.4, whole=layout == "whole")
        e0 = edits(ref, hyp)
        out["ref_words"] += len(ref)
        out["r0_edits"] += e0
        if res.decision != "keep":
            out["drops"][res.decision] = out["drops"].get(res.decision, 0) + 1
            continue
        e1 = edits(ref, res.label)
        out["kept"] += 1
        out["kept_ref_words"] += len(ref)
        out["kept_r0_edits"] += e0
        out["kept_label_edits"] += e1
        out["kept_exact"] += e1 == 0
        out["kept_r0_exact"] += e0 == 0
        out["worse"] += e1 > e0
        out["better"] += e1 < e0
        out["repairs"] += res.head_repair + res.tail_repair
        out["filled"] += res.filled
        out["edge_sub"] += res.edge_sub
    out["r0_wer"] = round(100.0 * out["r0_edits"] / out["ref_words"], 3)
    out["kept_r0_wer"] = round(100.0 * out["kept_r0_edits"] / max(1, out["kept_ref_words"]), 3)
    out["kept_label_wer"] = round(100.0 * out["kept_label_edits"] / max(1, out["kept_ref_words"]), 3)
    out["layout"], out["seed"] = layout, seed
    return out


if __name__ == "__main__":
    for layout in ("one_cut", "own_cut", "whole"):
        print(json.dumps(run(sys.argv[1], layout=layout), sort_keys=True))
