"""plbt/v1: put book text on a Libri-Light row (H61).

A row is a span (t0, t1) of one recording with a recogniser's transcript `hyp` (R0's v1.1 label for
the rows cut in H52; Libriheavy's own transcript `texts[1]` for rows cut on Libriheavy boundaries).
`cuts` are the Libriheavy cuts of that recording near the span, each with its book text. `project`
aligns `hyp` inside the concatenated book tokens (the book may extend freely before and after) and
writes the label from the alignment:

    book word equals the heard word (apostrophes aside) -> the book word; an edge apostrophe is kept
                                                           only when both have it ('TIS)
    book word differs from the heard word               -> the book word   (this repairs R0's errors)
    book word with no heard word, inside the span        -> the book word   (R0 dropped it)
    heard word with no book word, inside the span        -> nothing         (R0 inserted it)
    ANYONE against ANY ONE, TO DAY against TODAY         -> the book's form, no edit counted
    abbreviation (Mr., St.)                              -> its spoken form; the heard one if several
    number (1867, 3rd, $5, XIV)                          -> the 1..cap heard number words aligned to it

and repairs a truncated edge from timing: when the alignment stops one or two words short of the end
of a cut that ends inside the row's audio, those words were spoken and are appended (same at the head).

A row is dropped, with the reason counted, when the evidence is weak: no cut, more than MAX_MISMATCH
edits per aligned word, a run of more than MAX_RUN dropped or inserted words (the reader left the
text), a number or ambiguous abbreviation nothing suitable was heard for, or a number with an
inserted word beside it (`nonlexical`), or an unsettled edge (`edge`). The edge needs care: the book continues on both sides of the row, so the
first heard word can always be "substituted" by the book word before the true start. The text alone
cannot tell R0 mishearing the first word (AN for AND: take the book's AND) from R0 inventing one (AND
before THE DAY: taking the book's previous word NIGHT would add a word that is not in the audio). So
an edge substitution is accepted only when the two words are similar in spelling (EDGE_SIM); a
dissimilar edge substitution or an edge insertion drops the row to the arbitration queue.

`whole=True` is for a row that is exactly one cut (rows cut on Libriheavy's own boundaries): the whole
book text of the cut is the label, so the alignment is global, there is no edge to settle, and the
transcript only fills the numbers and picks the spoken form of ambiguous abbreviations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tools.plbt.norm import ABBR, ABBREV, LEX, NUM, NUMBER_KEYS, Word, key

MAX_MISMATCH = 0.30
MAX_RUN = 2
REPAIR = 2  # words an edge repair may add
SLACK = 0.05  # seconds
EDGE_SIM = 0.5  # least spelling similarity for accepting a substitution of the first / last heard word
BIG = 1 << 20  # one edit; matched words break ties (score = edits * BIG - matches)
PARAMS = {"MAX_MISMATCH": MAX_MISMATCH, "MAX_RUN": MAX_RUN, "REPAIR": REPAIR, "SLACK": SLACK,
          "EDGE_SIM": EDGE_SIM}


@dataclass
class Cut:
    start: float
    end: float
    words: list[Word]


@dataclass
class Result:
    decision: str  # "keep" or a drop reason
    label: list[str] = field(default_factory=list)
    edits: int = 0  # substitutions + dropped + inserted words inside the span
    sub: int = 0
    dele: int = 0
    ins: int = 0
    span: int = 0  # book tokens the alignment covers
    filled: int = 0  # heard words used for numbers
    head_repair: int = 0
    tail_repair: int = 0
    edge_sub: int = 0  # accepted substitutions of the first / last heard word

    def audit(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "label"}
        d["words"] = len(self.label)
        return d


def similarity(a: str, b: str) -> float:
    """1 - Levenshtein(a, b) / max(len): 1.0 for equal words, 0.0 for nothing in common."""
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def _align(hyp: list[str], book: list[Word], whole: bool = False) -> list[tuple]:
    """Semi-global alignment: all of `hyp`, any contiguous part of `book` (all of it when `whole`).
    Ops in reading order: (kind, i0, i1, j0, j1) = book[i0:i1] against hyp[j0:j1]."""
    n, m = len(book), len(hyp)
    hk = [key(w) for w in hyp]
    is_num = [k in NUMBER_KEYS for k in hk]
    INF = 1 << 60
    d = [[INF] * (m + 1) for _ in range(n + 1)]
    bp: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i * BIG if whole else 0  # semi-global: the book may start anywhere
        if whole and i:
            bp[i][0] = ("del", i - 1, i, 0, 0)
    for j in range(1, m + 1):
        d[0][j], bp[0][j] = j * BIG, ("ins", 0, 0, j - 1, j)
    for i in range(1, n + 1):
        w = book[i - 1]
        row, up, bpr = d[i], d[i - 1], bp[i]
        forms = ABBREV[w.core] if w.kind == ABBR else ()
        for j in range(1, m + 1):
            best, how = row[j - 1] + BIG, ("ins", i, i, j - 1, j)
            if w.kind == LEX:
                same = w.key == hk[j - 1]
                c = up[j - 1] + (-1 if same else BIG)
                if c <= best:
                    best, how = c, ("match" if same else "sub", i - 1, i, j - 1, j)
                if j >= 2 and hk[j - 2] + hk[j - 1] == w.key and up[j - 2] - 1 <= best:
                    best, how = up[j - 2] - 1, ("join_h", i - 1, i, j - 2, j)
                if i >= 2 and book[i - 2].kind == LEX and book[i - 2].key + w.key == hk[j - 1] \
                        and d[i - 2][j - 1] - 1 <= best:
                    best, how = d[i - 2][j - 1] - 1, ("join_b", i - 2, i, j - 1, j)
            elif w.kind == ABBR:
                if up[j - 1] + BIG <= best:
                    best, how = up[j - 1] + BIG, ("sub", i - 1, i, j - 1, j)
                for f in forms:
                    k = len(f)
                    if k <= j and tuple(hk[j - k:j]) == f and up[j - k] - k <= best:
                        best, how = up[j - k] - k, ("heard", i - 1, i, j - k, j)
            elif w.kind == NUM:
                for k in range(1, min(w.cap, j) + 1):
                    if not is_num[j - k]:
                        break
                    if up[j - k] < best:
                        best, how = up[j - k], ("heard", i - 1, i, j - k, j)
            if up[j] + BIG < best:
                best, how = up[j] + BIG, ("del", i - 1, i, j, j)
            row[j], bpr[j] = best, how
    # semi-global: the book may end anywhere; on a tie end later (a substitution, not an insertion)
    end = n if whole else min(range(n + 1), key=lambda i: (d[i][m], -i))
    ops, i, j = [], end, m
    while j > 0 or (whole and i > 0):
        op = bp[i][j]
        ops.append(op)
        i, j = op[1], op[3]
    ops.reverse()
    return ops


def _run(ops: list[tuple], kind: str) -> int:
    best = cur = 0
    for op in ops:
        cur = cur + 1 if op[0] == kind else 0
        best = max(best, cur)
    return best


def _spoken(w: Word) -> list[str] | None:
    """The label words of a book token nothing (or something else) was heard for; None = cannot say."""
    if w.kind == LEX:
        return [w.core]
    if w.kind == ABBR and len(ABBREV[w.core]) == 1:
        return list(ABBREV[w.core][0])
    return None


def project(hyp: list[str], cuts: list[Cut], t0: float, t1: float, whole: bool = False) -> Result:
    cuts = sorted((c for c in cuts if c.words), key=lambda c: c.start)
    if not cuts:
        return Result("no_cut")
    if not hyp:
        return Result("empty_hyp")
    book: list[Word] = []
    owner: list[int] = []
    for ci, c in enumerate(cuts):
        book.extend(c.words)
        owner.extend([ci] * len(c.words))
    ops = _align(hyp, book, whole)
    r = Result("keep")
    label: list[str] = []
    for kind, i0, i1, j0, j1 in ops:
        if kind == "match":
            w, h = book[i0], hyp[j0]
            label.append(("'" if w.lead and h.startswith("'") else "") + w.core
                         + ("'" if w.trail and h.endswith("'") else ""))
        elif kind == "heard":
            label.extend(hyp[j0:j1])
            r.filled += (j1 - j0) if book[i0].kind == NUM else 0
        elif kind == "ins":
            r.ins += 1
        else:  # sub, del, join_h, join_b: the book's words
            r.sub += kind == "sub"
            r.dele += kind == "del"
            for w in book[i0:i1]:
                words = _spoken(w)
                if words is None:
                    r.decision = "nonlexical"
                else:
                    label.extend(words)
    for a, b in zip(ops, ops[1:], strict=False):  # an inserted word next to a number: R0 garbled the number
        if (a[0] == "ins" and b[0] == "heard" and book[b[1]].kind == NUM) or \
                (b[0] == "ins" and a[0] == "heard" and book[a[1]].kind == NUM):
            r.decision = "nonlexical"
    used = [op for op in ops if op[2] > op[1]]
    if not used:
        return Result("mismatch", ins=r.ins, edits=r.ins)
    b0, b1 = used[0][1], used[-1][2]
    r.span, r.edits = b1 - b0, r.sub + r.dele + r.ins
    unsettled = False
    for op in () if whole else ((ops[0], ops[-1]) if len(ops) > 1 else (ops[0],)):
        if op[0] == "ins":
            unsettled = True
        elif op[0] == "sub":
            if similarity(book[op[1]].key, key(hyp[op[3]])) >= EDGE_SIM:
                r.edge_sub += 1
            else:
                unsettled = True
    # edge repair from timing: the cut ends (starts) inside the row's audio, so its last (first) words were spoken
    tail = [w for w, o in zip(book[b1:b1 + REPAIR + 1], owner[b1:b1 + REPAIR + 1], strict=True) if o == owner[b1 - 1]]
    if not whole and 1 <= len(tail) <= REPAIR and cuts[owner[b1 - 1]].end <= t1 + SLACK and ops[-1][0] != "ins" \
            and all(_spoken(w) for w in tail):
        label.extend(x for w in tail for x in _spoken(w))
        r.tail_repair = len(tail)
    lo = max(0, b0 - REPAIR - 1)
    head = [w for w, o in zip(book[lo:b0], owner[lo:b0], strict=True) if o == owner[b0]]
    if not whole and 1 <= len(head) <= REPAIR and cuts[owner[b0]].start >= t0 - SLACK and ops[0][0] != "ins" \
            and all(_spoken(w) for w in head):
        label[:0] = [x for w in head for x in _spoken(w)]
        r.head_repair = len(head)
    r.label = label
    if r.decision != "keep":
        return r
    if r.edits > MAX_MISMATCH * max(r.span, len(hyp)):
        r.decision = "mismatch"
    elif _run(ops, "del") > MAX_RUN:
        r.decision = "del_run"
    elif _run(ops, "ins") > MAX_RUN:
        r.decision = "ins_run"
    elif unsettled:
        r.decision = "edge"
    elif not label:
        r.decision = "empty_label"
    return r
