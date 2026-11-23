"""Text helpers shared by the generated sentences: one Unicode source string, printed as TeX or as Markdown.

Write sentences with x (U+00D7), en dash, minus (U+2212), thin space (U+2009) and no-break space; tex() turns
them into TeX (and "ULTRA" into the paper's macro), md() into README Markdown.
"""

from __future__ import annotations

import re

from .common import fmt

_TEX = (("\u00d7", "$\\times$"), ("%", "\\%"), ("\u2013", "--"), ("\u2212", "$-$"), ("\u2009", "\\,"),
        ("\u00a0", "~"), ("ULTRA", "\\ultra{}"))


def tex(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    for a, b in _TEX:
        s = s.replace(a, b)
    return s


def md(s: str) -> str:
    return s.replace("\u2009", " ").replace("\u00a0", " ").replace("\\,", " ")


def signed(x: float, nd: int) -> str:
    return ("+" if x >= 0 else "\u2212") + fmt(abs(x), nd)


def names(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def name(r: dict) -> str:
    """A system's name in prose (systems.json `text`, else its plot label), with the paper macro spelled out."""
    return (r.get("text") or r["short"]).replace("\\ultra{}", "ULTRA")


def checkpoint(r: dict) -> str:
    """The checkpoint's own name: systems.json `checkpoint` (e.g. "A1 step 30,000"), else the start of `detail`."""
    return r.get("checkpoint") or (r.get("detail") or r["short"]).split(",")[0]
