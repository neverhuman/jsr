"""What the paper says in words must hold in the data, and Figure 1 outside the PDF must be current."""

import json
import re

from tools import paper_data as pd
from tools.paper_data import publish

DATA = pd.REPO / pd.DATA
PAPER = pd.REPO / "docs" / "paper"
REFRESH = "run `just paper-refresh`"


def _records():
    return {r["id"]: r for r in map(json.loads, (DATA / "records.jsonl").read_text().splitlines())}


def test_claims_made_in_words_hold_in_the_measured_rows():
    """The section headed "Where ULTRA wins and where it loses" is generated, so its sentences stay true.
    The story around them (title, contributions) assumes three things; if one flips, rewrite the text."""
    rec = _records()
    if "ColdHeadSpeedOverFast" in rec:  # absent until another system has a row from the headline row's host
        assert rec["ColdHeadSpeedOverFast"]["raw"] > 1.0, "ULTRA is no longer the fastest system on its host"
    assert rec["ColdLightRssOverHead"]["raw"] > 1.0, "ULTRA no longer needs less memory than every other system"
    assert rec["ColdHeadWerGapBest"]["raw"] > 0.0, "ULTRA is no longer less accurate than the best other system"


def test_fallbacks_cover_every_macro_the_sections_use():
    sec = PAPER / "sections"
    used = set()
    for p in sorted(sec.glob("*.tex")):
        if p.name != "fallbacks.tex":
            used |= set(re.findall(r"\\([A-Z][A-Za-z]+)", re.sub(r"(?<!\\)%.*", "", p.read_text())))
    own = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "paper.tex").read_text()))
    own |= {"Cref", "IEEEkeywords", "IfFileExists", "Delta", "Sigma", "Omega", "Big", "Bigl", "Bigr", "LaTeX", "TeX"}
    listed = set(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", (sec / "fallbacks.tex").read_text()))
    assert used - own <= listed, f"add to sections/fallbacks.tex: {sorted(used - own - listed)}"
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (DATA / "metrics.tex").read_text()))
    assert used - own <= defined, f"used in sections but not generated (prints ??): {sorted(used - own - defined)}"


def test_readme_shows_the_current_figure_block():
    """README.md carries the generated Figure 1 block verbatim, so its numbers cannot go stale silently."""
    block = (DATA / "money_readme.md").read_text().rstrip("\n")
    readme = (pd.REPO / "README.md").read_text()
    assert readme.count(publish.START) == 1 and readme.count(publish.END) == 1
    shown = readme[readme.index(publish.START):readme.index(publish.END) + len(publish.END)]
    assert shown == block, f"README.md shows an old Figure 1 block; {REFRESH}"
    drawn = [ln for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]
             if ln.split(",")[5:7] == ["zen2", "1"] and ln.split(",")[21] == "1"]
    assert len(re.findall(r"^\| ", block, flags=re.M)) == len(drawn) + 1, "one table line per plotted system"


def test_money_png_was_exported_from_the_current_inputs():
    """docs/paper/money.png is stamped with the hash of the files Figure 1 is drawn from."""
    png = PAPER / "money.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"docs/paper/money.png is missing; {REFRESH}"
    stamp = (pd.REPO / publish.STAMP).read_text().strip()
    assert stamp == publish.figure_digest(pd.REPO), f"docs/paper/money.png is older than its data; {REFRESH}"
