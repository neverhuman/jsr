"""The arXiv stager (python/ultra/tools/paper_data/arxiv_stage.py, driven by ops/paper/build_arxiv.sh): comment stripping,
IfFileExists resolution, figure wrapping and the privacy scan. The docker verification is not run here."""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("arxiv_stage", REPO / "python/ultra/tools/paper_data/arxiv_stage.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_strip_keeps_escaped_percent_and_verb():
    src = "50\\% off % gone\n% whole line\nx \\verb|%| y % tail\n"
    out = st.strip_comments(src)
    assert out == "50\\% off %\nx \\verb|%| y %\n"


def test_strip_leaves_verbatim_alone():
    src = "\\begin{verbatim}\n% keep\n\\end{verbatim}\n% drop\n"
    assert st.strip_comments(src) == "\\begin{verbatim}\n% keep\n\\end{verbatim}\n"


def test_iffileexists_resolves_by_shipped_set():
    t = "\\IfFileExists{data/m.tex}{\\input{data/m.tex}}{}\\IfFileExists{gone.tex}{A}{\\B{x}}"
    assert st.resolve_iffileexists(t, {"data/m.tex"}) == "\\input{data/m.tex}\\B{x}"


def test_wrap_figures_only_tex_drawn_floats():
    names, n = [], [0]
    drawn = "\\begin{figure}[H]\n\\centering\n\\input{figures/a.tex}%\n\\caption{c}\n\\end{figure}"
    pic = "\\begin{figure}\n\\includegraphics{x.pdf}\n\\caption{c}\n\\end{figure}"
    out = st.wrap_figures(drawn + "\n" + pic, n, names)
    assert names == ["paperfig1"]
    assert "\\centering\n\\beginpgfgraphicnamed{paperfig1}\\input{figures/a.tex}%\n\\endpgfgraphicnamed" in out
    assert pic in out


@pytest.mark.parametrize("text", [
    "host xbabe3 ran", "ip " + ".".join(["10", "77", "0", "1"]) + " here", "see /home/ubuntu/x", "mail a@b.com", "ghp_" + "a" * 30,
    "github.com/neverhuman/jsr", "neverhumanbot",
])
def test_scan_flags_leaks(text):
    assert st.scan_text("t", text)


def test_scan_passes_clean_text_and_noreply():
    assert not st.scan_text("t", "Zen 2 core, 32.7x real time, v3.1.0, NEVERHUMAN Research, x@users.noreply.github.com")


def test_stage_only_builds_a_clean_tree():
    r = subprocess.run([str(REPO / "ops/paper/build_arxiv.sh"), "--stage-only"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    src = REPO / "target/arxiv/src"
    assert (src / "00README.json").is_file() and (src / "paper.bbl").is_file()
    assert (src / "paper.tex").read_text().startswith("\\pdfoutput=1\n")
    assert not list(src.rglob("*.bib"))

def test_plain_expands_geq_times_without_leftover():
    """arXiv upload abstract must not retain TeX for \\geq / \\times; leftover check stays strict."""
    out = st.plain(r"lowest WER at $\geq 14\times$ and $\le 1$", {})
    assert out == "lowest WER at >= 14x and <= 1"
    assert not re.findall(r"[\\{}$]", out)


def test_plain_leaves_unknown_command_as_leftover():
    """The leftover-markup gate still fails on unexpanded TeX."""
    out = st.plain(r"uses \unknowncmd here", {})
    assert re.findall(r"[\\{}$]", out)
