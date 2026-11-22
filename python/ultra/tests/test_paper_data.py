import json
import re
import shutil

import pytest

from tools import paper_data as pd

DATA = pd.REPO / pd.DATA


def test_regeneration_is_byte_identical(tmp_path):
    files, _ = pd.build(pd.REPO)
    assert files, "generator produced nothing"
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8", newline="\n")
    for name in files:
        committed = DATA / name
        assert committed.is_file(), f"{name} is not committed; run python -m tools.paper_data"
        assert committed.read_bytes() == (tmp_path / name).read_bytes(), (
            f"{name} is stale; run python -m tools.paper_data")


def test_generation_is_deterministic():
    a, _ = pd.build(pd.REPO)
    b, _ = pd.build(pd.REPO)
    assert a == b


def test_no_required_value_is_missing():
    _, missing = pd.build(pd.REPO)
    assert missing == [], f"records missing required fields: {missing}"


def test_red_marks_are_exactly_the_declared_pending_records():
    cur = json.loads((DATA / "curated.json").read_text())
    declared = sorted(k for k, e in cur["metrics"].items() if e.get("pending"))
    red = sorted(re.findall(r"\\PaperMissing\{([A-Za-z]+)\}", (DATA / "metrics.tex").read_text()))
    assert red == declared


def test_macro_names_are_letters_only_and_unique():
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", (DATA / "metrics.tex").read_text())
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names), [n for n in names if not n.isalpha()]


def test_every_record_has_label_and_provenance():
    for line in (DATA / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["label"] in pd.LABELS, r["id"]
        for k in pd.REQUIRED:
            if k == "value" and r.get("pending"):
                continue
            assert r[k] not in (None, ""), (r["id"], k)


def test_curated_quotes_still_appear_in_their_sources():
    cur = json.loads((DATA / "curated.json").read_text())
    checked = 0
    for rid, e in cur["metrics"].items():
        src = pd.REPO / e["source"]
        if not src.is_file():  # e.g. the gitignored AGENT_CHAT.md outside this checkout
            continue
        assert e["quote"] in src.read_text(encoding="utf-8"), f"{rid}: quote not found in {e['source']}"
        checked += 1
    assert checked > 50


def test_pairing_rule_rehearsal_rows_use_rehearsal_checkpoints():
    rows = (DATA / "rehearsal_cost.csv").read_text().splitlines()[1:]
    keys = [r.split(",")[0] for r in rows]
    assert keys == ["rzero", "dense", "mr"]
    text = (DATA / "rehearsal_cost.csv").read_text()
    for forbidden in ("Nemotron", "ORT", "60k", "beam", "0.04074", "0.17017"):
        assert forbidden not in text
    recs = {json.loads(x)["id"]: json.loads(x) for x in (DATA / "records.jsonl").read_text().splitlines()}
    for stem, cand in (("Rzero", "zipformer-int8"), ("DenseFortyFive", "d2-45000"), ("MR", "d2mr-50000")):
        assert cand in recs[f"WerRehearsal{stem}"]["source"]
        assert "frontier_panel.json" in recs[f"WerRehearsal{stem}"]["source"]
    # the ladder's dense row is the avg-3 (45/50/55k) finalist, so its WER must be that checkpoint's
    # Job E's dense row is the older avg-3 (45/50/55k): its WER must be that checkpoint's (F1, same harness)
    assert "d2-avg3-45-50-55" in pd.CANDIDATES[1][1]
    assert "dense_avg3_dev-clean.json" in recs["WerRehearsalDenseOld"]["source"]
    # the headline dense row is avg 55/60/65k (Job G): density and WER must both be that checkpoint's
    assert "d2-avg3-55-60-65" in recs["DenDenseCCX"]["source"] and "job_g" in recs["DenDenseCCX"]["source"]
    assert "dense_avg3new_clean-greedy.json" in recs["WerRehearsalDense"]["source"]
    assert "dense_avg3new_clean-beam4.json" in recs["WerRehearsalDenseBeam"]["source"]
    assert "Job G's own R0" in recs["DenRatioDense"]["configuration"]
    assert "d2-45000" in recs["FrontierDense"]["source"]


def test_manifest_refuses_tampered_raw(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(pd.REPO / pd.RAW, root / pd.RAW)
    pd.verify_manifest(root)
    p = root / pd.RAW / "frontier.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(pd.PaperDataError, match="sha256 mismatch"):
        pd.verify_manifest(root)


# Evidence rows (schema v2), their grouping and the Figure 1 layout: tests/test_paper_evidence.py.


# Narrow check: result numbers in the manuscript must come from metrics.tex macros. Flags a decimal
# number (3.14) or a percentage (12% / 12\%) in prose. Ignored: comments, \cite/\ref/\label/\url/
# \texttt/\input/\include arguments, section signs (\S 4.4), versions (v1.0), lengths (0.5\linewidth, 3.5in),
# the 95\% confidence level, and lines that carry a \cite (numbers quoted from the cited work).
_STRIP = re.compile(r"\\(?:cite[a-z]*|ref|eqref|label|url|href|texttt|input|include|includegraphics|"
                    r"bibliography|bibliographystyle|usepackage|documentclass)\*?(?:\[[^\]]*\])?\{[^}]*\}")
_LEN = re.compile(r"\d+(?:\.\d+)?\s*(?:\\(?:linewidth|columnwidth|textwidth|baselineskip)|in\b|pt\b|cm\b|mm\b|em\b|ex\b)")
_SEC = re.compile(r"(?:\\S|§|Sec\.|Section|Fig\.|Table|Eq\.|v)~?\s*\d+(?:\.\d+)*")
_BAD = re.compile(r"(?<![\w\\{])\d+\.\d+|(?<![\w\\{])\d+(?:\.\d+)?\s*\\?%")


def bare_numbers(tex: str) -> list[str]:
    hits = []
    for line in tex.splitlines():
        line = re.sub(r"(?<!\\)%.*", "", line)  # drop comments (unescaped %)
        if "\\cite" in line:
            continue
        line = re.sub(r"95\s*\\?%", "", line)
        line = _STRIP.sub("", line)
        line = _LEN.sub("", line)
        line = _SEC.sub("", line)
        hits += [m.group(0) for m in _BAD.finditer(line)]
    return hits


def test_bare_number_checker_is_narrow():
    assert bare_numbers("WER fell to 7.65\\% on dev-clean") == ["7.65"]
    assert bare_numbers("In 2026 we used \\S 4.4 and v1.0, one-sided 95\\% % 3.3 comment") == []
    assert bare_numbers("Moonshine & 123.4 & 2.49 \\cite{m}") == []
    assert bare_numbers("width=0.48\\columnwidth, 3.5in, Section 4.2, \\DtwoBestClean\\,\\%") == []
    assert bare_numbers("about 12\\% less") == ["12\\%"]


def test_sections_contain_no_bare_result_numbers():
    sec = pd.REPO / "docs" / "paper" / "sections"
    bad = {}
    for p in sorted(sec.glob("*.tex")) if sec.is_dir() else []:
        hits = bare_numbers(p.read_text(encoding="utf-8"))
        if hits:
            bad[p.name] = hits
    assert not bad, f"use metrics.tex macros instead of bare numbers: {bad}"


FIGURES = ("money", "frontier_warm", "density", "rehearsal", "curve", "sweep", "leftctx")
_HARNESS = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[table]{xcolor}\usepackage{tikz}\usepackage{pgfplots}\pgfplotsset{compat=1.17}
\usetikzlibrary{calc,arrows.meta}\usepgfplotslibrary{groupplots}
\input{data/metrics.tex}
\newsavebox\fbx
\newcommand{\measure}[1]{\sbox\fbx{\input{figures/#1.tex}}\typeout{FIGWIDTH #1 \the\wd\fbx}}
\begin{document}
FIGURES
x
\end{document}
"""


def test_figures_compile_standalone_within_column(tmp_path):
    import subprocess

    assert shutil.which("pdflatex"), "pdflatex is required for the paper figures (texlive-latex-extra, texlive-pictures)"
    paper = pd.REPO / "docs" / "paper"
    body = " ".join(f"\\measure{{{f}}}" for f in FIGURES)
    tex = tmp_path / "figs.tex"
    tex.write_text(_HARNESS.replace("FIGURES", body))
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmp_path),
                           str(tex)], cwd=paper, capture_output=True, text=True, timeout=180)
    log = (tmp_path / "figs.log").read_text(errors="replace")
    assert proc.returncode == 0, log[-3000:]
    widths = dict(re.findall(r"FIGWIDTH (\w+) ([\d.]+)pt", log))
    assert set(widths) == set(FIGURES)
    for name, w in widths.items():
        assert float(w) <= 252.0, f"{name} is {w}pt wide; the IEEE column is 252pt"
    for fig1 in ("money", "frontier_warm"):
        assert float(widths[fig1]) <= 243.0, f"Figure 1 candidate {fig1} must fit 243pt"
