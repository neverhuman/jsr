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
        if not src.is_file():
            continue
        assert e["quote"] in src.read_text(encoding="utf-8"), f"{rid}: quote not found in {e['source']}"
        checked += 1
    assert checked >= 1


def test_signed_curated_values_keep_their_sign_and_match_their_quote():
    """A difference is printed with its sign from the raw number, and the quoted source line shows that signed number."""
    cur = json.loads((DATA / "curated.json").read_text())
    recs = {r["id"]: r for r in map(json.loads, (DATA / "records.jsonl").read_text().splitlines())}
    signed = {k: e for k, e in cur["metrics"].items() if e.get("signed")}
    assert len(signed) >= 10 and all("text" not in e for e in signed.values())
    for rid, e in signed.items():
        dec, raw = int(e["decimals"]), float(e["raw"])
        from decimal import ROUND_HALF_UP, Decimal
        digits = Decimal(repr(abs(raw))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
        assert recs[rid]["value"] == ("+" if raw >= 0 else "$-$") + str(digits), rid
        quoted = [float(x) for x in re.findall(r"[+\-\u2212]?\d+\.\d+", e["quote"].replace("\u2212", "-"))]
        assert any(abs(q - raw) < 1e-9 for q in quoted), f"{rid}: {raw} is not a signed number of its quote {e['quote']!r}"


def test_v4_findings_receipts_are_the_exact_chat_posts():
    """Each v4 finding cites a copied chat post under raw/v4_findings: one `## <ts> | <agent> |` post per file."""
    cur = json.loads((DATA / "curated.json").read_text())
    cited = {e["source"] for k, e in cur["metrics"].items() if k.startswith("Vf")}
    assert cited and all(c.startswith("docs/paper/data/raw/v4_findings/") for c in cited)
    for c in cited:
        text = (pd.REPO / c).read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0].startswith("## 20") and sum(l.startswith("## ") for l in lines) == 1, c
    for k, e in cur["metrics"].items():
        if k.startswith("Vf"):
            assert e["quote"] in (pd.REPO / e["source"]).read_text(encoding="utf-8").splitlines()[e["line"] - 1], k


def test_manifest_refuses_tampered_raw(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(pd.REPO / pd.RAW, root / pd.RAW)
    pd.verify_manifest(root)
    p = root / pd.RAW / "release_v3" / "14p_wer.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(pd.PaperDataError, match="sha256 mismatch"):
        pd.verify_manifest(root)


# Narrow check: result numbers in the manuscript must come from metrics.tex macros.
_STRIP = re.compile(r"\\(?:cite[a-z]*|ref|eqref|label|url|href|texttt|input|include|includegraphics|"
                    r"bibliography|bibliographystyle|usepackage|documentclass)\*?(?:\[[^\]]*\])?\{[^}]*\}")
_LEN = re.compile(r"\d+(?:\.\d+)?\s*(?:\\(?:linewidth|columnwidth|textwidth|baselineskip)|in\b|pt\b|cm\b|mm\b|em\b|ex\b)")
_SEC = re.compile(r"(?:\\S|§|Sec\.|Section|Fig\.|Table|Eq\.|v)~?\s*\d+(?:\.\d+)*")
_BAD = re.compile(r"(?<![\w\\{])\d+\.\d+|(?<![\w\\{])\d+(?:\.\d+)?\s*\\?%")


def bare_numbers(tex: str) -> list[str]:
    hits = []
    for line in tex.splitlines():
        line = re.sub(r"(?<!\\)%.*", "", line)
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
    assert bare_numbers("Nemotron & 600.0 & 2.61 \\cite{m}") == []
    assert bare_numbers("width=0.48\\columnwidth, 3.5in, Section 4.2, \\RelFullFinalClean\\,\\%") == []
    assert bare_numbers("about 12\\% less") == ["12\\%"]


def test_sections_contain_no_bare_result_numbers():
    sec = pd.REPO / "docs" / "paper" / "sections"
    bad = {}
    for p in sorted(sec.glob("*.tex")) if sec.is_dir() else []:
        hits = bare_numbers(p.read_text(encoding="utf-8"))
        if hits:
            bad[p.name] = hits
    assert not bad, f"use metrics.tex macros instead of bare numbers: {bad}"


FIGURES = ("money",)
_HARNESS = r"""\documentclass[10pt,letterpaper,twocolumn]{article}
\input{house.tex}
\usepackage{pgfplotstable}
\usepgfplotslibrary{groupplots}
\newcommand{\ultra}{\textsc{Ultra}}
\input{data/metrics.tex}
\newsavebox\fbx
\newcommand{\measure}[1]{\sbox\fbx{\input{figures/#1.tex}}\typeout{FIGWIDTH #1 \the\wd\fbx}}
\begin{document}
\onecolumn
\typeout{BODYWIDTH \the\textwidth}
FIGURES
x
\end{document}
"""


def test_figures_compile_standalone_within_page(tmp_path):
    import subprocess

    assert shutil.which("pdflatex"), "pdflatex is required for the paper figures"
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
    body_width = re.search(r"BODYWIDTH ([\d.]+)pt", log)
    assert body_width
    assert float(widths["money"]) <= float(body_width[1]) + 0.1, "Figure 1 exceeds the paper text width"


def test_every_manifest_file_is_committed():
    import subprocess

    if shutil.which("git") is None or not (pd.REPO / ".git").exists():
        return
    tracked = set(subprocess.run(["git", "ls-files", "--", str(pd.RAW)], cwd=pd.REPO, check=True, capture_output=True,
                                 text=True).stdout.splitlines())
    listed = [line.split()[1] for line in (pd.REPO / pd.RAW / "MANIFEST.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=pd.REPO, capture_output=True, text=True,
                             input="".join(f"{pd.RAW}/{rel}\n" for rel in listed)).stdout.split()
    assert not ignored, f"MANIFEST pins files that git ignores: {ignored}"
    untracked = [rel for rel in listed if f"{pd.RAW}/{rel}" not in tracked]
    assert len(untracked) < len(listed), "no manifest file is tracked: run this test from a checkout"
    assert all(not rel.endswith(".log") for rel in listed), "a .log file is never evidence"
