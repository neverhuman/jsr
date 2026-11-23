"""What the paper says in words must hold in the data, and Figure 1 outside the PDF must be current."""

import json
import re
import shutil
import subprocess

from tools import paper_data as pd
from tools.paper_data import publish, records, reported

DATA = pd.REPO / pd.DATA
PAPER = pd.REPO / "docs" / "paper"
REFRESH = "run `just paper-refresh`"

# Zero legacy ULTRA models, zero third-party vendor benchmark tables, zero draft/stale phrasing
BANNED_LEGACY_RE = re.compile(
    r"\b(?:A1|H22ctl|h22ctl|D2|D2MR|D2e|D3MR[a-z]?|B0|F1|S_MR|multirate|Multirate|ternary|Ternary|"
    r"Picovoice|Cheetah|Redux|Photon|PRELIMINARY)\b|"
    r"earlier checkpoint|not drawn|training continues|Where we trail|tab:reported|"
    r"(?<![\d.])(?:28\.2|0\.350)(?!\d)"
)


def _records():
    return {r["id"]: r for r in map(json.loads, (DATA / "records.jsonl").read_text().splitlines())}


def test_claims_made_in_words_hold_in_the_measured_rows():
    """Verify every headline claim in the prose and README holds quantitatively in the admitted rows."""
    rec = _records()
    # Final point beats Zipformer on pooled WER and dev-other, trails Zipformer on dev-clean, and trails best comparator
    assert rec["ColdHeadWer"]["raw"] < rec["ColdZipWer"]["raw"], "ULTRA final no longer beats Zipformer on pooled WER"
    assert rec["ColdHeadWerOther"]["raw"] < rec["ColdZipWerOther"]["raw"], "ULTRA final no longer beats Zipformer on dev-other"
    assert rec["ColdHeadWerClean"]["raw"] > rec["ColdZipWerClean"]["raw"], "dev-clean claim flipped; update prose"
    assert rec["ColdHeadWerGapBest"]["raw"] > 0.0, "ULTRA final no longer trails the lowest-WER comparator"
    # Live point claims checked whenever the standalone live cold row (14q) is admitted
    if "ColdLiveSpeed" in rec:
        assert rec["ColdHeadSpeedOverFast"]["raw"] > 1.8, "ULTRA live is no longer ~2x faster than Zipformer"
        assert rec["ColdLightRssOverHead"]["raw"] > 1.5, "ULTRA live no longer uses less memory than every comparator"
        assert rec["ColdLiveWatts"]["raw"] < rec["ColdZipWatts"]["raw"], "ULTRA live is no longer lower watts than Zipformer"
    # INT8 parity within pre-registered +-0.10 pp gate on full DEV
    assert abs(rec["RelIntEightDelta"]["raw"]) <= 0.10, "INT8 vs FP32 parity exceeded 0.10 pp"


def test_front_page_pairs_each_modes_speed_and_accuracy():
    rec = _records()
    if "ColdLiveSpeed" not in rec:
        return
    live = rec["ColdLiveSpeed"]["value"]
    live_wer = rec["ColdLiveWer"]["value"]
    final = rec["ColdHeadSpeed"]["value"]
    final_wer = rec["ColdHeadWer"]["value"]
    subtitle = rec["ColdSubtitle"]["value"]
    assert f"Live: {live}$\\times$ real time, {live_wer}\\% WER" in subtitle
    assert f"Final (commit pass at utterance ends): {final}$\\times$, {final_wer}\\% WER" in subtitle
    live_abs = rec["ColdAbstractLive"]["value"]
    final_abs = rec["ColdAbstractFinal"]["value"]
    assert f"Live decoding runs at {live}$\\times$" in live_abs and f"{live_wer}\\% pooled WER" in live_abs
    assert "active core energy per audio-second" in live_abs
    assert f"reference utterance boundaries, reaches {final_wer}\\% pooled WER at {final}$\\times$" in final_abs
    assert "58\\%" in rec["ColdAbstractZip"]["value"]
    assert rec["ColdAbstractMoreAcc"]["value"] == "Five of the 13 public recognisers we measured are more accurate than the final mode."


def test_no_legacy_models_or_reports_in_hero_readme_or_paper():
    """Figure 1, Table 1, evidence, systems.json, sections, README, and PDF hold ONLY v3.0.0 + all 13 public comparators."""
    plotted = [ln.split(",") for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]
               if ln.split(",")[5:7] == ["zen2", "1"] and ln.split(",")[21] == "1"]
    variants = {c[1] for c in plotted}
    expected = {
        "rc1-ema-live", "rc1-ema-final", "r0-int8", "parakeet-tdt-v3-int8", "parakeet-v3-onnxasr-int8",
        "nemotron-en-560", "nemotron-en-160", "nemotron-35", "moonshine-small", "moonshine-tiny",
        "vosk-small", "vosk-large", "whispercpp-stream-tiny", "whispercpp-stream-base",
    }
    assert variants == expected, f"unexpected plotted set: {sorted(variants)}"
    # Dynamic complete-panel WER ranking includes both ULTRA modes + all 13 public comparators (15 total)
    rec = _records()
    assert rec["ColdCompletePanelCount"]["raw"] == 15
    assert rec["ColdHeadWerRank"]["raw"] == 6, f"expected RC1 final rank 6, got {rec['ColdHeadWerRank']['raw']}"
    readme_txt = (pd.REPO / "README.md").read_text(encoding="utf-8")
    for must in ("ULTRA v3.0.0 (live)", "ULTRA v3.0.0 (final, seg.)", "29.1", "0.344",
                 "Parakeet unified en q8 (stream 160 ms)", "Whisper.cpp Streaming Base",
                 "Also measured, not ranked (subset WER, other hardware)", "2.21%", "2.33%", "7.74%",
                 "in the quiet full-panel H1 cohort", "53.2%", "0.358 W/stream", "242/457 ms",
                 "docs/release.md#install-the-pinned-release", "docs/ultra/releases/v3.0.1.md"):
        assert must in readme_txt, f"README missing required comparator/install item: {must!r}"
    # Zero non-plotted legacy rows in cold_dev.csv
    all_csv = [ln.split(",")[1] for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    assert set(all_csv) == variants, f"cold_dev.csv holds unplotted legacy rows: {set(all_csv) - variants}"
    # Release provenance must never borrow older checkpoint receipts
    rel_ev = (DATA / "evidence" / "xbabe0-v3-release.jsonl").read_text(encoding="utf-8")
    for old_ck in ("a1-30000", "a1-20000", "h22ctl", "d2e", "4701787c"):
        assert old_ck not in rel_ev.lower(), f"release evidence references older checkpoint {old_ck!r}"
    # Zero banned legacy tokens across every rendered and source file
    targets = [
        pd.REPO / "README.md",
        PAPER / "paper.tex",
        DATA / "systems.json",
        DATA / "cold_dev.csv",
        DATA / "money_layout.tex",
        DATA / "money_readme.md",
        DATA / "table_measured.tex",
        DATA / "table_hardware.tex",
        DATA / "metrics.tex",
        DATA / "SOURCES.md",
        *sorted((PAPER / "sections").glob("*.tex")),
        *sorted((PAPER / "figures").glob("*.tex")),
        *sorted((DATA / "evidence").glob("*.jsonl")),
    ]
    hits = {}
    for p in targets:
        found = sorted(set(BANNED_LEGACY_RE.findall(p.read_text(encoding="utf-8"))))
        if found:
            hits[str(p.relative_to(pd.REPO))] = found
    if shutil.which("pdftotext") and (PAPER / "paper.pdf").is_file():
        pdf_txt = subprocess.check_output(["pdftotext", str(PAPER / "paper.pdf"), "-"], text=True)
        pdf_hits = sorted(set(BANNED_LEGACY_RE.findall(pdf_txt)))
        if pdf_hits:
            hits["docs/paper/paper.pdf"] = pdf_hits
    assert not hits, f"legacy models/reports/tokens leaked into clean-release presentation: {hits}"


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
    title, tagline, summary, image, caption, *_ = [
        line for line in readme.splitlines() if line and not line.startswith("<!--")
    ]
    paper_title = re.search(r"\\newcommand\{\\PaperTitle\}\{([^}]+)\}", (DATA / "metrics.tex").read_text())
    assert paper_title and title == f"# {paper_title[1]}"
    assert tagline == "Always listening, barely computing."
    subtitle = _records()["ColdSubtitle"]["value"]
    summary_text = subtitle.replace("$\\times$", "×").replace("$\\cdot$", "·").replace("\\%", "%").replace("\\,", " ")
    assert summary == f"**{summary_text}**"
    assert image.startswith(f"![{paper_title[1]}. ") and image.endswith("(docs/paper/money.png)")
    assert caption.startswith("**Figure 1.**") and "[Read the paper](docs/paper/paper.pdf)" in caption
    drawn = [ln for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]
             if ln.split(",")[5:7] == ["zen2", "1"] and ln.split(",")[21] == "1"]
    table = re.findall(r"^\| .*", block, flags=re.M)
    assert len(table) == len(drawn) + 2, "one table line per plotted system + 1 other-host complete-panel row + header"
    labels = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\.\*\*", block, flags=re.M)]
    assert labels == ["Accuracy", "Speed & memory", "Watts", "Integer-exact INT8"]
    layout = (DATA / "money_layout.tex").read_text()
    assert r"\def\MoneyOursLegend{ULTRA v3.0.0 (live / final)}" in layout
    assert "ULTRA v3.0.0\\\\(live)" in layout and "ULTRA v3.0.0\\\\(final, seg.)" in layout


def test_release_branding_is_consistent():
    """Binding brand spec: plain all-caps ULTRA, exact 4-place title, exact tagline, exact v3.0.0 labels, v3.0.1 install-only."""
    expected_title = "ULTRA: Reproducible Streaming Speech Recognition on One CPU Core"
    metrics_txt = (DATA / "metrics.tex").read_text(encoding="utf-8")
    assert f"\\newcommand{{\\PaperTitle}}{{{expected_title}}}" in metrics_txt
    paper_tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\ultra}{ULTRA\xspace}" in paper_tex
    assert r"pdftitle={\PaperTitle}" in paper_tex and r"pdfauthor={}" in paper_tex
    assert "Always listening, barely computing." in (PAPER / "sections" / "intro.tex").read_text(encoding="utf-8")

    readme = (pd.REPO / "README.md").read_text(encoding="utf-8")
    assert "| **ULTRA v3.0.0 (live)** |" in readme and "| **ULTRA v3.0.0 (final, seg.)** |" in readme
    install = re.search(r"(?ms)^## Install the signed release\n.*?(?=^## |\Z)", readme)
    assert install and "v3.0.1" in install[0], "README install section must mention v3.0.1"
    non_install_readme = readme[:install.start()] + readme[install.end():]
    assert "v3.0.1" not in non_install_readme, "v3.0.1 leaked outside README install section"

    brand_re = re.compile(r"\b(?:ULTRA-S|Ultra-S|Ossicle)\b|ULTRA v3\.0 \(|(?<!M3 )(?<!Apple M3 )\bUltra\b")
    targets: dict[str, str] = {"README.md": non_install_readme}
    for path in (
        PAPER / "paper.tex", DATA / "systems.json", DATA / "metrics.tex", DATA / "money_layout.tex",
        *sorted((PAPER / "sections").glob("*.tex")), *sorted((PAPER / "figures").glob("*.tex")),
        *sorted((DATA / "evidence").glob("*.jsonl")),
    ):
        txt = path.read_text(encoding="utf-8")
        targets[str(path.relative_to(pd.REPO))] = txt
        assert "v3.0.1" not in txt, f"v3.0.1 leaked into {path.relative_to(pd.REPO)}"
    if shutil.which("pdftotext") and (PAPER / "paper.pdf").is_file():
        pdf_txt = subprocess.check_output(["pdftotext", str(PAPER / "paper.pdf"), "-"], text=True)
        targets["docs/paper/paper.pdf"] = pdf_txt
        assert "v3.0.1" not in pdf_txt, "v3.0.1 leaked into docs/paper/paper.pdf"
    hits = {}
    for name, txt in targets.items():
        found = sorted(set(brand_re.findall(txt)))
        if found:
            hits[name] = found
    assert not hits, f"brand-spec violations found: {hits}"
    if shutil.which("pdfinfo") and (PAPER / "paper.pdf").is_file():
        info = subprocess.check_output(["pdfinfo", str(PAPER / "paper.pdf")], text=True)
        title_m = re.search(r"^Title:[ \t]*(.*)$", info, flags=re.M)
        author_m = re.search(r"^Author:[ \t]*(.*)$", info, flags=re.M)
        assert title_m and title_m.group(1).strip() == expected_title, f"unexpected pdfinfo Title: {title_m and title_m.group(1)!r}"
        assert not author_m or author_m.group(1).strip() == "", f"pdfinfo Author must be empty: {author_m and author_m.group(1)!r}"


def test_money_png_was_exported_from_the_current_inputs():
    """docs/paper/money.png is stamped with the hash of the files Figure 1 is drawn from."""
    png = PAPER / "money.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"docs/paper/money.png is missing; {REFRESH}"
    stamp = (pd.REPO / publish.STAMP).read_text().strip()
    assert stamp == publish.figure_digest(pd.REPO), f"docs/paper/money.png is older than its data; {REFRESH}"


def test_reported_numbers_never_reach_a_measured_file():
    for name in ("cold_dev.csv", "money_layout.tex", "money_frontier.csv", "table_measured.tex"):
        text = (DATA / name).read_text()
        assert "Rep" not in text and "reported" not in text, name
    figures = "".join(p.read_text() for p in sorted((PAPER / "figures").glob("*.tex")))
    assert "\\Rep" not in figures, "a published number is drawn in a figure"


def test_reported_values_are_printed_in_their_quoted_source_line(tmp_path):
    import pytest

    doc = json.loads((DATA / "reported.json").read_text())
    for rid, v in doc["values"].items():
        assert v["source"] in doc["sources"], rid
        assert re.search(r"(?<![\d.])" + re.escape(v["text"]) + r"(?!\d)", v["quote"].replace(",", "")), rid
    root = tmp_path / "repo"
    shutil.copytree(DATA, root / pd.DATA)
    doc["values"]["RepPvCheetahWerAvg"]["text"] = "6.6"  # not what the quoted line says
    (root / pd.DATA / "reported.json").write_text(json.dumps(doc))
    with pytest.raises(pd.PaperDataError, match="does not appear in its quoted source line"):
        reported.add_reported(root, records.Records())


def test_every_citation_key_exists():
    keys = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "refs.bib").read_text(), flags=re.M))
    used = set()
    for p in sorted((PAPER / "sections").glob("*.tex")):
        for group in re.findall(r"\\cite\{([^}]*)\}", p.read_text()):
            used |= {k.strip() for k in group.split(",")}
    systems = json.loads((DATA / "systems.json").read_text())
    used |= {k for v in systems["variants"].values() for k in v.get("cite", [])}
    assert used <= keys, f"cited but not in refs.bib: {sorted(used - keys)}"


def test_measured_table_has_one_line_per_measured_system_and_only_generated_cells():
    table = (DATA / "table_measured.tex").read_text()
    macros = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (DATA / "metrics.tex").read_text()))
    cells = set(re.findall(r"\\(Cold[A-Za-z]+)", table))
    assert cells and cells <= macros, f"table cells without a macro: {sorted(cells - macros)}"
    plain = [ln.split(",") for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    drawn = [c for c in plain if c[21] == "1"]
    assert len(re.findall(r"\\Cold[A-Za-z]+Speed[A-Za-z]* &", table)) == len(drawn)
