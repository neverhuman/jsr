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

# Zero legacy models, zero third-party vendor benchmark tables, zero draft/stale phrasing
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
    # Final point is not worse than Zipformer on pooled WER, beats it on dev-other, equals it on dev-clean (to the printed
    # precision), and trails the best comparator
    assert rec["ColdHeadWer"]["raw"] < rec["ColdZipWer"]["raw"], "Isosyll final no longer beats Zipformer on pooled WER"
    assert rec["ColdHeadWerOther"]["raw"] < rec["ColdZipWerOther"]["raw"], "Isosyll final no longer beats Zipformer on dev-other"
    assert abs(rec["ColdHeadWerClean"]["raw"] - rec["ColdZipWerClean"]["raw"]) < 0.005, "dev-clean is no longer equal; update prose"
    assert rec["ColdHeadWerGapBest"]["raw"] > 0.0, "Isosyll final no longer trails the lowest-WER comparator"
    # Live point claims checked whenever the standalone live cold row (14q) is admitted
    if "ColdLiveSpeed" in rec:
        assert rec["ColdHeadSpeedOverFast"]["raw"] > 1.8, "Isosyll live is no longer ~2x faster than Zipformer"
        assert rec["ColdLightRssOverHead"]["raw"] > 1.2, "Isosyll live no longer uses clearly less memory than every comparator"  # live +LM: 1.4x
        assert rec["ColdLiveWatts"]["raw"] < rec["ColdZipWatts"]["raw"], "Isosyll live is no longer lower watts than Zipformer"
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
    assert "57\\%" in rec["ColdAbstractZip"]["value"]
    assert rec["ColdAbstractMoreAcc"]["value"] == "Seven of the 15 public recognisers we measured are more accurate than the final mode."


def test_no_legacy_models_or_reports_in_hero_readme_or_paper():
    """Figure 1, Table 1, evidence, systems.json, sections, README, and PDF hold ONLY v3.1.0 (v3.0.0 only as the prior release) + all 15 public comparators."""
    plotted = [ln.split(",") for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]
               if ln.split(",")[5:7] == ["zen2", "1"] and ln.split(",")[21] == "1"]
    variants = {c[1] for c in plotted}
    expected = {
        "rc2-live-lm", "rc2-final", "r0-int8-r6", "parakeet-tdt-v3-int8", "parakeet-v3-onnxasr-int8",
        "nemotron-en-560", "nemotron-en-160", "nemotron-35", "moonshine-small", "moonshine-tiny",
        "vosk-small", "vosk-large", "whispercpp-stream-tiny", "whispercpp-stream-base",
    }
    assert variants == expected, f"unexpected plotted set: {sorted(variants)}"
    # Dynamic complete-panel WER ranking includes both Isosyll modes + all 15 public comparators (17 total)
    rec = _records()
    assert rec["ColdCompletePanelCount"]["raw"] == 17
    assert rec["ColdHeadWerRank"]["raw"] == 8, f"expected RC1 final rank 8, got {rec['ColdHeadWerRank']['raw']}"
    readme_txt = (pd.REPO / "README.md").read_text(encoding="utf-8")
    for must in ("Isosyll v3.1.0 live +LM (160 ms)", "32.7", "0.328", "Released by owner decision after missing the registered gates",
                 "docs/release.md#install-the-pinned-release", "docs/ultra/releases/v3.1.0.md", "docs/ultra/releases/v3.0.1.md",
                 "ultra-v3.1.0-int8-bundle.tar", "ultra-cold-wav --chunk 4" if False else "./ultra-cold-wav --chunk 4",
                 "./ultra-engine-commit --model int8"):
        assert must in readme_txt, f"README missing required comparator/install item: {must!r}"
    # Claims moved out of the lean README stay asserted, in the paper sources and records
    sections = {p.name: p.read_text(encoding="utf-8") for p in (PAPER / "sections").glob("*.tex")}
    assert "Head-to-head with Zipformer INT8" in (DATA / "table_versus.tex").read_text()
    assert rec["VsVerdictTwo"]["value"] == "not significantly different from" and "VsVerdictTwo" in sections["abstract.tex"]
    assert rec["VsPriorLiveSpeed"]["value"] == "29.1" and rec["VsPriorLiveWatts"]["value"] == "0.344" and rec["VsPriorFinalWatts"]["value"] == "0.720"
    assert all(m in sections["results.tex"] for m in ("VsPriorLiveSpeed", "VsPriorLiveWatts", "VsPriorFinalWatts"))
    assert rec["RelCapSixteenLagPninetyFive"]["value"] == "457" and "RelCapSixteenLagPninetyFive" in sections["results.tex"]
    assert rec["ColdCompleteCompCount"]["raw"] == 15 and "ColdCompleteCompCount" in sections["related.tex"]
    # Zero non-plotted legacy rows in cold_dev.csv
    all_csv = [ln.split(",")[1] for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    v31_unplotted = {"rc2-live-greedy", "rc2-live-lm320", "rc1-ema-live", "rc1-ema-final", "r0-int8"}  # Fig. 1 foreground and the prior release
    assert set(all_csv) == variants | v31_unplotted, f"cold_dev.csv holds unexpected rows: {set(all_csv) - variants - v31_unplotted}"
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
    descriptor = re.search(r"\\newcommand\{\\PaperDescriptor\}\{([^}]+)\}", (DATA / "metrics.tex").read_text())
    assert descriptor and tagline == f"**{descriptor[1]}**"
    subtitle = _records()["HeroSubtitle"]["value"]
    summary_text = subtitle.replace("$\\times$", "×").replace("$\\cdot$", "·").replace("\\%", "%").replace("\\,", " ")
    assert summary == f"**{summary_text}**"
    assert image.startswith(f"![{paper_title[1]}. ") and image.endswith("(docs/paper/money.png)")
    assert caption.startswith("**Figure 1.**") and "[Paper](docs/paper/paper.pdf)" in caption
    assert "selection panel" in caption and "unselected DEV" in caption and "docs/ultra/releases/v3.1.0.md" in caption
    table = re.findall(r"^\| .*", block, flags=re.M)
    pairs = json.loads((DATA / "systems.json").read_text())["versus"]["pairs"]
    assert len(table) == 1 + 1 + 3, "lean table: header, Zipformer, live greedy, live +LM at 160 ms, final"
    assert len(pairs) == 2
    labels = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\.\*\*", block, flags=re.M)]
    assert labels == ["Always-on", "Final text", "Bit-exact and portable"]
    assert all(len(b.splitlines()) == 1 for b in re.findall(r"^- \*\*.*$", block, flags=re.M)), "one source line per bullet"
    layout = (DATA / "money_layout.tex").read_text()
    assert (r"\def\MoneyOursLegend{\shortstack[l]{Isosyll v3.1.0 (live +LM)\\"
            "Isosyll v3.1.0 (final, seg.)}}") in layout
    assert "Isosyll v3.1.0\\\\(live +LM)" in layout and "Isosyll v3.1.0\\\\(final, seg.)" in layout


def test_release_branding_is_consistent():
    """Binding brand spec (owner, 2026-11-24): the system is Isosyll; exact title, descriptor and v3.0.0 labels;
    the historical name ULTRA appears only where release artifacts are named; v3.0.1 is install-only."""
    expected_title = "Isosyll: Always Listening, Barely Computing"
    expected_descriptor = "Bit-Exact Streaming Speech Recognition on One CPU Core"
    metrics_txt = (DATA / "metrics.tex").read_text(encoding="utf-8")
    assert f"\\newcommand{{\\PaperTitle}}{{{expected_title}}}" in metrics_txt
    assert f"\\newcommand{{\\PaperDescriptor}}{{{expected_descriptor}}}" in metrics_txt
    paper_tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\ultra}{Isosyll\xspace}" in paper_tex
    assert r"pdftitle={\PaperTitle}" in paper_tex and r"pdfauthor={}" in paper_tex
    assert "Always listening, barely computing." in (PAPER / "sections" / "intro.tex").read_text(encoding="utf-8")

    readme = (pd.REPO / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {expected_title}\n")
    assert "| **Isosyll v3.1.0 live +LM (160 ms)** |" in readme and "| **Isosyll v3.1.0 final** |" in readme
    install = re.search(r"(?ms)^## Install the signed release\n.*?(?=^## |\Z)", readme)
    assert install and "v3.0.1" in install[0], "README install section must mention v3.0.1"
    non_install_readme = readme[:install.start()] + readme[install.end():]
    assert "v3.0.1" not in non_install_readme, "v3.0.1 leaked outside README install section"
    assert not re.search(r"\bULTRA\b", non_install_readme), "display name ULTRA outside the README install section"
    for path in (DATA / "metrics.tex", DATA / "money_layout.tex", DATA / "money_readme.md"):
        assert not re.search(r"\bULTRA\b", path.read_text(encoding="utf-8")), f"display name ULTRA in {path.name}"

    brand_re = re.compile(r"\b(?:ULTRA-S|Ultra-S|Isosyll-S|Ossicle|ISOSYLL|IsoSyll)\b|ULTRA v3\.0 \(|(?<!M3 )(?<!Apple M3 )\bUltra\b")
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


def test_readme_v4_section_is_generated_outside_the_money_block():
    readme = (pd.REPO / "README.md").read_text()
    block = (DATA / "v4_readme.md").read_text().rstrip("\n")
    assert readme.count(publish.V4_START) == 1 and readme.count(publish.V4_END) == 1
    shown = readme[readme.index(publish.V4_START):readme.index(publish.V4_END) + len(publish.V4_END)]
    assert shown == block, f"README.md shows an old V4 section; {REFRESH}"
    assert readme.index(publish.END) < readme.index(publish.V4_START), "the V4 section sits after the money block, not inside it"
    assert "SPEC15" in block and "O3-LD" in block and "not adopted" in block and "fresh audio" in block
    assert "pending" not in block, "the O3-LD verdict landed; no pending wording"
    assert len([l for l in block.splitlines() if l.startswith("- ")]) == 3, "three-line V4 summary"
    assert "cannot reach" in block and "cannot close" not in block
    money = (DATA / "money_readme.md").read_text()
    assert "800 vs. 722" in money and "722 vs. 650" in money and "2.46×" in money
    v4 = (PAPER / "sections" / "v4_findings.tex").read_text()  # the detail the README no longer carries
    rec = _records()
    assert rec["VfOThreeDelta"]["value"] == "+0.0115" and "VfOThreeDelta" in v4 and "stopped at the read, as registered" in v4
    assert "160\\,ms chunk for RC2" in v4 and "VfTdtChunkMs" in v4 and "ReLU$^2$ kill readings" in v4 and "VfSingleSeedPp" in v4
    assert "5.288" not in block, "the PL3 final 5.288% is not tied to this verdict"


def test_release_is_stated_as_owner_decision_not_a_gate_pass():
    readme = (pd.REPO / "README.md").read_text()
    assert "Released by owner decision after missing the registered gates" in readme
    for name in ("abstract", "conclusion"):
        txt = (PAPER / "sections" / f"{name}.tex").read_text()
        assert "owner decision" in txt and "CandFinalFullEdits" in txt and "CandGateFinalFull" in txt, name
    rec = _records()
    assert rec["CandFinalFullEdits"]["raw"] > rec["CandGateFinalFull"]["raw"], "the gate is no longer missed; reword"
    assert rec["CandLiveLmCDelta"]["raw"] > 0, "live +LM no longer trails Zipformer on the C half; reword"


def test_ratios_to_zipformer_are_medians_of_within_block_ratios():
    """Paired ratios come from the same-block anchor (grok review): final speed prints 1.19x, not the 1.20x of the two medians."""
    rec = _records()
    summary = json.loads((DATA / "raw/release_v31/timing_interleaved/summary_report.json").read_text())["paired_ratios_vs_zipformer"]
    assert abs(rec["VsSpeedXOne"]["raw"] - summary["live_c4_lm"]["speed_ratio_median"]) < 5e-5
    assert abs(rec["VsSpeedXTwo"]["raw"] - summary["final_m5"]["speed_ratio_median"]) < 5e-5
    assert rec["VsSpeedXTwo"]["value"] == "1.19" and rec["VsEnergyPctTwo"]["value"] == "115"
    block = (DATA / "money_readme.md").read_text()
    assert "(1.19×)" in block and "1.20" not in block


def test_prior_release_rows_carry_their_own_record_not_the_new_intervals():
    """The README no longer lists v3.0.0; the paper does, with v3.0.0's own record (not the v3.1.0 intervals)."""
    rec = _records()
    assert (rec["VsPriorLiveDelta"]["value"], rec["VsPriorLiveLo"]["value"], rec["VsPriorLiveHi"]["value"]) == ("+1.59", "+1.27", "+1.92")
    assert (rec["VsPriorFinalDelta"]["value"], rec["VsPriorFinalLo"]["value"], rec["VsPriorFinalHi"]["value"]) == ("$-$0.19", "$-$0.53", "+0.14")
    results = (PAPER / "sections" / "results.tex").read_text()
    assert "that\nsession's Zipformer, so no ratio is formed" in results and "CandLiveGreedyOldDelta" in results and "live greedy" in results
    assert rec["VsWideRssMb"]["value"] == "110.6" and rec["CandLiveLmEightCDelta"]["value"] == "+0.65" and "VsWideRssMb" in results
    assert "v3.0.0" not in (DATA / "money_readme.md").read_text().split("<!-- money-plot:end -->")[0].split("| **Isosyll v3.1.0 final**")[0]



def test_four_backend_claim_is_scoped_to_main_not_the_release_binary():
    block = (DATA / "money_readme.md").read_text()
    rec = _records()
    assert rec["ScalarDiffs"]["raw"] == 0 and rec["ScalarUtts"]["raw"] == 730
    assert "not for the v3.1.0 binary" in block and rec["ScalarMainCommit"]["value"] in block
    results = (PAPER / "sections" / "results.tex").read_text()
    assert "does not\napply to the" in results and "ScalarMainCommit" in results and "ScalarRelCommit" in results


def test_headline_precision_gate_scope_and_install_sample_are_exact():
    readme = (pd.REPO / "README.md").read_text()
    block = (DATA / "money_readme.md").read_text()
    for must in ("32.67× vs. 14.04×", "57.3%", "109.47 MB", "live +LM scores 6.10% on full DEV", "the selection panel and on the unselected C half"):
        assert must in block, must
    assert "both halves" not in block and "reranking those lists cannot reach 5%" in (DATA / "v4_readme.md").read_text()
    # install sample: no `<wav-dir>` (a Bash redirection), staged names and output schemas stated, extraction and check on separate lines
    install = re.search(r"(?ms)^## Install the signed release\n.*?(?=^## |\Z)", readme)[0]
    assert "<wav-dir>" not in install and "./wav-dir ./live.jsonl" in install and "u00000.wav" in install
    assert "`provisional` and `committed`" in install and "`id` and `hyp`" in install
    assert "tar -xf ultra-v3.1.0-int8-bundle.tar\n  sha256sum -c ultra-v3.1.0-int8-bundle.SHA256SUMS" in install and "&& sha256sum" not in install
    release = (pd.REPO / "docs/release.md").read_text()
    assert "<wav-dir>" not in release and "{\"id\", \"hyp\"}" in release
    rc2 = (pd.REPO / "docs/ultra/releases/v3.1.0-rc2.md").read_text()
    assert "Historical record" in rc2 and "v3.1.0.md" in rc2 and "are complete" in rc2 and "Pending: the quiet-host" not in rc2
    # intro/limitations claims are scoped to the matched Zipformer, and benchmark energy is in J per audio-second
    rec = _records()
    for rid in ("ColdIntroLiveItem", "ColdSpeedSentence", "ColdMemorySentence", "ColdIntroContrib", "ColdLimitLive", "ColdConclusionResult", "ColdWattsSentence"):
        text = rec[rid]["value"]
        assert "every comparator" not in text and not re.search(r"\d\\,W\b", text), rid
    assert "J per audio-second" in rec["ColdIntroLiveItem"]["value"] and "matched streaming" in rec["ColdIntroLiveItem"]["value"]
    results = (PAPER / "sections" / "results.tex").read_text()
    assert "inconclusive on both the selection panel" in results and "Zipformer-class" not in results
    v4 = (PAPER / "sections" / "v4_findings.tex").read_text()
    assert "VfRtenRatio" in v4 and "development CPU" in v4 and "not energy-qualified" in v4 and "recorded fixed beam" in v4
