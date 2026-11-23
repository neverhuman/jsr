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
    """The comparison sentences are generated, so they stay true. The story around them (title, abstract,
    contributions, README lead) assumes three things; if one flips, rewrite the text."""
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
    title, image, caption, *_ = [line for line in readme.splitlines() if line and not line.startswith("<!--")]
    assert title.startswith("# ") and image.startswith("![") and image.endswith("(docs/paper/money.png)")
    assert caption.startswith("**Figure 1.**") and "[Read the paper](docs/paper/paper.pdf)" in caption
    drawn = [ln for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]
             if ln.split(",")[5:7] == ["zen2", "1"] and ln.split(",")[21] == "1"]
    table = [ln for ln in re.findall(r"^\| .*", block, flags=re.M) if ", not drawn" not in ln]
    assert len(table) == len(drawn) + 1, "one table line per plotted system (plus the always-on row, marked)"
    labels = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\.\*\*", block, flags=re.M)]
    assert labels == ["Cost", "Watts", "Capacity", "Silence", "Endless", "Same words everywhere", "Where we trail"]


def test_money_png_was_exported_from_the_current_inputs():
    """docs/paper/money.png is stamped with the hash of the files Figure 1 is drawn from."""
    png = PAPER / "money.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"docs/paper/money.png is missing; {REFRESH}"
    stamp = (pd.REPO / publish.STAMP).read_text().strip()
    assert stamp == publish.figure_digest(pd.REPO), f"docs/paper/money.png is older than its data; {REFRESH}"


def test_reported_numbers_never_reach_a_measured_file():
    rec = _records()
    reported = [rid for rid in rec if rid.startswith("Rep")]
    assert len(reported) > 90 and all(rec[rid]["label"] == "reported" for rid in reported)
    for name in ("cold_dev.csv", "money_layout.tex", "money_frontier.csv"):
        text = (DATA / name).read_text()
        assert "Rep" not in text and "reported" not in text, name
    assert "\\Rep" not in (DATA / "table_measured.tex").read_text()
    figures = "".join(p.read_text() for p in sorted((PAPER / "figures").glob("*.tex")))
    assert "\\Rep" not in figures, "a published number is drawn in a figure"


def test_reported_values_are_printed_in_their_quoted_source_line(tmp_path):
    import shutil

    doc = json.loads((DATA / "reported.json").read_text())
    for rid, v in doc["values"].items():
        assert v["source"] in doc["sources"], rid
        assert re.search(r"(?<![\d.])" + re.escape(v["text"]) + r"(?!\d)", v["quote"].replace(",", "")), rid
    root = tmp_path / "repo"
    shutil.copytree(DATA, root / pd.DATA)
    shutil.copytree(pd.REPO / "research" / "curves", root / "research" / "curves")
    doc["values"]["RepPvCheetahWerAvg"]["text"] = "6.6"  # not what the quoted line says
    (root / pd.DATA / "reported.json").write_text(json.dumps(doc))
    import pytest

    with pytest.raises(pd.PaperDataError, match="does not appear in its quoted source line"):
        pd.build(root)


def test_every_citation_key_exists():
    keys = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "refs.bib").read_text(), flags=re.M))
    used = set()
    for p in sorted((PAPER / "sections").glob("*.tex")):
        for group in re.findall(r"\\cite\{([^}]*)\}", p.read_text()):
            used |= {k.strip() for k in group.split(",")}
    systems = json.loads((DATA / "systems.json").read_text())
    used |= {k for v in systems["variants"].values() for k in v.get("cite", [])}
    used |= {s["cite"] for s in json.loads((DATA / "reported.json").read_text())["sources"].values()}
    assert used <= keys, f"cited but not in refs.bib: {sorted(used - keys)}"


def test_measured_table_has_one_line_per_measured_system_and_only_generated_cells():
    table = (DATA / "table_measured.tex").read_text()
    macros = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (DATA / "metrics.tex").read_text()))
    cells = set(re.findall(r"\\(Cold[A-Za-z]+)", table))
    assert cells and cells <= macros, f"table cells without a macro: {sorted(cells - macros)}"
    plain = [ln.split(",") for ln in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    drawn = [c for c in plain if c[21] == "1"]  # the plot column: every drawn system has its own table line
    assert len(re.findall(r"\\Cold[A-Za-z]+Speed[A-Za-z]* &", table)) >= len(drawn)


def test_related_work_claims_hold_in_the_reported_values():
    rec = _records()
    # related.tex and results.tex: two published sources differ by more than a factor of two on one model
    assert rec["RepPvMoonSmallWerClean"]["raw"] > 2.0 * rec["RepMsSmallWerShipped"]["raw"]


def test_page_one_facts_print_the_receipts_numbers_with_their_checkpoint():
    """The abstract and the README lead with facts generated from receipts; each names its checkpoint."""
    from tools.paper_data import alwayson

    rec = _records()
    cap = alwayson.capacity(pd.REPO)["systems"]["h22ctl-79800-ema-int8"]
    assert f"{cap['n']} live streams" in rec["AbsCapacity"]["value"]
    assert rec["AoCapacityStreams"]["raw"] == cap["n"] and cap["flat"]
    sil = alwayson.silence(pd.REPO)
    assert sil["blocks"] == 10 and sil["words"] == 0 and "no word" in rec["AbsSilence"]["value"]
    assert f"{sil['gap_words']} in the" in rec["AbsSilence"]["value"], "the stream's own pauses must be reported too"
    end = alwayson.endless(pd.REPO)
    assert end["resets"] == 0 and end["utts"] == 5567, "the endless run is no longer one unbroken stream"
    assert end["end"] <= end["peak"] < 1.05 * end["warm"], "memory grew over the endless run; rewrite the claim"
    assert abs(end["second"] - end["first"]) < 0.5, "WER drifted over the endless run; rewrite the claim"
    st = alwayson.stream(pd.REPO)
    assert abs(st["replay"]["delta"]) < 0.2 and st["replay"]["lower"] < 0 < st["replay"]["upper"]
    assert st["penalty"]["lower"] > 0, "the continuous-stream penalty is no longer outside its bounds"
    head = json.loads((DATA / "systems.json").read_text())["headline"]
    aon = rec["AoCheckpoint"]["value"]
    if head["ours"] != head["always_on"]:  # the always-on tests ran on another checkpoint: every fact says so
        assert aon in rec["AbsCapacity"]["value"] and aon in rec["AoCapacity"]["value"] and aon in rec["AoEndless"]["value"]
    if rec["MoneyOursWatts"]["value"] != "\\relax":  # no energy line of our point: Figure 1 says whose watts are quoted
        quoted = rec["AoWatts"]["source"].split("id=")[1].split("-zen2")[0]
        assert "an earlier" in rec["MoneyOursWatts"]["value"] and rec["AoWatts"]["value"] in rec["MoneyOursWatts"]["value"]
        assert quoted != head["ours"], "the text quotes watts of a row it says has none"
    readme = (DATA / "money_readme.md").read_text()
    for rid in ("AbsCost", "AbsWatts", "AbsCapacity", "AbsSilence", "AbsEndless", "AbsAccuracy"):
        assert "pending" not in rec[rid]["value"], rid
    assert f"{cap['n']} live streams on one core" in readme


def test_a_missing_receipt_makes_its_fact_pending(tmp_path):
    import shutil

    root = tmp_path / "repo"
    shutil.copytree(DATA, root / pd.DATA)
    shutil.copytree(pd.REPO / "research" / "curves", root / "research" / "curves")
    (root / pd.DATA / "raw/energy/derived/capacity-test-entry4b.jsonl").unlink()
    manifest = root / pd.DATA / "raw/MANIFEST.txt"
    manifest.write_text("".join(ln + "\n" for ln in manifest.read_text().splitlines() if "capacity-test-entry4b" not in ln))
    files, _ = pd.build(root)
    assert "\\newcommand{\\AbsCapacity}{Capacity: pending.}" in files["metrics.tex"]
    assert "- **Capacity.** pending" in files["money_readme.md"]
    assert "capacity test pending" in files["table_capacity.tex"]
