"""Measured comparison rows (schema v2), their grouping, and the Figure 1 label layout."""

import json
import re
import shutil
import subprocess

import pytest

from tools import paper_data as pd
from tools.paper_data import cold, money

DATA = pd.REPO / pd.DATA


def _mini_repo(tmp_path, rows):
    root = tmp_path / "repo"
    shutil.copytree(pd.REPO / pd.DATA, root / pd.DATA)
    shutil.copytree(pd.REPO / "research" / "curves", root / "research" / "curves")
    ev = root / pd.DATA / "evidence"
    shutil.rmtree(ev, ignore_errors=True)
    ev.mkdir()
    (ev / "cold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _row(**kw):
    r = {"variant": "sys-a", "system": "A", "runtime": "rt 1.0", "role": "comparator", "class": "streaming",
         "configuration": "c", "machine": "zen2", "host": "h0", "hardware": "H0 core 46", "threads": 1,
         "parallel": "intra_op", "session": "s1", "workload": "panel", "workload_digest": pd.PANEL_DIGEST,
         "scope": "cold",
         "cpu_s_per_audio_s": 0.2, "cpu_reps": [0.1, 0.2, 0.3], "wall_s_per_audio_s": 0.25,
         "wall_reps": [0.125, 0.25, 0.375], "wer_pct": 3.0, "wer_clean_pct": 2.0, "wer_other_pct": 4.5,
         "peak_rss_bytes": 150_000_000, "artifact_bytes": 70_000_000, "label": "exploratory", "source": "x",
         "receipt_sha256": "y"}
    r.update(kw)
    r.setdefault("id", f"{r['variant']}-{r['machine']}-t{r['threads']}")
    return r


def test_evidence_rows_are_ingested(tmp_path):
    rows = [_row(), _row(variant="sys-b", system="B", role="ours", cpu_s_per_audio_s=0.05, cpu_reps=[0.05, 0.05, 0.06],
                         wer_pct=9.0)]
    files, missing = pd.build(_mini_repo(tmp_path, rows))
    assert missing == []
    assert files["cold_dev.csv"].splitlines() == [
        ",".join(cold.CSV_HEADER),
        "sys-a-zen2-t1,sys-a,A,comparator,streaming,zen2,1,zen2-t1-comparator-streaming,0.20000,5.000,4.000,3.00,"
        "2.00,4.50,150.0,70.0,,,,exploratory,h0,1",
        "sys-b-zen2-t1,sys-b,B,ours,streaming,zen2,1,zen2-t1-ours-streaming,0.05000,20.000,4.000,9.00,2.00,4.50,"
        "150.0,70.0,,,,exploratory,h0,1",
    ]
    for macro in ("\\newcommand{\\ColdRowsCount}{2}", "\\newcommand{\\ColdSysASpeed}{5.00}",
                  "\\newcommand{\\ColdSysBSpeed}{20.0}", "\\newcommand{\\ColdSysAWer}{3.00}",
                  "\\newcommand{\\ColdSysARss}{150}", "\\newcommand{\\ColdRowsEnergy}{0}"):
        assert macro in files["metrics.tex"], macro


@pytest.mark.parametrize("bad, msg", [
    ({"wer_pct": None}, "missing"),
    ({"cpu_s_per_audio_s": 0.25}, "median"),
    ({"wall_reps": [0.25, 0.25]}, "at least 3"),
    ({"workload_digest": "other"}, "mix workloads"),
    ({"scope": "warm"}, "mix scopes"),
    ({"label": "reported"}, "reported"),
    ({"class": "offline"}, "bad class"),
    ({"machine": "laptop"}, "bad machine"),
    ({"id": "sys-b"}, "<variant>-<machine>-t<threads>"),
    ({"class": "whole_utterance", "lag_p50_ms": 300}, "throughput only"),
    ({"joules_per_audio_s": 0.4}, "all-or-none"),
    ({"short": "a label, with a comma"}, "plain characters"),
    ({"session": None}, "missing"),
    ({"variant": "sys-a", "id": "sys-a-zen2-t1-again"}, "anchor"),
    ({"anchor": True, "id": "sys-b-zen2-t1-anchor-h1"}, "no primary row"),
])
def test_evidence_rows_fail_visibly(tmp_path, bad, msg):
    root = _mini_repo(tmp_path, [_row(), _row(**{"variant": "sys-b", **bad})])
    with pytest.raises(pd.PaperDataError, match=msg):
        pd.build(root)


def test_rows_group_by_machine_and_threads_instead_of_refusing(tmp_path):
    rows = [
        _row(role="ours"),
        _row(variant="sys-b", role="ablation"),
        _row(variant="sys-c", **{"class": "whole_utterance"}),
        _row(threads=8, parallel="workers", scope="cold-eight"),  # another group: its own scope is allowed
        _row(machine="xeon8358", host="x6", scope="cold-xeon"),
        _row(variant="sys-c", machine="m3ultra", host="mac", **{"class": "whole_utterance"}),
    ]
    files, missing = pd.build(_mini_repo(tmp_path, rows))
    assert missing == []
    series = [line.split(",")[7] for line in files["cold_dev.csv"].splitlines()[1:]]
    assert series == ["m3ultra-t1-comparator-whole", "xeon8358-t1-comparator-streaming", "zen2-t1-ours-streaming",
                      "zen2-t1-ablation-streaming", "zen2-t1-comparator-whole", "zen2-t8-comparator-streaming"]
    tex = files["metrics.tex"]
    # Figure 1 draws one group (one pinned Zen 2 core) and never an ablation row
    assert "\\newcommand{\\ColdRowsCount}{2}" in tex and "\\newcommand{\\ColdRowsWhole}{1}" in tex
    for macro in ("\\ColdSysASpeedEight", "\\ColdSysASpeedXeon", "\\ColdSysCWerMthree", "\\ColdSysBWer"):
        assert f"\\newcommand{{{macro}}}" in tex, macro
    assert "sys-b" not in files["money_layout.tex"] and "sys-c-zen2-t1" in files["money_layout.tex"]


def test_a_comparator_is_drawn_only_beside_a_row_of_ours_from_its_session(tmp_path):
    rows = [
        _row(variant="ours-a", role="ours", cpu_s_per_audio_s=0.05, cpu_reps=[0.05, 0.05, 0.05]),
        _row(),                                                    # same session as ours: drawn
        _row(variant="sys-b", session="s2", host="h1"),            # its session has no row of ours: not drawn
        _row(variant="sys-c", session="s3", host="h2", wer_pct=2.0),  # s3 holds an anchor of ours, 30% off
        _row(variant="ours-a", role="ours", session="s3", host="h2", anchor=True, id="ours-a-zen2-t1-anchor-h2",
             cpu_s_per_audio_s=0.065, cpu_reps=[0.065, 0.065, 0.065]),
    ]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    plot = {line.split(",")[0]: line.split(",")[-1] for line in files["cold_dev.csv"].splitlines()[1:]}
    assert plot == {"ours-a-zen2-t1": "1", "sys-a-zen2-t1": "1", "sys-b-zen2-t1": "0", "sys-c-zen2-t1": "1",
                    "ours-a-zen2-t1-anchor-h2": "0"}
    tex = files["metrics.tex"]
    for macro in ("\\ColdRowsCount}{3}", "\\ColdRowsUndrawn}{1}", "\\ColdRowsOffFrontier}{1}",
                  "\\ColdAnchorCount}{1}", "by at most 30.0\\% in CPU cost; sessions beyond 10"):
        assert macro in tex, macro
    # the off-axis session's row is drawn, but the frontier line ignores it (at 2.0% sys-c would be on it)
    assert "ours-a-zen2-t1-anchor" not in files["money_layout.tex"] and "sys-c-zen2-t1" in files["money_layout.tex"]
    assert [ln.split(",")[1] for ln in files["money_frontier.csv"].splitlines()[1:]] == ["3.000"]


def test_lag_and_energy_sidecars_join_by_row_id(tmp_path):
    lag = {"kind": "lag", "row": "sys-a-zen2-t1", "lag_p50_ms": 310, "lag_p95_ms": 720, "lag_method": "replay",
           "lag_workload_digest": "d", "lag_receipt_sha256": "r"}
    core = {"kind": "energy", "row": "sys-a-zen2-t1", "joules_per_audio_s": 0.75, "energy_method": "core_msr_active",
            "energy_receipt_sha256": "e"}
    pkg = dict(core, joules_per_audio_s=1.25, energy_method="pkg_rapl_delta")
    head = {"kind": "energy_headline", "machine": "zen2", "threads": 1, "energy_method": "pkg_rapl_delta",
            "reason": "repeatable within 10%"}
    ours = _row(variant="sys-o", role="ours")
    files, _ = pd.build(_mini_repo(tmp_path, [_row(), ours, lag, core]))
    assert files["cold_dev.csv"].splitlines()[1].endswith(",310,720,0.750,exploratory,h0,1")
    for macro in ("\\ColdSysALagPfifty}{310}", "\\ColdSysALagPninetyFive}{720}", "\\ColdSysAWatts}{0.75}",
                  "\\ColdRowsEnergy}{1}", "\\ColdRowsLag}{1}"):
        assert macro in files["metrics.tex"], macro
    shutil.rmtree(tmp_path / "repo")
    files, _ = pd.build(_mini_repo(tmp_path, [_row(), ours, core, pkg, head]))  # two methods, one named headline
    assert "\\ColdSysAWatts}{1.25}" in files["metrics.tex"]
    for rows, msg in (([_row(), dict(lag, row="nope")], "unknown row"), ([_row(lag_p50_ms=1), lag], "given twice"),
                      ([_row(), core, core], "given twice"), ([_row(), core, pkg], "energy_headline"),
                      ([_row(), dict(core, watts_per_stream=0.5)], "same quantity")):
        shutil.rmtree(tmp_path / "repo")
        with pytest.raises(pd.PaperDataError, match=msg):
            pd.build(_mini_repo(tmp_path, rows))


def _boxes():
    """Label boxes, leaders and marker boxes of the committed Figure 1, in points inside the plot box."""
    layout = (DATA / "money_layout.tex").read_text()
    lim = {k: float(v) for k, v in re.findall(r"\\def\\Money(Xmin|Xmax|Ymax)\{([\d.]+)\}", layout)}
    labels = {m[4]: (float(m[0]), float(m[1]), float(m[5])) for m in re.findall(
        r"\\moneylabel\{([\d.-]+)\}\{([\d.-]+)\}\{(\w+)\}\{([^}]*)\}% (\S+) w=([\d.]+)", layout)}
    rows = [dict(zip(cold.CSV_HEADER, line.split(","))) for line in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    import math
    span = math.log10(lim["Xmax"] / lim["Xmin"])
    marks, boxes = {}, {}
    for r in rows:
        if r["id"] not in labels:
            continue
        px = money.AXIS_W * math.log10(float(r["speed"]) / lim["Xmin"]) / span
        py = money.AXIS_H * float(r["wer"]) / lim["Ymax"]
        rad = money.MARK_R[r["role"]]
        marks[r["id"]] = (px - rad, py - rad, px + rad, py + rad)
        cx, cy, w = labels[r["id"]]
        boxes[r["id"]] = (cx - w / 2, cy - money.LINE_PT / 2, cx + w / 2, cy + money.LINE_PT / 2)
    return rows, boxes, marks


def test_money_labels_do_not_collide():
    rows, boxes, marks = _boxes()
    plotted = [r for r in rows if r["machine"] == "zen2" and r["threads"] == "1" and r["role"] != "ablation"]
    assert set(boxes) == {r["id"] for r in plotted} and len(boxes) >= 8

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    ids = sorted(boxes)
    for i, a in enumerate(ids):
        assert 0 <= boxes[a][0] and boxes[a][2] <= money.AXIS_W and 0 <= boxes[a][1] and boxes[a][3] <= money.AXIS_H, a
        for b in ids[i + 1:]:
            assert not hit(boxes[a], boxes[b]), f"labels {a} and {b} overlap"
        for b in ids:
            assert not hit(boxes[a], marks[b]), f"label {a} covers the marker of {b}"


def test_money_placement_never_ships_a_collision():
    pts = [{"id": f"p{i:02d}", "px": 100.0 + 0.5 * (i % 5), "py": 70.0 + 0.5 * (i // 5), "r": 2.3,
            "w": money.text_width("a fairly long label"), "label": "a fairly long label"} for i in range(40)]
    with pytest.raises(pd.PaperDataError, match="cannot be placed"):
        money.place(pts, [], [])


def test_label_width_estimates_cover_the_typeset_labels(tmp_path):
    """The layout trusts money.text_width; TeX must not set any label wider than that estimate."""
    assert shutil.which("pdflatex"), "pdflatex is required for the paper figures"
    layout = (DATA / "money_layout.tex").read_text()
    labels = re.findall(r"\\moneylabel\{[\d.-]+\}\{[\d.-]+\}\{(\w+)\}\{([^}]*)\}% \S+ w=([\d.]+)", layout)
    assert labels
    body = "".join(f"\\setbox0\\hbox{{\\scriptsize{'' if role != 'ours' else chr(92) + 'bfseries'} {text}}}"
                   f"\\typeout{{LABELWIDTH {i} \\the\\wd0}}\n" for i, (role, text, _) in enumerate(labels))
    tex = tmp_path / "labels.tex"
    tex.write_text("\\documentclass[conference]{IEEEtran}\\usepackage[T1]{fontenc}\\usepackage{lmodern}\n"
                   "\\begin{document}\n" + body + "x\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmp_path),
                    str(tex)], capture_output=True, text=True, timeout=120, check=True)
    real = dict(re.findall(r"LABELWIDTH (\d+) ([\d.]+)pt", (tmp_path / "labels.log").read_text()))
    for i, (_, text, est) in enumerate(labels):
        typeset = float(real[str(i)]) + 1.2  # the node's inner sep on both sides
        assert typeset <= float(est), f"{text!r}: typeset {typeset:.1f} pt exceeds the estimate {est} pt"
        assert float(est) <= 1.3 * typeset + 2, f"{text!r}: estimate {est} pt is far too loose ({typeset:.1f} pt)"
