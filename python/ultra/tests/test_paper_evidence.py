"""Measured comparison rows (schema v2), their grouping, and the Figure 1 label layout."""

import json
import re
import shutil
import subprocess

import pytest

from tools import paper_data as pd
from tools.paper_data import cold, labels as labels_mod, money

DATA = pd.REPO / pd.DATA


def _mini_repo(tmp_path, rows):
    root = tmp_path / "repo"
    shutil.copytree(pd.REPO / pd.DATA, root / pd.DATA)
    shutil.copytree(pd.REPO / "research" / "curves", root / "research" / "curves")
    ev = root / pd.DATA / "evidence"
    shutil.rmtree(ev, ignore_errors=True)
    ev.mkdir()
    (root / pd.DATA / "systems.json").unlink()  # names and the headline row belong to the real variants
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
        "2.00,4.50,150.0,70.0,,,,exploratory,h0,1,,,,,0,full",
        "sys-b-zen2-t1,sys-b,B,ours,streaming,zen2,1,zen2-t1-ours-streaming,0.05000,20.000,4.000,9.00,2.00,4.50,"
        "150.0,70.0,,,,exploratory,h0,1,,,,,0,full",
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
    ({"final_latency_p50_ms": 5000, "final_latency_p95_ms": 9000}, "whole_utterance rows"),
    ({"class": "whole_utterance", "final_latency_p50_ms": 5000}, "all-or-none"),
    ({"lag_design_p50_ms": 250, "lag_design_p95_ms": 400}, "all-or-none"),  # no proof fields
    ({"lag_p50_ms": 300, "lag_method": "m", "lag_workload_digest": "d", "lag_receipt_sha256": "r"}, "all-or-none"),
    ({"timing_panel": "p73"}, "timing_workload_digest"),
    ({"timing_panel": "half"}, "bad timing_panel"),
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
    for macro in ("\\ColdSysASpeedEightWorkers", "\\ColdSysASpeedXeon", "\\ColdSysCWerMthree", "\\ColdSysBWer"):
        assert f"\\newcommand{{{macro}}}" in tex, macro
    assert "sys-b" not in files["money_layout.tex"] and "sys-c-zen2-t1" in files["money_layout.tex"]


def test_a_comparator_is_drawn_only_beside_a_row_of_ours_from_its_session(tmp_path):
    rows = [
        _row(variant="ours-a", role="ours", cpu_s_per_audio_s=0.05, cpu_reps=[0.05, 0.05, 0.05]),
        _row(),                                                    # same session as ours: drawn
        _row(variant="sys-b", session="s2", host="h1"),            # its session has no row of ours: not drawn
        _row(variant="sys-c", session="s3", host="h2", wer_clean_pct=1.0),  # s3 holds an anchor of ours, 30% off
        _row(variant="ours-a", role="ours", session="s3", host="h2", anchor=True, id="ours-a-zen2-t1-anchor-h2",
             cpu_s_per_audio_s=0.065, cpu_reps=[0.065, 0.065, 0.065]),
    ]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    plot = {line.split(",")[0]: line.split(",")[21] for line in files["cold_dev.csv"].splitlines()[1:]}
    assert plot == {"ours-a-zen2-t1": "1", "sys-a-zen2-t1": "1", "sys-b-zen2-t1": "0", "sys-c-zen2-t1": "1",
                    "ours-a-zen2-t1-anchor-h2": "0"}
    tex = files["metrics.tex"]
    for macro in ("\\ColdRowsCount}{3}", "\\ColdRowsUndrawn}{1}", "\\ColdRowsOffFrontier}{1}",
                  "\\ColdAnchorCount}{1}", "its CPU cost is 1.30 times that of the row printed here"):
        assert macro in tex, macro
    # the off-axis session's row is drawn, but the frontier line ignores it (at 1.0% sys-c would be on it)
    assert "ours-a-zen2-t1-anchor" not in files["money_layout.tex"] and "sys-c-zen2-t1" in files["money_layout.tex"]
    assert [ln.split(",")[1] for ln in files["money_frontier.csv"].splitlines()[1:]] == ["3.000"]


def test_each_system_is_drawn_from_its_quiet_host_row_when_one_exists(tmp_path):
    def loaded(variant, cpu, **kw):  # the shared host: slower, one session
        return _row(variant=variant, host="xbabe3", session="s-loaded", id=f"{variant}-zen2-t1-xbabe3",
                    cpu_s_per_audio_s=cpu, cpu_reps=[cpu, cpu, cpu], **kw)

    def quiet(variant, cpu, **kw):
        return _row(variant=variant, host="xbabe0", session="s-quiet", cpu_s_per_audio_s=cpu, cpu_reps=[cpu] * 3, **kw)

    interim = [loaded("ours-a", 0.06, role="ours"), loaded("sys-a", 0.14), loaded("sys-b", 0.3, wer_clean_pct=1.0)]
    files, _ = pd.build(_mini_repo(tmp_path, interim))
    tex = files["metrics.tex"]
    assert "\\ColdRowsCount}{3}" in tex and "\\ColdRowsInterim}{3}" in tex and "\\ColdRowsFaded}{0}" in tex
    assert "Interim: every point was measured on host H0" in tex
    shutil.rmtree(tmp_path / "repo")
    files, _ = pd.build(_mini_repo(tmp_path, interim + [quiet("ours-a", 0.03, role="ours"), quiet("sys-a", 0.08)]))
    cols = {line.split(",")[0]: line.split(",") for line in files["cold_dev.csv"].splitlines()[1:]}
    # quiet-host rows are drawn; the loaded-host rows of those variants become cross-checks; a system with no
    # quiet-host row yet is still drawn from its loaded-host row, lighter, and stays off the frontier line
    assert {k: (c[21], c[26]) for k, c in cols.items()} == {
        "ours-a-zen2-t1": ("1", "0"), "sys-a-zen2-t1": ("1", "0"), "ours-a-zen2-t1-xbabe3": ("0", "0"),
        "sys-a-zen2-t1-xbabe3": ("0", "0"), "sys-b-zen2-t1-xbabe3": ("1", "1")}
    tex = files["metrics.tex"]
    for text in ("\\ColdRowsCount}{3}", "\\ColdAnchorCount}{2}", "\\ColdRowsInterim}{1}", "\\ColdRowsFaded}{1}",
                 "\\ColdOursASpeed}{33.3}", "2 systems, measured on both hosts, cost 1.75 to 2.00 times as much CPU on the shared host H0 as on the quiet host H1",
                 "1 of the 3 points is still interim (lighter marks): measured on host H0"):
        assert text in tex, text
    assert [ln.split(",")[1] for ln in files["money_frontier.csv"].splitlines()[1:]] == ["3.000"]  # sys-b is not on it: quiet rows only


def test_a_variant_measured_in_two_sessions_on_one_host_keeps_the_earlier_row(tmp_path):
    rows = [_row(role="ours", host="xbabe0", session="2026-11-22 a"),
            _row(role="ours", host="xbabe0", session="2026-11-22 b", id="sys-a-zen2-t1-session-b",
                 cpu_s_per_audio_s=0.21, cpu_reps=[0.21, 0.21, 0.21])]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    plot = {ln.split(",")[0]: ln.split(",")[21] for ln in files["cold_dev.csv"].splitlines()[1:]}
    assert plot == {"sys-a-zen2-t1": "1", "sys-a-zen2-t1-session-b": "0"}
    assert "its CPU cost is 1.05 times that of the row printed here" in files["metrics.tex"]


def test_watts_are_called_out_beside_the_systems_that_have_an_energy_line(tmp_path):
    energy = {"kind": "energy", "row": "sys-o-zen2-t1", "host": "h0", "joules_per_audio_s": 0.343,
              "energy_method": "core_msr_active", "energy_receipt_sha256": "e"}
    rows = [_row(), _row(variant="sys-o", role="ours", system="Ours", cpu_s_per_audio_s=0.04, cpu_reps=[0.04] * 3), energy]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    layout = files["money_layout.tex"]
    assert "{Ours\\,$\\cdot$\\,0.343\\,W}" in layout and "{A}% sys-a-zen2-t1" in layout
    assert "\\def\\MoneyWattsOn{0}" in layout  # no axis: one system has no value yet
    assert "Active core W per stream (core energy registers, same run), not device power." in files["metrics.tex"]
    shutil.rmtree(tmp_path / "repo")
    files, _ = pd.build(_mini_repo(tmp_path, rows + [dict(energy, row="sys-a-zen2-t1", joules_per_audio_s=1.7)]))
    # every system has a value and 0.343 x 25 = 8.6 W and 1.7 x 5 = 8.5 W per busy core agree: a watts axis
    assert "\\def\\MoneyWattsOn{1}" in files["money_layout.tex"] and "$\\cdot$" not in files["money_layout.tex"]
    assert "Top axis: watts one live stream costs" in files["metrics.tex"]


def test_a_row_measured_on_a_busy_quiet_host_is_a_cross_check_not_a_primary_row(tmp_path):
    def on(host, cpu, **kw):
        return _row(role="ours", host=host, cpu_s_per_audio_s=cpu, cpu_reps=[cpu] * 3, **kw)

    quiet = on("xbabe0", 0.035, session="quiet", n_busy_cores_reps=[1, 1, 2])
    busy = on("xbabe0", 0.055, session="busy", id="sys-a-zen2-t1-busy", n_busy_cores_reps=[1, 16, 16])
    files, _ = pd.build(_mini_repo(tmp_path, [busy, quiet]))
    plot = {ln.split(",")[0]: ln.split(",")[21] for ln in files["cold_dev.csv"].splitlines()[1:]}
    assert plot == {"sys-a-zen2-t1": "1", "sys-a-zen2-t1-busy": "0"}
    assert "\\ColdSysASpeed}{28.6}" in files["metrics.tex"]  # the quiet row's speed, whatever the file order
    shutil.rmtree(tmp_path / "repo")
    files, _ = pd.build(_mini_repo(tmp_path, [busy]))  # only a loaded row: it is kept, but nothing is drawn
    assert "\\ColdRowsCount}{0}" in files["metrics.tex"] and "\\ColdSysASpeed}" not in files["metrics.tex"]


def test_a_slow_system_timed_on_the_subset_is_drawn_and_counted(tmp_path):
    slow = _row(variant="sys-s", timing_panel="p73", timing_workload_digest="p73-digest", wer_source="full panel on h2",
                cpu_s_per_audio_s=2.5, cpu_reps=[2.5] * 3)
    design = {"kind": "lag", "row": "sys-s-zen2-t1", "lag_design_p50_ms": 600, "lag_design_p95_ms": 900,
              "lag_method": "replay", "lag_workload_digest": "d", "lag_receipt_sha256": "r"}  # no paced lag yet
    files, _ = pd.build(_mini_repo(tmp_path, [_row(role="ours"), slow, design]))
    line = next(ln for ln in files["cold_dev.csv"].splitlines() if ln.startswith("sys-s-zen2-t1,"))
    assert line.endswith(",1,600,900,,,0,p73")
    tex = files["metrics.tex"]
    assert "\\ColdRowsSubset}{1}" in tex and "\\ColdSysSLagDesignPfifty}{600}" in tex and "ColdSysSLagPfifty" not in tex
    assert "1 slow system is timed on a 73-utterance subset of the panel (A)" in tex


def test_an_energy_headline_without_a_group_names_the_method_of_every_group(tmp_path):
    ours = _row(variant="sys-o", role="ours")
    line = {"kind": "energy", "row": "sys-a-zen2-t1", "host": "h0", "energy_receipt_sha256": "e"}
    rows = [_row(), ours, dict(line, joules_per_audio_s=0.6, energy_method="core_msr_active"),
            dict(line, joules_per_audio_s=0.5, energy_method="core_msr_dynamic"),
            {"kind": "energy_headline", "energy_method": "core_msr_active"}]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    assert "\\ColdSysAWatts}{0.600}" in files["metrics.tex"]


def test_quiet_host_rows_need_no_shared_session_and_eight_core_rows_count_their_own_cores(tmp_path):
    ours = _row(variant="sys-o", role="ours", host="xbabe0", session="entry-2", n_busy_cores_reps=[2, 1, 1])
    other = _row(host="xbabe0", session="entry-3", n_busy_cores_reps=[2, 2, 2])  # another entry of the same queue
    eight = _row(threads=8, parallel="workers", host="xbabe0", session="entry-20", n_busy_cores_reps=[9, 9, 10])
    ours8 = _row(variant="sys-o", role="ours", threads=8, parallel="workers", host="xbabe0", session="entry-20",
                 n_busy_cores_reps=[8, 8, 9])
    files, _ = pd.build(_mini_repo(tmp_path, [ours, other, eight, ours8]))
    plot = {ln.split(",")[0]: ln.split(",")[21] for ln in files["cold_dev.csv"].splitlines()[1:]}
    assert plot["sys-a-zen2-t1"] == "1"  # drawn: the quiet host's protocol makes its rows comparable
    assert plot["sys-a-zen2-t8"] == "1" and plot["sys-o-zen2-t8"] == "1"  # 8 busy cores of its own: not loaded


def test_eight_cores_used_three_ways_are_three_rows(tmp_path):
    def eight(parallel, wall, **kw):
        return _row(threads=8, parallel=parallel, id=f"sys-a-zen2-t8-{parallel}", wall_s_per_audio_s=wall,
                    wall_reps=[wall] * 3, **kw)

    rows = [_row(), eight("intra_op", 0.05), eight("workers", 0.03), eight("processes", 0.04)]
    files, _ = pd.build(_mini_repo(tmp_path, rows))
    tex = files["metrics.tex"]
    for macro in ("\\ColdSysARtfxEight}{20.0}", "\\ColdSysARtfxEightWorkers}{33.3}", "\\ColdSysARtfxEightProcs}{25.0}"):
        assert macro in tex, macro
    shutil.rmtree(tmp_path / "repo")
    with pytest.raises(pd.PaperDataError, match="bad parallel"):
        pd.build(_mini_repo(tmp_path, [_row(), eight("threads", 0.05)]))


def test_lag_and_energy_sidecars_join_by_row_id(tmp_path):
    lag = {"kind": "lag", "row": "sys-a-zen2-t1", "lag_p50_ms": 310, "lag_p95_ms": 720, "lag_method": "replay",
           "lag_workload_digest": "d", "lag_receipt_sha256": "r"}
    core = {"kind": "energy", "row": "sys-a-zen2-t1", "host": "h0", "joules_per_audio_s": 0.75,
            "energy_method": "core_msr_active", "energy_receipt_sha256": "e"}
    pkg = dict(core, joules_per_audio_s=1.25, energy_method="pkg_rapl_delta")
    head = {"kind": "energy_headline", "machine": "zen2", "threads": 1, "energy_method": "pkg_rapl_delta",
            "reason": "repeatable within 10%"}
    ours = _row(variant="sys-o", role="ours")
    whole = _row(variant="sys-w", **{"class": "whole_utterance"})
    final = {"kind": "lag", "row": "sys-w-zen2-t1", "final_latency_p50_ms": 5100, "final_latency_p95_ms": 9900,
             "lag_method": "replay", "lag_workload_digest": "d", "lag_receipt_sha256": "r"}
    design = dict(lag, lag_design_p50_ms=250, lag_design_p95_ms=480)
    files, _ = pd.build(_mini_repo(tmp_path, [_row(), ours, whole, design, final, core]))
    lines = {ln.split(",")[0]: ln for ln in files["cold_dev.csv"].splitlines()[1:]}
    assert lines["sys-a-zen2-t1"].endswith(",310,720,0.750,exploratory,h0,1,250,480,,,0,full")
    assert lines["sys-w-zen2-t1"].endswith(",,,,exploratory,h0,1,,,5100,9900,0,full")
    for macro in ("\\ColdSysALagPfifty}{310}", "\\ColdSysALagPninetyFive}{720}", "\\ColdSysAWatts}{0.750}",
                  "\\ColdSysALagDesignPfifty}{250}", "\\ColdSysWFinalLatencyPninetyFive}{9900}",
                  "\\ColdRowsEnergy}{1}", "\\ColdRowsLag}{1}"):
        assert macro in files["metrics.tex"], macro
    shutil.rmtree(tmp_path / "repo")
    files, _ = pd.build(_mini_repo(tmp_path, [_row(), ours, core, pkg, head]))  # two methods, one named headline
    assert "\\ColdSysAWatts}{1.250}" in files["metrics.tex"]
    shutil.rmtree(tmp_path / "repo")
    shared = dict(core, joules_per_audio_s=0.9, energy_method="core_msr_contended")  # a separate run, shared core
    files, _ = pd.build(_mini_repo(tmp_path, [_row(), ours, core, shared, dict(head, energy_method="core_msr_active")]))
    assert "\\ColdSysAWatts}{0.750}" in files["metrics.tex"]
    for rows, msg in (([_row(), dict(lag, row="nope")], "unknown row"), ([_row(lag_p50_ms=1), lag], "given twice"),
                      ([_row(), core, core], "given twice"), ([_row(), core, pkg], "energy_headline"),
                      ([_row(), dict(core, watts_per_stream=0.5)], "same quantity"),
                      ([_row(), dict(core, host="another-host")], "the row's own host")):
        shutil.rmtree(tmp_path / "repo")
        with pytest.raises(pd.PaperDataError, match=msg):
            pd.build(_mini_repo(tmp_path, rows))


def _boxes():
    """Label boxes and marker boxes of the committed Figure 1, in points inside the plot box."""
    layout = (DATA / "money_layout.tex").read_text()
    lim = {k: float(v) for k, v in re.findall(r"\\def\\Money(Xmin|Xmax|Ymax)\{([\d.]+)\}", layout)}
    labels = {m[2]: (float(m[0]), float(m[1]), float(m[3])) for m in re.findall(
        r"\\moneylabel\{([\d.-]+)\}\{([\d.-]+)\}\{\w+\}\{[^}]*\}% (\S+) w=([\d.]+)", layout)}
    rows = [dict(zip(cold.CSV_HEADER, line.split(","))) for line in (DATA / "cold_dev.csv").read_text().splitlines()[1:]]
    assert lim["Xmin"] == 0
    span = lim["Xmax"] - lim["Xmin"]
    marks, boxes = {}, {}
    for r in rows:
        if r["id"] not in labels:
            continue
        px = money.AXIS_W * (float(r["speed"]) - lim["Xmin"]) / span
        py = money.AXIS_H * float(r["wer"]) / lim["Ymax"]
        rad = money.MARK_R[r["role"]]
        marks[r["id"]] = (px - rad, py - rad, px + rad, py + rad)
        cx, cy, w = labels[r["id"]]
        boxes[r["id"]] = (cx - w / 2, cy - labels_mod.LINE_PT / 2, cx + w / 2, cy + labels_mod.LINE_PT / 2)
    return rows, boxes, marks


def test_money_labels_do_not_collide():
    rows, boxes, marks = _boxes()
    plotted = [r for r in rows if r["machine"] == "zen2" and r["threads"] == "1" and r["plot"] == "1"]
    assert set(boxes) == {r["id"] for r in plotted} and len(boxes) == 14

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    ids = sorted(boxes)
    for i, a in enumerate(ids):
        assert 0 <= boxes[a][0] and boxes[a][2] <= money.AXIS_W and 0 <= boxes[a][1] and boxes[a][3] <= money.AXIS_H, a
        assert not hit(boxes[a], money.BETTER), f"label {a} covers the 'better' arrow"
        for b in ids[i + 1:]:
            assert not hit(boxes[a], boxes[b]), f"labels {a} and {b} overlap"
        for b in ids:
            assert not hit(boxes[a], marks[b]), f"label {a} covers the marker of {b}"


def test_money_speed_comparison_uses_true_positions():
    layout = (DATA / "money_layout.tex").read_text()
    ticks = re.search(r"\\def\\MoneyXticks\{([\d,]+)\}", layout)
    assert ticks
    assert [int(tick) for tick in ticks[1].split(",")] == list(range(0, 36, 5))
    _, _, marks = _boxes()
    arrow = re.search(r"\\def\\MoneySpeedGap\{.*?xshift=([\d.]+)pt.*?xshift=([\d.]+)pt.*?2\.0\$\\times\$ as fast as Zipformer INT8", layout)
    assert arrow
    for x, row in zip(arrow.groups(), ("r0-int8-zen2-t1", "rc1-ema-live-zen2-t1")):
        left, _, right, _ = marks[row]
        assert float(x) == pytest.approx((left + right) / 2, abs=0.01)


def test_money_placement_never_ships_a_collision():
    """Forty markers on one spot leave no room even for numbers: the build must refuse, not overlap."""
    spec = tuple((f"p{i:02d}", "a fairly long label", False, 100.0 + 0.5 * (i % 5), 70.0 + 0.5 * (i // 5), 2.3)
                 for i in range(40))
    with pytest.raises(pd.PaperDataError, match="cannot be labelled"):
        money._layout(spec, ())


def test_label_width_estimates_cover_the_typeset_labels(tmp_path):
    """The layout trusts money.text_width; TeX must not set any label wider than that estimate."""
    assert shutil.which("pdflatex"), "pdflatex is required for the paper figures"
    layout = (DATA / "money_layout.tex").read_text()
    labels = re.findall(r"\\moneylabel\{[\d.-]+\}\{[\d.-]+\}\{(\w+)\}\{([^}]*)\}% \S+ w=([\d.]+)", layout)
    assert labels
    body = "".join(f"\\setbox0\\hbox{{\\scriptsize{'' if role == 'comparator' else chr(92) + 'bfseries'} {line}}}"
                   f"\\typeout{{LABELWIDTH {i}-{n} \\the\\wd0}}\n" for i, (role, text, _) in enumerate(labels)
                   for n, line in enumerate(text.split("\\\\")))  # a two-line label: each line on its own
    tex = tmp_path / "labels.tex"
    tex.write_text("\\documentclass[conference]{IEEEtran}\\usepackage[T1]{fontenc}\\usepackage{lmodern}\n"
                   "\\begin{document}\n" + body + "x\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmp_path),
                    str(tex)], capture_output=True, text=True, timeout=120, check=True)
    real: dict[str, float] = {}
    for key, width in re.findall(r"LABELWIDTH (\d+)-\d+ ([\d.]+)pt", (tmp_path / "labels.log").read_text()):
        real[key] = max(real.get(key, 0.0), float(width))
    for i, (_, text, est) in enumerate(labels):
        typeset = real[str(i)] + 1.2  # the node's inner sep on both sides
        assert typeset <= float(est), f"{text!r}: typeset {typeset:.1f} pt exceeds the estimate {est} pt"
        assert float(est) <= 1.3 * typeset + 2, f"{text!r}: estimate {est} pt is far too loose ({typeset:.1f} pt)"


def test_a_crowded_figure_falls_back_to_numbers_and_a_key():
    """Twenty systems cannot all be labelled directly in 197 x 140 pt: the rest become numbers with a key."""
    spec = tuple((f"p{i:02d}", f"A recogniser name {i:02d}", False, 60.0 + 9.0 * (i % 10), 30.0 + 7.0 * (i // 10), 2.3)
                 for i in range(20))
    placed = money._layout(spec, ())
    assert placed == money._layout(spec, ()), "the layout must be repeatable"
    numbered = [number for _, number, _, _ in placed if number is not None]
    assert numbered and sorted(numbered) == list(range(1, len(numbered) + 1))
    boxes = [box for _, _, box, _ in placed]

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for a in range(len(boxes)):
        assert not any(hit(boxes[a], boxes[b]) for b in range(a + 1, len(boxes))), a
        assert not any(hit(boxes[a], (px - r, py - r, px + r, py + r)) for _, _, _, px, py, r in spec), a


# Figure 1 as refresh 4 drew it: its Zipformer label sat above Moonshine Tiny's point, nearer that point than its own
FIG1_R4 = (("a1-20000-ema-int8-zen2-t1", "ULTRA-S (ours)", True, 162.56, 57.75, 3.3),
           ("r0-int8-zen2-t1", "Zipformer (baseline) - 0.59 W", False, 136.58, 37.62, 2.3),
           ("moonshine-tiny-zen2-t1", "Moonshine Tiny - 1.08 W", False, 113.56, 81.08, 2.3),
           ("moonshine-small-zen2-t1-xbabe3", "Moonshine Small", False, 56.66, 49.29, 2.3),
           ("nemotron-en-560-zen2-t1-xbabe3", "Nemotron 560 ms", False, 60.47, 32.37, 2.3),
           ("nemotron-en-160-zen2-t1-xbabe3", "Nemotron 160 ms", False, 38.14, 36.02, 2.3),
           ("nemotron-35-zen2-t1-xbabe3", "Nemotron 3.5", False, 21.21, 51.33, 2.3),
           ("parakeet-tdt-v3-int8-zen2-t1", "Parakeet TDT v3 - 1.71 W", False, 100.32, 30.19, 2.3))
FIG1_R4_LINES = (((136.58, 37.62), (136.58, 57.75)), ((136.58, 57.75), (162.56, 57.75)))


def test_every_label_sits_nearest_the_point_it_names():
    """A label (or the number that replaces it) is nearer its own point than any other point, by labels.NEAREST,
    and never lies on the frontier line, so no reader can pair a name with the wrong point."""
    money._layout.cache_clear()
    placed = money._layout(FIG1_R4, FIG1_R4_LINES)
    where = {i: (px, py) for i, _, _, px, py, _ in FIG1_R4}
    for i, _number, box, _leader in placed:
        own = labels_mod.gap(box, *where[i])
        for j, (x, y) in where.items():
            if j != i:
                assert labels_mod.gap(box, x, y) >= own + labels_mod.NEAREST, f"{i}'s label is nearer {j}"
        assert not any(labels_mod.seg_hits_box(seg, box) for seg in FIG1_R4_LINES), f"{i}'s label is on the frontier"


# Figure 1 as refresh 6 draws it: in refresh 5 the onnx-asr Parakeet label ran under the Zipformer point at a corner
FIG1_R6 = (("a1-30000-ema-int8-zen2-t1", "ULTRA-S (ours) - 0.35 W", True, 163.76, 63.23, 3.3),
           ("r0-int8-zen2-t1", "Zipformer - 0.59 W", False, 136.58, 41.72, 2.3),
           ("moonshine-tiny-zen2-t1", "Moonshine Tiny - 1.08 W", False, 113.56, 89.91, 2.3),
           ("moonshine-small-zen2-t1-xbabe3", "Moonshine Small", False, 56.66, 54.66, 2.3),
           ("nemotron-en-560-zen2-t1-xbabe3", "Nemotron 560 ms", False, 60.47, 35.9, 2.3),
           ("nemotron-en-160-zen2-t1-xbabe3", "Nemotron 160 ms", False, 38.14, 39.94, 2.3),
           ("nemotron-35-zen2-t1-xbabe3", "Nemotron 3.5", False, 21.21, 56.92, 2.3),
           ("parakeet-tdt-v3-int8-zen2-t1", "Parakeet v3 · sherpa-onnx - 1.71 W", False, 100.32, 33.47, 2.3),
           ("parakeet-v3-onnxasr-int8-zen2-t1", "Parakeet v3 · onnx-asr", False, 105.3, 34.6, 2.3),
           ("vosk-small-zen2-t1", "Vosk Small", False, 106.45, 147.64, 2.3),
           ("vosk-large-zen2-t1", "Vosk Large", False, 85.74, 82.15, 2.3),
           ("whispercpp-stream-tiny-zen2-t1", "Whisper.cpp Tiny", False, 24.47, 199.55, 2.3))
FIG1_R6_LINES = (((136.58, 41.72), (136.58, 63.23)), ((136.58, 63.23), (163.76, 63.23)))


def test_no_label_runs_under_a_frontier_system_and_ours_keeps_its_name():
    """Another system's label never sits right under or over a frontier system's point, a long label avoids a
    marker's corner when it can, and our own label is never replaced by a number."""
    money._layout.cache_clear()
    placed = {i: (number, box) for i, number, box, _ in money._layout(FIG1_R6, FIG1_R6_LINES)}
    frontier = [(px, py) for i, _, _, px, py, _ in FIG1_R6 if i in ("r0-int8-zen2-t1", "a1-30000-ema-int8-zen2-t1")]
    for i, (number, box) in placed.items():
        if i == "a1-30000-ema-int8-zen2-t1":
            assert number is None, "our own label was replaced by a number"
            continue
        for fx, fy in frontier:
            near = box[0] <= fx <= box[2] and not box[1] <= fy <= box[3]
            assert not (near and min(abs(fy - box[1]), abs(fy - box[3])) < 2 * labels_mod.LINE_PT), f"{i} runs under a frontier system"
    assert placed["r0-int8-zen2-t1"][0] is None, "the Zipformer, the frontier's other system, keeps its name"


def test_hero_requires_matched_full_panel_cold_repetitions():
    variants = ("rc1-ema-live", "rc1-ema-final", "r0-int8", "parakeet-v3-onnxasr-int8")
    rows = [
        _row(variant=variant, host="xbabe0", plot=True, interim=False, timing_panel="full")
        for variant in variants
    ]
    assert set(money.hero_rows(rows)) == {"Live", "Final", "Zip", "Parakeet"}
    for changes in (
        {"timing_panel": "p73"},
        {"interim": True},
        {"workload_digest": "unmatched panel"},
        {"cpu_reps": [0.2]},
    ):
        assert money.hero_rows([*rows[:-1], rows[-1] | changes]) == {}
