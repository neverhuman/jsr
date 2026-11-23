"""Energy meter arithmetic: counter wrap-around, unit conversion, statistics, and one end-to-end run."""

import json
import os
import sys
import time

import pytest

from tools import energy_bench as eb
from tools import energy_report as er

UNIT_MSR = 0xA1003  # the Zen 2 hosts: energy exponent 16
UNIT_J = 2.0 ** -16
PKG_MAX_UJ = 65_532_610_987


def test_cpu_list_parsing_keeps_order_and_refuses_nonsense():
    assert eb.parse_cpu_list("16-19,31") == [16, 17, 18, 19, 31]
    assert eb.parse_cpu_list("8") == [8]
    for bad in ("", "3-1", "4,4", "2-3,3"):
        with pytest.raises(ValueError):
            eb.parse_cpu_list(bad)


def test_energy_unit_comes_from_bits_12_to_8():
    assert eb.energy_unit_joules(UNIT_MSR) == UNIT_J
    assert eb.energy_unit_joules(UNIT_MSR) * 1e6 == pytest.approx(15.2588, abs=1e-4)
    assert eb.energy_unit_joules(0x0E00) == 2.0 ** -14


def test_wrap_delta_handles_the_32_bit_counter_and_the_powercap_range():
    assert eb.wrap_delta(10, 250, eb.COUNTER_MODULUS) == 240
    assert eb.wrap_delta(eb.COUNTER_MODULUS - 5, 7, eb.COUNTER_MODULUS) == 12
    assert eb.wrap_delta(PKG_MAX_UJ - 1_000_000, 2_500_000, PKG_MAX_UJ) == 3_500_000
    assert eb.wrap_delta(123, 123, eb.COUNTER_MODULUS) == 0
    with pytest.raises(ValueError):
        eb.wrap_delta(5, eb.COUNTER_MODULUS + 1, eb.COUNTER_MODULUS)


def test_accumulate_survives_several_wraps_and_converts_units():
    step = eb.COUNTER_MODULUS // 3 + 11
    readings = [(7 + i * step) % eb.COUNTER_MODULUS for i in range(10)]
    joules = eb.accumulate(readings, eb.COUNTER_MODULUS, UNIT_J)
    assert joules[0] == 0.0
    assert joules[-1] == pytest.approx(9 * step * UNIT_J)
    micro = eb.accumulate([PKG_MAX_UJ - 400_000, 600_000], PKG_MAX_UJ, 1e-6)
    assert micro == pytest.approx([0.0, 1.0])


def test_summarise_reports_median_and_spread():
    stats = eb.summarise([0.30, 0.31, 0.33])
    assert stats["median"] == 0.31
    assert (stats["min"], stats["max"], stats["n"]) == (0.30, 0.33, 3)
    assert stats["spread"] == pytest.approx(0.03)
    assert stats["spread_pct"] == pytest.approx(100 * 0.03 / 0.31)
    assert stats["stdev"] == pytest.approx(0.0152753, rel=1e-4)
    assert eb.summarise([4.0])["stdev"] is None
    assert eb.summarise([None, None]) is None


def test_linear_fit_recovers_slope_intercept_and_flags_a_bent_line():
    line = er.linear_fit([1, 2, 4, 8], [7.0, 12.0, 22.0, 42.0])
    assert line["slope"] == pytest.approx(5.0)
    assert line["intercept"] == pytest.approx(2.0)
    assert line["r2"] == pytest.approx(1.0)
    assert line["max_rel_residual"] == pytest.approx(0.0, abs=1e-12)
    bent = er.linear_fit([1, 2, 4, 8], [7.0, 12.0, 22.0, 30.0])
    assert bent["r2"] < 0.99
    assert bent["max_rel_residual"] > 0.05
    with pytest.raises(ValueError):
        er.linear_fit([2, 2], [1.0, 3.0])


def test_view_metrics_subtracts_idle_and_divides_by_audio_and_cpu():
    # 100 s run at 6 W over a 1 W idle floor, 2,000 s of audio, 98 CPU-seconds.
    view = eb.view_metrics(600.0, 100.0, 120.0, 120.0, 2000.0, 98.0)
    assert view["run_watts"] == 6.0
    assert view["idle_watts"] == 1.0
    assert view["net_watts"] == 5.0
    assert view["net_joules"] == 500.0
    assert view["joules_per_audio_s"] == 0.25       # = watts per live stream
    assert view["gross_joules_per_audio_s"] == 0.30
    assert view["joules_per_cpu_s"] == pytest.approx(500.0 / 98.0)
    assert eb.view_metrics(600.0, 100.0, 120.0, 120.0, None, None)["joules_per_audio_s"] is None


def synthetic_samples(schedule, idle_watts, run_watts, package_idle, package_run, start_counts):
    """Counter samples for a timeline of (time, label, running-after-this-sample) points."""
    samples, core_j, pkg_j = [], {16: 0.0, 17: 0.0}, 0.0
    previous_time, running = schedule[0][0], False
    for moment, label, running_after in schedule:
        dt = moment - previous_time
        for cpu in core_j:
            core_j[cpu] += (run_watts if running else idle_watts) * dt
        pkg_j += (package_run if running else package_idle) * dt
        samples.append({
            "t": float(moment), "label": label,
            "core": {str(c): (start_counts + round(j / UNIT_J)) % eb.COUNTER_MODULUS for c, j in core_j.items()},
            "pkg": (start_counts + round(pkg_j / UNIT_J)) % eb.COUNTER_MODULUS,
            "pkg_uj": (PKG_MAX_UJ - 5_000_000 + round(pkg_j * 1e6)) % PKG_MAX_UJ,
            "all": {"16": 0, "17": 0, "40": round(moment * 2.0 / UNIT_J) % eb.COUNTER_MODULUS} if label else None,
        })
        previous_time, running = moment, running_after
    return samples


def one_rep_schedule():
    points = {5: ("idle1_start", False), 65: ("run1_start", True), 165: ("run1_end", False),
              170: ("idle2_start", False), 230: ("idle2_end", False)}
    return [(t, *points.get(t, (None, 65 <= t < 165))) for t in range(0, 236)]


def test_trace_gives_exact_phase_energy_across_a_counter_wrap():
    header = {"energy_unit_j": UNIT_J, "cpus": [16, 17], "pkg_max_uj": PKG_MAX_UJ}
    # Start 200 J below the wrap: the package counter wraps during the run, the cores do not.
    samples = synthetic_samples(one_rep_schedule(), 0.5, 6.0, 85.0, 99.0, eb.COUNTER_MODULUS - round(200 / UNIT_J))
    trace = eb.Trace(header, samples)
    marks = trace.marks
    run_j, run_s = trace.span(trace.core_sum(), marks["run1_start"], marks["run1_end"])
    assert run_s == 100.0
    assert run_j == pytest.approx(2 * 6.0 * 100.0, rel=1e-6)
    pkg_j, _ = trace.span(trace.package, marks["run1_start"], marks["run1_end"])
    assert pkg_j == pytest.approx(9900.0, rel=1e-6)
    sysfs_j, _ = trace.span(trace.package_sysfs, marks["run1_start"], marks["run1_end"])
    assert sysfs_j == pytest.approx(9900.0, rel=1e-6)
    view = eb.view_for_rep(trace, trace.core_sum(), marks, 1, 1, 2000.0, 198.0, 0.0, None, 2)
    assert view["net_watts"] == pytest.approx(11.0, rel=1e-6)          # two cores, 6.0 - 0.5 each
    assert view["joules_per_audio_s"] == pytest.approx(0.55, rel=1e-6)
    assert view["idle_before_watts"] == pytest.approx(1.0, rel=1e-6)
    assert view["idle_after_watts"] == pytest.approx(1.0, rel=1e-6)
    assert view["per_second_watts_stdev"] == pytest.approx(0.0, abs=1e-3)
    # Every other physical core, from the boundary snapshots only (core 40 burns 2 W throughout).
    other = trace.other_cores_joules("run1_start", "run1_end", {16, 17})
    assert other == pytest.approx(200.0, rel=1e-6)


def test_trim_and_audio_rate_give_watts_per_paced_stream():
    header = {"energy_unit_j": UNIT_J, "cpus": [16, 17], "pkg_max_uj": PKG_MAX_UJ}
    samples = synthetic_samples(one_rep_schedule(), 0.5, 2.5, 85.0, 90.0, 12345)
    trace = eb.Trace(header, samples)
    first, last = trace.trimmed(trace.marks["run1_start"], trace.marks["run1_end"], 10.0)
    assert (trace.times[first], trace.times[last]) == (75.0, 155.0)
    # Eight streams paced at 1x on the pinned cores: (2 * 2.0 W net) / 8 streams.
    view = eb.view_for_rep(trace, trace.core_sum(), trace.marks, 1, 1, None, 3.0, 10.0, 1.0, 8)
    assert view["run_window_seconds"] == 80.0
    assert view["joules_per_audio_s"] == pytest.approx(0.5, rel=1e-6)
    assert view["joules_per_cpu_s"] is None
    with pytest.raises(ValueError):
        trace.trimmed(trace.marks["run1_start"], trace.marks["run1_end"], 60.0)


def test_busy_fraction_and_distrust_flags():
    before = {80: (100, 1000), 81: (50, 1000)}
    after = {80: (130, 2000), 81: (60, 2000)}
    assert eb.busy_fraction(before, after, [80, 81]) == pytest.approx(0.02)
    thresholds = dict(eb.DEFAULT_THRESHOLDS, foreign_watts=3.0, idle_drift_package=3.0, on_cpu=0.95)
    clean = {
        "busy": {"sibling_busy_fraction_run": 0.001},
        "other_cores": {"run_watts": 21.0, "idle_watts": 20.5},
        "core": {"idle_before_watts": 0.50, "idle_after_watts": 0.55},
        "package": {"idle_before_watts": 85.0, "idle_after_watts": 86.0},
        "cpu_seconds": 99.0, "wall_seconds": 100.0, "exit_codes": [0],
    }
    assert eb.flags_for(clean, thresholds, 1, 1) == []
    dirty = dict(clean, busy={"sibling_busy_fraction_run": 0.2},
                 other_cores={"run_watts": 30.0, "idle_watts": 20.0},
                 core={"idle_before_watts": 0.5, "idle_after_watts": 1.0},
                 package={"idle_before_watts": 85.0, "idle_after_watts": 95.0},
                 cpu_seconds=60.0, exit_codes=[1])
    assert eb.flags_for(dirty, thresholds, 1, 1) == [
        "sibling_busy", "foreign_load_changed", "idle_drift_core", "idle_drift_package",
        "worker_not_on_cpu", "nonzero_exit",
    ]
    # A paced run waits on purpose: no on-CPU rule.
    assert "worker_not_on_cpu" not in eb.flags_for(dict(clean, cpu_seconds=5.0), thresholds, 1, 0)
    # Foreign load that differs between the two idle windows counts, even if the run sits between them.
    stepped = dict(clean, other_cores={"run_watts": 22.0, "idle_watts": 22.0,
                                       "idle_before_watts": 20.0, "idle_after_watts": 24.0})
    assert eb.flags_for(stepped, thresholds, 1, 1) == ["foreign_load_changed"]


def rep_with(flags, core_j, package_j):
    def view(value):
        return {"joules_per_audio_s": value, "gross_joules_per_audio_s": value * 1.1, "joules_per_cpu_s": value * 20,
                "net_watts": value * 10, "run_watts": value * 11, "idle_watts": value}
    return {"flags": flags, "core": view(core_j), "package": view(package_j),
            "cpu_s_per_audio_s": 0.05, "wall_seconds": 100.0, "mean_mhz": 4200.0}


def test_headline_uses_only_the_repetitions_that_are_clean_for_each_view():
    reps = [
        rep_with([], 0.30, 0.50),
        rep_with(["foreign_load_changed"], 0.31, 0.90),      # package view spoiled, core view fine
        rep_with(["sibling_busy"], 0.60, 0.95),              # both spoiled
        rep_with([], 0.32, 0.52),
    ]
    head = eb.headline(reps, eb.DEFAULT_THRESHOLDS)
    assert head["clean_repetitions"] == {"core": 3, "package": 2}
    assert head["joules_per_audio_s"] == 0.31                      # median of 0.30, 0.31, 0.32
    assert head["joules_per_audio_s_package"] == pytest.approx(0.51)
    assert head["watts_per_stream"] == head["joules_per_audio_s"]
    assert head["flags"] == ["foreign_load_changed", "sibling_busy"]
    assert head["repetitions_disagree"] is True                    # 0.30 to 0.32 is 6.5% of the median
    none_clean = eb.headline([rep_with(["sibling_busy"], 0.6, 0.9)], eb.DEFAULT_THRESHOLDS)
    assert none_clean["joules_per_audio_s"] == 0.6                 # nothing clean: report it, flagged
    assert none_clean["flags"] == ["no_clean_core_rep", "no_clean_package_rep", "sibling_busy"]


def test_least_squares_and_phase_points_recover_package_watts_per_core_watt():
    fit = er.least_squares([[1.0, 4.0], [2.0, 4.0], [4.0, 9.0], [8.0, 5.0], [0.5, 30.0]],
                           [55 + 1.4 * 1 + 1.5 * 4, 55 + 1.4 * 2 + 1.5 * 4, 55 + 1.4 * 4 + 1.5 * 9,
                            55 + 1.4 * 8 + 1.5 * 5, 55 + 1.4 * 0.5 + 1.5 * 30])
    assert fit["intercept"] == pytest.approx(55.0)
    assert fit["coefficients"] == pytest.approx([1.4, 1.5])
    assert fit["r2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        er.least_squares([[1.0], [1.0], [1.0]], [1.0, 2.0, 3.0])
    receipt = {"arm": "spin-2", "reps": [
        {"rep": 1, "core": {"run_watts": 20.0, "idle_before_watts": 1.0, "idle_after_watts": 1.2, "idle_watts": 1.1},
         "other_cores": {"run_watts": 4.0, "idle_watts": 4.0, "idle_before_watts": 4.0, "idle_after_watts": 4.1},
         "package": {"run_watts": 90.0, "idle_before_watts": 62.0, "idle_after_watts": 62.4, "idle_watts": 62.2}},
        {"rep": 2, "core": {"run_watts": 20.5, "idle_before_watts": 1.2, "idle_after_watts": 1.0, "idle_watts": 1.1},
         "other_cores": {"run_watts": 9.0, "idle_watts": 4.0, "idle_before_watts": 4.1, "idle_after_watts": 3.9},
         "package": {"run_watts": 98.0, "idle_before_watts": 62.4, "idle_after_watts": 61.8, "idle_watts": 62.1}},
    ]}
    labels = [point[0] for point in er.phase_points(receipt)]
    assert labels == ["spin-2:run1", "spin-2:idle1", "spin-2:run2", "spin-2:idle2", "spin-2:idle3"]


def test_sha256_path_hashes_files_and_whole_directories(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "a.bin").write_bytes(b"a")
    (tmp_path / "model" / "b.bin").write_bytes(b"b")
    single = eb.sha256_path(tmp_path / "model" / "a.bin")
    assert single == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    whole = eb.sha256_path(tmp_path / "model")
    (tmp_path / "model" / "b.bin").write_bytes(b"c")
    assert eb.sha256_path(tmp_path / "model") != whole


class FakeSampler:
    """Stands in for the root helper: one core at 1 W idle and 6 W while a run is marked."""

    def __init__(self, cpus, observer_cpu, interval, max_seconds, frequency_cpus, sudo=True):
        self.cpus = list(cpus)
        self.header = {"energy_unit_j": UNIT_J, "pkg_max_uj": PKG_MAX_UJ, "cpus": self.cpus,
                       "all_cores": self.cpus, "msr": True, "powercap": True, "interval": interval}
        self.samples, self.frequency = [], []
        self.joules, self.running, self.last = 0.0, False, time.monotonic()
        self.mark("start")

    def mark(self, label):
        now = time.monotonic()
        self.joules += (6.0 if self.running else 1.0) * (now - self.last)
        self.last = now
        counts = round(self.joules / UNIT_J)
        self.samples.append({
            "t": now, "label": label, "core": {str(c): counts for c in self.cpus},
            "pkg": counts * 20, "pkg_uj": round(self.joules * 20e6),
            "all": {str(c): counts for c in self.cpus},
        })
        self.frequency.append((now, 4200.0))
        if label.startswith("run") and label.endswith("_start"):
            self.running = True
        elif label.startswith("run"):
            self.running = False

    def stop(self):
        self.mark("stop")


def test_run_writes_a_receipt_with_the_agreed_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "Sampler", FakeSampler)
    cpu = min(os.sched_getaffinity(0))
    out = tmp_path / "receipt.json"
    model = tmp_path / "model.bin"
    model.write_bytes(b"weights")
    code = eb.main([
        "run", "--arm", "fake-arm", "--cpus", str(cpu), "--reps", "2", "--idle-seconds", "0.3",
        "--settle-seconds", "0.05", "--audio-seconds", "10", "--hash", str(model), "--env", "OMP_NUM_THREADS=1",
        "--out", str(out), "--", sys.executable, "-c", "import time; time.sleep(0.4)  # {rep} {worker} {cpu}",
    ])
    assert code == 0
    receipt = json.loads(out.read_text())
    assert receipt["schema"] == eb.SCHEMA
    for field in ("joules_per_audio_s", "watts_per_stream", "idle_watts", "machine", "energy_method"):
        assert receipt[field] is not None
    assert receipt["watts_per_stream"] == receipt["joules_per_audio_s"]
    assert receipt["repetitions"] == 2 and len(receipt["reps"]) == 2
    first = receipt["reps"][0]
    assert first["exit_codes"] == [0]
    assert first["core"]["net_watts"] == pytest.approx(5.0, rel=0.02)
    assert first["package"]["net_watts"] == pytest.approx(100.0, rel=0.02)
    assert first["core"]["joules_per_audio_s"] == pytest.approx(5.0 * first["core"]["run_window_seconds"] / 10, rel=0.02)
    assert receipt["sha256"][str(model)] == eb.sha256_path(model)
    assert receipt["env"] == {"OMP_NUM_THREADS": "1"}
    assert receipt["host"]["kernel"]
    assert "worker_not_on_cpu" in receipt["flags"]  # the stand-in command sleeps instead of computing
    assert receipt["clean_repetitions"] == {"core": 0, "package": 0}

    # Re-applying the rules to the stored repetitions changes no measured value.
    again = tmp_path / "again.json"
    assert er.main(["resummarise", str(out), "--out", str(again)]) == 0
    redone = json.loads(again.read_text())
    assert redone["reps"] == receipt["reps"]
    # (the stored values are rounded to six digits, so a median of two can move in the last one)
    assert redone["joules_per_audio_s"] == pytest.approx(receipt["joules_per_audio_s"], rel=1e-5)
    assert redone["resummarised"]["measured_with_tool_sha256"] == receipt["tool_sha256"]

    lines = er.sidecar_lines(receipt, "fake-arm-zen2-t1", "f" * 64, "2026-11-22 xbabe0 e2ef608")
    assert [line["energy_method"] for line in lines] == ["core_msr_active", "core_msr_dynamic", "pkg_rapl_delta"]
    active, dynamic, package = lines
    assert all(line["row"] == "fake-arm-zen2-t1" and line["kind"] == "energy" for line in lines)
    assert active["joules_per_audio_s"] > dynamic["joules_per_audio_s"] > 0       # active keeps the idle floor
    assert active["energy_active_w"] == pytest.approx(6.0, rel=0.02)
    assert dynamic["energy_idle_w"] == pytest.approx(1.0, rel=0.02)
    assert package["energy_idle_w"] == pytest.approx(20.0, rel=0.02)
    assert package["joules_per_audio_s"] == pytest.approx(20 * dynamic["joules_per_audio_s"], rel=0.02)
    assert len(dynamic["energy_reps"]) == 2 and dynamic["watts_per_stream"] == dynamic["joules_per_audio_s"]


def test_fit_and_table_read_receipts(tmp_path, capsys):
    paths = []
    for workers, core, package in ((1, 6.0, 16.0), (2, 12.0, 26.0), (4, 24.0, 46.0)):
        receipt = {
            "schema": eb.SCHEMA, "arm": f"spin-{workers}", "workers": workers, "repetitions": 3, "flags": [],
            "cpus": [16, 17, 18, 19], "reps": [],
            "joules_per_audio_s": 0.3 * workers, "joules_per_audio_s_package": 0.5 * workers,
            "watts_per_busy_core": 6.0, "watts_per_busy_core_package": 10.0, "cpu_s_per_audio_s": 0.05 * workers,
            "summary": {"net_watts": {"median": core}, "net_watts_package": {"median": package},
                        "joules_per_audio_s": {"spread_pct": 1.5}},
        }
        path = tmp_path / f"{workers}.json"
        path.write_text(json.dumps(receipt))
        paths.append(str(path))
    assert er.main(["fit", *paths]) == 0
    fit = json.loads(capsys.readouterr().out)
    assert fit["core"]["slope"] == pytest.approx(6.0)
    assert fit["package"]["slope"] == pytest.approx(10.0)
    assert fit["package"]["intercept"] == pytest.approx(6.0)
    assert fit["package_over_core_slope"] == pytest.approx(10.0 / 6.0, rel=1e-5)
    assert er.main(["table", "--ours", "spin-1", *paths]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[2].startswith("| spin-1 | 0.05 | 0.3 | 0.5 | 6 / 10 | 1x |")
    assert "| 4x |" in table[4]


RECEIPTS = sorted((eb.Path(__file__).resolve().parents[3] / "docs/paper/data/raw/energy").glob("*.json"))


@pytest.mark.parametrize("path", RECEIPTS, ids=lambda p: p.stem)
def test_committed_receipt_is_consistent(path):
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == eb.SCHEMA  # derived files (fits, tables) live in a subdirectory
    assert receipt["arm"] == path.stem
    assert receipt["machine"] and receipt["energy_method"] and receipt["sha256"]
    head = eb.headline(receipt["reps"], receipt["thresholds"])
    for field in ("joules_per_audio_s", "joules_per_audio_s_package", "watts_per_busy_core", "idle_watts"):
        if head[field] is None:  # an arm without audio (idle, busy loop) has no joules per audio-second
            assert receipt[field] is None, field
        else:
            assert receipt[field] == pytest.approx(head[field], rel=1e-5), field
    for field in ("clean_repetitions", "flags", "repetitions"):
        assert receipt[field] == head[field], field
    assert receipt["host"]["kernel"] and receipt["host"]["boost"] is not None and receipt["host"]["governor"]
    assert len(receipt["reps"]) == receipt["repetitions"] >= 1
    assert all(rep["loadavg_before"] and rep["loadavg_after"] for rep in receipt["reps"])
    for view in ("core", "package"):
        for rep in receipt["reps"]:
            net = rep[view]["run_watts"] - rep[view]["idle_watts"]
            assert rep[view]["net_watts"] == pytest.approx(net, rel=1e-3, abs=2e-3)
    assert receipt["watts_per_stream"] == receipt["joules_per_audio_s"]


def test_power_bench_rows_become_the_same_evidence_lines(capsys, tmp_path):
    # One row as power_bench.py writes it (turbostat CorWatt on a pinned core, one run).
    summary = {"schema": "jsr.power-bench/v1", "timestamps_sha256": "e" * 64, "core_power": [
        {"system": "jsr-d2e-16457-int8", "cpu_s_per_audio_s": 0.06071, "idle_core_w": 0.755, "active_core_w": 5.841,
         "dynamic_core_w": 5.086, "mw_per_live_stream": 354.6, "dynamic_mw_per_live_stream": 308.8},
        {"system": "moonshine-tiny", "cpu_s_per_audio_s": 0.20369, "idle_core_w": 0.8, "active_core_w": 5.507,
         "dynamic_core_w": 4.707, "mw_per_live_stream": 1121.6, "dynamic_mw_per_live_stream": 958.7},
    ]}
    lines = er.power_bench_lines(summary, {"jsr-d2e-16457-int8": "d2e-16457-int8-zen2-t1-xbabe3"}, "a" * 64, "xbabe3")
    assert [(line["row"], line["energy_method"]) for line in lines] == [
        ("d2e-16457-int8-zen2-t1-xbabe3", "core_msr_active"), ("d2e-16457-int8-zen2-t1-xbabe3", "core_msr_dynamic")]
    active, dynamic = lines
    assert active["joules_per_audio_s"] == pytest.approx(0.3546) == active["watts_per_stream"]
    assert active["joules_per_audio_s"] == pytest.approx(5.841 * 0.06071, rel=1e-3)   # watts x seconds of one run
    assert dynamic["joules_per_audio_s"] == pytest.approx(0.3088)
    assert active["energy_tool"] == "jsr.power-bench/v1" and active["energy_flags"] == ["single_run"]
    path = tmp_path / "power_summary.json"
    path.write_text(json.dumps(summary))
    assert er.main(["sidecar", "--power-bench", "moonshine-tiny=moonshine-tiny-zen2-t1-xbabe3", "--host", "xbabe3",
                    str(path)]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["energy_method"] for line in printed] == ["core_msr_active", "core_msr_dynamic"]
    assert printed[0]["joules_per_audio_s"] == pytest.approx(1.1216)
    with pytest.raises(SystemExit):
        er.power_bench_lines({"schema": "other"}, {}, "a" * 64, "xbabe3")


def write_power_log(path, seconds, watts_of, start_wall=1_795_000_000.0, restart_at=None):
    """A continuous log as the sampler writes it: cores 0-3, counters starting just below the wrap."""
    unit = UNIT_J
    lines = [json.dumps({"type": "header", "energy_unit_j": unit, "cpus": [0, 1, 2, 3]})]
    counts = {c: eb.COUNTER_MODULUS - 5000 for c in range(4)}
    package = eb.COUNTER_MODULUS - 9000
    for second in range(seconds + 1):
        if restart_at is not None and second == restart_at:
            lines.append(json.dumps({"type": "header", "energy_unit_j": unit, "cpus": [0, 1, 2, 3]}))
        lines.append(json.dumps({"t": 100.0 + second, "label": None, "pkg": package % eb.COUNTER_MODULUS,
                                 "core": {str(c): v % eb.COUNTER_MODULUS for c, v in counts.items()},
                                 "wall": start_wall + second + (40.0 if restart_at is not None and second >= restart_at else 0.0)}))
        for c in counts:
            counts[c] += round(watts_of(c, second) / unit)
        package += round((60.0 + 1.3 * sum(watts_of(c, second) for c in counts)) / unit)
    path.write_text("\n".join(lines) + "\n")
    return start_wall


def test_join_gives_another_agents_run_its_joules_per_audio_second(tmp_path):
    # Core 2 runs an arm from second 100 to 300 at 8 W over a 0.25 W idle floor; core 0 is busy all along.
    def watts_of(core, second):
        if core == 0:
            return 9.0
        return 8.0 if core == 2 and 100 <= second < 300 else 0.25
    path = tmp_path / "power-20261122.jsonl"
    origin = write_power_log(path, 400, watts_of)
    log = er.PowerLog([path])
    # The other agent's receipt names service CPU 6: in this four-core toy machine that is the SMT sibling of
    # core 2 and shares its register (on the real host thread 80 is the sibling of core 16).
    row = er.join_run(log, [6], origin + 100, origin + 300, 2000.0)
    assert row["covered"] and row["cores"] == [2]
    assert row["core_watts"] == pytest.approx((8.0 * 200 + 0.25 * 3) / 200, rel=1e-3)
    assert row["joules_per_audio_s_active"] == pytest.approx((8.0 * 200 + 0.25 * 3) / 2000, rel=1e-3)
    assert row["joules_per_audio_s_dynamic"] == pytest.approx((8.0 - 0.25) * 200 / 2000, rel=1e-3)
    assert row["idle_core_watts"] == pytest.approx(0.25, rel=1e-3) and row["idle_is_trustworthy"]
    assert row["other_busy_cores"] == 1                                   # core 0
    assert row["package_watts_median"] == pytest.approx(60 + 1.3 * (9.0 + 8.0 + 0.5), rel=1e-3)
    assert er.join_run(log, [2], origin + 350, origin + 450, 100.0) == {"covered": False}   # runs past the log
    lines = er.join_lines([row, dict(row, joules_per_audio_s_active=0.81)], "sys-zen2-t1", "c" * 64, "xbabe0", "d" * 64)
    assert [line["energy_method"] for line in lines] == ["core_msr_active", "core_msr_dynamic"]
    assert lines[0]["row"] == "sys-zen2-t1" and lines[0]["host"] == "xbabe0"
    assert lines[0]["joules_per_audio_s"] == pytest.approx((row["joules_per_audio_s_active"] + 0.81) / 2)
    assert lines[0]["energy_reps"] == [row["joules_per_audio_s_active"], 0.81]
    busy_neighbour = dict(row, idle_is_trustworthy=False)                # the core ran another arm just before
    assert [line["energy_method"] for line in er.join_lines([busy_neighbour], "r", "c" * 64, "xbabe0")] == ["core_msr_active"]


def test_join_refuses_an_interval_that_spans_a_restart_of_the_log(tmp_path):
    path = tmp_path / "power-20261123.jsonl"
    origin = write_power_log(path, 300, lambda core, second: 5.0, restart_at=150)
    log = er.PowerLog([path])
    assert log.gaps == [(origin + 149, origin + 150 + 40)]
    assert er.join_run(log, [1], origin + 100, origin + 220, 100.0) == {"covered": False}
    assert er.join_run(log, [1], origin + 20, origin + 120, 100.0)["covered"]


def test_stair_levels_read_each_plateau_and_the_package_line():
    # Idle 60 s, then 1, 2 and 4 busy workers for 30 s each, idle 60 s. Package = 80 + 1.3 x core watts when
    # busy, 66 W when idle, with a 40 W burst from other work in one second of the middle plateau.
    times, cores, package, labels = [0.0], [0.0], [0.0], {"idle1_start": 0}
    def run(seconds, core_watts, package_watts, burst_at=None):
        for second in range(seconds):
            times.append(times[-1] + 1.0)
            cores.append(cores[-1] + core_watts)
            package.append(package[-1] + package_watts + (40.0 if burst_at == second else 0.0))
    run(60, 0.3, 66.0)
    labels["run1_start"] = len(times) - 1
    run(30, 10.0, 80 + 1.3 * 10.0)
    run(30, 19.0, 80 + 1.3 * 19.0, burst_at=12)
    run(30, 28.0, 80 + 1.3 * 28.0)
    labels["run1_end"] = len(times) - 1
    labels["idle2_start"] = len(times) - 1
    run(60, 0.3, 66.0)
    labels["idle2_end"] = len(times) - 1
    series = {"t": times, "labels": labels, "core_joules": cores, "package_joules": package}
    result = er.stair_levels(series, [("one", 1, 0, 30), ("two", 2, 30, 60), ("four", 4, 60, 90)])
    assert [level["core_register_watts"] for level in result["levels"]] == pytest.approx([10.0, 19.0, 28.0])
    assert [level["package_watts"] for level in result["levels"]] == pytest.approx([93.0, 104.7, 116.4])  # burst ignored
    assert result["package_watts_per_core_register_watt"] == pytest.approx(1.3)
    assert result["package_floor_watts_when_busy"] == pytest.approx(80.0)
    assert result["idle_package_watts"] == pytest.approx(66.0)
    assert result["package_floor_watts_when_busy"] - result["idle_package_watts"] == pytest.approx(14.0)


def test_idle_pairs_report_the_noise_floor_of_run_minus_idle():
    def rep(core_before, core_after, package_before, package_after):
        return {"core": {"idle_before_watts": core_before, "idle_after_watts": core_after},
                "package": {"idle_before_watts": package_before, "idle_after_watts": package_after}}
    receipts = [{"measured_cores": [16, 17], "reps": [rep(0.6, 1.0, 60.0, 75.0), rep(1.0, 0.8, 75.0, 61.0)]},
                {"measured_cores": [16], "reps": [rep(0.3, 0.2, 61.0, 99.0)]}]
    result = er.idle_pairs(receipts)
    assert result["core_per_pinned_core"]["pairs"] == 3
    assert result["core_per_pinned_core"]["median_abs_watts"] == pytest.approx(0.1)   # 0.2, 0.1 and 0.1 W per core
    assert result["package"]["median_abs_watts"] == pytest.approx(15.0)
    assert result["package"]["max_abs_watts"] == pytest.approx(38.0)


def test_stable_word_lag_from_bounded_snapshot_windows():
    # Reference "a b c d e f" ending at 1..6 s; the hypothesis grows one word per second, revises "x" to "d",
    # and every snapshot shows only the last three words (the first may be cut).
    ref = list("abcdef")
    ends = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    snapshots = [(1.2, "a"), (2.2, "a b"), (3.2, "a b c"), (4.2, "b c x"), (4.6, "b c d"),
                 (5.3, "c d e"), (6.4, "d e f"), (6.5, "d e f")]
    lags, total = er.stable_word_lags(snapshots, "A B C D E F", ref, ends)
    assert total == 6
    assert lags == pytest.approx([0.2, 0.2, 0.2, 0.6, 0.3, 0.4])     # "d" counts from 4.6 s, when "x" was replaced
    assert er.align_matches(["a", "b", "c"], ["a", "c"]) == [(0, 0), (2, 1)]
    # A cut first word ("ister" of "mister") is not mistaken for a different word.
    counts = er.confirmed_counts([(1.0, "ister quilter has")], ["mister", "quilter", "has", "missed"])
    assert counts == [3]


def test_stitching_bounded_windows_rebuilds_the_whole_hypothesis_across_segments():
    def event(t, seg, text, kind="Preview"):
        return {"t_s": t, "event": {"event_type": kind, "segment_id": seg, "snapshot_utf8": text}}
    # Windows of at most 12 characters; "SHARK" is revised to "SHARP"; a second segment starts at 5 s.
    events = [event(1.0, 0, "WE LOOK"), event(2.0, 0, "WE LOOK SHAR"), event(2.5, 0, "LOOK SHARK W"),
              event(3.0, 0, "LOOK SHARP W"), event(3.5, 0, "SHARP WE MAY"), event(4.0, 0, "SHARP WE MAY", "Final"),
              {"t_s": 4.1, "event": {"event_type": "SegmentIncomplete", "segment_id": 0}},
              event(5.0, 1, "BE ABLE"), event(6.0, 1, "BE ABLE TO GO", "Final")]
    texts = er.stitched_texts(events)
    assert texts[-1][1] == "WE LOOK SHARP WE MAY BE ABLE TO GO"
    ref = "we look sharp we may be able to go".split()
    ends = [0.5, 0.9, 1.4, 1.6, 1.9, 4.2, 4.6, 5.5, 5.9]
    lags, total, final = er.stable_lags_from_events(events, ref, ends)
    assert total == 9 and len(lags) == 9 and final.startswith("WE LOOK SHARP")
    assert lags[2] == pytest.approx(3.0 - 1.4)          # "sharp" counts from 3.0 s, after "SHARK" was revised
    assert lags[-1] == pytest.approx(6.0 - 5.9)
