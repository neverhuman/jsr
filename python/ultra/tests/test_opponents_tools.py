"""The opponent benchmark tooling under docs/paper/opponents (bench-cross): views, shards,
side-by-side workers, the macOS meter's parser, plan phases and the receipt table."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "docs/paper/opponents"


def load(name):
    spec = importlib.util.spec_from_file_location(f"opponents_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_panel(root, count=730):
    """A staged panel with tiny WAVs whose lengths differ, and its manifest digest."""
    (root / "wav").mkdir(parents=True)
    rows, index = [], []
    for position in range(count):
        ident = f"u{position:05d}"
        samples = 160 * (1 + position % 7)
        with wave.open(str(root / "wav" / f"{ident}.wav"), "wb") as out:
            out.setparams((1, 2, 16000, samples, "NONE", "not compressed"))
            out.writeframes(bytes(2 * samples))
        rows.append({"id": ident, "samples": samples, "pcm_sha256": "0" * 64, "wav_sha256": "0" * 64})
        index.append({"id": ident, "uid": f"1-2-{position:04d}", "speaker": str(position // 10),
                      "dataset": "dev-clean" if position < 400 else "dev-other", "ref": "A B",
                      "duration_s": samples / 16000})
    (root / "audio.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return hashlib.sha256((root / "audio.json").read_bytes()).hexdigest()


def test_views_and_balanced_shards(tmp_path, monkeypatch, capsys):
    workload = load("workload")
    monkeypatch.setattr(workload, "PANEL_DIGEST", fake_panel(tmp_path / "panel"))
    workload.stage(tmp_path / "panel", tmp_path / "views", "panel73", 8)
    capsys.readouterr()
    view = tmp_path / "views/panel73"
    names = sorted(path.name for path in (view / "wav").iterdir())
    assert names == [f"u{position:05d}.wav" for position in range(0, 730, 10)]
    layout = json.loads((view / "shards-8/shards.json").read_text())
    assert sorted(ident for part in layout["ids"] for ident in part) == [name[:-4] for name in names]
    assert max(layout["audio_seconds"]) - min(layout["audio_seconds"]) <= 7 * 160 / 16000
    assert os.stat(view / "wav/u00000.wav").st_ino == os.stat(tmp_path / "panel/wav/u00000.wav").st_ino
    identity = json.loads((view / "workload.json").read_text())
    assert identity["utterances"] == 73 and identity["panel_digest"] == workload.PANEL_DIGEST


def test_view_refuses_a_changed_panel(tmp_path):
    workload = load("workload")
    fake_panel(tmp_path / "panel", count=20)
    with pytest.raises(ValueError, match="frozen digest"):
        workload.stage(tmp_path / "panel", tmp_path / "views", "panel10", 0)


def test_workers_fill_tokens_and_propagate_failure(tmp_path):
    shards = tmp_path / "view/shards-2"
    for index in range(2):
        (shards / f"shard-{index}").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    base = [sys.executable, str(TOOLS / "workers.py"), "none:2", str(shards), str(run), "--"]
    subprocess.run([*base, "sh", "-c", "echo {wav} {engine} > {out}"], check=True)
    for index in range(2):
        wav, engine = (run / f"worker-{index}.out").read_text().split()
        assert wav == str(shards / f"shard-{index}")
        assert engine == str(tmp_path / f"view/engine-shards-2/shard-{index}")
    again = tmp_path / "again"
    again.mkdir()
    failed = subprocess.run([*base[:4], str(again), "--", "sh", "-c", "exit 3"], check=False)
    assert failed.returncode != 0


def test_bsd_time_block_is_split_from_the_command_output():
    meter = load("bsd_time_measure")
    text = ("{\"runtime\": \"x\"}\nwarning\n"
            "       69.23 real        68.77 user         0.23 sys\n"
            "           330481664  maximum resident set size\n"
            "                   0  average shared memory size\n"
            "           329400944  peak memory footprint\n")
    own, block = meter.split_time_block(text)
    assert own == "{\"runtime\": \"x\"}\nwarning\n"
    assert block == {"real": 69.23, "user": 68.77, "sys": 0.23, "maximum_resident_set_size": 330481664,
                     "peak_memory_footprint": 329400944}


def test_plan_runs_cpu_sets_side_by_side_and_phases_in_order(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "plan.sh").write_text((TOOLS / "plan.sh").read_text())
    (scripts / "queue.sh").write_text(
        "#!/usr/bin/env bash\nwhile read -r arm budget round rest; do\n"
        "  echo \"$1 ${OPP_SERVICE_CPUS:-none} ${OPP_OBSERVER_CPU:-none} $arm $budget $round\" >> \"$OPP_ROOT/seen\"\n"
        "done\n")
    (tmp_path / "bench-cross/logs").mkdir(parents=True)
    plan = "32 48 a t1 1\n36 49 a t1 2\n32 48 b t1 1\n--\n32,33 48 a t8 1\n# note\n"
    subprocess.run(["bash", str(scripts / "plan.sh"), "demo"], input=plan, text=True, check=True,
                   env={**os.environ, "OPP_ROOT": str(tmp_path), "OPP_MACHINE": "test"})
    seen = (tmp_path / "seen").read_text().splitlines()
    assert sorted(seen[:3]) == ["demo-p0-32 32 48 a t1 1", "demo-p0-32 32 48 b t1 1", "demo-p0-36 36 49 a t1 2"]
    assert seen.index("demo-p0-32 32 48 a t1 1") < seen.index("demo-p0-32 32 48 b t1 1")
    assert seen[3] == "demo-p1-32_33 32,33 48 a t8 1"
    log = (tmp_path / "bench-cross/logs/plan-demo.log").read_text()
    assert log.count("phase") == 2 and "plan demo done" in log


def test_shell_runners_parse():
    for name in ("run-opponent.sh", "queue.sh", "plan.sh", "setup-rust.sh", "setup-python.sh", "setup-uv.sh"):
        subprocess.run(["bash", "-n", str(TOOLS / name)], check=True)


def receipt(run, cpu, wall, rss, wer, text):
    run.mkdir(parents=True)
    (run / "round").write_text(run.name.split("-")[1] + "\n")
    (run / "started").write_text("2026-11-22T21:00:00Z\n")
    (run / "finished").write_text("2026-11-22T21:10:00Z\n")
    (run / "budget.json").write_text(json.dumps({
        "arm": "demo", "budget": "t1", "class": "streaming", "machine": "zen2", "threads": 1,
        "parallel": "intra_op", "service_cpus": "32", "observer_cpu": "48", "workload": "panel10", "trace": 0}))
    (run / "measurement.json").write_text(json.dumps({
        "cpu_seconds_user": cpu, "cpu_seconds_sys": 0.0, "wall_seconds": wall, "peak_rss_bytes": rss,
        "exit_code": 0}))
    (run / "score.json").write_text(json.dumps({"system_a": {
        "utterances": 10, "pooled_wer": wer,
        "datasets": {"dev-clean": {"wer": wer / 2}, "dev-other": {"wer": wer * 2}}}}))
    (run / "scores.jsonl").write_text(text)
    stat = "cpu 0 0 0 0 0 0 0 0\n" + "".join(f"cpu{n} 10 0 0 {idle} 0 0 0 0\n" for n, idle in ((32, 0), (33, 90), (40, 2)))
    (run / "cpustat.before").write_text(stat.replace(" 10 ", " 0 ").replace(" 90 ", " 0 ").replace(" 2 ", " 0 "))
    (run / "cpustat.after").write_text(stat)
    (run / "freq.tsv").write_text("1 4000000\n2 4200000\n")


def test_table_reports_every_repetition_and_the_median(tmp_path):
    collect = load("collect")
    arm = tmp_path / "zen2/demo-t1-panel10"
    receipt(arm / "round-1", 6.0, 6.5, 100, 0.10, "same")
    receipt(arm / "round-2", 5.0, 5.5, 300, 0.10, "same")
    receipt(arm / "round-3", 9.0, 9.5, 200, 0.10, "same")
    (row,) = collect.table(tmp_path, {"panel10": 50.0})
    assert row["cpu_reps"] == [0.12, 0.1, 0.18] and row["cpu_s_per_audio_s"] == 0.12
    assert row["median_round"] == "1" and row["peak_rss_bytes"] == 300
    assert row["wall_s_per_audio_s"] == pytest.approx(0.13) and row["rtfx"] == pytest.approx(1 / 0.13)
    assert row["identical_hypotheses"] and row["wer_pct"] == pytest.approx(10.0)
    assert row["wer_clean_pct"] == pytest.approx(5.0) and row["wer_other_pct"] == pytest.approx(20.0)
    # cpu40 was fully busy and is neither the service nor the meter CPU; cpu33 was idle.
    assert row["busy_neighbours"] == [[40]] * 3 and row["n_busy_cores"] == [1, 1, 1]
    assert row["service_mhz"] == [4100.0] * 3
    (arm / "round-3/scores.jsonl").write_text("different")
    (row,) = collect.table(tmp_path, {"panel10": 50.0})
    assert not row["identical_hypotheses"]
