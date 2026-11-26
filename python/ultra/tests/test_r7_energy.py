"""R7 energy ingest: evidence kind, one finite positive value per target, grey arms, receipt hashes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools" / "paper_data" / "r7_summarize.py"
spec = importlib.util.spec_from_file_location("r7_summarize", MODULE)
r7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r7)


def receipt(j, clean=True):
    return {
        "joules_per_audio_s": j, "cpu_s_per_audio_s": 0.05, "flags": [] if clean else ["idle_drift_core"],
        "clean_repetitions": {"core": 1 if clean else 0, "package": 0},
        "reps": [{"wall_seconds": 10.0, "audio_seconds": 100.0, "core": {"idle_watts": 0.1},
                  "package": {"idle_watts": 60.0}, "mean_mhz": 4200}],
    }


def build(tmp_path, values):
    """values: {arm key: [(block, j, clean), ...]}"""
    for key, rows in values.items():
        for block, j, clean in rows:
            d = tmp_path / f"block{block}"
            d.mkdir(exist_ok=True)
            (d / f"{key}.json").write_text(json.dumps(receipt(j, clean)))


def run(tmp_path, monkeypatch, extra=()):
    out = tmp_path / "evidence.jsonl"
    monkeypatch.setattr(sys, "argv", ["r7_summarize.py", str(tmp_path), "--evidence", str(out), *extra])
    r7.main()
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_kind_median_over_clean_blocks_and_grey_arm(tmp_path, monkeypatch):
    build(tmp_path, {
        "vosk-small": [(1, 1.0, True), (2, 9.0, False), (3, 1.2, True)],
        "vosk-large": [(1, 5.0, False), (2, 6.0, False)],
    })
    rows = run(tmp_path, monkeypatch)
    assert [r["id"] for r in rows] == ["vosk-small-zen2-t1"]
    row = rows[0]
    assert row["kind"] == "r7_energy" and row["n_clean"] == 2 and row["clean_blocks"] == [1, 3]
    assert row["energy_j_per_audio_s"] == pytest.approx(1.1)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["arms"]["vosk-large"]["grey"] is True


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_or_non_finite(tmp_path, monkeypatch, bad):
    build(tmp_path, {"vosk-small": [(1, bad, True)]})
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch)


def test_rejects_duplicate_targets(tmp_path, monkeypatch):
    build(tmp_path, {"vosk-small": [(1, 1.0, True)], "vosk-large": [(1, 2.0, True)]})
    monkeypatch.setitem(r7.IDS, "vosk-large", r7.IDS["vosk-small"])
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch)
