import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.sp_a_fleet import summarise  # noqa: E402


def _worker(tmp, i, run_id="r1", p99=100.0, last=50.0, streams=10):
    w = dict(run_id=run_id, worker_id=i, streams=streams, lateness_ms={"p99": p99, "last": last},
             wer={"ref_words": 100, "sub": 1, "del": 0, "ins": 1}, audio_accepted_s=60.0, cpu_seconds=6.0,
             max_ready_batch=streams)
    with open(os.path.join(tmp, f"worker_{i}.json"), "w") as fh:
        json.dump(w, fh)


def test_valid_run_counts_only_sustaining_workers(tmp_path):
    _worker(tmp_path, 0)
    _worker(tmp_path, 1, p99=900.0)
    s = summarise(str(tmp_path), "r1", 2, [0, 0], 320.0)
    assert s["valid"] and s["sustaining_workers"] == 1 and s["sustained_streams"] == 10
    assert s["offered_streams"] == 20 and s["pooled_wer"] == 2.0 and s["cpu_s_per_audio_s"] == 0.1


def test_missing_stale_or_failed_workers_invalidate(tmp_path):
    _worker(tmp_path, 0)
    _worker(tmp_path, 1, run_id="old")
    s = summarise(str(tmp_path), "r1", 3, [0, 0, 1], 320.0)
    assert not s["valid"]
    assert any("another run" in p for p in s["problems"])
    assert any("no result file" in p for p in s["problems"])
    assert any("exit 1" in p for p in s["problems"])
