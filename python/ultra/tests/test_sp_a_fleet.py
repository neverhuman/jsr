import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.sp_a_fleet import summarise  # noqa: E402

LAT = {"p99": 100.0, "last": 50.0, "last_third_mean": 60.0}


def _worker(tmp, i, run_id="r1", lat=None, streams=10, core=None, drained=True, cpu=6.0):
    w = dict(run_id=run_id, worker_id=i, streams=streams, lateness_ms=lat or dict(LAT), drained=drained,
             wer={"ref_words": 100, "sub": 1, "del": 0, "ins": 1}, audio_accepted_s=60.0, cpu_seconds=cpu,
             rtf_cpu=0.1, max_ready_batch=streams, affinity=[40 + i if core is None else core])
    with open(os.path.join(tmp, f"worker_{i}.json"), "w") as fh:
        json.dump(w, fh)


def test_valid_run_counts_only_sustaining_workers(tmp_path):
    _worker(tmp_path, 0)
    _worker(tmp_path, 1, lat={"p99": 900.0, "last": 50.0, "last_third_mean": 60.0})
    _worker(tmp_path, 2, lat={"p99": 100.0, "last": 50.0, "last_third_mean": 500.0})  # growing backlog
    _worker(tmp_path, 3, drained=False)
    s = summarise(str(tmp_path), "r1", 4, [40, 41, 42, 43], [0, 0, 0, 0], 320.0)
    assert s["valid"] and s["sustaining_workers"] == 1 and s["sustained_streams"] == 10
    assert s["offered_streams"] == 40 and s["pooled_wer"] == 2.0 and s["cpu_s_per_audio_s"] == 0.1


def test_invalid_runs_claim_no_capacity(tmp_path):
    _worker(tmp_path, 0)
    _worker(tmp_path, 1, run_id="old")
    s = summarise(str(tmp_path), "r1", 3, [40, 41, 42], [0, 0, 9], 320.0)
    assert not s["valid"] and s["sustained_streams"] == 0
    joined = " ".join(s["problems"])
    assert "another run" in joined and "no result file" in joined and "exit 9" in joined


def test_short_core_list_or_exit_list_is_invalid(tmp_path):
    _worker(tmp_path, 0)
    s = summarise(str(tmp_path), "r1", 2, [40], [0], 320.0)
    assert not s["valid"] and s["sustained_streams"] == 0


def test_wrong_affinity_and_nonfinite_are_invalid(tmp_path):
    _worker(tmp_path, 0, core=7)
    _worker(tmp_path, 1, cpu=float("nan"))
    s = summarise(str(tmp_path), "r1", 2, [40, 41], [0, 0], 320.0)
    assert not s["valid"]
    joined = " ".join(s["problems"])
    assert "expected [40]" in joined and "non-finite" in joined
