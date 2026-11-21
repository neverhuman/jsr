import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train", "ops"))
import divergence_watch as dw  # noqa: E402


def _log(tmp_path, losses, evals=()):
    p = tmp_path / "train.jsonl"
    rows = [{"kind": "train", "step": 50 * (i + 1), "loss": x} for i, x in enumerate(losses)]
    rows += [{"kind": "eval", "step": s, "wer": w} for s, w in evals]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{\"kind\": \"tr")  # torn last line
    return str(p)


def _args(path, *extra):
    return dw.parse_args(["T", path, "--pid", "4242", "--after", "100", "--patience", "200", *extra])


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(SystemExit):
        dw.parse_args(["T", str(tmp_path / "missing.jsonl"), "--pid", "4242"])
    p = _log(tmp_path, [1.0])
    for bad in (["--pid", "1"], [], ["--pid", "5", "--host", "h", "--pid-file", "/r/PID"],
                ["--host", "h;rm -rf /", "--pid-file", "/r/PID"], ["--host", "h", "--pid-file", "r/PID"],
                ["--host", "h", "--pid-file", "/r/$(id)"], ["--pid", "5", "--ratio", "0"]):
        with pytest.raises(SystemExit):
            dw.parse_args(["T", p, *bad])


def test_commands_are_argument_lists(tmp_path):
    p = _log(tmp_path, [1.0])
    local = dw.parse_args(["T", p, "--pid", "4242"])
    assert dw.trainer_cmd(local, "stop") == ["kill", "-TERM", "4242"]
    remote = dw.parse_args(["T", p, "--host", "xbabe2", "--pid-file", "/runs/D2MR/PID"])
    cmd = dw.trainer_cmd(remote, "stop")
    assert cmd[:5] == ["ssh", "-n", "-o", "ConnectTimeout=20", "xbabe2"]
    assert cmd[5].startswith("P=$(cat /runs/D2MR/PID) && ") and cmd[5].endswith('kill -TERM "$P"')


def test_malformed_record_is_an_error():
    assert dw.record("not json") is None
    assert dw.record('{"kind": "other"}') is None
    with pytest.raises(ValueError, match="malformed train"):
        dw.record('{"kind": "train", "step": 5, "loss": "nan?"}')


def test_stable_run_passes_and_runaway_stops(tmp_path):
    assert dw.check(_args(_log(tmp_path, [2.0 - 0.001 * i for i in range(100)]))) is None
    why = dw.check(_args(_log(tmp_path, [1.0] * 20 + [3.0] * 20)))
    assert why is not None and why.startswith("train-loss EMA")


def test_wer_jump_stops(tmp_path):
    why = dw.check(_args(_log(tmp_path, [1.0] * 5, evals=[(100, 20.0), (200, 26.0)])))
    assert why == "dev WER 26.0 at step 200 > best 20.0 + 5.0"
