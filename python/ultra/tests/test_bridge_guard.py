"""The launch policy covers separate worker groups and always stops actual guards."""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from bridge import run_guarded  # noqa: E402


def test_group_discovery_includes_worker_group_and_excludes_other_jobs(monkeypatch):
    class Entry:
        def __init__(self, pid, ppid, pgid):
            self.name = str(pid)
            self.stat = f"{pid} (name with (parens)) S {ppid} {pgid} 0"

        def __truediv__(self, _):
            return self

        def read_text(self):
            return self.stat

    entries = [Entry(100, 1, 100), Entry(101, 100, 100), Entry(102, 101, 102), Entry(200, 1, 200)]
    monkeypatch.setattr(run_guarded.Path, "iterdir", lambda _: iter(entries))
    assert run_guarded.process_groups(100) == {100, 102}


def test_every_group_gets_one_guard_and_cleanup_terminates_actual_processes(monkeypatch):
    root = Path(__file__).absolute().parents[3] / "target" / "bridge-guard-test"
    root.mkdir(parents=True, exist_ok=True)
    job = root / "job.json"
    job.write_text(json.dumps(dict(out=str(root / "run"), timeout_s=120)))
    children = []

    class Process:
        def __init__(self, command, **_):
            self.command = command
            self.pid = 100 + len(children)
            self.is_root = not children
            self.polls = 0
            self.terminated = False
            self.returncode = 0 if self.is_root else None
            children.append(self)

        def poll(self):
            self.polls += 1
            return 0 if self.is_root and self.polls > 2 else None

        def terminate(self):
            self.terminated = True

        def wait(self):
            return 0

    monkeypatch.setattr(run_guarded.subprocess, "Popen", Process)
    monkeypatch.setattr(run_guarded, "process_groups", lambda _: {100, 102})
    monkeypatch.setattr(run_guarded.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["guard", "/test/binary", str(job)])
    try:
        assert run_guarded.main() == 0
        assert len(children) == 3
        assert all(c.terminated for c in children[1:])
        facts = json.loads((root / "run.governance" / "governance.json").read_text())
        assert {r["pgid"] for r in facts["groups"]} == {100, 102}
        assert facts["wall_timing_evidence"] is False
        effective = json.loads((root / "run.governance" / "job.json").read_text())
        assert effective["guarded"] is True
        assert effective["timeout_s"] == 3600
    finally:
        shutil.rmtree(root)
