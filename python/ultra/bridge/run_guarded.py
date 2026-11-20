"""Apply the owner's CPU fleet policy to every group in one exploratory Rust job.

This launch/host-policy helper never reads audio, references or scores. Rust remains
the measurement/scoring authority. Governance processes live outside the measured
service tree; guarded wall time is not timing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_groups(root):
    """PGIDs in the current descendant tree, including dedicated worker/CLI groups."""
    rows = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            fields = stat[stat.rfind(")") + 1:].split()
            rows[int(p.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, ValueError, IndexError):
            continue
    owned = {root}
    while True:
        new = {pid for pid, (ppid, _) in rows.items() if ppid in owned} - owned
        if not new:
            break
        owned |= new
    return {rows[p][1] for p in owned if p in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("job")
    p.add_argument("--nemotron", action="store_true")
    p.add_argument("--export", action="store_true", help="verified A2 export preparation, never timed inference")
    a = p.parse_args()
    job_path = Path(a.job).absolute()
    job = json.loads(job_path.read_text())
    job["guarded"] = True
    if not a.export:
        # Host-policy pauses can exceed an ordinary inference timeout. A remaining
        # timeout is still an invalid run, never an empty successful outcome.
        job["timeout_s"] = max(3600, job["timeout_s"])
    out = Path(job["out"]).absolute()
    governance = out.with_name(out.name + ".governance")
    governance.mkdir()
    effective = governance / "job.json"
    effective.write_text(json.dumps(job, indent=2))
    guard_source = Path(__file__).absolute().parents[1] / "ultra" / "fleet_guard.py"
    guard_bytes = guard_source.read_bytes()
    retained_guard = governance / "fleet_guard.py"
    retained_guard.write_bytes(guard_bytes)
    command = [str(Path(a.binary).absolute()), "run-nemotron" if a.nemotron else "run", str(effective)]
    environment = os.environ.copy()
    if a.export:
        command = [str(Path(a.binary).absolute()), str(Path(__file__).with_name("export_verified.py")), str(effective)]
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    root = subprocess.Popen(command, start_new_session=True, env=environment)
    guards, records = {}, []
    code = None
    try:
        while root.poll() is None:
            for pgid in process_groups(root.pid):
                previous = guards.get(pgid)
                if previous is None or previous.poll() is not None:
                    log = governance / f"pgid-{pgid}-{len(records)}.jsonl"
                    guard = subprocess.Popen([sys.executable, str(retained_guard), "--pgid", str(pgid),
                        "--gpu", "-1", "--start-duty", "1.0", "--cpu-target", "0.85",
                        "--gpu-target", "0.82", "--log-every", "1", "--log", str(log)],
                        start_new_session=True)
                    guards[pgid] = guard
                    records.append(dict(pgid=pgid, guard_python_pid=guard.pid, log=str(log)))
            time.sleep(0.1)
        code = root.returncode
    finally:
        if root.poll() is None:
            for pgid in process_groups(root.pid):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            root.wait()
        # Stop the actual guard Python processes, never only a wrapper.
        for guard in guards.values():
            if guard.poll() is None:
                guard.terminate()
        for guard in guards.values():
            guard.wait()
        (governance / "governance.json").write_text(json.dumps(dict(
            schema="ultra.bridge-governance/v1", job_exit_status=code,
            guard_source_sha256=hashlib.sha256(guard_bytes).hexdigest(),
            cpu_target=0.85, start_duty=1.0, gpu=-1, groups=records,
            wall_timing_evidence=False, governance_cpu_scope="outside measured inference service tree"), indent=2))
    return code if code is not None else 1


if __name__ == "__main__":
    sys.exit(main())
