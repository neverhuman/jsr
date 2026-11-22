"""Run a plan of guarded repetitions (open-run.sh) phase by phase, one job per CPU set at a time.

    open_schedule.py PLAN LOG_DIR

PLAN lines are `PHASE ARM ROUND SERVICE_CPUS OBSERVER_CPU` (`#` starts a comment). Phases run in
the order they first appear; a phase starts when the previous one has no job left. Inside a
phase, jobs that name the same SERVICE_CPUS run one after another in file order and different
CPU sets run side by side. Launches are 20 s apart so that each repetition's own gate sees
the load of the one started before it. A failed job is recorded and the plan continues.
State is rewritten to LOG_DIR/schedule.json after every change.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGGER_S = 20


def read_plan(text):
    jobs = []
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"expected PHASE ARM ROUND SERVICE_CPUS OBSERVER_CPU: {line!r}")
        phase, arm, round_id, service, observer = fields
        jobs.append({"phase": phase, "arm": arm, "round": int(round_id), "service": service,
                     "observer": int(observer), "status": "pending"})
    return jobs


def next_jobs(jobs):
    """Jobs that may start now: the first pending job of every idle CPU set in the current phase."""
    open_jobs = [job for job in jobs if job["status"] in ("pending", "running")]
    if not open_jobs:
        return []
    phase = open_jobs[0]["phase"]
    busy = {job["service"] for job in open_jobs if job["status"] == "running"}
    ready = []
    for job in open_jobs:
        if job["phase"] == phase and job["status"] == "pending" and job["service"] not in busy:
            busy.add(job["service"])
            ready.append(job)
    return ready


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    plan, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = read_plan(plan.read_text())
    script = Path(__file__).resolve().with_name("open-run.sh")
    running = {}

    def save():
        state = {"plan": str(plan), "scheduler_pid": os.getpid(), "updated_utc": now(), "jobs": jobs}
        temporary = log_dir / "schedule.json.tmp"
        temporary.write_text(json.dumps(state, indent=1) + "\n")
        temporary.replace(log_dir / "schedule.json")

    save()
    while True:
        for job in list(running):
            code = running[job][0].poll()
            if code is not None:
                entry = running.pop(job)[1]
                entry.update(status="done" if code == 0 else "failed", exit_code=code, ended_utc=now())
                save()
        ready = next_jobs(jobs)
        if ready:
            job = ready[0]
            name = f"{job['arm']}.r{job['round']}"
            with open(log_dir / f"{name}.log", "ab") as log:
                child = subprocess.Popen(
                    ["bash", str(script), job["arm"], str(job["round"]), job["service"], str(job["observer"])],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            job.update(status="running", pid=child.pid, started_utc=now())
            running[id(job)] = (child, job)
            save()
            time.sleep(STAGGER_S)
        elif running:
            time.sleep(5)
        else:
            break
    failed = [f"{job['arm']} round {job['round']}" for job in jobs if job["status"] != "done"]
    print(f"{now()} plan finished; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
