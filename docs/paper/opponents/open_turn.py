"""Run timed repetitions on the shared host, one arm on the whole host at a time.

    open_turn.py PLAN LOG_DIR

Protocol of the timing host (`$PAPER_OPPONENTS_ROOT/TIMING.order`): an entry N may start only
when every earlier entry is `DONE` or `SKIP` in `TIMING.done`; the host lock is the directory
`TIMING.lock` (made atomically, with an `owner` file, held for at most 30 minutes, removed when
the block ends); the agent appends `DONE N <UTC> <agent>` when its entry is finished.

PLAN lines (`#` starts a comment):
    ENTRY PANEL ARM ROUND SERVICE_CPUS OBSERVER_CPU EST_MIN   one repetition through open-run.sh
    anchor ARM EST_MIN_FULL EST_MIN_P73                       run ARM once at the start of every
                                                              hold, on the panel and the first CPU
                                                              of the hold's first repetition
ARM may be `A|B`: the one of the two with the shorter wall time in round 1 of this plan (both
clean); if neither has such a round-1 receipt, A.

Entries run in file order; an entry this agent already marked DONE is skipped, so the runner
can be restarted. Inside an entry the repetitions run one after another; a new hold starts when
the next repetition's estimate no longer fits in the current one. Once an arm has run, its
measured duration (plus 5%) replaces the plan's estimate. A repetition that was not alone on
the host (exit 3) is repeated once at the end of its entry. State: LOG_DIR/turn.json.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

AGENT = os.environ.get("OPEN_AGENT", "bench-open")
HOLD_S = float(os.environ.get("OPEN_HOLD_S", 30 * 60))
MARGIN_S = float(os.environ.get("OPEN_HOLD_MARGIN_S", 60))
POLL_S = float(os.environ.get("OPEN_POLL_S", 10))


def utc(moment=None):
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_plan(text):
    """`(jobs, anchor)` from the plan text; anchor is None or `{arm, est_s: {panel: seconds}}`."""
    jobs, anchor = [], None
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "anchor":
            if len(fields) != 4:
                raise ValueError(f"expected anchor ARM EST_MIN_FULL EST_MIN_P73: {line!r}")
            anchor = {"arm": fields[1], "est_s": {"full": 60 * float(fields[2]), "p73": 60 * float(fields[3])}}
            continue
        if len(fields) != 7:
            raise ValueError(f"expected ENTRY PANEL ARM ROUND SERVICE_CPUS OBSERVER_CPU EST_MIN: {line!r}")
        entry, panel, arm, round_id, service, observer, minutes = fields
        jobs.append({"entry": int(entry), "panel": panel, "arm": arm, "round": int(round_id), "service": service,
                     "observer": int(observer), "est_s": 60 * float(minutes), "status": "pending"})
    return jobs, anchor


def finished_entries(text, agent=None):
    """Entry numbers recorded as DONE or SKIP (by `agent` only, when given)."""
    done = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ("DONE", "SKIP") and fields[1].isdigit():
            if agent is None or fields[-1] == agent:
                done.add(int(fields[1]))
    return done


def turn_has_come(entry, done_text):
    return set(range(1, entry)) <= finished_entries(done_text)


def fits(job, held_s, ran_in_hold):
    """Whether `job` may start in the current hold (the first job of a hold always may)."""
    return not ran_in_hold or held_s + job["est_s"] <= HOLD_S - MARGIN_S


def cpu_count(service):
    count = 0
    for part in service.split(","):
        low, _, high = part.partition("-")
        count += int(high or low) - int(low) + 1
    return count


def receipt_dir(runs, panel, service, round_id, arm):
    """Where open-run.sh writes this repetition's receipt."""
    group = ("1c" if cpu_count(service) == 1 else "8c") + ("" if panel == "full" else f"-{panel}")
    return runs / group / "cold" / f"round-{round_id}" / arm


def clean_wall_s(run):
    """Wall seconds of a finished repetition that was alone on the host, else None."""
    try:
        if json.loads((run / "host.cpu.json").read_text())["verdict"] != "clean":
            return None
        return json.loads((run / "measurement.json").read_text())["wall_seconds"]
    except (OSError, ValueError, KeyError):
        return None


def resolve(runs, job):
    """The arm to run for a job whose ARM may name two alternatives."""
    choices = job["arm"].split("|")
    walls = [(clean_wall_s(receipt_dir(runs, job["panel"], job["service"], 1, arm)), arm) for arm in choices]
    measured = [(wall, arm) for wall, arm in walls if wall is not None]
    return min(measured)[1] if measured else choices[0]


def next_round(runs, panel, service, arm):
    """One more than the highest round already on disk for this arm, panel and budget."""
    base = receipt_dir(runs, panel, service, 1, arm).parent.parent
    rounds = [int(path.parent.name.split("-")[1]) for path in base.glob(f"round-*/{arm}")
              if path.parent.name.split("-")[1].isdigit()]
    return max(rounds, default=0) + 1


class Host:
    def __init__(self, root):
        self.lock = root / "TIMING.lock"
        self.done = root / "TIMING.done"
        self.held = False

    def done_text(self):
        return self.done.read_text() if self.done.is_file() else ""

    def acquire(self, entry):
        while True:
            if turn_has_come(entry, self.done_text()):
                try:
                    self.lock.mkdir()
                except FileExistsError:
                    pass
                else:
                    self.held = True
                    start = datetime.now(timezone.utc)
                    until = start + timedelta(seconds=HOLD_S)
                    (self.lock / "owner").write_text(
                        f"{AGENT} {entry} pid={os.getpid()} start={utc(start)} until={utc(until)}\n")
                    return utc(start)
            time.sleep(POLL_S)

    def release(self):
        if self.held:
            (self.lock / "owner").unlink(missing_ok=True)
            self.lock.rmdir()
            self.held = False
        return utc()

    def mark_done(self, entry):
        with self.done.open("a") as out:
            out.write(f"DONE {entry} {utc()} {AGENT}\n")


def main():
    plan, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs, anchor = read_plan(plan.read_text())
    root = Path(os.environ.get("PAPER_OPPONENTS_ROOT",
                               Path(os.environ.get("JSR_DATA", Path.home() / "jsr-data")) / "opponents"))
    runs = Path(os.environ.get("OPEN_RUNS", root / "bench-open" / "runs"))
    script = os.environ.get("OPEN_RUN_SCRIPT", str(Path(__file__).resolve().with_name("open-run.sh")))
    host = Host(root)
    holds, measured = [], {}
    child = None

    def save():
        state = {"plan": str(plan), "pid": os.getpid(), "updated_utc": utc(), "holds": holds, "jobs": jobs}
        temporary = log_dir / "turn.json.tmp"
        temporary.write_text(json.dumps(state, indent=1) + "\n")
        temporary.replace(log_dir / "turn.json")

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        host.release()
        save()
        sys.exit(128 + signum)

    def run(job, hold):
        nonlocal child
        job["arm"] = resolve(runs, job)
        name = f"{job['arm']}.{job['panel']}.r{job['round']}"
        job.update(status="running", started_utc=utc())
        hold["jobs"].append(name)
        save()
        began = time.monotonic()
        with open(log_dir / f"{name}.log", "ab") as log:
            child = subprocess.Popen(
                ["bash", script, job["arm"], str(job["round"]), job["service"], str(job["observer"])],
                env={**os.environ, "OPEN_PANEL": job["panel"], "OPEN_RUNS": str(runs)}, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code in (0, 3):
            measured[(job["arm"], job["panel"], cpu_count(job["service"]))] = time.monotonic() - began
        job.update(status={0: "done", 3: "not-alone"}.get(code, "failed"), exit_code=code, ended_utc=utc())
        save()
        return code

    def estimate(job):
        arms = job["arm"].split("|")
        known = [measured[key] for key in ((arm, job["panel"], cpu_count(job["service"])) for arm in arms)
                 if key in measured]
        return 1.05 * max(known) if known else job["est_s"]

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        mine_done = finished_entries(host.done_text(), AGENT)
        for entry in dict.fromkeys(job["entry"] for job in jobs):
            pending = [job for job in jobs if job["entry"] == entry]
            if entry in mine_done:
                for job in pending:
                    job["status"] = "skipped-already-done"
                save()
                continue
            repeated = set()
            while pending:
                hold = {"entry": entry, "waiting_since_utc": utc(), "jobs": []}
                holds.append(hold)
                save()
                hold["acquired_utc"] = host.acquire(entry)
                began = time.monotonic()
                if anchor is not None:
                    first = pending[0]
                    service = first["service"].split(",")[0].split("-")[0]
                    marker = {"entry": entry, "panel": first["panel"], "arm": anchor["arm"], "service": service,
                              "observer": first["observer"], "anchor": True,
                              "est_s": anchor["est_s"][first["panel"]], "status": "pending"}
                    marker["round"] = next_round(runs, marker["panel"], service, anchor["arm"])
                    jobs.append(marker)
                    run(marker, hold)
                while pending:
                    job = pending[0]
                    job["est_s"] = estimate(job)
                    if not fits(job, time.monotonic() - began, bool(hold["jobs"]) and not (
                            anchor is not None and len(hold["jobs"]) == 1)):
                        break
                    pending.pop(0)
                    if run(job, hold) == 3 and id(job) not in repeated:
                        again = {**job, "status": "pending", "repeat_of": f"{job['arm']}.r{job['round']}"}
                        repeated.add(id(again))
                        jobs.append(again)
                        pending.append(again)
                hold["released_utc"] = host.release()
                hold["held_s"] = round(time.monotonic() - began, 1)
                save()
            host.mark_done(entry)
            save()
    finally:
        host.release()
        save()
    bad = [f"{job['arm']} {job['panel']} round {job['round']}" for job in jobs
           if job["status"] not in ("done", "skipped-already-done")]
    print(f"{utc()} plan finished; not done: {bad or 'none'}")
    sys.exit(1 if any(job["status"] == "failed" for job in jobs) else 0)


if __name__ == "__main__":
    main()
