"""Run timed repetitions on the shared host, one arm on the whole host at a time.

    open_turn.py PLAN LOG_DIR

Protocol of the timing host (`$PAPER_OPPONENTS_ROOT/TIMING.order`): an entry N may start only
when every entry listed before it in TIMING.order (ids such as `4b` included; the file is re-read
on every poll) is `DONE` or `SKIP` in `TIMING.done`; the host lock is the directory
`TIMING.lock` (made atomically, with an `owner` file, held for at most 30 minutes, removed when
the block ends); the agent appends `DONE N <UTC> <agent>` when its entry is finished.

PLAN lines (`#` starts a comment):
    ENTRY PANEL ARM ROUND SERVICE_CPUS OBSERVER_CPU EST_MIN   one repetition through open-run.sh
    anchor ARM EST_MIN_FULL EST_MIN_P73 [ENTRIES]             run ARM once at the start of every
                                                              hold (of the listed entries only, if a
                                                              comma-separated list is given; an entry's
                                                              own anchor wins over the general one), on
                                                              the panel of the hold's first repetition
                                                              and on its first CPU (or on its whole CPU
                                                              list for a multi-core anchor: -t8/-w8/-s8)
    require ENTRY PATH                                        the entry needs PATH (relative to the
                                                              opponents root) at its turn
ARM may be `A|B`: the one of the two with the shorter wall time in round 1 of this plan (both
clean); if neither has such a round-1 receipt, A.

The runner takes whichever of its entries has its turn first (normally file order); an entry
that is already DONE is skipped, so the runner can be restarted. Inside an entry the repetitions run one after another; a new hold starts when
the next repetition's estimate no longer fits in the current one. Once an arm has run, its
measured duration (plus 5%) replaces the plan's estimate. An entry that is DONE or SKIP is not
run again; its re-queued ids of this agent in TIMING.order (`Nr`, then `Nrr`, ...) run instead. A repetition that was not alone on
the host (exit 3) is repeated once at the end of its entry. If OPEN_AFTER_HOLD is set, that
shell command runs at the end of every hold, after its last repetition and before the lock is
released (for example, to copy the receipts off the host without loading anyone else's
measurement); a failure is logged and ignored. State: LOG_DIR/turn.json.

An entry with `require` lines is checked inside its first hold, after the shell command
OPEN_PREPARE (if set; for example, to copy a model in under the lock). If a required path is
missing, the runner appends `SKIP N <UTC> <agent> (reason; re-queued as Nr after M)` to
TIMING.done and re-queues the entry in TIMING.order as `Nr`, right after the next entry of
another agent (lead, 2026-11-23T03:20Z), then releases the lock; at most OPEN_REQUEUE_MAX (3)
times per plan entry, after which the entry is only skipped.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

AGENT = os.environ.get("OPEN_AGENT", "bench-open")
HOLD_S = float(os.environ.get("OPEN_HOLD_S", 30 * 60))
MARGIN_S = float(os.environ.get("OPEN_HOLD_MARGIN_S", 60))
POLL_S = float(os.environ.get("OPEN_POLL_S", 10))
AFTER_HOLD = os.environ.get("OPEN_AFTER_HOLD")
PREPARE = os.environ.get("OPEN_PREPARE")
REQUEUE_MAX = int(os.environ.get("OPEN_REQUEUE_MAX", 3))
ENTRY_ID = r"[0-9]+[a-z]*"  # 12, 6b, and re-queued ids such as 7r or 14br
ORDER_LINE = re.compile(r"\s*(" + ENTRY_ID + r")\s*\|\s*([^|]*?)\s*(?:\||$)")


def utc(moment=None):
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


MULTI_CORE = ("-t8", "-w8", "-s8")


def read_plan(text):
    """`(jobs, anchors)`: anchors maps an entry (or None, for every entry) to `{arm, est_s: {panel: seconds}}`."""
    jobs, anchors = [], {}
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "require":
            continue  # read_requirements
        if fields[0] == "anchor":
            if len(fields) not in (4, 5):
                raise ValueError(f"expected anchor ARM EST_MIN_FULL EST_MIN_P73 [ENTRIES]: {line!r}")
            spec = {"arm": fields[1], "est_s": {"full": 60 * float(fields[2]), "p73": 60 * float(fields[3])}}
            for entry in (fields[4].split(",") if len(fields) == 5 else [None]):
                anchors[entry] = spec
            continue
        if len(fields) != 7:
            raise ValueError(f"expected ENTRY PANEL ARM ROUND SERVICE_CPUS OBSERVER_CPU EST_MIN: {line!r}")
        entry, panel, arm, round_id, service, observer, minutes = fields
        if not re.fullmatch(ENTRY_ID, entry):
            raise ValueError(f"entry ids look like 12 or 6b: {line!r}")
        jobs.append({"entry": entry, "panel": panel, "arm": arm, "round": int(round_id), "service": service,
                     "observer": int(observer), "est_s": 60 * float(minutes), "status": "pending"})
    return jobs, anchors


def read_requirements(text):
    """`require ENTRY PATH` lines: entry -> the paths (relative to the opponents root) it needs at its turn."""
    requires = {}
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if fields and fields[0] == "require":
            if len(fields) != 3 or not re.fullmatch(ENTRY_ID, fields[1]):
                raise ValueError(f"expected require ENTRY PATH: {line!r}")
            requires.setdefault(fields[1], []).append(fields[2])
    return requires


def anchor_for(anchors, entry):
    """The anchor of an entry: its own, else the general one, else None."""
    return anchors.get(str(entry), anchors.get(None))


def anchor_service(arm, service):
    """A one-core anchor runs on the first CPU of the hold's first repetition, a multi-core one on all of them."""
    return service if arm.endswith(MULTI_CORE) else service.split(",")[0].split("-")[0]


def finished_entries(text, agent=None, kinds=("DONE", "SKIP")):
    """Entry ids (strings such as "5" or "4b") recorded as DONE or SKIP (by `agent` only, when given).

    Lines read `DONE N <UTC> <agent>` or `SKIP N <UTC> <agent> (reason)`."""
    done = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in kinds and re.fullmatch(ENTRY_ID, fields[1]):
            if agent is None or (fields[3] if len(fields) > 3 else fields[-1]) == agent:
                done.add(fields[1])
    return done


def order_entries(text):
    """`(id, agent, line_number)` of every `N | agent | ...` line of TIMING.order, in order."""
    return [(match.group(1), match.group(2), number)
            for number, match in enumerate(map(ORDER_LINE.match, text.splitlines())) if match]


def order_ids(text):
    """Entry ids in the order TIMING.order lists them (`N | agent | ...` lines)."""
    return [ident for ident, _, _ in order_entries(text)]


def family_ids(entry, order_text):
    """A plan entry and its re-queued ids of this agent in TIMING.order: N, Nr, Nrr, ..."""
    later = [ident for ident, agent, _ in order_entries(order_text)
             if agent == AGENT and re.fullmatch(re.escape(str(entry)) + "r+", ident)]
    return [str(entry)] + sorted(later, key=len)


def active_id(entry, order_text, done_text):
    """The id a plan entry runs under now: the first of its family that is neither DONE nor SKIP."""
    finished = finished_entries(done_text)
    return next((ident for ident in family_ids(entry, order_text) if ident not in finished), None)


def turn_has_come(entry, done_text, order_text=""):
    """Every entry before `entry` in the order file is finished; without the entry there, every lower number."""
    finished, order, entry = finished_entries(done_text), order_ids(order_text), str(entry)
    if entry in order:
        return set(order[:order.index(entry)]) <= finished
    return {str(number) for number in range(1, int(re.match(r"[0-9]+", entry).group()))} <= finished


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
        self.order = root / "TIMING.order"
        self.held = False

    def done_text(self):
        return self.done.read_text() if self.done.is_file() else ""

    def order_text(self):
        return self.order.read_text() if self.order.is_file() else ""

    def acquire(self, entry):
        """Wait for the entry's turn and take the lock; None if the entry was finished meanwhile."""
        while True:
            done_text = self.done_text()
            if str(entry) in finished_entries(done_text):
                return None
            if turn_has_come(entry, done_text, self.order_text()):
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

    def skip(self, entry, reason, requeue=True):
        """Append SKIP for the entry; with `requeue`, first list it again as `<entry>r` right after the next
        entry of another agent in TIMING.order (a backup of the old file stays beside it). The new id or None."""
        text = self.order_text()
        lines, entries = text.splitlines(), order_entries(text)
        position = next((index for index, (ident, _, _) in enumerate(entries) if ident == str(entry)), None)
        new_id, note = None, reason
        if requeue and position is not None:
            new_id = f"{entry}r"
            later = entries[position + 1:]
            index = next((number for number, item in enumerate(later) if item[1] != AGENT), None)
            after = None if index is None else later[index]
            at = len(lines) if after is None else after[2] + 1
            for ident, agent, number in ([] if index is None else later[index + 1:]):
                if agent != AGENT or not ident.endswith("r") or number != at:
                    break
                at += 1  # behind entries of this agent re-queued there before, in the order they were skipped
            line = re.sub(r"\s+# re-queued by .*$", "", lines[entries[position][2]])
            line = re.sub(r"^(\s*)" + re.escape(str(entry)) + r"(\s*\|)", r"\g<1>" + new_id + r"\g<2>", line, count=1)
            lines.insert(at, f"{line}  # re-queued by {AGENT} {utc()} after SKIP {entry}: {reason}")
            stamp = datetime.now(timezone.utc).strftime("%H%M%S")
            self.order.with_name(f"TIMING.order.bak-{AGENT}-{stamp}").write_text(text)
            temporary = self.order.with_name(f".TIMING.order.{os.getpid()}.tmp")
            temporary.write_text("\n".join(lines) + "\n")
            temporary.replace(self.order)
            note = f"{reason}; re-queued as {new_id} after {after[0] if after else 'the last entry'}"
        with self.done.open("a") as out:
            out.write(f"SKIP {entry} {utc()} {AGENT} ({note})\n")
        return new_id


def main():
    plan, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs, anchors = read_plan(plan.read_text())
    requires = read_requirements(plan.read_text())
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

    def hook(command, log_name):
        with open(log_dir / log_name, "ab") as log:
            try:
                return subprocess.run(["bash", "-c", command], stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired:
                return "timeout"

    def run_entry(family, active, pending):
        """The pending repetitions of plan entry `family`, run as TIMING entry `active`: "done", "skipped"
        (a required path was missing; the entry was skipped and re-queued) or "finished-elsewhere"."""
        repeated, checked = set(), family not in requires
        while pending:
            hold = {"entry": active, "waiting_since_utc": utc(), "jobs": []}
            holds.append(hold)
            save()
            acquired = host.acquire(active)
            if acquired is None:
                hold["outcome"] = "finished elsewhere"
                save()
                return "finished-elsewhere"
            hold["acquired_utc"] = acquired
            began = time.monotonic()
            if not checked:
                checked = True
                if PREPARE:
                    hold["prepare_exit"] = hook(PREPARE, "prepare.log")
                missing = [path for path in requires[family] if not (root / path).exists()]
                if missing:
                    hold["skipped"] = reason = "not staged at its turn: " + ", ".join(missing)
                    hold["requeued_as"] = host.skip(active, reason, len(active) - len(family) < REQUEUE_MAX)
                    hold["released_utc"] = host.release()
                    save()
                    return "skipped"
            anchor = anchor_for(anchors, family)
            if anchor is not None:
                first = pending[0]
                service = anchor_service(anchor["arm"], first["service"])
                marker = {"entry": active, "panel": first["panel"], "arm": anchor["arm"], "service": service,
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
            if AFTER_HOLD:
                hold["after_hold_exit"] = hook(AFTER_HOLD, "after-hold.log")
            hold["released_utc"] = host.release()
            hold["held_s"] = round(time.monotonic() - began, 1)
            save()
        return "done"

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    families = {}
    for job in jobs:
        families.setdefault(job["entry"], []).append(job)
    try:
        while True:
            order_text, done_text = host.order_text(), host.done_text()
            ready, open_count = [], 0
            for family, plan_jobs in families.items():
                active = active_id(family, order_text, done_text)
                pending = [job for job in plan_jobs if job["status"] == "pending"]
                if active is None or not pending:
                    ran = finished_entries(done_text, kinds=("DONE",)) & set(family_ids(family, order_text))
                    for job in pending:
                        job["status"] = "skipped-already-done" if ran else "skipped"
                    continue
                open_count += 1
                if turn_has_come(active, done_text, order_text):
                    ready.append((family, active, pending))
            if not open_count:
                break
            if not ready:
                time.sleep(POLL_S)
                continue
            order = order_ids(order_text)
            family, active, pending = min(ready, key=lambda item: order.index(item[1]) if item[1] in order else len(order))
            if run_entry(family, active, pending) == "done":
                host.mark_done(active)
            save()
    finally:
        host.release()
        save()
    bad = [f"{job['arm']} {job['panel']} round {job['round']}" for job in jobs
           if job["status"] not in ("done", "skipped-already-done", "skipped")]
    print(f"{utc()} plan finished; not done: {bad or 'none'}")
    sys.exit(1 if any(job["status"] == "failed" for job in jobs) else 0)


if __name__ == "__main__":
    main()
