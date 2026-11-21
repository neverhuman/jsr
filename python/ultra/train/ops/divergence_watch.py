"""External divergence guard for runs started before the in-trainer guard (D2, D2MR).

  python3 divergence_watch.py <label> <train.jsonl (local or synced copy)> --pid <pid>
  python3 divergence_watch.py <label> <train.jsonl> --host <ssh host> --pid-file <remote PID file>

Same rules as train.trainer: after `after` steps, if the EMA of the logged train loss stays above
(1 + ratio) x its best for `patience` steps, or a dev eval is > wer_points worse than the best,
send SIGTERM to the trainer (it checkpoints and exits) and post an ALERT. Exits when the trainer
is gone. The process is signalled only if its command line is a train.trainer, so a reused PID is
never touched. No local shell is involved: commands are argument lists, and the host, PID file and
PID are validated before use.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time

CHAT = "/home/ubuntu/jsr/AGENT_CHAT.md"
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,62}")
PATH_RE = re.compile(r"/[A-Za-z0-9_./-]{1,511}")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("path", help="train.jsonl of the watched run")
    ap.add_argument("--pid", type=int, help="local trainer PID")
    ap.add_argument("--host", help="ssh host of a remote trainer (with --pid-file)")
    ap.add_argument("--pid-file", help="absolute path of the remote run's PID file")
    ap.add_argument("--after", type=int, default=3000)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--wer-points", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--chat", default=CHAT, help="file the ALERT is appended to")
    a = ap.parse_args(argv)
    if not a.label.strip():
        ap.error("label must be non-empty")
    if not os.path.isfile(a.path):
        ap.error(f"train log not found: {a.path}")
    if (a.pid is None) == (a.host is None):
        ap.error("give exactly one of --pid (local) or --host with --pid-file (remote)")
    if a.pid is not None and (a.pid <= 1 or a.pid_file):
        ap.error("--pid must be > 1 and is not combined with --pid-file")
    if a.host is not None:
        if not HOST_RE.fullmatch(a.host):
            ap.error(f"bad ssh host: {a.host!r}")
        if not a.pid_file or not PATH_RE.fullmatch(a.pid_file):
            ap.error("--host needs an absolute --pid-file of [A-Za-z0-9_./-] characters")
    if a.after < 0 or a.patience <= 0 or a.interval <= 0 or not 0 < a.ratio < 10 or a.wer_points <= 0:
        ap.error("need after >= 0, patience > 0, interval > 0, 0 < ratio < 10, wer-points > 0")
    return a


def record(line):
    """One train.jsonl line -> (kind, step, value) or None; a malformed line is skipped, a record
    with the right kind but missing or non-numeric fields is an error (the log format changed)."""
    try:
        r = json.loads(line)
    except ValueError:
        return None  # a partially written last line
    if not isinstance(r, dict):
        return None
    kind = r.get("kind")
    if kind == "train":
        key = "loss"
    elif kind == "eval" and r.get("wer") is not None:
        key = "wer"
    else:
        return None
    step, value = r.get("step"), r.get(key)
    if not isinstance(step, int) or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"malformed {kind} record: {line.strip()[:200]}")
    return kind, step, float(value)


def check(a):
    ema = best = None
    over_since = None
    best_wer = None
    with open(a.path) as fh:
        for line in fh:
            rec = record(line)
            if rec is None:
                continue
            kind, step, x = rec
            if kind == "train":
                ema = x if ema is None else 0.9 * ema + 0.1 * x  # records are 50-step means
                if step < a.after:
                    continue
                best = ema if best is None else min(best, ema)
                if ema > (1 + a.ratio) * best:
                    over_since = over_since or step
                    if step - over_since >= a.patience:
                        return (f"train-loss EMA {ema:.3f} > {1 + a.ratio} x best {best:.3f} "
                                f"since step {over_since}")
                else:
                    over_since = None
            else:
                if best_wer is not None and x > best_wer + a.wer_points:
                    return f"dev WER {x} at step {step} > best {best_wer} + {a.wer_points}"
                best_wer = x if best_wer is None else min(best_wer, x)
    return None


def trainer_cmd(a, action):
    """Argument list that checks the trainer is alive ("alive") or SIGTERMs it ("stop")."""
    if a.pid is not None:
        pid = str(a.pid)
        check = ["ps", "-p", pid, "-o", "args="]
        return check if action == "alive" else ["kill", "-TERM", pid]
    pf = shlex.quote(a.pid_file)
    remote = f'P=$(cat {pf}) && ps -p "$P" -o args= | grep -q train.trainer'
    if action == "stop":
        remote += ' && kill -TERM "$P"'
    return ["ssh", "-n", "-o", "ConnectTimeout=20", a.host, remote]


def alive(a):
    if a.pid is not None:
        r = subprocess.run(trainer_cmd(a, "alive"), capture_output=True, text=True)
        return r.returncode == 0 and "train.trainer" in r.stdout
    # remote: one retry so a transient ssh failure does not end the watch
    return any(subprocess.run(trainer_cmd(a, "alive")).returncode == 0 for _ in range(2))


def stop(a):
    if alive(a):
        subprocess.run(trainer_cmd(a, "stop"))


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    while True:
        why = check(a)
        if why:
            stop(a)
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            with open(a.chat, "a") as fh:
                fh.write(f"\n## {ts} | trainer-lane | ALERT {a.label} stopped by the external "
                         f"divergence guard\n- {why}\n")
            return 1
        if not alive(a):
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
