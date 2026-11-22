"""Pull a remote run's logs to xbabe3 every 10 min (ionice rsync) and post new dev evals to AGENT_CHAT.

  python3 sync_and_post.py <host> <remote_run_dir> <local_run_dir> <label>
Stops once the remote trainer PID is gone (after one last sync/post), leaving <local_run_dir>/.final.
Posted items are keyed by kind+step in <local_run_dir>/.posted.json, so a restarted mirror (e.g. by
mirror_pattern.sh) never re-posts an eval or a divergence alert.
"""
import datetime, json, os, subprocess, sys, time
host, rdir, ldir, label = sys.argv[1:5]
chat = "/home/ubuntu/jsr/AGENT_CHAT.md"
os.makedirs(ldir, exist_ok=True)
state_path = os.path.join(ldir, ".posted.json")
def _keys(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") in ("eval", "diverged"):
            out.add(f"{r['kind']}:{r.get('step')}")
    return out
try:
    posted = set(json.load(open(state_path)))
except (OSError, ValueError):
    posted = _keys(os.path.join(ldir, "train.jsonl"))  # first run: what the mirror already holds was posted
def _mark(key):
    posted.add(key)
    with open(state_path + ".tmp", "w") as fh:
        json.dump(sorted(posted), fh)
    os.replace(state_path + ".tmp", state_path)
def alive():
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"kill -0 $(cat {rdir}/PID)"], capture_output=True)
    return r.returncode == 0
while True:
    up = alive()
    subprocess.run(["ionice", "-c3", "nice", "-n", "10", "rsync", "-rt", "--include=*.jsonl", "--include=*.json",
                    "--include=*.log", "--include=PID", "--include=PGID", "--include=GUARD_PID", "--exclude=*",
                    f"{host}:{rdir}/", f"{ldir}/"], capture_output=True)
    path = os.path.join(ldir, "train.jsonl")
    if os.path.exists(path):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        evals = [r for r in recs if r.get("kind") == "eval"]
        trains = [r for r in recs if r.get("kind") == "train"]
        for r in evals:
            if f"eval:{r['step']}" in posted:
                continue
            t = max((x for x in trains if x["step"] <= r["step"]), key=lambda x: x["step"], default={})
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            with open(chat, "a") as fh:
                fh.write(f"\n## {ts} | trainer-lane | {label} dev-clean eval (auto, {host})\n"
                         f"- step {r['step']}, epoch {r.get('epoch')}: **WER {r['wer']}%** (sub {r['sub']} del {r['del']} "
                         f"ins {r['ins']}, 400 utts, greedy, chunk-masked), forced advances {r['forced_advances']}; "
                         f"train RNNT {t.get('rnnt', 0):.3f} CTC {t.get('ctc', 0):.3f}/token; "
                         f"{t.get('audio_h_per_h', 0):.0f} audio-h/h (guarded wall clock).\n")
            _mark(f"eval:{r['step']}")
        for r in [x for x in recs if x.get("kind") == "diverged"]:
            if f"diverged:{r.get('step')}" in posted:
                continue
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            with open(chat, "a") as fh:
                fh.write(f"\n## {ts} | trainer-lane | ALERT {label} stopped by the divergence guard ({host})\n"
                         f"- {r.get('error')}\n")
            _mark(f"diverged:{r.get('step')}")
    if not up:
        open(os.path.join(ldir, ".final"), "w").close()
        break
    time.sleep(600)
