"""Pull a remote run's logs to xbabe3 every 10 min (ionice rsync) and post new dev evals to AGENT_CHAT.

  python3 sync_and_post.py <host> <remote_run_dir> <local_run_dir> <label>
Stops once the remote trainer PID is gone (after one last sync/post).
"""
import datetime, json, os, subprocess, sys, time
host, rdir, ldir, label = sys.argv[1:5]
chat = "/home/ubuntu/jsr/AGENT_CHAT.md"
os.makedirs(ldir, exist_ok=True)
seen = 0
posted_div = 0
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
        for r in evals[seen:]:
            t = max((x for x in trains if x["step"] <= r["step"]), key=lambda x: x["step"], default={})
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            with open(chat, "a") as fh:
                fh.write(f"\n## {ts} | trainer-lane | {label} dev-clean eval (auto, {host})\n"
                         f"- step {r['step']}, epoch {r.get('epoch')}: **WER {r['wer']}%** (sub {r['sub']} del {r['del']} "
                         f"ins {r['ins']}, 400 utts, greedy, chunk-masked), forced advances {r['forced_advances']}; "
                         f"train RNNT {t.get('rnnt', 0):.3f} CTC {t.get('ctc', 0):.3f}/token; "
                         f"{t.get('audio_h_per_h', 0):.0f} audio-h/h (guarded wall clock).\n")
        seen = len(evals)
        for r in [x for x in recs if x.get("kind") == "diverged"][posted_div:]:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            with open(chat, "a") as fh:
                fh.write(f"\n## {ts} | trainer-lane | ALERT {label} stopped by the divergence guard ({host})\n"
                         f"- {r.get('error')}\n")
            posted_div += 1
    if not up:
        break
    time.sleep(600)
