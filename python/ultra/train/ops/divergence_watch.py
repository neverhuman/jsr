"""External divergence guard for runs started before the in-trainer guard (D2, D2MR).

  python3 divergence_watch.py <label> <train.jsonl (local or synced copy)> <stop command> <alive command>

Same rules as train.trainer: after `after` steps, if the EMA of the logged train loss stays above
(1 + ratio) x its best for `patience` steps, or a dev eval is > wer_points worse than the best,
run the stop command (SIGTERM to the trainer: it checkpoints and exits) and post an ALERT.
Exits when the alive command fails (trainer finished). Both commands check the process is a
train.trainer before acting, so a reused PID is never signalled.
"""
import datetime, json, subprocess, sys, time

label, path, stop_cmd, alive_cmd = sys.argv[1:5]
after, ratio, patience, wer_points = 3000, 0.3, 300, 5.0
CHAT = "/home/ubuntu/jsr/AGENT_CHAT.md"


def check():
    ema = best = None
    over_since = None
    best_wer = None
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == "train":
            x = r["loss"]
            ema = x if ema is None else 0.9 * ema + 0.1 * x  # records are 50-step means
            if r["step"] < after:
                continue
            best = ema if best is None else min(best, ema)
            if ema > (1 + ratio) * best:
                over_since = over_since or r["step"]
                if r["step"] - over_since >= patience:
                    return f"train-loss EMA {ema:.3f} > {1 + ratio} x best {best:.3f} since step {over_since}"
            else:
                over_since = None
        elif r.get("kind") == "eval" and r.get("wer") is not None:
            if best_wer is not None and r["wer"] > best_wer + wer_points:
                return f"dev WER {r['wer']} at step {r['step']} > best {best_wer} + {wer_points}"
            best_wer = r["wer"] if best_wer is None else min(best_wer, r["wer"])
    return None


while True:
    why = check()
    if why:
        subprocess.run(stop_cmd, shell=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        with open(CHAT, "a") as fh:
            fh.write(f"\n## {ts} | trainer-lane | ALERT {label} stopped by the external divergence guard\n- {why}\n")
        break
    if subprocess.run(alive_cmd, shell=True).returncode != 0:
        break
    time.sleep(300)
