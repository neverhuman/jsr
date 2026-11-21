"""A1: DEV WER curve of D2 on eval-only snapshots model-<step>.pt every 10k steps (CPU, cores 12-15).

Appends one JSON row per (step, split) to dev_curve.jsonl under an exclusive flock, verifies the
row is valid JSON with the expected step/split before appending, and posts a summary to AGENT_CHAT
(also under flock). Exits when the trainer is gone and every 10k snapshot is evaluated.
"""
import datetime, fcntl, glob, json, os, re, subprocess, time

D2 = "/mnt/xbabe3-external/jsr/runs/train/D2_S_960h"
REPO = "/home/ubuntu/jsr/.agent/worktrees/trainer"
CHAT = "/home/ubuntu/jsr/AGENT_CHAT.md"
OUT = os.path.join(D2, "dev_curve.jsonl")
SPLITS = ("dev-clean", "dev-other")


def locked_append(path, text):
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def done_rows():
    rows = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                rows.add((r["step"], r["split"]))
            except (ValueError, KeyError):
                continue
    return rows


def trainer_alive():
    try:
        os.kill(int(open(os.path.join(D2, "PID")).read()), 0)
        return True
    except (OSError, ValueError):
        return False


def evaluate(ckpt, split):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="/mnt/xbabe3-external/jsr/venvs/torch270-cpu")
    cmd = ["taskset", "-c", "12-15", "nice", "-n", "10", f"{REPO}/.venv/bin/python", "-m", "train.evaluate",
           "--run-dir", D2, "--checkpoint", ckpt, "--split", split, "--limit", "400", "--threads", "4"]
    out = subprocess.run(cmd, cwd=f"{REPO}/python/ultra", env=env, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no result for {ckpt} {split}")


while True:
    have = done_rows()
    for sha in sorted(glob.glob(os.path.join(D2, "model-*.sha256"))):
        step = int(re.search(r"model-(\d+)", sha).group(1))
        if step % 10000:
            continue
        todo = [s for s in SPLITS if (step, s) not in have]
        if not todo:
            continue
        parts = []
        for split in SPLITS:
            if split in todo:
                r = evaluate(sha[:-7] + ".pt", split)
                if r.get("step") != step or r.get("split") != split:
                    raise RuntimeError(f"unexpected row {r}")
                locked_append(OUT, json.dumps(r) + "\n")
            else:
                r = next(json.loads(l) for l in open(OUT) if f'"step": {step},' in l and f'"split": "{split}"' in l)
            parts.append(f"{split} **{r['wer']}%** (S {r['sub']} D {r['del']} I {r['ins']}; CTC {r['ctc_wer']}%)")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        locked_append(CHAT, f"\n## {ts} | trainer-lane | A1 D2 DEV curve (auto): model-{step:08d}.pt\n"
                            f"- 400 utts each, greedy, chunk-masked (== streaming): " + "; ".join(parts) + "\n")
    if not trainer_alive():
        break
    time.sleep(300)
