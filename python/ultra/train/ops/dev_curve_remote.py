"""DEV curve for a run on another host: every `every` steps pull model-<step>.pt (+sha256) to H0,
verify it, score dev-clean and dev-other (400 utts each, greedy, chunk-masked) on H0 CPU cores
12-15 with the MKL torch build, append rows to <local>/dev_curve.jsonl under flock (validated
JSON with the expected step/split), and post a summary to AGENT_CHAT.

  python3 dev_curve_remote.py <host> <remote_run_dir> <local_dir> <label> [every=10000]
"""
import datetime, fcntl, hashlib, json, os, re, subprocess, sys, time

host, rdir, ldir, label = sys.argv[1:5]
every = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
REPO = "/home/ubuntu/jsr/.agent/worktrees/trainer"
CHAT = "/home/ubuntu/jsr/AGENT_CHAT.md"
OUT = os.path.join(ldir, "dev_curve.jsonl")
SPLITS = ("dev-clean", "dev-other")
os.makedirs(ldir, exist_ok=True)


def locked_append(path, text):
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def done():
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                s.add((r["step"], r["split"]))
            except (ValueError, KeyError):
                pass
    return s


def ssh(cmd):
    return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)


def evaluate(ckpt, split, step):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="/mnt/xbabe3-external/jsr/venvs/torch270-cpu")
    out = subprocess.run(["taskset", "-c", "12-15", "nice", "-n", "10", f"{REPO}/.venv/bin/python", "-m",
                          "train.evaluate", "--run-dir", ldir, "--checkpoint", ckpt, "--split", split,
                          "--limit", "400", "--threads", "4"], cwd=f"{REPO}/python/ultra", env=env,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("step") == step and r.get("split") == split:
                return r
    raise RuntimeError(f"no valid row for {ckpt} {split}")


while True:
    alive = ssh(f"kill -0 $(cat {rdir}/PID)").returncode == 0
    names = ssh(f"ls {rdir} | grep -E '^model-[0-9]+\\.sha256$'").stdout.split()
    have = done()
    for n in sorted(names):
        step = int(re.search(r"(\d+)", n).group(1))
        if step % every or all((step, s) in have for s in SPLITS):
            continue
        base = n[:-7]
        subprocess.run(["ionice", "-c3", "nice", "-n", "10", "rsync", "-t", f"{host}:{rdir}/{base}.pt",
                        f"{host}:{rdir}/{base}.sha256", ldir + "/"], check=True)
        pt = os.path.join(ldir, base + ".pt")
        h = hashlib.sha256(open(pt, "rb").read()).hexdigest()
        if h != open(os.path.join(ldir, base + ".sha256")).read().strip():
            raise RuntimeError(f"sha256 mismatch for {pt}")
        parts = []
        for split in SPLITS:
            if (step, split) in have:
                continue
            r = evaluate(pt, split, step)
            locked_append(OUT, json.dumps(r) + "\n")
            parts.append(f"{split} **{r['wer']}%** (S {r['sub']} D {r['del']} I {r['ins']}; CTC {r['ctc_wer']}%)")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        locked_append(CHAT, f"\n## {ts} | trainer-lane | {label} DEV curve (auto): {base}.pt\n"
                            f"- 400 utts each, greedy, chunk-masked, H0 CPU: " + "; ".join(parts) + "\n")
    if not alive:
        break
    time.sleep(600)
