"""Self-test for h50_measure.py (codex-lead 06:32Z fixtures): verdict needs 3 windows on both sides;
missing / partial GPU samples mark energy unavailable (never 0); time-to-WER only from pinned
canonical rows."""
import json, os, subprocess, sys, tempfile

TOOL = os.path.join(os.path.dirname(__file__), "..", "h50_measure.py")

def run_dir(d, rate, minutes, world=2, step_s=0.2, log_every=25):
    os.makedirs(d, exist_ok=True)
    t0, recs, step = 1.8e9, [], 0
    n = int(minutes * 60 / (step_s * log_every))
    for i in range(n + 1):
        recs.append({"kind": "train", "step": step, "time": t0 + i * step_s * log_every,
                     "audio_s": rate * step_s * log_every, "rank_step_s": [step_s * log_every] * world})
        step += log_every
    with open(f"{d}/train.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
    with open(f"{d}/config.json", "w") as fh:
        json.dump({"world": world, "train": {"ckpt_every_steps": 0}}, fh)
    return t0

def measure(*args):
    out = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)

with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tmp:
    run_dir(f"{tmp}/ref", 1000, 26, world=1)          # warm-up + 3 windows
    run_dir(f"{tmp}/one", 2000, 11)                   # warm-up + 1 window only
    r = measure(f"{tmp}/one", "--ref", f"{tmp}/ref", "--gate", "1.6")
    assert r["gate"]["verdict"] == "INSUFFICIENT_WINDOWS", r["gate"]
    run_dir(f"{tmp}/three", 2000, 26)
    r = measure(f"{tmp}/three", "--ref", f"{tmp}/ref", "--gate", "1.6")
    assert r["gate"]["verdict"] == "PASS" and r["n_windows"] == 3, r.get("gate")
    open(f"{tmp}/empty.csv", "w").close()
    r = measure(f"{tmp}/three", "--gpu-csv", f"{tmp}/empty.csv")
    assert all(w.get("energy") == "unavailable" and "total_J" not in w for w in r["windows"]), r["windows"][0]
    r = measure(f"{tmp}/three", "--target-wer", "5.0")
    assert r["time_to_wer"]["verdict"] == "REFUSED"
    with open(f"{tmp}/wer.jsonl", "w") as fh:
        fh.write(json.dumps({"step": 1000, "wer": 5.5, "ckpt_sha256": "a", "cohort": "dev-v2", "scorer": "ultra-eval@x", "weights": "raw"}) + "\n")
        fh.write(json.dumps({"step": 2000, "wer": 4.9, "ckpt_sha256": "b", "cohort": "dev-v2", "scorer": "ultra-eval@x", "weights": "raw"}) + "\n")
        fh.write(json.dumps({"step": 1500, "wer": 4.0, "ckpt_sha256": "c", "cohort": "dev-clean400", "scorer": "py", "weights": "raw"}) + "\n")
    r = measure(f"{tmp}/three", "--target-wer", "5.0", "--wer-file", f"{tmp}/wer.jsonl", "--cohort", "dev-v2",
                "--scorer", "ultra-eval@x", "--weights", "raw")
    assert r["time_to_wer"]["step"] == 2000 and r["time_to_wer"]["ckpt_sha256"] == "b", r["time_to_wer"]
print("h50_measure self-test: 5 checks passed")
