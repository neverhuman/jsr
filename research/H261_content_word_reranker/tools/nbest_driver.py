"""H261: native c32 B16 M5 n-best on an arbitrary PCM npz (pcm/offsets/uids), reusing codex's dev_eval worker protocol.
nbest_driver.py PCM.npz OUT_DIR host=N[,host=N]   (env: MODEL, CHUNK=32, BEAM=16)
Writes OUT_DIR/<host>.json worker reports and OUT_DIR/nbest.jsonl ({uid, nbest:[[tokens, score, hyp]]})."""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/jsr/.agent/worktrees/wer-codex/python/ultra")
from train.ops import dev_eval as dev  # noqa: E402

PUSH = Path("/mnt/xbabe3-external/jsr/wer-push")
pcm_path, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
workers = dev.worker_counts(sys.argv[3])
args = argparse.Namespace(
    weights=Path(os.environ.get("MODEL", PUSH / "H234/step2000/ema-int8")).resolve(),
    decoder=(PUSH / "H133/target/release/ultra-dev-worker").resolve(),
    mac_decoder=dev.WAVE / "target-mac/aarch64-apple-darwin/release/ultra-dev-worker",
    chunk=int(os.environ.get("CHUNK", 32)), beam=int(os.environ.get("BEAM", 16)), precision="int8",
    lm=PUSH / "shared/isosyll/isosyll-model/lm/final.ulm", lm_weight=0.80, lodr=0.60, blank_penalty=0.0,
    nbest=True, features=False, batch=8)
uids = dev.mapped_array(pcm_path, "uids"); offs = dev.mapped_array(pcm_path, "offsets")
clips = [dev.Clip(str(u), "set", i, int(offs[i]), int(offs[i + 1])) for i, u in enumerate(uids)]
sources = {"set": pcm_path}
arch = json.loads((args.weights / "arch.json").read_text())
assign = {h: b for h, b in dev.assign_batches(clips, args.batch, workers).items() if b}
out.mkdir(parents=True, exist_ok=False)
def run(host):
    dec = args.mac_decoder if host == "xbabe4" else args.decoder
    src = lambda stream: dev.write_pcm(stream, assign[host], clips, sources)
    t = dev.run_worker(args, host, dec, arch, src, out / f"{host}.json", workers[host])
    return host, t
with ThreadPoolExecutor(len(assign)) as pool:
    times = dict(pool.map(run, assign))
rows = []
for h in assign:
    rep = json.loads((out / f"{h}.json").read_text()); rows += rep["rows"]
    print(h, "cpu_s", rep["cpu_s"], "wall", round(times[h]), flush=True)
with open(out / "nbest.jsonl", "w") as fh:
    for r in sorted(rows, key=lambda r: r["index"]):
        fh.write(json.dumps({"uid": clips[r["index"]].uid, "nbest": [[c["tokens"], c["score"], c["hyp"]] for c in r["nbest"]]}) + "\n")
print("rows", len(rows), "of", len(clips))
