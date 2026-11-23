"""H52 student hyps for the agreement filter: A1-49000 EMA INT8 (native engine) on Libri-Light shards.

Runs on a worker host (xbabe1 / xbabe2 / x5 ...). It claims ready shards in H0's hand-off dir
with `mkdir NNNN.student.claim` (the R0 labellers' NNNN.claim is never touched), pulls the parquet
over the LAN, checks its sha256 against NNNN.ready, splits it into npz chunks (the engine's
reference-blind format: pcm <i2, offsets <i8, uids <U, speakers <i8, texts <U, all empty texts),
runs one single-threaded `ultra-engine-eval --precision int8` per core, and writes back
NNNN.student.jsonl (header + {"id", "hyp", "tokens"} per utterance) then NNNN.student.done.
The number of concurrent engines follows a host CPU cap (busy fraction from /proc/stat): it
shrinks above --cpu-cap and grows below cap - 0.1, between 1 and --max-procs. All at nice 19.

usage: python -I ll_student.py --remote xbabe3:/dev/shm/h52/raw/librilight.small --engine BIN
        --weights DIR --work /dev/shm/h52-student --max-procs 60 [--cpu-cap 0.80] [--chunk 100]
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def ssh(host, cmd, check=False):
    if host == "local":
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=check)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd],
                          capture_output=True, text=True, check=check)


def copy(host, src, dst):
    if host == "local":
        return subprocess.run(["cp", src, dst], capture_output=True, text=True)
    return subprocess.run(["rsync", "-a", f"{host}:{src}", dst], capture_output=True, text=True)


def cpu_busy(prev):
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle, total = v[3] + v[4], sum(v[:8])
    if prev is None:
        return None, (idle, total)
    di, dt = idle - prev[0], total - prev[1]
    return (1.0 - di / dt) if dt > 0 else 0.0, (idle, total)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def to_npz_chunks(parquet, outdir, key, chunk):
    pf = pq.ParquetFile(parquet)
    t = pf.read(columns=["id", "speaker_id", "audio"])
    ids = t.column("id").to_pylist()
    spk = t.column("speaker_id").to_pylist()
    auds = t.column("audio").to_pylist()
    paths = []
    for c0 in range(0, len(ids), chunk):
        pcms, offs = [], [0]
        for a in auds[c0:c0 + chunk]:
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
            assert sr == 16000 and x.ndim == 1
            pcms.append(x)
            offs.append(offs[-1] + len(x))
        sub = ids[c0:c0 + chunk]
        p = os.path.join(outdir, f"{key}_c{c0 // chunk:03d}.npz")
        np.savez(p, pcm=np.concatenate(pcms).astype("<i2"), offsets=np.asarray(offs, dtype="<i8"),
                 uids=np.asarray(sub, dtype=f"<U{max(len(s) for s in sub)}"),
                 speakers=np.asarray(spk[c0:c0 + chunk], dtype="<i8"), texts=np.asarray([""] * len(sub), dtype="<U1"))
        paths.append((p, len(sub)))
    return paths, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True, help="host:/dev/shm/h52/raw/librilight.<subset>")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--max-procs", type=int, default=32)
    ap.add_argument("--start-procs", type=int, default=0)
    ap.add_argument("--cpu-cap", type=float, default=0.80)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--ahead", type=int, default=0, help="shards in flight (default: procs/8 + 2)")
    ap.add_argument("--exit-when-done", action="store_true", help="exit when PASS2_DONE.json exists and nothing is left")
    ap.add_argument("--min-avail-gb", type=float, default=12.0, help="start no engine below this MemAvailable")
    ap.add_argument("--max-file", default="", help="optional file holding a runtime max-procs override")
    ap.add_argument("--loader", default="", help="dir with ld-linux-x86-64.so.2 + libc/libm/libgcc_s of the build host")
    a = ap.parse_args()
    host, rdir = a.remote.split(":", 1)
    me = os.uname().nodename
    os.makedirs(a.work, exist_ok=True)
    eng_sha = sha256_file(a.engine)
    wsha = {n: sha256_file(os.path.join(a.weights, n)) for n in sorted(os.listdir(a.weights))}
    header = {"kind": "header", "model": "A1-49000 EMA INT8", "engine_sha256": eng_sha, "weights_sha256": wsha,
              "args": ["--precision", "int8", "--streams", str(a.streams)], "host": me,
              "note": "student greedy hyps for the H52 agreement filter"}
    target = a.start_procs or max(1, a.max_procs // 2)
    import signal

    def _term(signum, frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, _term)
    try:
        _loop(a, host, rdir, me, header, target)
    finally:  # kill our engines and release the claims of unfinished shards
        for pr in list(_RUNNING):
            try:
                pr.kill()
            except Exception:
                pass
        for k in list(_SHARDS):
            ssh(host, f"test -e {rdir}/{k}.student.done || rm -rf {rdir}/{k}.student.claim")
            log(f"released claim {k}")


_RUNNING, _SHARDS = {}, {}


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def _loop(a, host, rdir, me, header, target):
    running = _RUNNING  # Popen -> (key, chunk_path, outdir)
    queue = deque()  # (key, chunk_path, n)
    shards = _SHARDS  # key -> {"left": n_chunks, "rows": [], "ready": meta}
    prev = None
    _, prev = cpu_busy(None)
    last_adj, last_list, last_log = time.time(), 0.0, 0.0
    done_count = 0
    fails = {}
    while True:
        now = time.time()
        # --- CPU cap feedback
        if now - last_adj > 20:
            busy, prev = cpu_busy(prev)
            last_adj = now
            mx = a.max_procs
            if a.max_file and os.path.exists(a.max_file):
                try:
                    mx = int(open(a.max_file).read().strip())
                except ValueError:
                    pass
            avail = mem_avail_gb()
            if avail < a.min_avail_gb / 2 and target > 1:
                target = max(1, int(target * 0.85))
            elif avail < a.min_avail_gb:
                target = min(target, len(running))
            target = min(target, mx)
            if busy is not None and avail >= a.min_avail_gb:
                if busy > a.cpu_cap and target > 1:
                    target = max(1, int(target * 0.85))
                elif busy < a.cpu_cap - 0.10 and target < mx:
                    target = min(mx, target + 4)
            if now - last_log > 120:
                last_log = now
                log(f"busy={busy if busy is None else round(busy, 3)} avail={avail:.1f}GB target={target} running={len(running)} "
                    f"queued_chunks={len(queue)} shards_in_flight={len(shards)} done={done_count}")
        # --- claim more shards
        ahead = a.ahead or (target // 8 + 2)
        if len(shards) < ahead and now - last_list > 10:
            last_list = now
            r = ssh(host, f"ls {rdir}")
            names = set(r.stdout.split()) if r.returncode == 0 else set()
            ready = sorted(n[:-6] for n in names if n.endswith(".ready"))
            cands = [k for k in ready if k + ".student.done" not in names and k + ".student.claim" not in names
                     and k + ".student.onprem" not in names and k not in shards
                     and (host != "local" or k + ".parquet" in names)]
            for k in cands:
                if len(shards) >= ahead:
                    break
                if ssh(host, f"mkdir {rdir}/{k}.student.claim && echo {me} > {rdir}/{k}.student.claim/owner").returncode != 0:
                    continue
                meta = json.loads(ssh(host, f"cat {rdir}/{k}.ready", check=True).stdout)
                lp = os.path.join(a.work, k + ".parquet")
                rr = copy(host, f"{rdir}/{k}.parquet", lp)
                if rr.returncode == 0 and os.path.exists(lp) and sha256_file(lp) == meta["sha256"]:
                    ssh(host, f"touch {rdir}/{k}.student.fetched")
                if rr.returncode != 0 or not os.path.exists(lp) or sha256_file(lp) != meta["sha256"]:
                    log(f"fetch failed / sha mismatch for {k}; releasing claim")
                    ssh(host, f"rm -rf {rdir}/{k}.student.claim")
                    if os.path.exists(lp):
                        os.remove(lp)
                    continue
                cd = os.path.join(a.work, k)
                os.makedirs(cd, exist_ok=True)
                chunks, n = to_npz_chunks(lp, cd, k, a.chunk)
                os.remove(lp)
                shards[k] = {"left": len(chunks), "rows": [], "ready": meta, "n": n, "t0": time.time()}
                for p, m in chunks:
                    queue.append((k, p, m))
                log(f"claimed {k}: {n} rows, {len(chunks)} chunks")
        # --- start engines up to target
        while queue and len(running) < target:
            k, p, m = queue.popleft()
            od = p[:-4] + ".out"
            os.makedirs(od, exist_ok=True)
            pre = ([os.path.join(a.loader, "ld-linux-x86-64.so.2"), "--library-path", a.loader] if a.loader else [])
            cmd = ["nice", "-n", "19"] + pre + [a.engine, "--weights", a.weights, "--npz", p, "--precision", "int8",
                   "--streams", str(a.streams), "--limit", "100000", "--out", od]
            pr = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(od + ".err", "w"))
            running[pr] = (k, p, od)
        # --- reap
        for pr in list(running):
            if pr.poll() is None:
                continue
            k, p, od = running.pop(pr)
            if pr.returncode != 0:
                fails[p] = fails.get(p, 0) + 1
                log(f"engine failed rc={pr.returncode} ({fails[p]}x) on {p}: {open(od + '.err').read()[-300:]}")
                if fails[p] < 2:
                    queue.append((k, p, 0))
                    continue
                if k in shards:  # give the shard up: drop its queued chunks, release the claim
                    shards.pop(k)
                    for item in [q for q in queue if q[0] == k]:
                        queue.remove(item)
                    ssh(host, f"rm -rf {rdir}/{k}.student.claim")
                    shutil.rmtree(os.path.join(a.work, k), ignore_errors=True)
                    log(f"GAVE UP on {k}; claim released")
                continue
            if k not in shards:  # a chunk of a shard we gave up on
                continue
            res = json.load(open(os.path.join(od, "engine_eval.json")))
            uids = np.load(p)["uids"].tolist()
            for row in res["rows"]:
                shards[k]["rows"].append({"id": uids[row["index"]] if "index" in row else row["id"],
                                          "hyp": row.get("hyp", ""), "tokens": row.get("tokens")})
            shards[k]["left"] -= 1
            shutil.rmtree(od, ignore_errors=True)
            os.remove(p)
            if shards[k]["left"] == 0:
                s = shards.pop(k)
                if len(s["rows"]) != s["n"]:
                    log(f"ROW COUNT MISMATCH {k}: {len(s['rows'])} != {s['n']}")
                lj = os.path.join(a.work, k + ".student.jsonl")
                with open(lj, "w") as f:
                    f.write(json.dumps({**header, "shard": k + ".parquet", "shard_sha256": s["ready"]["sha256"],
                                        "rows": len(s["rows"])}) + "\n")
                    for row in sorted(s["rows"], key=lambda x: x["id"]):
                        f.write(json.dumps(row) + "\n")
                rr = (subprocess.run(["cp", lj, f"{rdir}/{k}.student.jsonl.part"], capture_output=True, text=True)
                      if host == "local" else
                      subprocess.run(["rsync", "-a", lj, f"{host}:{rdir}/{k}.student.jsonl.part"], capture_output=True, text=True))
                if rr.returncode == 0 and ssh(host, f"mv {rdir}/{k}.student.jsonl.part {rdir}/{k}.student.jsonl && "
                                                    f"echo {me} > {rdir}/{k}.student.done").returncode == 0:
                    done_count += 1
                    log(f"done {k}: {len(s['rows'])} rows in {time.time() - s['t0']:.0f}s (total {done_count})")
                    os.remove(lj)
                else:
                    log(f"write-back failed for {k}; kept {lj}")
                shutil.rmtree(os.path.join(a.work, k), ignore_errors=True)
        if a.exit_when_done and not running and not queue and not shards and now - last_list < 1:
            r = ssh(host, f"ls {rdir}")
            names = set(r.stdout.split())
            if "PASS2_DONE.json" in names and all(n[:-6] + ".student.done" in names for n in names if n.endswith(".ready")):
                log(f"all shards have student hyps; exiting (this host did {done_count})")
                return
        time.sleep(0.5)


if __name__ == "__main__":
    main()
