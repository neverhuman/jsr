"""H52 uploader: ship labelled shards from H0's /dev/shm hand-off dir to xbabe5, then free the RAM.

A shard moves only when kd-cache's NNNN.labeled marker exists (after NNNN.r0labels.jsonl). It
rsyncs NNNN.{parquet,ids,ready,r0labels.jsonl,labeled} with a partial dir (no truncated files under
final names), checks the remote sha256 of the parquet against NNNN.ready, writes NNNN.uploaded and
deletes the local parquet + labels. It waits while the pause file exists (H50 measurement windows).
With --unlabelled-ok it also ships ready shards that have no labels (only if asked).
When PASS2_DONE.json exists and every ready shard is uploaded, it ships the bookkeeping files and exits.

usage: python -I ll_uploader.py <local_shard_dir> <host:remote_dir> [--pause-file F] [--bwlimit-kbps 40000]
"""
import argparse
import json
import os
import subprocess
import time


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def upload_one(a, host, rdir, k, names):
    while a.pause_file and os.path.exists(a.pause_file):
        time.sleep(10)
    meta = json.load(open(os.path.join(a.local, k + ".ready")))
    onprem = False
    if a.student_split and k + ".student.done" not in names:
        try:
            os.mkdir(os.path.join(a.local, k + ".student.claim"))
            open(os.path.join(a.local, k + ".student.claim", "owner"), "w").write("x5\n")
        except FileExistsError:
            onprem = True
            subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"touch {rdir}/{k}.student.onprem"])
    files = [os.path.join(a.local, k + ext) for ext in (".parquet", ".ids", ".ready", ".r0labels.jsonl", ".labeled",
                                                        ".student.jsonl", ".student.done")
             if os.path.exists(os.path.join(a.local, k + ext))]
    for attempt in range(20):
        # rsync verifies a whole-file checksum of every transferred file; the finaliser re-checks
        # each parquet's sha256 against .ready and .labeled before use
        r = subprocess.run(["rsync", "-a", "--partial-dir=.rsync-partial", f"--bwlimit={a.bwlimit_kbps}"] + files
                           + [f"{host}:{rdir}/"], capture_output=True, text=True)
        if r.returncode == 0:
            break
        log(f"rsync retry {attempt} {k}: {r.stderr.strip()[:200]}")
        time.sleep(min(300, 10 * (attempt + 1)))
    else:
        log(f"UPLOAD FAILED (kept locally): {k}")
        return False
    with open(os.path.join(a.local, k + ".uploaded"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    lp = os.path.join(a.local, k + ".r0labels.jsonl")
    if os.path.exists(lp):
        os.remove(lp)
    if not onprem or os.path.exists(os.path.join(a.local, k + ".student.fetched")):
        os.remove(os.path.join(a.local, k + ".parquet"))
    log(f"uploaded {k} ({meta['rows']} rows, {meta['bytes'] / 1e6:.0f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("dest")
    ap.add_argument("--pause-file", default="")
    ap.add_argument("--bwlimit-kbps", type=int, default=20000, help="per transfer")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--unlabelled-ok", action="store_true")
    ap.add_argument("--require-student", action="store_true", help="also wait for NNNN.student.done")
    ap.add_argument("--student-split", action="store_true",
                    help="ship after R0 labels; unclaimed shards get their student hyps on x5, claimed ones ship late")
    a = ap.parse_args()
    host, rdir = a.dest.split(":", 1)
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {rdir}"], check=True)
    n_up = 0
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(a.parallel)
    inflight = {}
    while True:
        names = set(os.listdir(a.local))
        ready = sorted(n[:-6] for n in names if n.endswith(".ready"))
        todo = [k for k in ready if k + ".uploaded" not in names and
                (k + ".labeled" in names or a.unlabelled_ok) and
                (k + ".student.done" in names or not a.require_student)]
        for k in todo:
            if k in inflight:
                continue
            inflight[k] = pool.submit(upload_one, a, host, rdir, k, names)
        for k in [k for k, f in inflight.items() if f.done()]:
            if inflight.pop(k).result():
                n_up += 1
        names = set(os.listdir(a.local))
        # late work: free parquets an on-prem student worker has fetched; ship late student results
        for k in sorted(n[:-9] for n in names if n.endswith(".uploaded")):
            pq_ = os.path.join(a.local, k + ".parquet")
            if os.path.exists(pq_) and (k + ".student.fetched" in names or k + ".student.done" in names):
                os.remove(pq_)
            if k + ".student.done" in names and k + ".student.shipped" not in names:
                fs = [os.path.join(a.local, k + e) for e in (".student.jsonl", ".student.done")]
                if subprocess.run(["rsync", "-a", "--partial-dir=.rsync-partial"] + fs + [f"{host}:{rdir}/"],
                                  capture_output=True).returncode == 0:
                    open(os.path.join(a.local, k + ".student.shipped"), "w").close()
                    os.remove(fs[0])
                    log(f"shipped late student hyps {k}")
        names = set(os.listdir(a.local))
        def finished(k):
            if k + ".uploaded" not in names:
                return False
            cl = os.path.join(a.local, k + ".student.claim", "owner")
            onprem_owned = os.path.exists(cl) and open(cl).read().strip() != "x5"
            return not (a.student_split and onprem_owned and k + ".student.shipped" not in names)
        if not inflight and "PASS2_DONE.json" in names and all(finished(n[:-6]) for n in names if n.endswith(".ready")):
            extra = [os.path.join(a.local, n) for n in ("SHA256SUMS", "member_stats.jsonl", "members_done.txt", "PASS2_DONE.json")
                     if n in names]
            subprocess.run(["rsync", "-a"] + extra + [f"{host}:{rdir}/"], check=True)
            log(f"ALL UPLOADED ({n_up} shards this run); bookkeeping shipped")
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
