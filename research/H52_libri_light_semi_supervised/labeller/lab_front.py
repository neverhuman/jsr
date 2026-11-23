"""h52-labeler frontend stage: fbank features for Libri-Light shards, exactly as r0-fp32-gpu-v1.

Uses kd-cache's own frontend workers from r0_gpu.py (_winit imports train/teacher_cache.py's R0.fbank and
perturb unchanged; _wfeat decodes the FLAC to int16, perturbs (speed 1.0 = identity), runs R0.fbank and
returns n_samples + the int16 PCM sha256). Nothing here changes a number; it only moves features.

  compute_feats(pool, parquet, shard, shard_sha256) -> dict of arrays (ids, n_samples, pcm_sha256, offsets,
                                                       feats float32 [sum n_fbank, 80], meta json)
  serve: python lab_front.py serve --inbox DIR --stage DIR [--workers N]
         inbox holds NNNN.go (JSON {"parquet", "shard", "sha256", "dest_host", "dest_dir"}); the features
         go to dest_host:dest_dir/NNNN.feats.npz, then NNNN.fready is created there.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.environ.get("H52LAB_KD", os.path.join(os.path.dirname(HERE), "kdcode"))
os.environ.setdefault("KD_CODE_PARTS", KD)
sys.path.insert(0, KD)
import r0_gpu  # noqa: E402  kd-cache's teacher module, unchanged


def _winit_nice(code):
    os.nice(max(0, 19 - os.nice(0)))  # producer workers yield to everyone (owner rule: background at nice 19)
    r0_gpu._winit(code)


def make_pool(workers: int):
    import multiprocessing as mp

    return mp.get_context("spawn").Pool(workers, initializer=_winit_nice, initargs=(KD,))


def frontend_code():
    import kaldi_native_fbank as knf

    return {"r0_gpu_sha256": r0_gpu.sha256_file(os.path.abspath(r0_gpu.__file__)),
            "teacher_cache_sha256": r0_gpu.sha256_file(os.path.join(KD, "train/teacher_cache.py")),
            "lab_front_sha256": r0_gpu.sha256_file(os.path.abspath(__file__)), "knf": knf.__version__,
            "numpy": np.__version__, "speed": 1.0}


def compute_feats(pool, parquet: str, shard: str, shard_sha: str | None):
    import pyarrow.parquet as pq

    t0 = time.time()
    tab = pq.read_table(parquet, columns=["id", "audio"])
    ids = tab.column("id").to_pylist()
    auds = tab.column("audio").to_pylist()
    del tab
    res = pool.map(r0_gpu._wfeat, [(ids[i], "", auds[i]["bytes"], 1.0) for i in range(len(ids))], chunksize=2)
    del auds
    assert [r[0] for r in res] == ids, "frontend order"
    lens = np.array([len(r[2]) for r in res], np.int64)
    off = np.zeros(len(res) + 1, np.int64)
    off[1:] = np.cumsum(lens)
    feats = np.concatenate([r[2] for r in res]) if res else np.zeros((0, 80), np.float32)
    assert feats.dtype == np.float32 and feats.shape[1] == 80
    meta = {"shard": shard, "shard_sha256": shard_sha, "host": socket.gethostname(), "frontend_code": frontend_code(),
            "front_s": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return {"ids": np.array(ids), "n_samples": np.array([r[3] for r in res], np.int64),
            "pcm_sha256": np.array([r[4] for r in res]), "offsets": off, "feats": feats, "meta": np.array(json.dumps(meta))}


def save_npz(path: str, fz: dict):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **fz)
    os.replace(tmp, path)


def ship(path: str, name: str, host: str, ddir: str, log):
    """rsync the feature file to host:ddir, then mark it ready there (atomic rename + fready). host "local"
    = this host's GPU consumer: a rename into its feature dir (same filesystem), then fready."""
    t0 = time.time()
    if host in ("local", socket.gethostname()):
        os.makedirs(ddir, exist_ok=True)
        os.replace(path, os.path.join(ddir, name + ".feats.npz"))
        open(os.path.join(ddir, name + ".fready"), "w").close()
        log({"ship": name, "host": "local", "s": round(time.time() - t0, 2)})
        return
    subprocess.run(["rsync", "-a", "--whole-file", path, f"{host}:{ddir}/{name}.feats.npz.part"], check=True, timeout=600)
    subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                    f"mv {ddir}/{name}.feats.npz.part {ddir}/{name}.feats.npz && touch {ddir}/{name}.fready"],
                   check=True, timeout=60)
    os.remove(path)
    log({"ship": name, "host": host, "s": round(time.time() - t0, 2), "bytes": None})


def serve(a):
    sys.path.insert(0, HERE)
    pool_box = [None]
    src_pool = [None]
    os.makedirs(a.stage, exist_ok=True)
    lf = open(os.path.join(a.stage, "lab_front.log.jsonl"), "a")

    def log(d):
        d["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lf.write(json.dumps(d) + "\n")
        lf.flush()

    q: queue.Queue = queue.Queue(maxsize=a.max_unshipped)

    def shipper():
        while True:
            item = q.get()
            if item is None:
                return
            for attempt in range(5):
                try:
                    ship(*item, log)
                    break
                except Exception as e:  # noqa: BLE001  transient LAN / ssh failure: retry, then leave the file
                    log({"ship_error": item[1], "attempt": attempt, "err": repr(e)[:300]})
                    time.sleep(3 + 5 * attempt)

    ths = [threading.Thread(target=shipper, daemon=True) for _ in range(max(1, a.ship_threads))]
    for th in ths:
        th.start()
    inflight = []  # from-source units queued on the pool at once (--ahead): [g, name, job, async result, t0]
    pid = os.getpid()
    for p in glob.glob(os.path.join(a.inbox, "*.go.*")):  # claims of dead producers go back to the queue
        try:
            os.kill(int(p.rsplit(".", 1)[1]), 0)
        except (ValueError, ProcessLookupError):
            os.replace(p, p.rsplit(".", 1)[0])
        except PermissionError:
            pass
    while not os.path.exists(os.path.join(a.inbox, "STOP")):
        while len(inflight) < max(1, a.ahead):
            g = None
            for cand in sorted(glob.glob(os.path.join(a.inbox, "*.go"))):
                try:  # several producers may share one inbox: claim by atomic rename
                    os.rename(cand, cand + f".{pid}")
                except FileNotFoundError:
                    continue
                g = cand + f".{pid}"
                break
            if g is None:
                break
            job = json.load(open(g))
            ar = None
            if "members" in job:  # from-source unit (medium), queued without waiting
                import lab_source

                if src_pool[0] is None:
                    src_pool[0] = lab_source.make_pool(a.workers)
                ar = lab_source.submit_unit(src_pool[0], job)
            inflight.append([g, os.path.basename(g).split(".")[0], job, ar, time.time()])
        if not inflight:
            time.sleep(0.3)
            continue
        g, name, job, ar, t0 = inflight[0]
        if ar is not None and not ar.ready():
            ar.wait(0.5)
            continue
        inflight.pop(0)
        try:
            if ar is not None:
                import lab_source

                fz = lab_source.assemble_unit(job, ar.get(), t0)
            else:
                if pool_box[0] is None:
                    pool_box[0] = make_pool(a.workers)
                fz = compute_feats(pool_box[0], job["parquet"], job["shard"], job.get("sha256"))
        except Exception as e:  # noqa: BLE001  transient fetch failure: requeue the same job
            log({"front_error": name, "err": repr(e)[:300]})
            time.sleep(5)
            os.replace(g, g.rsplit(".", 1)[0])
            continue
        fs = json.loads(str(fz["meta"]))["front_s"]
        dest = os.path.join(a.stage, name + ".feats.npz")
        save_npz(dest, fz)
        audio_s = float(fz["n_samples"].sum()) / 16000.0
        del fz
        log({"feats": name, "front_s": fs, "audio_s": round(audio_s, 1), "x_rt": round(audio_s / max(fs, 1e-9), 1)})
        if job.get("remove_parquet"):
            os.remove(job["parquet"])
        os.remove(g)
        q.put((dest, name, job.get("dest_host", "local"), job.get("dest_dir") or a.default_dest_dir))
    for th in ths:
        q.put(None)
    for th in ths:
        th.join()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["serve"])
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-unshipped", type=int, default=2)
    ap.add_argument("--ahead", type=int, default=3, help="from-source units queued on the pool at once")
    ap.add_argument("--ship-threads", type=int, default=1, help="parallel rsync streams to the GPU host")
    ap.add_argument("--default-dest-dir", default="", help="feature dir for jobs without dest_dir (local consumer)")
    a = ap.parse_args(argv)
    return serve(a)


if __name__ == "__main__":
    sys.exit(main())
