"""h52-labeler driver (runs on H0): h52's hand-off contract, shard by shard.

  * takes only shards with NNNN.ready, claims each with `mkdir NNNN.claim` (atomic; owner.json inside);
  * routes it to the least-loaded GPU worker:
      local   (H0 3090): NNNN.go in a local inbox, the GPU process computes features itself;
      parquet (xbabe1 4090): rsync the parquet to the host, NNNN.go there, features + labels on that host;
      ship    (xbabe2 3090): rsync the parquet to a front host (xbabe1 CPUs), which ships features to the
              GPU host (xbabe2's CPU is saturated, so it runs only the GPU stage);
  * collects NNNN.r0labels.jsonl (sha256 checked against the worker's NNNN.ldone, ids checked against
    NNNN.ids, rows against NNNN.ready), renames it into place, then writes NNNN.labeled last.

  python3 lab_drive.py --contract /dev/shm/h52/raw/librilight.small --workers workers.json
Stop: touch <state>/STOP.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

CONVENTION = "r0-fp32-gpu-v1"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ControlPath=/dev/shm/h52lab/ssh-%r@%h:%p",
       "-o", "ConnectTimeout=15"]
RSYNC_E = "ssh " + " ".join(SSH[1:])


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sh(host, cmd, timeout=60, check=True):
    r = subprocess.run(SSH + [host, cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh {host} {cmd!r}: rc {r.returncode} {r.stderr[-300:]}")
    return r.stdout


def rsync(src, dst, timeout=900):
    subprocess.run(["rsync", "-a", "--whole-file", "-e", RSYNC_E, src, dst], check=True, timeout=timeout,
                   capture_output=True)


def write_atomic(path, text):
    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class Driver:
    def __init__(self, a):
        self.a = a
        self.C = a.contract
        self.W = json.load(open(a.workers))
        self.inflight = {}  # shard name -> {"worker", "t", "fut"}
        self.pool = cf.ThreadPoolExecutor(max_workers=4)
        os.makedirs(a.state, exist_ok=True)
        self.logf = open(os.path.join(a.state, "drive.log.jsonl"), "a")
        self.done_count = 0
        self.recover()

    def log(self, d):
        d["time"] = now()
        self.logf.write(json.dumps(d) + "\n")
        self.logf.flush()
        print(json.dumps(d), flush=True)

    def recover(self):
        for cl in glob.glob(os.path.join(self.C, "*.claim")):
            name = os.path.basename(cl)[:-6]
            ow = os.path.join(cl, "owner.json")
            if os.path.exists(os.path.join(self.C, name + ".labeled")) or not os.path.exists(ow):
                continue
            o = json.load(open(ow))
            if o.get("owner") == "h52-labeler" and o.get("worker") in self.W:
                self.inflight[name] = {"worker": o["worker"], "t": time.time(), "fut": None, "recovered": True}
                self.log({"recovered": name, "worker": o["worker"]})

    def count(self, w):
        return sum(1 for v in self.inflight.values() if v["worker"] == w)

    def pipeline_names(self, wname):
        """Shard names visible anywhere in a worker's pipeline (inboxes, stages, feature dir, outbox)."""
        w = self.W[wname]
        places = []  # (host or "local", dir)
        if w["kind"] == "local":
            places += [("local", w["inbox"]), ("local", w["out"])]
        elif w["kind"] == "parquet":
            places += [(w["host"], w["inbox"]), (w["host"], w["out"])]
        else:
            places += [(w["front_host"], w["front_inbox"]), (w["host"], w["feats"]), (w["host"], w["out"])]
            if w.get("front_stage"):
                places.append((w["front_host"], w["front_stage"]))
        places += [tuple(x) for x in w.get("extra_dirs", [])]
        names = set()
        for host, d in places:
            if host == "local":
                files = os.listdir(d) if os.path.isdir(d) else []
            else:
                files = sh(host, f"ls -1 {d} 2>/dev/null || true", timeout=30).split()
            names |= {f.split(".")[0] for f in files if f[:1].isdigit()}
        return names

    def redispatch_orphans(self, min_age):
        """Claimed by us, not labeled, older than min_age, no dispatch running, and absent from the
        worker's pipeline: dispatch again (outputs are idempotent; the first .labeled wins)."""
        cands = [(n, v) for n, v in self.inflight.items()
                 if time.time() - v["t"] > min_age and (v.get("fut") is None or v["fut"].done())
                 and not os.path.exists(os.path.join(self.C, n + ".labeled"))]
        if not cands:
            return
        seen = {}
        for n, v in cands:
            wname = v["worker"]
            if wname not in seen:
                try:
                    seen[wname] = self.pipeline_names(wname)
                except Exception as e:  # noqa: BLE001
                    self.log({"orphan_check_error": wname, "err": repr(e)[:300]})
                    seen[wname] = None
            if seen[wname] is not None and n not in seen[wname]:
                self.log({"redispatch": n, "worker": wname, "age_s": round(time.time() - v["t"], 1)})
                v["t"] = time.time()
                v["fut"] = self.pool.submit(self.dispatch, n, wname)

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, name, wname):
        w = self.W[wname]
        ready = json.load(open(os.path.join(self.C, name + ".ready")))
        pq = os.path.join(self.C, name + ".parquet")
        job = {"shard": ready.get("shard", name + ".parquet"), "sha256": ready.get("sha256"), "rows": ready.get("rows")}
        if w["kind"] == "local":
            job["parquet"] = pq
            os.makedirs(w["inbox"], exist_ok=True)
            write_atomic(os.path.join(w["inbox"], name + ".go"), json.dumps(job))
            return
        if w["kind"] == "ship" and w.get("front_host") == "local":
            # local feature shipper (H0 CPUs) reads the parquet in place and ships features to the GPU host
            job.update({"parquet": pq, "dest_host": w["host"], "dest_dir": w["feats"]})
            os.makedirs(w["front_inbox"], exist_ok=True)
            write_atomic(os.path.join(w["front_inbox"], name + ".go"), json.dumps(job))
            return
        host = w["front_host"] if w["kind"] == "ship" else w["host"]
        rin = w["front_inbox"] if w["kind"] == "ship" else w["inbox"]
        rsync(pq, f"{host}:{rin}/{name}.parquet")
        job.update({"parquet": f"{rin}/{name}.parquet", "remove_parquet": True})
        if w["kind"] == "ship":
            job.update({"dest_host": w["host"], "dest_dir": w["feats"]})
        js = json.dumps(job).replace("'", "")
        sh(host, f"printf '%s' '{js}' > {rin}/.{name}.go.tmp && mv {rin}/.{name}.go.tmp {rin}/{name}.go")

    def claim_and_dispatch(self):
        readies = sorted(glob.glob(os.path.join(self.C, "*.ready")))
        for r in readies:
            name = os.path.basename(r)[:-6]
            if name in self.inflight or os.path.exists(os.path.join(self.C, name + ".labeled")) \
                    or os.path.exists(os.path.join(self.C, name + ".claim")):
                continue
            cands = [(self.count(w) / max(self.W[w].get("weight", 1.0), 1e-6), i, w) for i, w in enumerate(self.W)
                     if not self.W[w].get("paused") and self.count(w) < self.W[w]["depth"]]
            if not cands:
                return
            wname = min(cands)[2]
            try:
                os.mkdir(os.path.join(self.C, name + ".claim"))
            except FileExistsError:
                continue
            write_atomic(os.path.join(self.C, name + ".claim", "owner.json"),
                         json.dumps({"owner": "h52-labeler", "worker": wname, "host": socket.gethostname(),
                                     "pid": os.getpid(), "time": now()}))
            fut = self.pool.submit(self.dispatch, name, wname)
            self.inflight[name] = {"worker": wname, "t": time.time(), "fut": fut}
            self.log({"claimed": name, "worker": wname})

    # ------------------------------------------------------------------ collect
    def list_done(self, wname):
        w = self.W[wname]
        if w["kind"] == "local":
            return sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(w["out"], "*.ldone")))
        out = sh(w["host"], f"cd {w['out']} && ls -1 | grep '\\.ldone$' || true", timeout=30)
        return sorted(x[:-6] for x in out.split())

    def collect(self, wname, name):
        w = self.W[wname]
        final = os.path.join(self.C, name + ".r0labels.jsonl")
        tmp = os.path.join(self.C, "." + name + ".r0labels.jsonl.tmp")
        if w["kind"] == "local":
            ld = json.load(open(os.path.join(w["out"], name + ".ldone")))
            os.replace(os.path.join(w["out"], name + ".r0labels.jsonl"), tmp)
        else:
            ld = json.loads(sh(w["host"], f"cat {w['out']}/{name}.ldone"))
            rsync(f"{w['host']}:{w['out']}/{name}.r0labels.jsonl", tmp)
        sha = sha256_file(tmp)
        if sha != ld["sha256"]:
            raise RuntimeError(f"{name}: labels sha256 {sha} != worker's {ld['sha256']}")
        ready = json.load(open(os.path.join(self.C, name + ".ready")))
        with open(tmp) as fh:
            head = json.loads(fh.readline())
            got = [json.loads(line)["id"] for line in fh]
        want = [x for x in open(os.path.join(self.C, name + ".ids")).read().split() if x]
        if got != want or len(got) != ready.get("rows", len(got)):
            raise RuntimeError(f"{name}: ids / rows mismatch ({len(got)} labelled, {len(want)} ids, ready rows {ready.get('rows')})")
        if head.get("convention") != CONVENTION:
            raise RuntimeError(f"{name}: convention {head.get('convention')}")
        os.replace(tmp, final)
        lab = {"shard": ready.get("shard", name + ".parquet"), "shard_sha256": ready.get("sha256"), "rows": len(got),
               "labels": os.path.basename(final), "labels_sha256": sha, "teacher_convention": CONVENTION,
               "labeller": "h52-labeler", "worker": wname, "gpu_host": head.get("gpu_host"),
               "frontend_host": head.get("frontend_host"), "beam": head.get("beam"),
               "model_sha256": head.get("model_sha256"), "tokenizer_sha256": head.get("tokenizer_sha256"),
               "code_sha256": head.get("code_sha256"), "teacher_cache_code_sha256": head.get("teacher_cache_code_sha256"),
               "labeller_sha256": (head.get("labeller") or {}).get("lab_gpu_sha256"),
               "stats": {k: ld.get(k) for k in ("audio_s", "wall_s", "enc_s", "greedy_s", "beam_s", "x_rt", "front_s",
                                                "nbest_mean", "max_mem_gb")}, "time": now()}
        write_atomic(os.path.join(self.C, name + ".labeled"), json.dumps(lab))
        if w["kind"] == "local":
            os.remove(os.path.join(w["out"], name + ".ldone"))
        else:
            sh(w["host"], f"rm -f {w['out']}/{name}.ldone {w['out']}/{name}.r0labels.jsonl", timeout=30)
        info = self.inflight.pop(name, {})
        self.done_count += 1
        self.log({"labeled": name, "worker": wname, "rows": len(got), "audio_s": ld.get("audio_s"),
                  "gpu_wall_s": ld.get("wall_s"), "x_rt": ld.get("x_rt"), "front_s": ld.get("front_s"),
                  "turnaround_s": round(time.time() - info.get("t", time.time()), 1)})

    def run(self):
        last_err = {}
        self.redispatch_orphans(min_age=-1)  # recovered claims that are nowhere in their pipeline
        last_orphan = time.time()
        while not os.path.exists(os.path.join(self.a.state, "STOP")):
            if time.time() - last_orphan > 60:
                self.redispatch_orphans(min_age=self.a.orphan_age)
                last_orphan = time.time()
            for wname in self.W:
                try:
                    for name in self.list_done(wname):
                        self.collect(wname, name)
                except Exception as e:  # noqa: BLE001  keep driving; the error is logged and retried
                    if last_err.get(wname) != repr(e):
                        self.log({"collect_error": wname, "err": repr(e)[:400]})
                    last_err[wname] = repr(e)
            for name, v in list(self.inflight.items()):
                f = v.get("fut")
                if f is not None and f.done() and f.exception() is not None:
                    self.log({"dispatch_error": name, "worker": v["worker"], "err": repr(f.exception())[:400]})
                    v["fut"] = self.pool.submit(self.dispatch, name, v["worker"])  # retry same worker
            try:
                self.W = json.load(open(self.a.workers))  # live edits: depth / weight / paused
            except (OSError, ValueError):
                pass
            self.claim_and_dispatch()
            time.sleep(self.a.poll)
        self.log({"stopped": True, "labeled_this_run": self.done_count})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--contract", default="/dev/shm/h52/raw/librilight.small")
    ap.add_argument("--workers", required=True)
    ap.add_argument("--state", default="/dev/shm/h52lab")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--orphan-age", type=float, default=240.0, help="re-dispatch in-flight shards absent from their pipeline after this")
    a = ap.parse_args(argv)
    Driver(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
