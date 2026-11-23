"""h52-labeler driver for from-source units (Libri-Light medium), runs on H0.

Units come from `lab_source.py plan` (14 members each, h52's pass-2 member selection). Each unit is claimed
locally (mkdir <state>/claims/UNNNNN), sent as a job (members + unit name) to a GPU worker's parquet inbox
(its lab_gpu re-cuts from CloudFront with h52's code and labels), collected (sha256 vs the worker's ldone,
unique ids, every src in the unit), then shipped to the landing dir on x5:
  <dest>/UNNNNN.r0labels.jsonl  (rsync to a dot-part name, then mv)   then   <dest>/UNNNNN.labeled (last).

  python3 lab_drive_units.py --units units.json --workers workers-medium.json --dest xbabe5:/path/r0src
Stop: touch <state>/STOP.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subprocess  # noqa: E402

from lab_drive import CONVENTION, RSYNC_E, now, rsync, sh, sha256_file, write_atomic  # noqa: E402


class UnitDriver:
    def __init__(self, a):
        self.a = a
        plan = json.load(open(a.units))
        self.units = {u["unit"]: u for u in plan["units"]}
        self.order = [u["unit"] for u in plan["units"]]
        self.W = json.load(open(a.workers))
        self.S = a.state
        os.makedirs(os.path.join(self.S, "claims"), exist_ok=True)
        os.makedirs(os.path.join(self.S, "jobs"), exist_ok=True)
        os.makedirs(os.path.join(self.S, "labels"), exist_ok=True)
        self.logf = open(os.path.join(self.S, "drive_units.log.jsonl"), "a")
        self.pool = cf.ThreadPoolExecutor(max_workers=4)
        self.inflight = {}
        self.seg = {}
        if a.seg_index:  # x5's pass-2 shard rows: src -> [[id, t0, t1], ...] (exact floats), lets workers skip plan()
            for line in open(a.seg_index):
                src, uid, t0, t1 = line.rstrip("\n").split("\t")
                self.seg.setdefault(src, []).append([uid, float(t0), float(t1)])
            for v in self.seg.values():
                v.sort(key=lambda r: r[0])
        self.dest_host, self.dest_dir = a.dest.split(":", 1)
        sh(self.dest_host, f"mkdir -p {self.dest_dir}")
        for cl in glob.glob(os.path.join(self.S, "claims", "U*")):
            u = os.path.basename(cl)
            if os.path.exists(os.path.join(cl, "done.json")) or not os.path.exists(os.path.join(cl, "owner.json")):
                continue
            o = json.load(open(os.path.join(cl, "owner.json")))
            if o.get("worker") in self.W:
                self.inflight[u] = {"worker": o["worker"], "t": 0.0, "fut": None}
                self.log({"recovered": u, "worker": o["worker"]})

    def log(self, d):
        d["time"] = now()
        self.logf.write(json.dumps(d) + "\n")
        self.logf.flush()
        print(json.dumps(d), flush=True)

    def done(self, u):
        return os.path.exists(os.path.join(self.S, "claims", u, "done.json"))

    def count(self, w):
        return sum(1 for v in self.inflight.values() if v["worker"] == w)

    def is_local(self, host):
        return host in ("local", socket.gethostname())

    def dispatch(self, u, wname):
        w = self.W[wname]
        members = self.units[u]["members"]
        if self.seg:
            members = [m[:7] + [self.seg[m[1]]] if m[1] in self.seg else m[:7] for m in members]
        job = {"unit": u, "shard": u, "members": members}
        if w["kind"] == "ship":  # a producer re-cuts + fbanks, then delivers features to the GPU consumer's feats dir
            job.update({"dest_host": w.get("dest_host", w["host"]), "dest_dir": w["feats"]})
        jp = os.path.join(self.S, "jobs", u + ".go")
        write_atomic(jp, json.dumps(job))
        if w["kind"] == "local":
            os.makedirs(w["inbox"], exist_ok=True)
            write_atomic(os.path.join(w["inbox"], u + ".go"), open(jp).read())
        elif w["kind"] == "ship":
            if self.is_local(w.get("front_host", "local")):
                os.makedirs(w["front_inbox"], exist_ok=True)
                write_atomic(os.path.join(w["front_inbox"], u + ".go"), open(jp).read())
            else:
                rsync(jp, f"{w['front_host']}:{w['front_inbox']}/{u}.go")
        else:
            rsync(jp, f"{w['host']}:{w['inbox']}/{u}.go")  # rsync writes a temp name, then renames

    def list_done(self, wname):
        w = self.W[wname]
        if w["kind"] == "local" or self.is_local(w.get("host", "")):
            names = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(w["out"], "*.ldone"))]
        else:
            names = sh(w["host"], f"cd {w['out']} && ls -1 | grep '\\.ldone$' || true", timeout=30).split()
            names = [x[:-6] for x in names]
        return sorted(x for x in names if x in self.units)  # only this driver's own plan (several drivers may share outboxes)

    def collect(self, wname, u):
        w = self.W[wname]
        loc = os.path.join(self.S, "labels", u + ".r0labels.jsonl")
        local_out = w["kind"] == "local" or self.is_local(w.get("host", ""))
        if local_out:
            ld = json.load(open(os.path.join(w["out"], u + ".ldone")))
            os.replace(os.path.join(w["out"], u + ".r0labels.jsonl"), loc)
        else:
            ld = json.loads(sh(w["host"], f"cat {w['out']}/{u}.ldone"))
            rsync(f"{w['host']}:{w['out']}/{u}.r0labels.jsonl", loc)
        sha = sha256_file(loc)
        if sha != ld["sha256"]:
            raise RuntimeError(f"{u}: sha256 {sha} != worker's {ld['sha256']}")
        floc = None
        if ld.get("frames_file"):  # frame-KD top-8 targets beside the labels
            floc = os.path.join(self.S, "labels", ld["frames_file"])
            if local_out:
                os.replace(os.path.join(w["out"], ld["frames_file"]), floc)
            else:
                rsync(f"{w['host']}:{w['out']}/{ld['frames_file']}", floc)
            if sha256_file(floc) != ld["frames_sha256"]:
                raise RuntimeError(f"{u}: frames sha256 differs from the worker's")
        members = {m[1] for m in self.units[u]["members"]}
        ids = set()
        with open(loc) as fh:
            head = json.loads(fh.readline())
            n = 0
            for line in fh:
                d = json.loads(line)
                n += 1
                if d["id"] in ids or d.get("src") not in members:
                    raise RuntimeError(f"{u}: duplicate id or foreign src in {d['id']}")
                ids.add(d["id"])
        if head.get("convention") != CONVENTION or n == 0:
            raise RuntimeError(f"{u}: convention {head.get('convention')} rows {n}")
        dd = self.dest_dir
        if self.a.frames_only and not floc:
            raise RuntimeError(f"{u}: frames-only run but the worker wrote no frames file")
        frames = None
        if floc:
            frames = {"file": ld["frames_file"], "sha256": ld["frames_sha256"], "bytes": ld.get("frames_bytes"), "k": 8,
                      "ids_dtype": "int16", "lp_dtype": "float16", "emitted_dtype": "int16", "frame_rate_hz": 25,
                      "frame_shift_s": 0.04, "speed": 1.0,
                      "layout": "kd-cache frame-cache npz (uid, text, hyp, n_frames, topk_ids, topk_lp, emitted, meta) "
                                "+ n_samples, pcm_sha256, n_fbank; train/kd.py KdStore reads it",
                      "text": "R0 greedy hyp (= the pseudo-label)",
                      "frame_rule": "frame t in chunk c = t // 8 sees fbank frames [32c, 32c + 45); "
                                    "T_R0 = r0_frames(n_mel) = 8 * ((n_mel - 13) // 32 + 1)"}
        lab = {"unit": u, "rows": n, "labels": u + ".r0labels.jsonl", "labels_sha256": sha,
               "teacher_convention": CONVENTION, "labeller": "h52-labeler", "mode": "from-source",
               "members": [m[1] for m in self.units[u]["members"]],
               "member_sha256": [m[4] for m in self.units[u]["members"]], "source": self.units[u]["members"][0][0],
               "worker": wname, "gpu_host": head.get("gpu_host"), "frontend_host": head.get("frontend_host"),
               "frontend_code": head.get("frontend_code"), "beam": head.get("beam"),
               "model_sha256": head.get("model_sha256"), "code_sha256": head.get("code_sha256"),
               "labeller_sha256": (head.get("labeller") or {}).get("lab_gpu_sha256"),
               "stats": {k: ld.get(k) for k in ("audio_s", "wall_s", "enc_s", "greedy_s", "beam_s", "x_rt", "front_s")},
               "frames": frames, "time": now()}
        lp = os.path.join(self.S, "labels", u + ".labeled")
        write_atomic(lp, json.dumps(lab))
        # two WAN connections per unit: one rsync of labels + frames + .labeled into .incoming-U, one rename pass
        # that publishes labels and frames first and .labeled last
        inc = f"{dd}/.incoming-{u}"
        ship_frames_later = bool(floc) and bool(self.a.frames_outbox)
        files = [loc] + ([floc] if floc and not ship_frames_later else []) + [lp]
        subprocess.run(["rsync", "-a", "--whole-file", "-e", RSYNC_E] + files + [f"{self.dest_host}:{inc}/"],
                       check=True, timeout=900, capture_output=True)
        mv = f"mv {inc}/{u}.r0labels.jsonl {dd}/"
        if floc and not ship_frames_later:
            mv += f" && mv {inc}/{ld['frames_file']} {dd}/"
        mv += f" && mv {inc}/{u}.labeled {dd}/ && rmdir {inc}"
        sh(self.dest_host, mv)
        if ship_frames_later:  # frames (75% of the bytes) leave the critical path: a background rsync streams them
            os.makedirs(self.a.frames_outbox, exist_ok=True)
            os.replace(floc, os.path.join(self.a.frames_outbox, ld["frames_file"]))
            floc = None
        write_atomic(os.path.join(self.S, "claims", u, "done.json"), json.dumps(lab))
        if local_out:
            os.remove(os.path.join(w["out"], u + ".ldone"))
        else:
            sh(w["host"], f"rm -f {w['out']}/{u}.ldone {w['out']}/{u}.r0labels.jsonl {w['out']}/{u}.r0frames.npz", timeout=30)
        os.remove(loc)
        os.remove(lp)
        if floc:
            os.remove(floc)
        info = self.inflight.pop(u, {})
        self.log({"labeled": u, "worker": wname, "rows": n, "audio_s": ld.get("audio_s"), "gpu_wall_s": ld.get("wall_s"),
                  "x_rt": ld.get("x_rt"), "front_s": ld.get("front_s"),
                  "turnaround_s": round(time.time() - info.get("t", time.time()), 1)})

    def claim_and_dispatch(self):
        for u in self.order:
            if u in self.inflight or self.done(u) or os.path.exists(os.path.join(self.S, "claims", u)):
                continue
            cands = [(self.count(w) / max(self.W[w].get("weight", 1.0), 1e-6), i, w) for i, w in enumerate(self.W)
                     if not self.W[w].get("paused") and self.count(w) < self.W[w]["depth"]]
            if not cands:
                return
            wname = min(cands)[2]
            try:  # partitioned drivers share the claim namespace: losing the race just means it is taken
                os.mkdir(os.path.join(self.S, "claims", u))
            except FileExistsError:
                continue
            write_atomic(os.path.join(self.S, "claims", u, "owner.json"),
                         json.dumps({"owner": "h52-labeler", "worker": wname, "host": socket.gethostname(), "time": now()}))
            self.inflight[u] = {"worker": wname, "t": time.time(), "fut": self.pool.submit(self.dispatch, u, wname)}
            self.log({"claimed": u, "worker": wname})

    def pipeline_units(self, wname):
        """Unit names visible anywhere in a worker's pipeline (inboxes, front stage, feature dir, outbox)."""
        w = self.W[wname]
        if w["kind"] == "local":
            places = [("local", w["inbox"]), ("local", w["out"])]
        elif w["kind"] == "ship":
            places = [(w.get("front_host", "local"), w["front_inbox"]), (w.get("front_host", "local"), w.get("front_stage", "")),
                      (w["host"], w["feats"]), (w["host"], w["out"])]
        else:
            places = [(w["host"], w["inbox"]), (w["host"], w["out"])]
        names = set()
        for host, d in places:
            if not d:
                continue
            if self.is_local(host):
                files = os.listdir(d) if os.path.isdir(d) else []
            else:
                files = sh(host, f"ls -1 {d} 2>/dev/null || true", timeout=30).split()
            names |= {f.split(".")[0] for f in files if f.split(".")[0] in self.units}
        return names

    def run(self):
        last_err = {}
        present = {}
        for u, v in self.inflight.items():  # recovered: re-send only units that are nowhere in their pipeline
            if v["worker"] not in present:
                present[v["worker"]] = self.pipeline_units(v["worker"])
            v["t"] = time.time()
            if u not in present[v["worker"]]:
                self.log({"resend": u, "worker": v["worker"]})
                v["fut"] = self.pool.submit(self.dispatch, u, v["worker"])
        while not os.path.exists(os.path.join(self.S, "STOP")):
            for wname in list(self.W):
                try:
                    names = self.list_done(wname)
                except Exception as e:  # noqa: BLE001
                    if last_err.get(wname) != repr(e):
                        self.log({"list_error": wname, "err": repr(e)[:400]})
                    last_err[wname] = repr(e)
                    continue
                for u in names:  # one bad unit must not block the rest of the outbox
                    if self.done(u):
                        continue
                    try:
                        self.collect(wname, u)
                    except Exception as e:  # noqa: BLE001
                        if last_err.get(u) != repr(e):
                            self.log({"collect_error": wname, "unit": u, "err": repr(e)[:400]})
                        last_err[u] = repr(e)
            for u, v in list(self.inflight.items()):
                f = v.get("fut")
                if f is not None and f.done() and f.exception() is not None:
                    self.log({"dispatch_error": u, "worker": v["worker"], "err": repr(f.exception())[:400]})
                    time.sleep(2)
                    v["fut"] = self.pool.submit(self.dispatch, u, v["worker"])
            try:
                self.W = json.load(open(self.a.workers))
            except (OSError, ValueError):
                pass
            self.claim_and_dispatch()
            if all(self.done(u) for u in self.order):
                self.log({"all_done": len(self.order)})
                break
            time.sleep(self.a.poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--units", required=True)
    ap.add_argument("--workers", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--state", default="/dev/shm/h52lab/medium")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--seg-index", default="", help="x5 pass-2 shard rows TSV (src, id, t0, t1)")
    ap.add_argument("--frames-only", action="store_true", help="back-fill: ship only r0frames.npz + UNNNNN.frames.labeled")
    ap.add_argument("--frames-outbox", default="", help="local dir: frames wait here for a background rsync to --dest")
    a = ap.parse_args(argv)
    UnitDriver(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
