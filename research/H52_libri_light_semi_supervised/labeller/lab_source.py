"""h52-labeler from-source frontend (Libri-Light medium): re-cut the utterances exactly as h52's pass 2
does, then compute R0 features, so R0 labels need no parquet shipped over the WAN.

  * fetch + sha256 check + segmentation + ids = h52's own code, imported unchanged:
      ll_pass2._fetch (HTTPS byte range, retried), ll_pass2._member_tag (id prefix),
      ll_segment.decode_flac / plan / cut;   uid = f"ll{subset[0]}-{tag}-{k:04d}" exactly as ll_pass2.work.
    cut() returns the int16 slice that pass 2 FLAC-encodes losslessly, so the PCM here equals the PCM in
    x5's shards bit for bit; every record carries pcm_sha256, n_samples, t0, t1 and src for the join.
  * features = teacher_cache.R0.fbank (kd-cache's frontend, unchanged), speed 1.0.

  plan:  python lab_source.py plan --index DIR --exclude TSV --subset medium --source URL --unit-mb 250 --out units.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.environ.get("H52LAB_KD", os.path.join(os.path.dirname(HERE), "kdcode"))
H52BIN = os.environ.get("H52LAB_H52BIN", os.path.join(os.path.dirname(HERE), "h52bin"))
os.environ.setdefault("KD_CODE_PARTS", KD)

_S = {}


def _sinit(kd, h52bin):
    os.nice(max(0, 19 - os.nice(0)))  # producer workers at nice 19
    sys.path.insert(0, kd)
    sys.path.insert(0, h52bin)
    from train.teacher_cache import R0, perturb  # kd-cache's verified frontend

    import ll_pass2
    import ll_segment

    _S.update(fbank=R0.fbank, perturb=perturb, P=ll_pass2, S=ll_segment)


def src_member(task):
    """task = (source, name, off, size, sha, meta_text, prefix[, seg]). seg = [[uid, t0, t1], ...] from x5's pass-2
    shard rows for this member: then the utterances are cut with ll_segment.cut at exactly those t0 / t1 (the
    values pass 2 cut and FLAC-encoded), skipping plan(); the PCM is identical by construction."""
    seg = task[7] if len(task) > 7 else None
    source, name, off, size, sha, meta_text, prefix = task[:7]
    P, S = _S["P"], _S["S"]
    if seg:
        t0 = time.time()
        b = None
        for _ in range(3):
            b = P._fetch(source, off, size)
            if hashlib.sha256(b).hexdigest() == sha:
                break
            b = None
        if b is None:
            raise RuntimeError(f"{name}: sha256 mismatch after 3 fetches")
        t1 = time.time()
        pcm = S.decode_flac(b)
        del b
        out = []
        for uid, a, z in seg:
            if not uid.startswith(prefix + "-"):
                raise RuntimeError(f"{name}: seg id {uid} does not match prefix {prefix}")
            x = _S["perturb"](S.cut(pcm, {"t0": a, "t1": z}), 1.0)
            out.append((uid, _S["fbank"](x), int(len(x)), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                        float(a), float(z)))
        return {"name": name, "utts": out, "fetch_s": t1 - t0, "cpu_s": time.time() - t1, "dur": len(pcm) / S.SR,
                "seg": "x5-shard-rows"}
    t0 = time.time()
    b = None
    for _ in range(3):
        b = P._fetch(source, off, size)
        if hashlib.sha256(b).hexdigest() == sha:
            break
        b = None
    if b is None:
        raise RuntimeError(f"{name}: sha256 mismatch after 3 fetches")
    t1 = time.time()
    meta = json.loads(meta_text)
    pcm = S.decode_flac(b)
    del b
    dur = len(pcm) / S.SR
    utts, _st = S.plan(meta.get("voice_activity") or [], dur, pcm)
    out = []
    for k, u in enumerate(utts):
        x = _S["perturb"](S.cut(pcm, u), 1.0)
        f = _S["fbank"](x)
        out.append((f"{prefix}-{k:04d}", f, int(len(x)), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                    float(u["t0"]), float(u["t1"])))
    return {"name": name, "utts": out, "fetch_s": t1 - t0, "cpu_s": time.time() - t1, "dur": dur, "seg": "ll_segment.plan"}


def make_pool(workers):
    import multiprocessing as mp

    return mp.get_context("spawn").Pool(workers, initializer=_sinit, initargs=(KD, H52BIN))


def code_meta():
    import kaldi_native_fbank as knf

    sys.path.insert(0, KD)
    import r0_gpu

    return {"r0_gpu_sha256": r0_gpu.sha256_file(os.path.join(KD, "r0_gpu.py")),
            "teacher_cache_sha256": r0_gpu.sha256_file(os.path.join(KD, "train/teacher_cache.py")),
            "ll_pass2_sha256": r0_gpu.sha256_file(os.path.join(H52BIN, "ll_pass2.py")),
            "ll_segment_sha256": r0_gpu.sha256_file(os.path.join(H52BIN, "ll_segment.py")),
            "lab_source_sha256": r0_gpu.sha256_file(os.path.abspath(__file__)), "knf": knf.__version__,
            "numpy": np.__version__, "speed": 1.0}


def submit_unit(pool, job):
    """Queue a unit's members on the pool without waiting (units pipeline: no idle tail per unit)."""
    return pool.map_async(src_member, [tuple(m) for m in job["members"]], chunksize=1)


def compute_unit_feats(pool, job):
    """job: {"unit": "U00042", "members": [[source, name, off, size, sha, meta_text, prefix], ...]}"""
    t0 = time.time()
    return assemble_unit(job, pool.map(src_member, [tuple(m) for m in job["members"]], chunksize=1), t0)


def assemble_unit(job, res, t0):
    ids, feats, ns, sh, a, b, src = [], [], [], [], [], [], []
    fetch_s = cpu_s = dur = 0.0
    seg_src = {}
    for r in res:
        seg_src[r.get("seg", "?")] = seg_src.get(r.get("seg", "?"), 0) + 1
        fetch_s += r["fetch_s"]
        cpu_s += r["cpu_s"]
        dur += r["dur"]
        for uid, f, n, h, u0, u1 in r["utts"]:
            ids.append(uid)
            feats.append(f)
            ns.append(n)
            sh.append(h)
            a.append(u0)
            b.append(u1)
            src.append(r["name"])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{job['unit']}: duplicate ids")
    off = np.zeros(len(ids) + 1, np.int64)
    off[1:] = np.cumsum([len(f) for f in feats])
    meta = {"shard": job["unit"], "shard_sha256": None, "host": socket.gethostname(), "frontend_code": code_meta(),
            "source": job["members"][0][0] if job["members"] else None, "members": len(job["members"]),
            "member_audio_s": round(dur, 1), "fetch_s_sum": round(fetch_s, 1), "cpu_s_sum": round(cpu_s, 1),
            "seg_source": seg_src,
            "front_s": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return {"ids": np.array(ids), "n_samples": np.array(ns, np.int64), "pcm_sha256": np.array(sh), "offsets": off,
            "feats": np.concatenate(feats) if feats else np.zeros((0, 80), np.float32),
            "t0": np.array(a, np.float64), "t1": np.array(b, np.float64), "src": np.array(src),
            "meta": np.array(json.dumps(meta))}


def plan_units(index, exclude, subset, source, unit_mb, h52bin=H52BIN):
    """Members exactly as ll_pass2.main selects them (official-md5 index, regular .flac with metadata, minus
    the exclude list), in tar order, grouped into units of ~unit_mb MB of FLAC (~800 utterances at 250 MB)."""
    sys.path.insert(0, h52bin)
    import ll_pass2

    rec = json.load(open(os.path.join(index, "receipt.json")))
    if not rec.get("md5_matches_official"):
        raise SystemExit("index receipt does not show an official md5 match; refusing")
    excluded = set()
    if exclude:
        for line in open(exclude).read().splitlines()[1:]:
            excluded.add(line.split("\t")[0][:-5])
    metas = {}
    for line in open(os.path.join(index, "json_meta.jsonl")):
        r = json.loads(line)
        metas[r["name"][:-5]] = r["text"]
    members = []
    for line in open(os.path.join(index, "members.tsv")).read().splitlines()[1:]:
        name, kind, size, oh, od, sha = line.split("\t")
        if kind == "reg" and name.endswith(".flac"):
            members.append((int(od), int(size), name, sha))
    members.sort()
    todo = [m for m in members if m[2][:-5] in metas and m[2][:-5] not in excluded]
    letter = subset[0]
    units, ms, acc = [], [], 0
    for od, size, name, sha in todo:
        mt = metas[name[:-5]]
        ms.append([source, name, od, size, sha, mt, f"ll{letter}-{ll_pass2._member_tag(name, mt)}"])
        acc += size
        if acc >= unit_mb * 1e6:
            units.append({"unit": f"U{len(units):05d}", "members": ms})
            ms, acc = [], 0
    if ms:
        units.append({"unit": f"U{len(units):05d}", "members": ms})
    return units, {"members": len(members), "excluded": len(excluded), "todo": len(todo), "units": len(units),
                   "index_receipt": rec}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("--index", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--unit-mb", type=float, default=250.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    units, info = plan_units(a.index, a.exclude, a.subset, a.source, a.unit_mb)
    with open(a.out, "w") as fh:
        json.dump({"info": info, "units": units}, fh)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
