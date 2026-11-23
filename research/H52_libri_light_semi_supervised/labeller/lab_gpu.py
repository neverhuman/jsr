"""h52-labeler GPU stage: R0 pseudo-labels (teacher convention r0-fp32-gpu-v1) for Libri-Light shards.

The teacher is kd-cache's r0_gpu.py, imported UNCHANGED (R0GPU: hash-checked fp32 ONNX encoder on
onnxruntime-gpu CUDA EP in batched lock-step chunk-16 / left-128 streams; PyTorch fp32 decoder / joiner;
beam_nbest_generic: modified beam search). Features come from lab_front.py, which runs teacher_cache.py's
own R0.fbank through r0_gpu's own frontend workers (_winit / _wfeat), speed 1.0 only.

Greedy here is greedy_topk's exact law (same ops in the same order: log_softmax(joiner), topk(8), emit
top-1, context update for live rows, decoder recomputed for the batch when any row emits); it also keeps
the fp32 top-1 / top-2 / blank log-probs per frame instead of fp16 top-8 arrays, so per-token
log-probs and margins are fp32.

  daemon:  python lab_gpu.py serve --feats DIR --out DIR [--max-frames N] [--beam 8] [--mem-gb G]
  one:     python lab_gpu.py one --feats-file X.feats.npz --out-file Y.jsonl

Output per shard: NNNN.r0labels.jsonl = 1 header line + 1 line per id; NNNN.ldone (sha256, rows, timing).
Results only; no weights are written. Libri-Light only (no LibriSpeech dev / test input is accepted).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.environ.get("H52LAB_KD", os.path.join(os.path.dirname(HERE), "kdcode"))
os.environ.setdefault("KD_CODE_PARTS", KD)  # teacher_cache.py lives in KD/train (hash recorded in meta)
os.environ.setdefault("KD_MODEL_FP32", os.path.join(os.path.dirname(HERE), "model"))
sys.path.insert(0, KD)
import r0_gpu  # noqa: E402  kd-cache's teacher, unchanged

SCHEMA = "h52.r0labels/v1"


def sha256_file(p):
    return r0_gpu.sha256_file(p)


def greedy_stats(r0, enc, nfr, k=8):
    """greedy_topk's law; returns CPU arrays emit (B,T) int16, top1/top2/blank (B,T) float32, and the frame-KD
    targets exactly as greedy_topk stores them: topk ids (B,T,k) int16 and topk log-probs (B,T,k) float16."""
    torch = r0.torch
    B, Tm, _ = enc.shape
    dev = r0.device
    ctx = torch.zeros((B, 2), dtype=torch.long, device=dev)
    dec = r0.decoder(ctx)
    emit = torch.zeros((B, Tm), dtype=torch.int16, device=dev)
    top1 = torch.zeros((B, Tm), dtype=torch.float32, device=dev)
    top2 = torch.zeros((B, Tm), dtype=torch.float32, device=dev)
    blank = torch.zeros((B, Tm), dtype=torch.float32, device=dev)
    kids = torch.zeros((B, Tm, k), dtype=torch.int16, device=dev)
    klps = torch.zeros((B, Tm, k), dtype=torch.float16, device=dev)
    lens = torch.tensor(nfr, device=dev)
    for t in range(Tm):
        live = lens > t
        lp = torch.log_softmax(r0.joiner(enc[:, t], dec), dim=-1)
        v, ix = torch.topk(lp, k, dim=-1)
        kids[:, t], klps[:, t] = ix.to(torch.int16), v.to(torch.float16)
        y = ix[:, 0]
        emit[:, t] = y.to(torch.int16)
        top1[:, t], top2[:, t], blank[:, t] = v[:, 0], v[:, 1], lp[:, 0]
        upd = live & (y != 0)
        if bool(upd.any()):
            ctx = torch.where(upd[:, None], torch.stack([ctx[:, 1], y], 1), ctx)
            dec = torch.where(upd[:, None], r0.decoder(ctx), dec)
    return (emit.cpu().numpy(), top1.cpu().numpy(), top2.cpu().numpy(), blank.cpu().numpy(), kids.cpu().numpy(),
            klps.cpu().numpy())


def r5(x):
    return float(round(float(x), 5))


def utt_record(r0, uid, n_samples, pcm_sha, n_fb, n, emit, top1, top2, blank, nbest):
    e = emit[:n].astype(np.int64)
    pos = np.nonzero(e > 0)[0]
    toks = [int(x) for x in e[pos]]
    t1, t2, bl = top1[:n].astype(np.float64), top2[:n].astype(np.float64), blank[:n].astype(np.float64)
    marg = t1 - t2
    tok_lp = t1[pos]
    rec = {"id": uid, "n_samples": int(n_samples), "pcm_sha256": pcm_sha, "n_fbank": int(n_fb), "n_frames": int(n),
           "hyp": r0.sp.decode(toks), "ids": toks, "tok_frame": [int(x) for x in pos],
           "tok_lp": [r5(x) for x in tok_lp],
           "tok_margin": [r5(x) for x in marg[pos]],
           "path_lp": r5(t1.sum()) if n else 0.0,
           "tok_lp_mean": r5(tok_lp.mean()) if len(pos) else None,
           "tok_lp_min": r5(tok_lp.min()) if len(pos) else None,
           "margin_mean": r5(marg.mean()) if n else None,
           "margin_min": r5(marg.min()) if n else None,
           "blank_lp_mean_on_blank": r5(bl[e == 0].mean()) if (n and (e == 0).any()) else None}
    if nbest is not None:
        rec["nbest"] = [{"ids": s, "text": r0.sp.decode(s), "score": r5(sc)} for s, sc in nbest]
    return rec


def label_shard(r0, fz, args, meta, out_path):
    """fz: dict-like from np.load(feats npz). Writes out_path (tmp then rename). Returns stats."""
    torch = r0.torch
    ids, nsamp, psha, off = fz["ids"], fz["n_samples"], fz["pcm_sha256"], fz["offsets"]
    feats_all = fz["feats"]
    fmeta = json.loads(str(fz["meta"]))
    N = len(ids)
    feats = [feats_all[off[i]:off[i + 1]] for i in range(N)]
    items = [(None, None, f) for f in feats]  # batches_by_length looks at len(items[i][2])
    recs = [None] * N
    frames = [None] * N  # per utterance (topk ids, topk lp, emitted) for the frame-KD file
    t0 = time.time()
    t_enc = t_gr = t_bm = 0.0
    nb_total = 0
    for idx in r0_gpu.batches_by_length(items, args.max_frames):
        torch.cuda.synchronize()
        a = time.time()
        enc, nfr = r0.encode([feats[i] for i in idx])
        torch.cuda.synchronize()
        b = time.time()
        emit, top1, top2, blank, kids, klps = greedy_stats(r0, enc, nfr, 8)
        c = time.time()
        nb = r0_gpu.beam_nbest_generic(r0.decoder, r0.joiner, enc, nfr, args.beam) if args.beam > 0 else None
        torch.cuda.synchronize()
        d = time.time()
        t_enc += b - a
        t_gr += c - b
        t_bm += d - c
        for j, i in enumerate(idx):
            recs[i] = utt_record(r0, str(ids[i]), nsamp[i], str(psha[i]), len(feats[i]), nfr[j], emit[j], top1[j],
                                 top2[j], blank[j], nb[j] if nb is not None else None)
            if "src" in fz:  # from-source units: the join keys / checks for x5's pass-2 rows
                recs[i].update({"src": str(fz["src"][i]), "t0": float(fz["t0"][i]), "t1": float(fz["t1"][i])})
            n = nfr[j]
            frames[i] = (kids[j, :n], klps[j, :n], emit[j, :n])
            nb_total += len(nb[j]) if nb is not None else 0
        del enc
    audio_s = float(np.sum(nsamp)) / 16000.0
    header = {"header": True, "schema": SCHEMA, **meta, "speed": 1.0, "beam": args.beam,
              "shard": fmeta.get("shard"), "shard_sha256": fmeta.get("shard_sha256"), "rows": N,
              "frontend_host": fmeta.get("host"), "frontend_code": fmeta.get("frontend_code"),
              "frontend_seg_source": fmeta.get("seg_source"),
              "gpu_host": socket.gethostname(),
              "fields": {"ids": "R0 bpe.model piece ids (blank 0), greedy one symbol per frame",
                         "tok_frame": "encoder frame (25 Hz) of each emission",
                         "tok_lp": "fp32 log-prob of each emitted token at its emission frame (= frame top-1)",
                         "tok_margin": "top-1 minus top-2 log-prob at each emission frame",
                         "path_lp": "sum over frames of the greedy choice's log-prob (blanks + tokens)",
                         "margin_mean/min": "per-frame top-1 minus top-2 log-prob over all n_frames",
                         "blank_lp_mean_on_blank": "mean blank log-prob on frames where blank was chosen",
                         "nbest": "beam_nbest_generic (modified beam, merged by log-sum-exp) beam scores, best first"}}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)
    fr = {}
    if args.frames:  # frame-KD targets in kd-cache's frame-cache layout (train/kd.py KdStore reads these keys)
        fpath = out_path[:-len(".r0labels.jsonl")] + ".r0frames.npz"
        fm = {"teacher": meta["teacher"] + " (" + meta["convention"] + ")", "convention": meta["convention"],
              "speed": 1.0, "topk": 8, "split": "librilight", "shard": fmeta.get("shard"),
              "tokenizer_sha256": meta["tokenizer_sha256"], "model_sha256": meta["model_sha256"],
              "ids_dtype": "int16", "lp_dtype": "float16", "frame_rate_hz": 25, "frame_shift_s": 0.04,
              "time_support": meta["time_support"]["rule"],
              "text": "R0 greedy hyp (the pseudo-label; KdStore's ok check compares hyp with text)",
              "law": "greedy_topk: top-k log_softmax(joiner) at the greedy context, one symbol per frame"}
        np.savez(fpath + ".tmp.npz", uid=np.array([str(x) for x in ids]), text=np.array([r["hyp"] for r in recs]),
                 hyp=np.array([r["hyp"] for r in recs]), n_frames=np.array([len(f[2]) for f in frames], np.int32),
                 topk_ids=np.concatenate([f[0] for f in frames]) if N else np.zeros((0, 8), np.int16),
                 topk_lp=np.concatenate([f[1] for f in frames]) if N else np.zeros((0, 8), np.float16),
                 emitted=np.concatenate([f[2] for f in frames]) if N else np.zeros(0, np.int16),
                 n_samples=np.asarray(nsamp, np.int64), pcm_sha256=np.array([str(x) for x in psha]),
                 n_fbank=np.array([len(f) for f in feats], np.int32), meta=np.array(json.dumps(fm)))
        os.replace(fpath + ".tmp.npz", fpath)
        fr = {"frames_file": os.path.basename(fpath), "frames_sha256": sha256_file(fpath),
              "frames_bytes": os.path.getsize(fpath)}
    wall = time.time() - t0
    return {**fr, "rows": N, "audio_s": round(audio_s, 1), "wall_s": round(wall, 2), "enc_s": round(t_enc, 2),
            "greedy_s": round(t_gr, 2), "beam_s": round(t_bm, 2), "x_rt": round(audio_s / max(wall, 1e-9), 1),
            "nbest_mean": round(nb_total / max(N, 1), 2), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def load_r0(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.torch_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.torch_frac, 0)
    r0 = r0_gpu.R0GPU(model_dir=os.environ["KD_MODEL_FP32"], mem_gb=args.mem_gb)
    meta = r0_gpu.run_meta(r0, args)
    meta["labeller"] = {"lane": "h52-labeler", "lab_gpu_sha256": sha256_file(os.path.abspath(__file__)),
                        "ort_gpu_mem_limit_gb": args.mem_gb}
    return r0, meta


def feats_jobs(args, stop):
    """Feature inbox: NNNN.feats.npz + NNNN.fready (shipped by lab_front.py). Several consumers may share one
    inbox: a unit is claimed by atomically renaming NNNN.fready -> NNNN.fready.<pid>; claims of dead
    consumers are returned at start."""
    for p in glob.glob(os.path.join(args.feats, "*.fready.*")):
        try:
            os.kill(int(p.rsplit(".", 1)[1]), 0)
        except (ValueError, ProcessLookupError):
            os.replace(p, p.rsplit(".", 1)[0])
        except PermissionError:
            pass
    while not stop():
        todo = sorted(glob.glob(os.path.join(args.feats, "*.fready")))
        rd = None
        for cand in todo:
            try:
                os.rename(cand, cand + f".{os.getpid()}")
            except FileNotFoundError:
                continue  # another consumer took it
            rd = cand + f".{os.getpid()}"
            break
        if rd is None:
            time.sleep(0.3)
            continue
        name = os.path.basename(rd).split(".")[0]
        fpath = os.path.join(args.feats, name + ".feats.npz")
        with np.load(fpath, allow_pickle=False) as fz:
            fz = {k: fz[k] for k in fz.files}

        def done(rd=rd, fpath=fpath):
            os.remove(rd)
            os.remove(fpath)
        yield name, fz, done


def parquet_jobs(args, stop):
    """Parquet inbox: NNNN.go (JSON {"parquet", "shard", "sha256"}); features computed here with kd-cache's
    frontend workers in a prefetch thread, one shard ahead of the GPU."""
    import queue
    import threading

    sys.path.insert(0, HERE)
    import lab_front

    q = queue.Queue(maxsize=1)

    def worker():
        pools = {}
        inflight = []  # [name, taken, job, async result or None, t0]: up to --ahead units queued on the pool
        while not stop():
            while len(inflight) < max(1, args.ahead):
                gos = sorted(glob.glob(os.path.join(args.parquet_in, "*.go")))
                if not gos:
                    break
                g = gos[0]
                taken = g[:-3] + ".taken"
                os.replace(g, taken)
                job = json.load(open(taken))
                ar = None
                if "members" in job:  # from-source unit (medium): h52's re-cut + R0.fbank, queued without waiting
                    import lab_source

                    if "src" not in pools:
                        pools["src"] = lab_source.make_pool(args.workers)
                    ar = lab_source.submit_unit(pools["src"], job)
                inflight.append([os.path.basename(g)[:-3], taken, job, ar, time.time()])
            if not inflight:
                time.sleep(0.3)
                continue
            name, taken, job, ar, t0 = inflight[0]
            if ar is not None and not ar.ready():
                ar.wait(0.5)
                continue
            inflight.pop(0)
            try:
                if ar is not None:
                    import lab_source

                    fz = lab_source.assemble_unit(job, ar.get(), t0)
                else:
                    if "pq" not in pools:
                        pools["pq"] = lab_front.make_pool(args.workers)
                    fz = lab_front.compute_feats(pools["pq"], job["parquet"], job["shard"], job.get("sha256"))
            except Exception as e:  # noqa: BLE001  transient fetch / decode failure: requeue the job and keep serving
                print(json.dumps({"front_error": name, "err": repr(e)[:300]}), flush=True)
                time.sleep(5)
                os.replace(taken, taken[:-6] + ".go")
                continue
            if job.get("remove_parquet"):
                os.remove(job["parquet"])
            q.put((name, fz, taken))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        name, fz, taken = item
        yield name, fz, (lambda taken=taken: os.remove(taken))


def serve(args):
    r0, meta = load_r0(args)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "lab_gpu.log.jsonl"), "a")
    inbox = args.parquet_in or args.feats

    def stop():
        return os.path.exists(os.path.join(inbox, "STOP"))

    jobs = parquet_jobs(args, stop) if args.parquet_in else feats_jobs(args, stop)
    idle_since = time.time()
    for name, fz, done in jobs:
        idle = time.time() - idle_since
        out_path = os.path.join(args.out, name + ".r0labels.jsonl")
        for attempt in range(5):
            try:
                st = label_shard(r0, fz, args, meta, out_path)
                break
            except Exception as e:  # noqa: BLE001  GPU memory pressure from a co-tenant: halve the batch, retry
                msg = repr(e)
                if attempt == 4 or not any(k in msg for k in ("out of memory", "Failed to allocate", "OUT_OF_MEMORY",
                                                              "MemoryAllocation", "OutOfMemory")):
                    raise
                r0.torch.cuda.empty_cache()
                args.max_frames = max(50_000, args.max_frames // 2)
                meta["runtime"]["batching"] = f"length-sorted, max {args.max_frames} padded fbank frames per batch"
                print(json.dumps({"oom_retry": name, "max_frames": args.max_frames, "err": msg[:200]}), flush=True)
        st["front_s"] = json.loads(str(fz["meta"])).get("front_s")
        del fz
        st.update({"shard": name, "sha256": sha256_file(out_path), "idle_before_s": round(idle, 2),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        with open(os.path.join(args.out, name + ".ldone.tmp"), "w") as fh:
            json.dump(st, fh)
        os.replace(os.path.join(args.out, name + ".ldone.tmp"), os.path.join(args.out, name + ".ldone"))
        log.write(json.dumps(st) + "\n")
        log.flush()
        print(json.dumps(st), flush=True)
        done()
        idle_since = time.time()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "one"])
    ap.add_argument("--feats", default="", help="feature inbox (NNNN.feats.npz + NNNN.fready)")
    ap.add_argument("--parquet-in", default="", help="parquet inbox (NNNN.go); frontend runs here")
    ap.add_argument("--workers", type=int, default=8, help="frontend workers for --parquet-in")
    ap.add_argument("--ahead", type=int, default=3, help="from-source units queued on the frontend pool at once")
    ap.add_argument("--out", default="")
    ap.add_argument("--feats-file", default="")
    ap.add_argument("--out-file", default="")
    ap.add_argument("--max-frames", type=int, default=1_500_000, help="padded fbank frames per batch (B * Tmax)")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--frames", type=int, default=1, help="also write NNNN.r0frames.npz (frame-KD top-8 targets)")
    ap.add_argument("--mem-gb", type=float, default=8.0, help="onnxruntime CUDA arena limit")
    ap.add_argument("--torch-frac", type=float, default=0.0, help="torch per-process memory fraction (0 = off)")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return serve(a)
    r0, meta = load_r0(a)
    with np.load(a.feats_file, allow_pickle=False) as fz:
        fz = {k: fz[k] for k in fz.files}
    st = label_shard(r0, fz, a, meta, a.out_file)
    print(json.dumps(st), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
