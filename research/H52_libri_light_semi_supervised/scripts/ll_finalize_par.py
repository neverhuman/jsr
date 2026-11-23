"""H52 parallel finaliser: the same output as ll_finalize.py, with the slow passes spread over --workers processes.

Unchanged from ll_finalize.py (imported, not copied): RULES / H4, the input checks, the label-integrity policy, the
pre-registered filter (H1-H5, S1, S2), conf, the seeded bucket assignment and the per-bucket permutation.
Parallel:
  * verify pass: raw parquet sha256 in a process pool (checked against .ready exactly as before);
  * bucket pass: worker w buckets a contiguous block of raw shards into _buckets/wNN/bBBBBB.parquet, so bucket b's
    rows, concatenated in worker order, keep the single-process order (raw shard order, then row order);
  * write pass: buckets in a process pool; each final shard is written, re-opened and row-count checked, hashed,
    and only then are that bucket's parts deleted (disk peak ~ one copy of the kept data, never buckets + finals).
Guard: the trainer guard (v2: rate -5% vs baseline, data wait > --max-wait-pct) plus a disk floor (--min-free-gb on
the output filesystem). The parent checks it every 10 s and workers wait on a shared event between units of work.

usage: python -I ll_finalize_par.py --raw DIR --out DIR --subset medium [--workers 8] [ll_finalize.py options]
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir; ll_finalize lives beside us

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ll_finalize as F  # noqa: E402

log = F.log
CTX = mp.get_context("fork")
G = {}  # state shared with forked workers (copy-on-write)


def free_gb(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free / 1e9


class Guard(F.Guard):
    """ll_finalize.Guard (v1 / v2 trainer checks) plus a disk floor; logs GUARD PAUSE / RESUME with every reason."""

    def __init__(self, mode, glob_pat, max_wait, rate_drop, disk_path, min_free_gb):
        super().__init__(mode, glob_pat, max_wait, rate_drop)
        self.disk_path, self.min_free = disk_path, min_free_gb

    def reasons(self):
        bad = []
        fg = free_gb(self.disk_path)
        if fg < self.min_free:
            bad.append(f"disk free {fg:.0f} GB < {self.min_free:.0f} GB")
        if self.mode != "v2":
            if not F.trainer_wait_ok(self.glob, self.max_wait):
                bad.append(f"a trainer's data wait > {self.max_wait}% (v1)")
            return bad
        for rd in F._live_trainers():
            name = os.path.basename(rd)
            recs = F._train_records(rd, 200_000)
            if rd not in self.base:
                allr = F._train_records(rd, 5_000_000)
                head = [x for x in (F._rate(r) for r in allr[2:12]) if x]
                if len(head) >= 5:
                    self.base[rd] = statistics.median(head)
                    log(f"GUARD v2 baseline {name} = {self.base[rd]:.1f} (records 3-12; run started during the finaliser)")
            if len(recs) < 3:
                continue
            cur = [x for x in (F._rate(r) for r in recs[-3:]) if x]
            waits = [float(r["data_wait_pct"]) for r in recs[-3:] if "data_wait_pct" in r]
            if rd in self.base and cur and statistics.median(cur) < (1 - self.rate_drop) * self.base[rd]:
                bad.append(f"{name} rate {statistics.median(cur):.0f} < {(1 - self.rate_drop):.2f} x {self.base[rd]:.0f}")
            if waits and statistics.median(waits) > self.max_wait:
                bad.append(f"{name} data wait {statistics.median(waits):.2f}% > {self.max_wait}%")
        return bad

    def ok(self):
        bad = self.reasons()
        if bad and not self.paused:
            log("GUARD PAUSE " + "; ".join(bad))
            self.paused = True
        elif not bad and self.paused:
            log(f"GUARD RESUME (all checks recovered; disk free {free_gb(self.disk_path):.0f} GB)")
            self.paused = False
        return not bad


def run_workers(phase, target, jobs, guard, go):
    """Start one forked process per job, keep `go` in step with the guard, fail fast if any worker fails."""
    procs = [CTX.Process(target=target, args=job, name=f"{phase}-{i}") for i, job in enumerate(jobs)]
    for p in procs:
        p.start()
    last = time.time()
    while True:
        alive = [p for p in procs if p.is_alive()]
        failed = [p for p in procs if p.exitcode not in (None, 0)]
        if failed:
            for p in alive:
                p.terminate()
            raise SystemExit(f"{phase}: worker(s) failed: {[(p.name, p.exitcode) for p in failed]}")
        if not alive:
            break
        if guard.ok():
            go.set()
        else:
            go.clear()
        if time.time() - last > 300:
            log(f"{phase}: {len(alive)} workers running; disk free {free_gb(guard.disk_path):.0f} GB")
            last = time.time()
        time.sleep(10)
    go.set()


def _sha_job(p):
    G["go"].wait()
    return F.sha256_file(p)


def build_table(t, rows, extra):
    """The single-process transform of one raw row group's kept rows (text = R0 hyp + the label columns)."""
    ids = t.column("id").to_pylist()
    f = [rows[u] for u in ids]
    conf = [math.exp(x["lp"]) * (1.0 - (0.0 if math.isnan(x["agree"]) else min(1.0, x["agree"]))) for x in f]
    t = t.set_column(t.schema.get_field_index("text"), "text", pa.array([x["hyp"] for x in f], pa.string()))
    for (name, typ), vals in zip(extra, (conf, [x["lp"] for x in f], [x["lp_min"] for x in f],
                                       [x["m_mean"] for x in f], [x["m_min"] for x in f],
                                       [x["agree"] for x in f], [x["a1"] for x in f],
                                       ["r0-fp32-gpu-v1"] * len(f))):
        t = t.append_column(pa.field(name, typ), pa.array(vals, typ))
    return t, ids


def selected(pf, rg, bucket_of):
    t = pf.read_row_group(rg)
    ids = t.column("id").to_pylist()
    sel = [i for i, u in enumerate(ids) if u in bucket_of]
    return (t.take(sel) if sel else None), sel


def bucket_worker(w, shard_keys):
    g = G
    wdir = os.path.join(g["tmp_dir"], f"w{w:02d}")
    os.makedirs(wdir, exist_ok=True)
    writers = {}
    for j, k in enumerate(shard_keys):
        g["go"].wait()
        pf = pq.ParquetFile(os.path.join(g["raw"], k + ".parquet"))
        for rg in range(pf.metadata.num_row_groups):
            t, sel = selected(pf, rg, g["bucket_of"])
            if t is None:
                continue
            t, ids = build_table(t, g["rows"], g["extra"])
            t = t.cast(g["schema"])
            bk = np.array([g["bucket_of"][u] for u in ids])
            for b in np.unique(bk):
                part = t.take(np.nonzero(bk == b)[0])
                if b not in writers:
                    writers[b] = pq.ParquetWriter(os.path.join(wdir, f"b{b:05d}.parquet"), g["schema"], compression="snappy")
                writers[b].write_table(part)
        if j % 25 == 0 or j == len(shard_keys) - 1:
            log(f"[w{w:02d}] bucketed {k} ({j + 1}/{len(shard_keys)}; disk free {free_gb(g['tmp_dir']):.0f} GB)")
    for wr in writers.values():
        wr.close()


def write_worker(w, buckets):
    g = G
    res = open(os.path.join(g["tmp_dir"], f"res-w{w:02d}.jsonl"), "a")
    for j, b in enumerate(buckets):
        g["go"].wait()
        parts = [os.path.join(g["tmp_dir"], f"w{x:02d}", f"b{b:05d}.parquet") for x in range(g["n_workers"])]
        parts = [p for p in parts if os.path.exists(p)]
        if not parts:
            continue
        t = pa.concat_tables([pq.read_table(p) for p in parts])
        perm = np.random.default_rng([g["seed"], b]).permutation(t.num_rows)
        t = t.take(perm)
        op = os.path.join(g["out_dir"], f"{b:04d}.parquet")
        pq.write_table(t, op + ".part", row_group_size=g["rows_per_group"], compression="snappy")
        os.replace(op + ".part", op)
        n = pq.ParquetFile(op).metadata.num_rows
        if n != t.num_rows:
            raise SystemExit(f"{op}: wrote {n} rows but bucket {b} has {t.num_rows}; its parts are kept")
        res.write(json.dumps({"b": int(b), "sha256": F.sha256_file(op), "rows": n}) + "\n")
        res.flush()
        for p in parts:
            os.remove(p)
        if j % 25 == 0:
            log(f"[w{w:02d}] wrote {b:04d}.parquet ({j + 1}/{len(buckets)}; disk free {free_gb(g['out_dir']):.0f} GB)")
    res.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--rows-per-group", type=int, default=100)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--allow-missing-student", action="store_true")
    ap.add_argument("--allow-unlabeled", action="store_true", help="drop rows without an R0 label (reported)")
    ap.add_argument("--guard-runs", default="")
    ap.add_argument("--label-policy", default="")
    ap.add_argument("--r0src-dir", default="r0src")
    ap.add_argument("--ignore-shard-labels", action="store_true")
    ap.add_argument("--max-wait-pct", type=float, default=5.0)
    ap.add_argument("--guard-mode", default="v1", choices=["v1", "v2"])
    ap.add_argument("--rate-drop", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="pause while the output disk has less free space")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, f"librilight.{a.subset}")
    guard = Guard(a.guard_mode, a.guard_runs, a.max_wait_pct, a.rate_drop, a.out, a.min_free_gb)
    go = CTX.Event()
    go.set()
    G["go"] = go
    log(f"parallel finaliser: {a.workers} workers; disk floor {a.min_free_gb:.0f} GB; disk free {free_gb(a.out):.0f} GB")
    waited = 0
    while not guard.ok():
        if waited % 300 == 0:
            log("guard says hold; holding before any work")
        time.sleep(30)
        waited += 30
    names = set(os.listdir(a.raw))
    keys = sorted(n[:-6] for n in names if n.endswith(".ready"))
    # --- labels and student hyps: exactly ll_finalize.main ---
    src_hyp = {}
    sdir = os.path.join(a.raw, "student_src")
    n_src_lines = 0
    if os.path.isdir(sdir):
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(os.path.join(sdir, fn)):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "id" in r:
                    src_hyp[r["id"]] = (r["hyp"], r["t0"], r["t1"])
                    n_src_lines += 1
    log(f"from-source student hyps: {len(src_hyp)} ids ({n_src_lines} lines)")
    glab = {}
    ldir = os.path.join(a.raw, a.r0src_dir)
    label_header = None
    pol = json.load(open(a.label_policy)) if a.label_policy else None
    unit_stats = {"accepted_code": 0, "accepted_witness": 0, "rejected": 0}
    rank = {}
    if os.path.isdir(ldir):
        for fn in sorted(os.listdir(ldir)):
            if not fn.endswith(".labeled"):
                continue
            mk = json.load(open(os.path.join(ldir, fn)))
            unit, code = mk.get("unit", fn[:-8]), mk.get("code_sha256", "")
            if pol is not None:
                if code in pol.get("accept_code", []):
                    pr = 2
                    unit_stats["accepted_code"] += 1
                elif unit in set(pol.get("accept_units_v1", [])):
                    pr = 1
                    unit_stats["accepted_witness"] += 1
                else:
                    unit_stats["rejected"] += 1
                    continue
            else:
                pr = 1
            lf = os.path.join(ldir, mk.get("labels", fn[:-8] + ".r0labels.jsonl"))
            if F.sha256_file(lf) != mk["labels_sha256"]:
                raise SystemExit(f"{lf}: labels sha256 mismatch")
            for line in open(lf):
                r = json.loads(line)
                if r.get("header"):
                    label_header = label_header or {x: r.get(x) for x in ("schema", "convention", "teacher", "model_sha256", "tokenizer_sha256")}
                    continue
                if pr >= rank.get(r["id"], 0):
                    glab[r["id"]] = r
                    rank[r["id"]] = pr
    log(f"from-source R0 labels: {len(glab)} ids; unit policy {unit_stats if pol is not None else 'none'}")
    # --- verify pass: parquet sha256 in parallel, then the single-process per-shard checks ---
    paths = [os.path.join(a.raw, k + ".parquet") for k in keys]
    with CTX.Pool(a.workers) as pool:
        shas = dict(zip(keys, pool.map(_sha_job, paths, chunksize=4)))
    log(f"raw parquet sha256: {len(shas)} shards hashed")
    rows = {}
    checks = {"shards": 0, "missing_labels": 0, "missing_student": 0, "rows_unlabeled": 0, "glabel_cut_mismatch": 0}
    for k in keys:
        ready = json.load(open(os.path.join(a.raw, k + ".ready")))
        p = os.path.join(a.raw, k + ".parquet")
        sh = shas[k]
        assert sh == ready["sha256"], f"{k}: parquet sha256 != .ready"
        lab = {}
        if k + ".labeled" in names and not a.ignore_shard_labels:
            labeled = json.load(open(os.path.join(a.raw, k + ".labeled")))
            assert sh == labeled["shard_sha256"], f"{k}: parquet sha256 != .labeled"
            lp = os.path.join(a.raw, k + ".r0labels.jsonl")
            assert F.sha256_file(lp) == labeled["labels_sha256"], f"{k}: labels sha256 mismatch"
            for line in open(lp):
                r = json.loads(line)
                if r.get("header"):
                    label_header = label_header or {x: r.get(x) for x in ("schema", "convention", "teacher", "model_sha256", "tokenizer_sha256")}
                    continue
                lab[r["id"]] = r
        elif not glab:
            checks["missing_labels"] += 1
            continue
        stu = {}
        sp = os.path.join(a.raw, k + ".student.jsonl")
        if os.path.exists(sp):
            with open(sp) as f:
                hdr = json.loads(f.readline())
                assert hdr["shard_sha256"] == ready["sha256"], f"{k}: student shard sha mismatch"
                for line in f:
                    r = json.loads(line)
                    stu[r["id"]] = r["hyp"]
        t = pq.read_table(p, columns=["id", "t0", "t1", "src"])
        for uid, t0, t1, src_ in zip(*(t.column(c).to_pylist() for c in ("id", "t0", "t1", "src"))):
            if uid not in lab and uid in glab:
                g = glab[uid]
                if g.get("src") == src_ and g.get("t0") == t0 and g.get("t1") == t1:
                    lab[uid] = g
                else:
                    checks["glabel_cut_mismatch"] += 1
            if uid not in stu and uid in src_hyp:
                h_, s0, s1 = src_hyp[uid]
                if s0 == t0 and s1 == t1:
                    stu[uid] = h_
                else:
                    checks["src_cut_mismatch"] = checks.get("src_cut_mismatch", 0) + 1
        missing = sum(1 for uid in t.column("id").to_pylist() if uid not in stu)
        if missing:
            checks["rows_missing_student"] = checks.get("rows_missing_student", 0) + missing
            if not a.allow_missing_student:
                raise SystemExit(f"{k}: {missing} rows without student hyps (use --allow-missing-student)")
        for uid, t0, t1 in zip(t.column("id").to_pylist(), t.column("t0").to_pylist(), t.column("t1").to_pylist()):
            r = lab.get(uid)
            if r is None:
                checks["rows_unlabeled"] += 1
                if not a.allow_unlabeled:
                    raise SystemExit(f"{k}: no R0 label for {uid} (use --allow-unlabeled)")
                continue
            hyp = F.norm(r["hyp"])
            dur = t1 - t0
            cut_len = int(round(t1 * 16000)) - int(round(t0 * 16000))
            lp = float(r["tok_lp_mean"]) if r.get("tok_lp_mean") is not None else float("-inf")
            if math.isnan(lp):  # H52 amendment 4: an undefined (NaN) mean log-prob ranks lowest, so S1 drops it
                lp = float("-inf")
                checks["lp_nan_as_lowest"] = checks.get("lp_nan_as_lowest", 0) + 1
            f = {"hyp": hyp, "nw": len(hyp.split()), "dur": dur, "ok_len": r["n_samples"] == cut_len,
                 "lp": lp,
                 "lp_min": r.get("tok_lp_min"), "m_mean": r.get("margin_mean"), "m_min": r.get("margin_min"),
                 "a1": F.norm(stu[uid]) if uid in stu else None}
            f["agree"] = F.wer(hyp, f["a1"]) if f["a1"] is not None else float("nan")
            rows[uid] = f
        checks["shards"] += 1
    del glab, src_hyp, rank
    log(f"loaded {len(rows)} rows from {checks['shards']} shards; {checks}")
    # --- the pre-registered filter: exactly ll_finalize.main ---
    RULES, H4 = F.RULES, F.H4
    drop = {}
    for uid, f in rows.items():
        if f["nw"] == 0:
            drop[uid] = "H1_empty"
        elif not f["ok_len"]:
            drop[uid] = "H2_length"
        elif not (RULES["H3_min_wps"] <= f["nw"] / f["dur"] <= RULES["H3_max_wps"]):
            drop[uid] = "H3_rate"
        elif H4.search(f["hyp"]):
            drop[uid] = "H4_librivox"
    groups = {}
    for uid, f in rows.items():
        if uid not in drop:
            groups.setdefault(f["hyp"], []).append(uid)
    for hyp, ids in groups.items():
        if len(ids) > RULES["H5_max_copies"]:
            for uid in sorted(ids, key=lambda x: F.hashlib.sha256(x.encode()).hexdigest())[RULES["H5_max_copies"]:]:
                drop[uid] = "H5_duplicate"
    alive = [u for u in rows if u not in drop]
    lps = np.array([rows[u]["lp"] for u in alive])
    thr = float(np.quantile(lps, RULES["S1_drop_frac"])) if len(lps) else float("-inf")
    for u in alive:
        if rows[u]["lp"] < thr:
            drop[u] = "S1_low_lp"
    for u in rows:
        if u not in drop and not math.isnan(rows[u]["agree"]) and rows[u]["agree"] > RULES["S2_max_disagree"]:
            drop[u] = "S2_disagree"
    kept = [u for u in rows if u not in drop]
    rc = {}
    for v in drop.values():
        rc[v] = rc.get(v, 0) + 1
    kept_h = sum(rows[u]["dur"] for u in kept) / 3600
    all_h = sum(f["dur"] for f in rows.values()) / 3600
    log(f"kept {len(kept)}/{len(rows)} rows ({len(kept) / max(1, len(rows)):.3f}), {kept_h:.1f}/{all_h:.1f} h; drops {rc}; S1 thr {thr:.4f}")
    rng = np.random.default_rng(a.seed)
    n_out = max(1, round(len(kept) / a.rows_per_shard))
    order = rng.permutation(len(kept))
    bucket_of = {kept[i]: int(b) for b, idx in enumerate(np.array_split(order, n_out)) for i in idx}
    tmp_dir = os.path.join(out_dir, "_buckets")
    if os.path.isdir(tmp_dir):  # parts from an interrupted run: everything is re-derived from raw
        log(f"removing stale {tmp_dir} ({free_gb(a.out):.0f} GB free before)")
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    extra = [("conf", pa.float32()), ("r0_lp_mean", pa.float32()), ("r0_lp_min", pa.float32()),
             ("r0_margin_mean", pa.float32()), ("r0_margin_min", pa.float32()), ("agree_wer", pa.float32()),
             ("a1_hyp", pa.string()), ("label_src", pa.string())]
    # canonical schema = the single-process one: from the first raw row group (in key order) with a kept row
    schema = None
    for k in keys:
        pf = pq.ParquetFile(os.path.join(a.raw, k + ".parquet"))
        for rg in range(pf.metadata.num_row_groups):
            t, sel = selected(pf, rg, bucket_of)
            if t is None:
                continue
            t, _ = build_table(t, rows, extra)
            meta = dict(pf.schema_arrow.metadata or {})
            meta[b"h52_labels"] = json.dumps({"labels": label_header, "rules": RULES, "seed": a.seed,
                                              "result_label": "semi-supervised (R0 labels)"}).encode()
            schema = t.schema.with_metadata(meta)
            break
        if schema is not None:
            break
    raw_gb = sum(os.path.getsize(p) for p in paths) / 1e9
    need_gb = raw_gb * len(kept) / max(1, len(rows)) * 1.10
    log(f"DISK plan: raw parquet {raw_gb:.0f} GB -> buckets peak ~{need_gb:.0f} GB (parts deleted as each final shard is "
        f"verified); free {free_gb(a.out):.0f} GB; floor {a.min_free_gb:.0f} GB")
    if free_gb(a.out) - need_gb < a.min_free_gb:
        raise SystemExit(f"disk: {free_gb(a.out):.0f} GB free cannot hold ~{need_gb:.0f} GB above the {a.min_free_gb:.0f} GB floor")
    W = max(1, min(a.workers, len(keys)))
    blocks = [list(x) for x in np.array_split(np.array(keys, dtype=object), W)]
    G.update(raw=a.raw, rows=rows, bucket_of=bucket_of, extra=extra, schema=schema, tmp_dir=tmp_dir, out_dir=out_dir,
             seed=a.seed, rows_per_group=a.rows_per_group, n_workers=W)
    log(f"bucket pass: {W} workers x ~{len(blocks[0])} raw shards -> {n_out} buckets")
    run_workers("bucket", bucket_worker, [(w, blocks[w]) for w in range(W)], guard, go)
    log(f"bucket pass done; disk free {free_gb(a.out):.0f} GB")
    for w in range(W):
        rp = os.path.join(tmp_dir, f"res-w{w:02d}.jsonl")
        if os.path.exists(rp):
            os.remove(rp)
    run_workers("write", write_worker, [(w, list(range(w, n_out, W))) for w in range(W)], guard, go)
    res = {}
    for w in range(W):
        for line in open(os.path.join(tmp_dir, f"res-w{w:02d}.jsonl")):
            r = json.loads(line)
            res[r["b"]] = r
        os.remove(os.path.join(tmp_dir, f"res-w{w:02d}.jsonl"))
    for w in range(W):
        wd = os.path.join(tmp_dir, f"w{w:02d}")
        if os.path.isdir(wd):
            left = os.listdir(wd)
            if left:
                raise SystemExit(f"{wd}: {len(left)} bucket parts left after the write pass")
            os.rmdir(wd)
    os.rmdir(tmp_dir)
    n_rows_out = sum(r["rows"] for r in res.values())
    if n_rows_out != len(kept):
        raise SystemExit(f"final shards hold {n_rows_out} rows, expected {len(kept)}")
    sums = [f"{res[b]['sha256']}  {b:04d}.parquet" for b in range(n_out) if b in res]
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    lpk = np.array([rows[u]["lp"] for u in kept]) if kept else np.array([0.0])
    agr = np.array([rows[u]["agree"] for u in kept if not math.isnan(rows[u]["agree"])])
    report = {"subset": a.subset, "result_label": "semi-supervised (R0 labels)", "rules": RULES, "seed": a.seed,
              "input_shards": checks["shards"], "checks": checks, "rows_in": len(rows), "rows_kept": len(kept),
              "kept_fraction_rows": round(len(kept) / max(1, len(rows)), 4), "hours_in": round(all_h, 2),
              "hours_kept": round(kept_h, 2), "kept_fraction_hours": round(kept_h / max(1e-9, all_h), 4),
              "drop_counts": rc, "S1_lp_threshold": thr, "output_shards": len(sums),
              "kept_lp_mean_quantiles": np.quantile(lpk, [0.01, 0.1, 0.5, 0.9]).round(4).tolist(),
              "kept_agree_wer_mean": round(float(agr.mean()), 4) if len(agr) else None,
              "all_agree_wer_mean": round(float(np.nanmean([f["agree"] for f in rows.values()])), 4) if rows else None,
              "labels": label_header, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "FILTER_REPORT.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(json.dumps({k: report[k] for k in ("rows_in", "rows_kept", "kept_fraction_rows", "hours_in", "hours_kept",
                                            "drop_counts", "output_shards", "kept_agree_wer_mean", "all_agree_wer_mean")}))


if __name__ == "__main__":
    main()
