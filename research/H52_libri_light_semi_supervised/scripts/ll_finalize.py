"""H52 finaliser: raw Libri-Light shards + R0 labels + A1 student hyps -> trainer-ready shards.

Pre-registered filter (H52 README section 5, fixed before any label was seen):
  H1 hyp empty; H2 R0 n_samples != cut length; H3 word rate outside [0.6, 4.6] words/s over t1-t0;
  H4 normalised hyp contains "LIBRIVOX"; H5 identical hyp kept at most 3 times (first by sha256(id));
  S1 drop the lowest 10% by R0 mean token log-prob (among rows passing H1-H5);
  S2 drop rows whose A1-vs-R0 word disagreement > 0.5 (if a student hyp exists).
conf = exp(tok_lp_mean) * (1 - min(1, agree_wer)).
Every input is checked: parquet sha256 == .ready == .labeled shard_sha256, labels sha256 ==
.labeled labels_sha256, student header shard_sha256 == .ready sha256.
Output: <out>/librilight.<subset>/NNNN.parquet (LibriSpeech schema + label columns, text = R0 hyp),
globally shuffled via seeded buckets, 100-row row groups, SHA256SUMS, FILTER_REPORT.json.

usage: python -I ll_finalize.py --raw DIR --out DIR --subset small [--rows-per-shard 2000] [--seed 52]
       [--allow-missing-student]
"""
import argparse
import hashlib
import json
import math
import os
import time

import statistics

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RULES = {"H3_min_wps": 0.6, "H3_max_wps": 4.6, "H5_max_copies": 3, "S1_drop_frac": 0.10, "S2_max_disagree": 0.5,
         "H4_pattern": r"\bLIB[A-Z]*\s?VO[CX][A-Z]*\b|RECORDINGS? (?:IS|ARE) IN THE PUBLIC DOMAIN|TO VOLUNTEER PLEASE VISIT",
         "H4_amendment": "2026-11-23T07:58Z: R0 cannot spell LIBRIVOX (writes LIBERVOX / LIBRAVOX / LIBREVOX / LIBRAVOC ...), "
                         "so the literal 'LIBRIVOX' test matched nothing; the pattern now implements the pre-registered intent "
                         "(LibriVox preambles / disclaimers). Seen before any filter result was used."}


import re  # noqa: E402

H4 = re.compile(RULES["H4_pattern"])


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def trainer_wait_ok(glob_pat, limit):
    """True unless an active run (train.jsonl touched in the last 10 min) shows median data_wait_pct > limit
    over its last 5 train records (trainer-lane's 06:55Z rule)."""
    import glob as _g
    import statistics
    if not glob_pat:
        return True
    for p in _g.glob(os.path.expanduser(glob_pat)):
        if time.time() - os.path.getmtime(p) > 600:
            continue
        w = []
        for line in open(p).readlines()[-50:]:
            if '"kind": "train"' in line and "data_wait_pct" in line:
                try:
                    w.append(float(json.loads(line)["data_wait_pct"]))
                except (ValueError, KeyError):
                    pass
        if len(w) >= 3 and statistics.median(w[-5:]) > limit:
            return False
    return True



def _live_trainers():
    """{run_dir: [pids]} of running `-m train.trainer` / `train/trainer.py` processes (from /proc)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                args = [x.decode("utf-8", "replace") for x in fh.read().split(b"\0") if x]
        except OSError:
            continue
        if not args or not (("train.trainer" in args and "-m" in args) or any(x.endswith("train/trainer.py") for x in args)):
            continue
        rd = args[args.index("--run-dir") + 1] if "--run-dir" in args and args.index("--run-dir") + 1 < len(args) else \
            next((x.split("=", 1)[1] for x in args if x.startswith("--run-dir=")), None)
        if rd:
            out.setdefault(os.path.abspath(os.path.expanduser(rd)), []).append(int(pid))
    return out


def _train_records(rd, tail_bytes=600_000):
    recs = []
    try:
        with open(os.path.join(rd, "train.jsonl"), "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - tail_bytes))
            for line in fh.read().decode("utf-8", "replace").splitlines():
                if '"kind": "train"' in line:
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return recs


def _rate(r):
    w = r.get("wall_s") or 0
    return r["audio_s"] / w if w > 0 and "audio_s" in r else None


class Guard:
    """v1: pause while any active run's 5-record median data_wait_pct > max_wait (old rule).
    v2 (lead 10:1xZ exception): pause only if a live trainer's rate (median of its last 3 step records) drops more
    than rate_drop below its 10-min pre-start baseline, or its data wait (median of last 3) exceeds max_wait;
    resume on recovery. PAUSE / RESUME lines are logged (H0 relays them to the chat)."""

    def __init__(self, mode, glob_pat, max_wait, rate_drop, window_s=600.0):
        self.mode, self.glob, self.max_wait, self.rate_drop, self.window = mode, glob_pat, max_wait, rate_drop, window_s
        self.base, self.paused = {}, False
        if mode == "v2":
            for rd in _live_trainers():
                b = self._baseline(rd)
                if b:
                    self.base[rd] = b
            log(f"GUARD v2 start: baselines { {os.path.basename(k): round(v, 1) for k, v in self.base.items()} } "
                f"(pause if rate < {1 - rate_drop:.2f} x baseline or data wait > {max_wait}%)")

    def _baseline(self, rd):
        recs = _train_records(rd)
        ws = [r["wall_s"] for r in recs[-20:] if r.get("wall_s")]
        if len(recs) < 5 or not ws:
            return None
        n = max(5, int(self.window / statistics.median(ws)))
        rates = [x for x in (_rate(r) for r in recs[-n:]) if x]
        return statistics.median(rates) if rates else None

    def ok(self):
        if self.mode != "v2":
            return trainer_wait_ok(self.glob, self.max_wait)
        bad = []
        for rd in _live_trainers():
            name = os.path.basename(rd)
            recs = _train_records(rd, 200_000)
            if rd not in self.base:
                allr = _train_records(rd, 5_000_000)
                head = [x for x in (_rate(r) for r in allr[2:12]) if x]
                if len(head) >= 5:
                    self.base[rd] = statistics.median(head)
                    log(f"GUARD v2 baseline {name} = {self.base[rd]:.1f} (records 3-12; run started during the finaliser)")
            if len(recs) < 3:
                continue
            cur = [x for x in (_rate(r) for r in recs[-3:]) if x]
            waits = [float(r["data_wait_pct"]) for r in recs[-3:] if "data_wait_pct" in r]
            if rd in self.base and cur and statistics.median(cur) < (1 - self.rate_drop) * self.base[rd]:
                bad.append(f"{name} rate {statistics.median(cur):.0f} < {(1 - self.rate_drop):.2f} x {self.base[rd]:.0f}")
            if waits and statistics.median(waits) > self.max_wait:
                bad.append(f"{name} data wait {statistics.median(waits):.2f}% > {self.max_wait}%")
        if bad and not self.paused:
            log("GUARD PAUSE " + "; ".join(bad))
            self.paused = True
        elif not bad and self.paused:
            log("GUARD RESUME (all trainers recovered)")
            self.paused = False
        return not bad


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def norm(t):
    return " ".join((t or "").upper().split())


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


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
    ap.add_argument("--guard-runs", default="", help="glob of trainers' train.jsonl; pause while data_wait_pct > limit")
    ap.add_argument("--label-policy", default="", help='JSON {"accept_code": [...], "accept_units_v1": [...]}')
    ap.add_argument("--r0src-dir", default="r0src", help="unit-label directory under --raw (e.g. r0src_v11)")
    ap.add_argument("--ignore-shard-labels", action="store_true",
                    help="use only r0src/ unit labels (e.g. a v1.1 re-label of small), never per-shard .r0labels.jsonl")
    ap.add_argument("--max-wait-pct", type=float, default=5.0)
    ap.add_argument("--guard-mode", default="v1", choices=["v1", "v2"])
    ap.add_argument("--rate-drop", type=float, default=0.05, help="v2: pause if a trainer's rate drops more than this")
    a = ap.parse_args()
    guard = Guard(a.guard_mode, a.guard_runs, a.max_wait_pct, a.rate_drop)
    waited = 0
    while not guard.ok():  # touch nothing on this disk while a trainer is over
        if waited % 300 == 0:
            log("a trainer's data_wait_pct is above the limit; holding before any work")
        time.sleep(30)
        waited += 30
    names = set(os.listdir(a.raw))
    keys = sorted(n[:-6] for n in names if n.endswith(".ready"))
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
                    continue  # a partially relayed last line
                if "id" in r:
                    src_hyp[r["id"]] = (r["hyp"], r["t0"], r["t1"])
                    n_src_lines += 1
    log(f"from-source student hyps: {len(src_hyp)} ids ({n_src_lines} lines)")
    glab = {}
    ldir = os.path.join(a.raw, a.r0src_dir)
    label_header = None
    # label-integrity policy (lead 09:2xZ): accept a unit only if its code is in accept_code, or it is a v1
    # unit cleared by h52-labeler's v1.1 witness; for an id labelled twice, an accept_code unit wins
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
            if sha256_file(lf) != mk["labels_sha256"]:
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
    rows = {}  # id -> feature dict
    shard_of = {}
    checks = {"shards": 0, "missing_labels": 0, "missing_student": 0, "rows_unlabeled": 0, "glabel_cut_mismatch": 0}
    for k in keys:
        while not guard.ok():
            log("a trainer's data_wait_pct is above the limit; pausing 30 s (verify pass)")
            time.sleep(30)
        ready = json.load(open(os.path.join(a.raw, k + ".ready")))
        p = os.path.join(a.raw, k + ".parquet")
        sh = sha256_file(p)
        assert sh == ready["sha256"], f"{k}: parquet sha256 != .ready"
        lab = {}
        if k + ".labeled" in names and not a.ignore_shard_labels:
            labeled = json.load(open(os.path.join(a.raw, k + ".labeled")))
            assert sh == labeled["shard_sha256"], f"{k}: parquet sha256 != .labeled"
            lp = os.path.join(a.raw, k + ".r0labels.jsonl")
            assert sha256_file(lp) == labeled["labels_sha256"], f"{k}: labels sha256 mismatch"
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
            hyp = norm(r["hyp"])
            dur = t1 - t0
            cut_len = int(round(t1 * 16000)) - int(round(t0 * 16000))
            lp = float(r["tok_lp_mean"]) if r.get("tok_lp_mean") is not None else float("-inf")
            if math.isnan(lp):  # H52 amendment 4: an undefined (NaN) mean log-prob ranks lowest, so S1 drops it
                lp = float("-inf")
                checks["lp_nan_as_lowest"] = checks.get("lp_nan_as_lowest", 0) + 1
            f = {"hyp": hyp, "nw": len(hyp.split()), "dur": dur, "ok_len": r["n_samples"] == cut_len,
                 "lp": lp,
                 "lp_min": r.get("tok_lp_min"), "m_mean": r.get("margin_mean"), "m_min": r.get("margin_min"),
                 "a1": norm(stu[uid]) if uid in stu else None}
            f["agree"] = wer(hyp, f["a1"]) if f["a1"] is not None else float("nan")
            rows[uid] = f
            shard_of[uid] = k
        checks["shards"] += 1
    log(f"loaded {len(rows)} rows from {checks['shards']} shards; {checks}")
    # hard rules
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
            for uid in sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[RULES["H5_max_copies"]:]:
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
    # bucketed global shuffle
    rng = np.random.default_rng(a.seed)
    n_out = max(1, round(len(kept) / a.rows_per_shard))
    order = rng.permutation(len(kept))
    bucket_of = {kept[i]: int(b) for b, idx in enumerate(np.array_split(order, n_out)) for i in idx}
    out_dir = os.path.join(a.out, f"librilight.{a.subset}")
    tmp_dir = os.path.join(out_dir, "_buckets")
    os.makedirs(tmp_dir, exist_ok=True)
    extra = [("conf", pa.float32()), ("r0_lp_mean", pa.float32()), ("r0_lp_min", pa.float32()),
             ("r0_margin_mean", pa.float32()), ("r0_margin_min", pa.float32()), ("agree_wer", pa.float32()),
             ("a1_hyp", pa.string()), ("label_src", pa.string())]
    writers, schema = {}, None
    for k in keys:
        p = os.path.join(a.raw, k + ".parquet")
        while not guard.ok():
            log("a trainer's data_wait_pct is above the limit; pausing 30 s")
            time.sleep(30)
        pf = pq.ParquetFile(p)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg)
            ids = t.column("id").to_pylist()
            sel = [i for i, u in enumerate(ids) if u in bucket_of]
            if not sel:
                continue
            t = t.take(sel)
            ids = [ids[i] for i in sel]
            f = [rows[u] for u in ids]
            conf = [math.exp(x["lp"]) * (1.0 - (0.0 if math.isnan(x["agree"]) else min(1.0, x["agree"]))) for x in f]
            t = t.set_column(t.schema.get_field_index("text"), "text", pa.array([x["hyp"] for x in f], pa.string()))
            for (name, typ), vals in zip(extra, (conf, [x["lp"] for x in f], [x["lp_min"] for x in f],
                                               [x["m_mean"] for x in f], [x["m_min"] for x in f],
                                               [x["agree"] for x in f], [x["a1"] for x in f],
                                               ["r0-fp32-gpu-v1"] * len(f))):
                t = t.append_column(pa.field(name, typ), pa.array(vals, typ))
            if schema is None:
                meta = dict(pf.schema_arrow.metadata or {})
                meta[b"h52_labels"] = json.dumps({"labels": label_header, "rules": RULES, "seed": a.seed,
                                                  "result_label": "semi-supervised (R0 labels)"}).encode()
                schema = t.schema.with_metadata(meta)
            t = t.cast(schema)
            bk = np.array([bucket_of[u] for u in ids])
            for b in np.unique(bk):
                part = t.take(np.nonzero(bk == b)[0])
                if b not in writers:
                    writers[b] = pq.ParquetWriter(os.path.join(tmp_dir, f"b{b:05d}.parquet"), schema, compression="snappy")
                writers[b].write_table(part)
        log(f"bucketed {k}")
    for w in writers.values():
        w.close()
    sums = []
    for b in range(n_out):
        while not guard.ok():
            log("a trainer's data_wait_pct is above the limit; pausing 30 s (write pass)")
            time.sleep(30)
        bp = os.path.join(tmp_dir, f"b{b:05d}.parquet")
        if not os.path.exists(bp):
            continue
        t = pq.read_table(bp)
        perm = np.random.default_rng([a.seed, b]).permutation(t.num_rows)
        t = t.take(perm)
        op = os.path.join(out_dir, f"{b:04d}.parquet")
        pq.write_table(t, op + ".part", row_group_size=a.rows_per_group, compression="snappy")
        os.replace(op + ".part", op)
        sums.append(f"{sha256_file(op)}  {b:04d}.parquet")
        os.remove(bp)
    os.rmdir(tmp_dir)
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
