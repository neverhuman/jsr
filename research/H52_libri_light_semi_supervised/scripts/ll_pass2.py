"""H52 pass 2: segment Libri-Light FLAC members into LibriSpeech-schema parquet shards (text empty).

Every member is fetched by byte range (HTTPS Range or a local tar) and used only if its sha256
equals pass 1's index (which is itself tied to the official md5). Files excluded by the overlap
check are never fetched. Utterances are spread over K concurrent shard writers by crc32(id), so
each 100-row row group mixes ~100 different recordings (a Libri-Light file is one speaker for
10-60 min). Closed shards get a .ids file and a line in SHA256SUMS; with --push they are rsynced
to HOST:DIR and removed locally. A restart skips members whose utterances are all in closed shards
and never writes an id twice.

usage: python -I ll_pass2.py --index DIR --exclude TSV --source URL|TAR --subset small|medium --out DIR
        [--push HOST:DIR] [--workers 12] [--k 16] [--rows-per-group 100] [--rows-per-shard 1000]
        [--limit-members N] [--first-shard 0]
"""
import argparse
import hashlib
import http.client
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_segment as S  # noqa: E402

UA = "research-data-client/1.0"


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# ---------------------------------------------------------------- worker side
_W = {}


def _fetch(source, off, size):
    if not source.startswith("https://"):
        with open(source, "rb") as f:
            f.seek(off)
            return f.read(size)
    u = urllib.parse.urlsplit(source)
    for attempt in range(12):
        try:
            conn = _W.get("conn")
            if conn is None:
                conn = _W["conn"] = http.client.HTTPSConnection(u.netloc, timeout=120)
            conn.request("GET", u.path, headers={"User-Agent": UA, "Range": f"bytes={off}-{off + size - 1}"})
            r = conn.getresponse()
            if r.status != 206:
                r.read()
                raise IOError(f"HTTP {r.status}")
            b = r.read()
            if len(b) != size:
                raise IOError(f"short read {len(b)} != {size}")
            return b
        except Exception as ex:  # reconnect and retry with backoff
            try:
                _W["conn"].close()
            except Exception:
                pass
            _W["conn"] = None
            if attempt == 11:
                raise
            time.sleep(min(60, 2 ** attempt))


def work(task):
    """task: (source, name, off, size, sha256, meta_json_text, subset, uid_prefix, skip_ids)"""
    source, name, off, size, sha, meta_text, subset, prefix, skip = task
    t0 = time.time()
    b = None
    for _ in range(3):
        b = _fetch(source, off, size)
        if hashlib.sha256(b).hexdigest() == sha:
            break
        b = None
    if b is None:
        return {"name": name, "error": "sha256 mismatch after 3 fetches"}
    t1 = time.time()
    meta = json.loads(meta_text)
    pcm = S.decode_flac(b)
    del b
    dur = len(pcm) / S.SR
    utts, st = S.plan(meta.get("voice_activity") or [], dur, pcm)
    spk = int(str(meta["speaker"]).strip())
    bm = meta.get("book_meta") or {}
    try:
        book = int(str(bm.get("id", "")).strip())
    except ValueError:
        book = -1
    snr = meta.get("snr")
    rows = []
    for k, u in enumerate(utts):
        uid = f"{prefix}-{k:04d}"
        if uid in skip:
            continue
        rows.append(dict(id=uid, flac=S.encode_flac(S.cut(pcm, u)), speaker_id=spk, chapter_id=book, src=name,
                         t0=u["t0"], t1=u["t1"], v0=u["v0"], v1=u["v1"], head=u["head"], tail=u["tail"],
                         n_units=u["n_units"], ecut=u["eh"] | (u["et"] << 1), snr=float(snr) if snr is not None else float("nan"),
                         speech_s=u["speech"]))
    return {"name": name, "rows": rows, "stats": st, "dur": dur, "n_utts": len(utts),
            "fetch_s": t1 - t0, "cpu_s": time.time() - t1}


# ---------------------------------------------------------------- writer side
def schema():
    import pyarrow as pa
    return pa.schema([
        ("file", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text", pa.string()),
        ("speaker_id", pa.int64()),
        ("chapter_id", pa.int64()),
        ("id", pa.string()),
        ("subset", pa.string()),
        ("src", pa.string()),
        ("t0", pa.float64()), ("t1", pa.float64()), ("v0", pa.float64()), ("v1", pa.float64()),
        ("head", pa.float32()), ("tail", pa.float32()), ("n_units", pa.int16()), ("ecut", pa.int8()),
        ("snr", pa.float32()), ("speech_s", pa.float32()),
    ])


class Slot:
    def __init__(self, w):
        self.w, self.path, self.writer, self.buf, self.rows_in_shard, self.ids = w, None, None, [], 0, []
        self.srcs = set()


class Shards:
    def __init__(self, out, subset, k, rpg, rps, first, meta, pusher):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq = pa, pq
        self.out, self.subset, self.k, self.rpg, self.rps = out, subset, k, rpg, rps
        self.dir = os.path.join(out, f"librilight.{subset}")
        os.makedirs(self.dir, exist_ok=True)
        self.next = first
        while os.path.exists(os.path.join(self.dir, f"{self.next:04d}.parquet")) or \
                os.path.exists(os.path.join(self.dir, f"{self.next:04d}.ids")):
            self.next += 1
        self.slots = [Slot(i) for i in range(k)]
        self.schema = schema().with_metadata({b"h52": json.dumps(meta).encode()})
        self.pusher = pusher
        self.handoff = False
        self.closed_ids = set()
        self.on_close = None

    def add(self, r):
        s = self.slots[zlib.crc32(r["id"].encode()) % self.k]
        s.buf.append(r)
        if len(s.buf) >= self.rpg:
            self._flush(s)

    def _flush(self, s):
        if not s.buf:
            return
        if s.writer is None:
            s.path = os.path.join(self.dir, f"{self.next:04d}.parquet")
            self.next += 1
            s.writer = self.pq.ParquetWriter(s.path + ".part", self.schema, compression="snappy",
                                             use_dictionary=["subset", "src"])
        pa = self.pa
        b = s.buf
        tbl = pa.table({
            "file": [r["id"] + ".flac" for r in b],
            "audio": [{"bytes": r["flac"], "path": r["id"] + ".flac"} for r in b],
            "text": [""] * len(b),
            "speaker_id": [r["speaker_id"] for r in b], "chapter_id": [r["chapter_id"] for r in b],
            "id": [r["id"] for r in b], "subset": [self.subset] * len(b), "src": [r["src"] for r in b],
            "t0": [r["t0"] for r in b], "t1": [r["t1"] for r in b], "v0": [r["v0"] for r in b],
            "v1": [r["v1"] for r in b], "head": [r["head"] for r in b], "tail": [r["tail"] for r in b],
            "n_units": [r["n_units"] for r in b], "ecut": [r["ecut"] for r in b], "snr": [r["snr"] for r in b],
            "speech_s": [r["speech_s"] for r in b]}, schema=self.schema)
        s.writer.write_table(tbl, row_group_size=len(b))
        s.rows_in_shard += len(b)
        s.ids.extend(r["id"] for r in b)
        s.srcs.update(r["src"] for r in b)
        s.buf = []
        if s.rows_in_shard >= self.rps:
            self._close(s)

    def _close(self, s):
        if s.writer is None:
            return
        s.writer.close()
        h = hashlib.sha256()
        with open(s.path + ".part", "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        os.replace(s.path + ".part", s.path)
        with open(s.path[:-8] + ".ids", "w") as f:
            f.write("\n".join(s.ids) + "\n")
        with open(os.path.join(self.dir, "SHA256SUMS"), "a") as f:
            f.write(f"{h.hexdigest()}  {os.path.basename(s.path)}\n")
        if self.handoff:  # member list (for from-source labellers), then the ready marker last, atomically
            with open(s.path[:-8] + ".members", "w") as f:
                f.write("\n".join(sorted(s.srcs)) + "\n")
            ready = {"shard": os.path.basename(s.path), "subset": self.subset, "rows": s.rows_in_shard,
                     "bytes": os.path.getsize(s.path), "sha256": h.hexdigest(),
                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with open(s.path[:-8] + ".ready.part", "w") as f:
                json.dump(ready, f)
            os.replace(s.path[:-8] + ".ready.part", s.path[:-8] + ".ready")
        self.closed_ids.update(s.ids)
        if self.on_close:
            self.on_close(s.ids)
        if self.pusher:
            self.pusher.put(s.path)
        s.writer, s.path, s.rows_in_shard, s.ids = None, None, 0, []
        s.srcs = set()

    def finish(self):
        for s in self.slots:
            self._flush(s)
        for s in self.slots:
            self._close(s)


def pusher_thread(q, dest, stop, pause_file="", bwlimit_kbps=0):
    host, rdir = dest.split(":", 1)
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {rdir}"], check=True)
    while not (stop.is_set() and q.empty()):
        try:
            p = q.get(timeout=2)
        except queue.Empty:
            continue
        while pause_file and os.path.exists(pause_file):
            time.sleep(10)
        for attempt in range(20):
            cmd = ["rsync", "-a", "--partial"] + ([f"--bwlimit={bwlimit_kbps}"] if bwlimit_kbps else [])
            r = subprocess.run(cmd + [p, p[:-8] + ".ids", f"{host}:{rdir}/"], capture_output=True, text=True)
            if r.returncode == 0:
                os.remove(p)  # keep the .ids and SHA256SUMS lines locally
                break
            log(f"push retry {attempt} {p}: {r.stderr.strip()[:200]}")
            time.sleep(min(300, 10 * (attempt + 1)))
        else:
            log(f"PUSH FAILED (kept locally): {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--subset", required=True, choices=["small", "medium", "large"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default="")
    ap.add_argument("--handoff", action="store_true", help="write NNNN.ready markers for the labellers; no push")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rows-per-group", type=int, default=100)
    ap.add_argument("--rows-per-shard", type=int, default=1000)
    ap.add_argument("--limit-members", type=int, default=0)
    ap.add_argument("--first-shard", type=int, default=0)
    ap.add_argument("--throttle-file", default="", help="file holding the max in-flight tasks (0 = pause)")
    ap.add_argument("--push-pause-file", default="", help="pushes wait while this file exists")
    ap.add_argument("--push-bwlimit-kbps", type=int, default=0)
    ap.add_argument("--staging-limit-mb", type=int, default=0, help="no new tasks while local shards exceed this")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(a.index, "receipt.json")))
    if not rec.get("md5_matches_official"):
        sys.exit("index receipt does not show an official md5 match; refusing")
    excluded = set()
    for line in open(a.exclude).read().splitlines()[1:]:
        excluded.add(line.split("\t")[0][:-5])
    metas = {}
    for line in open(os.path.join(a.index, "json_meta.jsonl")):
        r = json.loads(line)
        metas[r["name"][:-5]] = r["text"]
    members = []
    for line in open(os.path.join(a.index, "members.tsv")).read().splitlines()[1:]:
        name, kind, size, oh, od, sha = line.split("\t")
        if kind == "reg" and name.endswith(".flac"):
            members.append((int(od), int(size), name, sha))
    members.sort()
    stem = lambda n: n[:-5]
    missing_meta = [n for _, _, n, _ in members if stem(n) not in metas]
    todo = [m for m in members if stem(m[2]) in metas and stem(m[2]) not in excluded]
    out_dir = os.path.join(a.out, f"librilight.{a.subset}")
    os.makedirs(out_dir, exist_ok=True)
    done_members = set()
    dm_path = os.path.join(out_dir, "members_done.txt")
    if os.path.exists(dm_path):
        done_members = set(open(dm_path).read().split())
    skip_ids = set()
    for fn in os.listdir(out_dir):
        if fn.endswith(".ids"):
            skip_ids.update(open(os.path.join(out_dir, fn)).read().split())
    for fn in os.listdir(out_dir):  # incomplete shards from a crash are discarded
        if fn.endswith(".part"):
            os.remove(os.path.join(out_dir, fn))
    todo = [m for m in todo if m[2] not in done_members]
    if a.limit_members:
        todo = todo[:a.limit_members]
    letter = a.subset[0]
    meta = {"tool": "ll_pass2.py", "subset": a.subset, "source": a.source, "archive_md5": rec["md5"],
            "archive_sha256": rec["sha256"], "segment_params": S.PARAMS,
            "label_status": "unlabelled (text empty); R0 labels joined later",
            "note": "semi-supervised (R0 labels) once labelled; chapter_id = LibriVox project id"}
    log(f"members {len(members)} with_meta {len(members) - len(missing_meta)} excluded {len(excluded)} "
        f"done {len(done_members)} todo {len(todo)} skip_ids {len(skip_ids)}")
    stop = threading.Event()
    q = queue.Queue()
    th = None
    if a.push:
        th = threading.Thread(target=pusher_thread, args=(q, a.push, stop, a.push_pause_file, a.push_bwlimit_kbps),
                              daemon=True)
        th.start()
    sh = Shards(a.out, a.subset, a.k, a.rows_per_group, a.rows_per_shard, a.first_shard, meta, q if a.push else None)
    # member -> ids still sitting in open slots; a member is done when none remain
    open_ids_of = {}
    id_member = {}

    def on_close(ids):
        for i in ids:
            m = id_member.pop(i, None)
            if m is not None:
                s = open_ids_of[m]
                s.discard(i)
                if not s:
                    del open_ids_of[m]
                    with open(dm_path, "a") as f:
                        f.write(m + "\n")
    sh.on_close = on_close
    sh.handoff = a.handoff
    stats_path = os.path.join(out_dir, "member_stats.jsonl")
    tot = {"members": 0, "utts": 0, "dur_s": 0.0, "utt_s": 0.0, "errors": 0, "fetch_s": 0.0, "cpu_s": 0.0}
    t_start = time.time()
    last = 0.0
    skip_by_prefix = {}
    for i in skip_ids:
        skip_by_prefix.setdefault(i.rsplit("-", 1)[0], set()).add(i)
    import multiprocessing as mp
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("forkserver")) as ex, \
            open(stats_path, "a") as stf:
        pending = set()
        it = iter(todo)
        exhausted = False

        cap_state = {"t": 0.0, "cap": 2 * a.workers}

        def cap():
            if a.throttle_file and time.time() - cap_state["t"] > 10:
                cap_state["t"] = time.time()
                try:
                    v = int(open(a.throttle_file).read().strip())
                    if v != cap_state["cap"]:
                        log(f"throttle: in-flight cap {cap_state['cap']} -> {v}")
                    cap_state["cap"] = max(0, min(v, 2 * a.workers))
                except (OSError, ValueError):
                    pass
            return cap_state["cap"]

        def staged_mb():
            n = 0
            for e in os.scandir(out_dir):
                if e.name.endswith(".parquet") or e.name.endswith(".part"):
                    n += e.stat().st_size
            return n / 1e6

        def submit_more():
            nonlocal exhausted
            if a.staging_limit_mb and staged_mb() > a.staging_limit_mb:
                return
            while not exhausted and len(pending) < cap():
                try:
                    od, size, name, sha = next(it)
                except StopIteration:
                    exhausted = True
                    return
                prefix = f"ll{letter}-{_member_tag(name, metas[stem(name)])}"
                pending.add(ex.submit(work, (a.source, name, od, size, sha, metas[stem(name)], a.subset, prefix,
                                             frozenset(skip_by_prefix.get(prefix, ())))))
        submit_more()
        from concurrent.futures import FIRST_COMPLETED, wait
        while pending or not exhausted:
            if not pending:
                time.sleep(10)  # paused by the throttle file
                submit_more()
                continue
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for fu in done:
                pending.discard(fu)
                res = fu.result()
                if "error" in res:
                    tot["errors"] += 1
                    log(f"ERROR {res['name']}: {res['error']}")
                    stf.write(json.dumps({"name": res["name"], "error": res["error"]}) + "\n")
                    continue
                name = res["name"]
                ids = [r["id"] for r in res["rows"]]
                if ids:
                    open_ids_of[name] = set(ids)
                    for i in ids:
                        id_member[i] = name
                else:
                    with open(dm_path, "a") as f:
                        f.write(name + "\n")
                for r in res["rows"]:
                    sh.add(r)
                tot["members"] += 1
                tot["utts"] += len(res["rows"])
                tot["dur_s"] += res["dur"]
                tot["utt_s"] += sum(r["t1"] - r["t0"] for r in res["rows"])
                tot["fetch_s"] += res["fetch_s"]
                tot["cpu_s"] += res["cpu_s"]
                stf.write(json.dumps({"name": name, "dur": round(res["dur"], 2), "n_utts": res["n_utts"],
                                      "written": len(res["rows"]), **res["stats"]}) + "\n")
            submit_more()
            if time.time() - last > 60:
                last = time.time()
                el = time.time() - t_start
                log(f"members {tot['members']}/{len(todo)} utts {tot['utts']} audio {tot['dur_s'] / 3600:.1f} h "
                    f"utt {tot['utt_s'] / 3600:.1f} h rate {tot['dur_s'] / max(1, el):.0f}x errors {tot['errors']} "
                    f"shards_next {sh.next} pushq {q.qsize()}")
        sh.finish()
    stop.set()
    if th:
        th.join()
    log(f"DONE {json.dumps(tot)} shards_next {sh.next} elapsed {time.time() - t_start:.0f}s")
    with open(os.path.join(out_dir, "PASS2_DONE.json"), "w") as f:
        json.dump({**tot, "shards_next": sh.next, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)


def _member_tag(name, meta_text):
    """<speaker>-<librivox project>-<sha1(member path)[:6]>, stable and unique per file."""
    meta = json.loads(meta_text)
    spk = str(meta["speaker"]).strip()
    book = str((meta.get("book_meta") or {}).get("id", "x")).strip() or "x"
    return f"{spk}-{book}-{hashlib.sha1(name.encode()).hexdigest()[:6]}"


if __name__ == "__main__":
    main()
