"""H52 student hyps "from source": A1-49000 INT8 on Libri-Light utterances re-cut from CloudFront.

For hosts that cannot reach the shard host. Segmentation is deterministic, so this re-creates
exactly the utterances of ll_pass2 (same index, skip list, ll_segment.plan, ids from
ll_pass2._member_tag) for the members with index % mod == rem, runs one single-threaded
`ultra-engine-eval --precision int8` per core on 100-utterance npz chunks of the cut PCM, and
appends {"id", "hyp", "tokens", "t0", "t1", "src"} per utterance plus {"member_done", "n_utts"} per
member to --out (append-only; a restart skips completed members). The finaliser joins on id and
uses a hyp only when t0 / t1 equal the shard row's.

usage: python -I ll_student_src.py --index DIR --skip TSV --source URL --subset medium --mod 4 --rem 0
        --engine BIN [--loader DIR] --weights DIR --work DIR --out FILE --max-procs 24 [--cpu-cap 0.80]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_pass2 as P  # noqa: E402
import ll_segment as S  # noqa: E402


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def cpu_busy(prev):
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle, total = v[3] + v[4], sum(v[:8])
    if prev is None:
        return None, (idle, total)
    di, dt = idle - prev[0], total - prev[1]
    return (1.0 - di / dt) if dt > 0 else 0.0, (idle, total)


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--skip", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--mod", type=int, default=1)
    ap.add_argument("--rem", type=int, default=0)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--loader", default="")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-procs", type=int, default=24)
    ap.add_argument("--start-procs", type=int, default=0)
    ap.add_argument("--cpu-cap", type=float, default=0.80)
    ap.add_argument("--min-avail-gb", type=float, default=40.0)
    ap.add_argument("--max-file", default="")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--members-file", default="", help="process exactly these member names (overrides --mod/--rem)")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(a.index, "receipt.json")))
    if not rec.get("md5_matches_official"):
        sys.exit("index receipt does not show an official md5 match; refusing")
    skip = {l.split("\t")[0][:-5] for l in open(a.skip).read().splitlines()[1:]}
    metas = {}
    for line in open(os.path.join(a.index, "json_meta.jsonl")):
        r = json.loads(line)
        metas[r["name"][:-5]] = r["text"]
    members = []
    for line in open(os.path.join(a.index, "members.tsv")).read().splitlines()[1:]:
        name, kind, size, oh, od, sha = line.split("\t")
        if kind == "reg" and name.endswith(".flac") and name[:-5] in metas and name[:-5] not in skip:
            members.append((int(od), int(size), name, sha))
    members.sort()
    if a.members_file:
        want = set(open(a.members_file).read().split())
        mine = [m for m in members if m[2] in want]
    else:
        mine = [m for i, m in enumerate(members) if i % a.mod == a.rem]
    done_members = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "member_done" in r:
                done_members.add(r["member_done"])
    todo = deque(m for m in mine if m[2] not in done_members)
    os.makedirs(a.work, exist_ok=True)
    out = open(a.out, "a", buffering=1)
    letter = a.subset[0]
    log(f"members {len(members)} mine {len(mine)} done {len(done_members)} todo {len(todo)}")
    pre = ([os.path.join(a.loader, "ld-linux-x86-64.so.2"), "--library-path", a.loader] if a.loader else [])
    target = a.start_procs or max(1, a.max_procs // 2)
    running, queue = {}, deque()
    pending_left = {}  # member -> utterances not yet decoded by the engine
    buf_pcm, buf_meta = [], []  # current chunk being filled
    chunk_no = 0
    _, prev = cpu_busy(None)
    last_adj = last_log = time.time()
    n_done_utts = n_done_members = 0
    t_start = time.time()
    audio_done = 0.0

    def flush_chunk():
        nonlocal chunk_no, buf_pcm, buf_meta
        if not buf_meta:
            return
        offs = np.cumsum([0] + [len(x) for x in buf_pcm]).astype("<i8")
        ids = [m["id"] for m in buf_meta]
        p = os.path.join(a.work, f"c{a.rem:02d}_{chunk_no:06d}.npz")
        np.savez(p, pcm=np.concatenate(buf_pcm).astype("<i2"), offsets=offs,
                 uids=np.asarray(ids, dtype=f"<U{max(len(i) for i in ids)}"),
                 speakers=np.asarray([m["spk"] for m in buf_meta], dtype="<i8"),
                 texts=np.asarray([""] * len(ids), dtype="<U1"))
        queue.append((p, list(buf_meta), 0))
        chunk_no += 1
        buf_pcm, buf_meta = [], []

    while todo or queue or running or buf_meta:
        now = time.time()
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
            elif busy is not None:
                if busy > a.cpu_cap and target > 1:
                    target = max(1, int(target * 0.85))
                elif busy < a.cpu_cap - 0.10:
                    target = min(mx, target + 4)
            target = min(target, mx)
            if now - last_log > 120:
                last_log = now
                el = now - t_start
                log(f"busy={busy if busy is None else round(busy, 3)} avail={avail:.0f}GB target={target} running={len(running)} "
                    f"queued={len(queue)} todo_members={len(todo)} done_members={n_done_members} utts={n_done_utts} "
                    f"rate={audio_done / max(1, el):.0f}x")
        # keep the engine queue fed: cut more members when fewer than target+4 chunks wait
        while todo and len(queue) < target + 4:
            od, size, name, sha = todo.popleft()
            try:
                b = P._fetch(a.source, od, size)
            except Exception as ex:
                log(f"fetch failed {name}: {ex}; retry later")
                todo.append((od, size, name, sha))
                break
            if hashlib.sha256(b).hexdigest() != sha:
                log(f"sha256 mismatch {name}; retry later")
                todo.append((od, size, name, sha))
                break
            meta_text = metas[name[:-5]]
            meta = json.loads(meta_text)
            pcm = S.decode_flac(b)
            del b
            utts, _ = S.plan(meta.get("voice_activity") or [], len(pcm) / S.SR, pcm)
            prefix = f"ll{letter}-{P._member_tag(name, meta_text)}"
            spk = int(str(meta["speaker"]).strip())
            if not utts:
                out.write(json.dumps({"member_done": name, "n_utts": 0}) + "\n")
                n_done_members += 1
                continue
            pending_left[name] = len(utts)
            for k, u in enumerate(utts):
                buf_pcm.append(S.cut(pcm, u))
                buf_meta.append({"id": f"{prefix}-{k:04d}", "t0": u["t0"], "t1": u["t1"], "src": name, "spk": spk})
                if len(buf_meta) >= a.chunk:
                    flush_chunk()
        if not todo and buf_meta:
            flush_chunk()
        while queue and len(running) < target:
            p, metas_, tries = queue.popleft()
            od_ = p[:-4] + ".out"
            os.makedirs(od_, exist_ok=True)
            cmd = ["nice", "-n", "19"] + pre + [a.engine, "--weights", a.weights, "--npz", p, "--precision", "int8",
                                               "--streams", str(a.streams), "--limit", "100000", "--out", od_]
            pr = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(od_ + ".err", "w"))
            running[pr] = (p, metas_, od_, tries)
        for pr in list(running):
            if pr.poll() is None:
                continue
            p, metas_, od_, tries = running.pop(pr)
            if pr.returncode != 0:
                log(f"engine failed rc={pr.returncode} on {p}: {open(od_ + '.err').read()[-200:]}")
                if tries < 1:
                    queue.append((p, metas_, tries + 1))
                    continue
                raise SystemExit("engine failed twice; stopping")
            res = json.load(open(os.path.join(od_, "engine_eval.json")))
            by_index = {row["index"]: row for row in res["rows"]}
            for i, m in enumerate(metas_):
                row = by_index[i]
                out.write(json.dumps({"id": m["id"], "hyp": row.get("hyp", ""), "tokens": row.get("tokens"),
                                      "t0": m["t0"], "t1": m["t1"], "src": m["src"]}) + "\n")
                n_done_utts += 1
                audio_done += m["t1"] - m["t0"]
                pending_left[m["src"]] -= 1
                if pending_left[m["src"]] == 0:
                    del pending_left[m["src"]]
                    out.write(json.dumps({"member_done": m["src"]}) + "\n")
                    n_done_members += 1
            shutil.rmtree(od_, ignore_errors=True)
            os.remove(p)
        time.sleep(0.2)
    log(f"DONE members {n_done_members} utts {n_done_utts} audio {audio_done / 3600:.1f} h in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
