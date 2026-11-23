#!/usr/bin/env python3
"""Per-row FLAC frame counts for every row group of the given parquet shards (exactly what the loader's _frames_after
reads, before speed perturbation), appended to a JSONL table: one line per file {file, size, mtime_ns, groups: [[[id, n], ...]]}.
Files already in the table with the same size and mtime are skipped, so it can run incrementally while shards arrive.
  dur_table.py <table.jsonl> <procs> <dir_or_glob> [...]"""
import glob, io, json, os, sys
from multiprocessing import Pool
import pyarrow.parquet as pq
import soundfile as sf


def scan(f):
    st = os.stat(f)
    pf = pq.ParquetFile(f)
    groups = []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["id", "audio"])
        ids = t.column("id").to_pylist()
        aud = t.column("audio").to_pylist()
        groups.append([[u, sf.info(io.BytesIO(a["bytes"])).frames] for u, a in zip(ids, aud)])
    return {"file": f, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "groups": groups}


if __name__ == "__main__":
    table, procs = sys.argv[1], int(sys.argv[2])
    files = []
    for p in sys.argv[3:]:
        files += sorted(glob.glob(os.path.join(p, "*.parquet")) if os.path.isdir(p) else glob.glob(p))
    have = {}
    if os.path.exists(table):
        for l in open(table):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            have[r["file"]] = (r["size"], r["mtime_ns"])
    todo = [f for f in files if have.get(f) != (os.stat(f).st_size, os.stat(f).st_mtime_ns)]
    with open(table, "a") as out, Pool(procs) as pool:
        for r in pool.imap_unordered(scan, todo):
            out.write(json.dumps(r) + "\n"); out.flush()
    print(json.dumps({"table": table, "files": len(files), "scanned": len(todo)}))
