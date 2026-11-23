"""H52: hash a tar while it downloads (reads from page cache as the file grows), list every member,
and copy out only the small .json metadata members. Never extracts audio, never executes anything.

usage: python -I follow_hash.py <tar> <expected_bytes> <official_md5> <subset> <out_dir>
Outputs in out_dir: receipt.json, members.tsv (name, type, size, offset_header, offset_data),
json_meta.jsonl ({"name", "text"} raw JSON text per metadata member), anomalies.txt.
"""
import hashlib
import json
import os
import sys
import tarfile
import time

path, want, official_md5, subset, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
os.makedirs(out_dir, exist_ok=True)
STALL_S = 6 * 3600


class Follow:
    """Sequential reader over a growing file; hashes every byte handed out."""

    def __init__(self, p, want):
        self.p, self.want, self.pos = p, want, 0
        self.md5, self.sha = hashlib.md5(), hashlib.sha256()
        while not os.path.exists(p):
            time.sleep(5)
        self.f = open(p, "rb")
        self.last_growth = time.time()
        self.last_size = 0

    def _avail(self):
        s = os.path.getsize(self.p)
        if s != self.last_size:
            self.last_size, self.last_growth = s, time.time()
        return s

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.want - self.pos
        n = min(n, self.want - self.pos)
        out = bytearray()
        while len(out) < n:
            avail = self._avail() - self.pos - len(out)
            if avail <= 0:
                if time.time() - self.last_growth > STALL_S:
                    raise RuntimeError(f"stalled at {self.pos + len(out)} for {STALL_S}s")
                time.sleep(2.0)
                continue
            b = self.f.read(min(n - len(out), avail))
            if not b:
                time.sleep(1.0)
                continue
            out += b
        b = bytes(out)
        self.md5.update(b)
        self.sha.update(b)
        self.pos += len(b)
        return b


t0 = time.time()
fo = Follow(path, want)
members = open(os.path.join(out_dir, "members.tsv"), "w")
members.write("name\ttype\tsize\toffset_header\toffset_data\n")
jout = open(os.path.join(out_dir, "json_meta.jsonl"), "w")
anomalies = []
counts = {"flac": 0, "json": 0, "dir": 0, "other": 0}
flac_bytes = 0
tf = tarfile.open(fileobj=fo, mode="r|", bufsize=1 << 20)
last_log = 0.0
for m in tf:
    name = m.name
    parts = name.split("/")
    bad = name.startswith("/") or ".." in parts or (parts[0] != subset)
    if m.isdir():
        counts["dir"] += 1
        kind = "dir"
    elif m.isreg():
        kind = "reg"
    else:
        kind = f"type{m.type!r}"
        bad = True
    members.write(f"{name}\t{kind}\t{m.size}\t{m.offset}\t{m.offset_data}\n")
    if bad:
        anomalies.append(f"{kind}\t{name}\t{m.size}")
        continue
    if kind == "reg":
        if name.endswith(".flac"):
            counts["flac"] += 1
            flac_bytes += m.size
        elif name.endswith(".json") and m.size < (16 << 20):
            counts["json"] += 1
            text = tf.extractfile(m).read().decode("utf-8", errors="replace")
            jout.write(json.dumps({"name": name, "text": text}) + "\n")
        else:
            counts["other"] += 1
            anomalies.append(f"unexpected\t{name}\t{m.size}")
    if time.time() - last_log > 300:
        last_log = time.time()
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pos={fo.pos} ({fo.pos / want:.1%}) {counts}", flush=True)
end_of_archive = fo.pos
# hash the remainder (end-of-archive blocks / record padding) up to the expected size
while fo.pos < want:
    fo.read(min(1 << 20, want - fo.pos))
members.close()
jout.close()
size = os.path.getsize(path)
md5 = fo.md5.hexdigest()
receipt = {
    "file": os.path.abspath(path), "subset": subset, "bytes_expected": want, "bytes_on_disk": size,
    "bytes_hashed": fo.pos, "end_of_archive_offset": end_of_archive, "md5": md5,
    "sha256": fo.sha.hexdigest(), "official_md5": official_md5, "md5_matches_official": md5 == official_md5,
    "counts": counts, "flac_bytes": flac_bytes, "anomalies": len(anomalies),
    "seconds": round(time.time() - t0, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join(out_dir, "anomalies.txt"), "w") as f:
    f.write("\n".join(anomalies) + ("\n" if anomalies else ""))
with open(os.path.join(out_dir, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=1)
print(json.dumps(receipt), flush=True)
sys.exit(0 if receipt["md5_matches_official"] and size == want and fo.pos == want else 3)
