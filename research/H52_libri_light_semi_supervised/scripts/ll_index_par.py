"""H52 pass 1 (parallel prefetch): stream a Libri-Light tar over HTTPS (or from a local file) and write its index.

Nothing from the archive is stored except the small .json metadata members (as text), and nothing
is executed. Outputs in out_dir:
  members.tsv      name, type, size, offset_header, offset_data, sha256 (of the member's data)
  json_meta.jsonl  {"name", "sha256", "text"} for every .json member
  anomalies.txt    unsafe names, unexpected types/extensions, header checksum errors
  receipt.json     url, size, md5, sha256, official md5 match, counts, timings
A connection error resumes with an HTTP Range request at the last member boundary, restoring the
hash state saved there, so the whole-file hashes cover every byte exactly once.

usage: python -I ll_index.py <url-or-path> <expected_bytes> <official_md5> <subset> <out_dir> [rate_MBps]
"""
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.parse

src, want, official_md5, subset, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rate = float(sys.argv[6]) * 1e6 if len(sys.argv) > 6 else 0.0
os.makedirs(out_dir, exist_ok=True)
CHUNK = int(os.environ.get("LL_INDEX_CHUNK", 16 << 20))
FAILED = []
UA = "research-data-client/1.0"


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


class Source:
    """Byte source from `offset`, over HTTPS (Range) or a local file."""

    def __init__(self, offset):
        self.offset = offset
        if src.startswith("https://"):
            u = urllib.parse.urlsplit(src)
            self.conn = http.client.HTTPSConnection(u.netloc, timeout=120)
            hdr = {"User-Agent": UA}
            if offset:
                hdr["Range"] = f"bytes={offset}-"
            self.conn.request("GET", u.path, headers=hdr)
            self.resp = self.conn.getresponse()
            ok = 206 if offset else 200
            if self.resp.status != ok:
                raise IOError(f"HTTP {self.resp.status} at offset {offset}")
            self.read = self.resp.read
        else:
            self.f = open(src, "rb")
            self.f.seek(offset)
            self.read = self._fread
        self.pos = offset

    def _fread(self, n):
        fail = int(os.environ.get("LL_INDEX_FAIL_AT", "0") or 0)
        if fail and self.pos < fail <= self.pos + n and not FAILED:
            FAILED.append(1)
            raise OSError("injected failure")
        b = self.f.read(n)
        self.pos += len(b)
        return b

    def close(self):
        try:
            (self.conn if hasattr(self, "conn") else self.f).close()
        except Exception:
            pass



import threading
from concurrent.futures import ThreadPoolExecutor

PAR = int(os.environ.get("LL_INDEX_PAR", "8"))
PCHUNK = int(os.environ.get("LL_INDEX_PCHUNK", str(32 << 20)))


class ParallelSource:
    """Ordered byte source from `offset`: PAR threads fetch consecutive PCHUNK ranges over keep-alive
    HTTPS connections; read() hands them out strictly in order (at most 2*PAR chunks buffered)."""

    def __init__(self, offset):
        self.u = urllib.parse.urlsplit(src)
        self.pos = offset
        self.next_start = offset
        self.done = {}
        self.cv = threading.Condition()
        self.stop = False
        self.err = None
        self.local = threading.local()
        self.cur = b""
        self.threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(PAR)]
        for t in self.threads:
            t.start()

    def _conn(self):
        c = getattr(self.local, "c", None)
        if c is None:
            c = self.local.c = http.client.HTTPSConnection(self.u.netloc, timeout=120)
        return c

    def _worker(self):
        while True:
            with self.cv:
                while not self.stop and self.next_start - self.pos > 2 * PAR * PCHUNK:
                    self.cv.wait(1.0)
                if self.stop or self.next_start >= want:
                    return
                start = self.next_start
                self.next_start = min(want, start + PCHUNK)
                end = self.next_start - 1
            for attempt in range(8):
                try:
                    c = self._conn()
                    c.request("GET", self.u.path, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
                    r = c.getresponse()
                    if r.status != 206:
                        r.read()
                        raise IOError(f"HTTP {r.status} for {start}-{end}")
                    b = r.read()
                    if len(b) != end - start + 1:
                        raise IOError(f"short range {len(b)} for {start}-{end}")
                    with self.cv:
                        self.done[start] = b
                        self.cv.notify_all()
                    break
                except Exception as ex:
                    try:
                        self.local.c.close()
                    except Exception:
                        pass
                    self.local.c = None
                    if attempt == 7:
                        with self.cv:
                            self.err = ex
                            self.cv.notify_all()
                        return
                    time.sleep(min(30, 2 ** attempt))

    def read(self, n):
        if not self.cur:
            with self.cv:
                while self.pos not in self.done:
                    if self.err is not None:
                        raise IOError(f"prefetch failed: {self.err}")
                    if self.pos >= want:
                        return b""
                    self.cv.wait(1.0)
                self.cur = self.done.pop(self.pos)
                self.cv.notify_all()
        b, self.cur = self.cur[:n], self.cur[n:]
        self.pos += len(b)
        return b

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()


_POOL = ThreadPoolExecutor(3)


class Stream:
    def __init__(self):
        self.pos = 0
        self.md5, self.sha = hashlib.md5(), hashlib.sha256()
        self.buf, self.bpos = b"", 0
        self.src = ParallelSource(0) if src.startswith("https://") else Source(0)
        self.t0 = time.time()
        self.bytes_since = 0

    def _fill(self):
        b = self.src.read(CHUNK)
        if not b:
            raise EOFError(f"source ended at {self.pos + len(self.buf) - self.bpos}")
        if rate:
            self.bytes_since += len(b)
            ahead = self.bytes_since / rate - (time.time() - self.t0)
            if ahead > 0:
                time.sleep(ahead)
        self.buf = self.buf[self.bpos:] + b
        self.bpos = 0

    def read(self, n, extra=None):
        """Exactly n bytes (hashed; also into `extra`, concurrently), or EOFError."""
        while len(self.buf) - self.bpos < n:
            self._fill()
        out = self.buf[self.bpos:self.bpos + n]
        self.bpos += n
        self.pos += n
        if n >= (1 << 20):  # hashlib releases the GIL for large buffers: all digests in parallel
            fs = [_POOL.submit(self.md5.update, out), _POOL.submit(self.sha.update, out)]
            if extra is not None:
                fs.append(_POOL.submit(extra.update, out))
            for f in fs:
                f.result()
        else:
            self.md5.update(out)
            self.sha.update(out)
            if extra is not None:
                extra.update(out)
        return out

    def checkpoint(self):
        return (self.pos, self.md5.copy(), self.sha.copy())

    def restore(self, ck):
        self.src.close()
        self.pos, md5, sha = ck
        self.md5, self.sha = md5.copy(), sha.copy()
        self.buf, self.bpos = b"", 0
        self.src = ParallelSource(self.pos) if src.startswith("https://") else Source(self.pos)
        self.t0, self.bytes_since = time.time(), 0


def octal(field):
    if field[:1] and field[0] & 0x80:  # base-256 (GNU) for big sizes
        v = int.from_bytes(field[1:], "big")
        return v
    s = field.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def header_ok(h):
    stored = octal(h[148:156])
    calc = sum(h[:148]) + 8 * 32 + sum(h[156:])
    return stored == calc


def safe(name):
    parts = name.split("/")
    return not (name.startswith("/") or ".." in parts or parts[0] != subset or "\\" in name)


members = open(os.path.join(out_dir, "members.tsv.part"), "w", buffering=1 << 16)
members.write("name\ttype\tsize\toffset_header\toffset_data\tsha256\n")
jout = open(os.path.join(out_dir, "json_meta.jsonl.part"), "w", buffering=1 << 16)
anomalies = []
counts = {"flac": 0, "json": 0, "dir": 0, "other": 0, "resumes": 0}
flac_bytes = 0
st = Stream()
ck = st.checkpoint()
pending = {}  # longname / pax overrides for the next header
last_log = time.time()
end_offset = None
while True:
    try:
        hoff = st.pos
        h = st.read(512)
        if h == b"\0" * 512:
            end_offset = hoff
            break
        if not header_ok(h):
            raise ValueError(f"bad header checksum at {hoff}")
        typ = h[156:157]
        size = octal(h[124:136])
        name = h[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        if h[257:262] == b"ustar":
            prefix = h[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
            if prefix:
                name = prefix + "/" + name
        dpos = st.pos
        padded = (size + 511) // 512 * 512
        if typ in (b"L", b"K", b"x", b"g"):
            data = st.read(padded)[:size]
            if typ == b"L":
                pending["name"] = data.split(b"\0", 1)[0].decode("utf-8", "replace")
            elif typ == b"x":
                i = 0
                while i < len(data):
                    sp = data.index(b" ", i)
                    ln = int(data[i:sp])
                    kv = data[sp + 1:i + ln - 1].decode("utf-8", "replace")
                    k, _, v = kv.partition("=")
                    if k == "path":
                        pending["name"] = v
                    elif k == "size":
                        pending["size"] = int(v)
                    i += ln
            # 'K' (long link) and 'g' (global pax) carry nothing we use
            continue
        name = pending.pop("name", name)
        if "size" in pending:
            size = pending.pop("size")
            padded = (size + 511) // 512 * 512
        pending.clear()
        kind = "dir" if typ == b"5" else ("reg" if typ in (b"0", b"\0", b"7") else f"type{typ!r}")
        msha = hashlib.sha256()
        keep = kind == "reg" and name.endswith(".json") and size < (16 << 20)
        got = bytearray() if keep else None
        left = size
        while left > 0:
            b = st.read(min(CHUNK, left), extra=msha)
            if keep:
                got += b
            left -= len(b)
        if padded > size:
            st.read(padded - size)
        digest = msha.hexdigest()
        members.write(f"{name}\t{kind}\t{size}\t{hoff}\t{dpos}\t{digest}\n")
        if kind == "dir":
            counts["dir"] += 1
        bad = not safe(name) or not (kind in ("dir", "reg"))
        if bad:
            anomalies.append(f"{kind}\t{name}\t{size}")
        elif kind == "reg":
            if name.endswith(".flac"):
                counts["flac"] += 1
                flac_bytes += size
            elif keep:
                counts["json"] += 1
                jout.write(json.dumps({"name": name, "sha256": digest,
                                       "text": bytes(got).decode("utf-8", "replace")}) + "\n")
            else:
                counts["other"] += 1
                anomalies.append(f"unexpected\t{name}\t{size}")
        ck = st.checkpoint()
        if time.time() - last_log > 60:
            last_log = time.time()
            el = time.time() - st.t0 if not counts["resumes"] else 0
            log(f"pos={st.pos} ({st.pos / want:.2%}) {counts}")
    except (OSError, EOFError, http.client.HTTPException, TimeoutError) as ex:
        if st.pos >= want and isinstance(ex, EOFError):
            raise
        counts["resumes"] += 1
        if counts["resumes"] > 200:
            raise
        log(f"resume #{counts['resumes']} after {type(ex).__name__}: {ex}; back to {ck[0]}")
        time.sleep(min(60, 5 * counts["resumes"]))
        # members.tsv / json lines are written only at boundaries, so nothing to undo
        pending.clear()
        st.restore(ck)
# hash the trailing zero blocks / record padding up to the expected size
while st.pos < want:
    try:
        st.read(min(CHUNK, want - st.pos))
    except (OSError, EOFError, http.client.HTTPException, TimeoutError) as ex:
        counts["resumes"] += 1
        log(f"tail resume after {type(ex).__name__}")
        ck2 = st.checkpoint()
        st.restore(ck2)
extra = st.buf[st.bpos:] + st.src.read(1)
members.close()
jout.close()
md5 = st.md5.hexdigest()
receipt = {"source": src, "subset": subset, "bytes_expected": want, "bytes_hashed": st.pos,
           "trailing_bytes_beyond_expected": len(extra), "end_of_archive_offset": end_offset,
           "md5": md5, "sha256": st.sha.hexdigest(), "official_md5": official_md5,
           "md5_matches_official": md5 == official_md5, "counts": counts, "flac_bytes": flac_bytes,
           "anomalies": len(anomalies), "seconds": round(time.time() - st.t0, 1),
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": "ll_index_par.py"}
with open(os.path.join(out_dir, "anomalies.txt"), "w") as f:
    f.write("\n".join(anomalies) + ("\n" if anomalies else ""))
ok = receipt["md5_matches_official"] and st.pos == want and not extra
for p in ("members.tsv", "json_meta.jsonl"):
    os.replace(os.path.join(out_dir, p + ".part"), os.path.join(out_dir, p if ok else p + ".UNVERIFIED"))
with open(os.path.join(out_dir, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=1)
log(json.dumps(receipt))
sys.exit(0 if ok else 3)
