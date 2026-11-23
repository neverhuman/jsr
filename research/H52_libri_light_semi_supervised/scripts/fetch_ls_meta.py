"""H52: stream LibriSpeech dev-clean.tar.gz, md5 the whole stream, and keep ONLY the corpus-level
metadata TXT files (SPEAKERS / CHAPTERS / BOOKS / README / LICENSE). No audio is written or decoded.
These files list every LibriSpeech speaker / chapter / book with its SUBSET column, so the test
speaker and book ids come from id lists only (no test audio or transcripts are read).

usage: python -I fetch_ls_meta.py <url> <expected_md5> <out_dir>
"""
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

url, want_md5, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
KEEP = {"LibriSpeech/SPEAKERS.TXT", "LibriSpeech/CHAPTERS.TXT", "LibriSpeech/BOOKS.TXT",
        "LibriSpeech/README.TXT", "LibriSpeech/LICENSE.TXT"}


class Hashing:
    def __init__(self, r):
        self.r, self.md5, self.n = r, hashlib.md5(), 0

    def read(self, n=-1):
        b = self.r.read(n)
        self.md5.update(b)
        self.n += len(b)
        return b


req = urllib.request.Request(url, headers={"User-Agent": "research-data-client/1.0"})
resp = urllib.request.urlopen(req, timeout=60)
h = Hashing(resp)
kept = {}
tf = tarfile.open(fileobj=h, mode="r|gz")
for m in tf:
    if m.name in KEEP and m.isreg():
        kept[m.name] = tf.extractfile(m).read()
while h.read(1 << 20):
    pass
md5 = h.md5.hexdigest()
ok = md5 == want_md5
receipt = {"url": url, "final_url": resp.geturl(), "bytes": h.n, "md5": md5, "official_md5": want_md5,
           "md5_matches_official": ok, "kept": {}}
if ok:
    for name, data in kept.items():
        p = os.path.join(out_dir, os.path.basename(name))
        with open(p, "wb") as f:
            f.write(data)
        receipt["kept"][os.path.basename(name)] = hashlib.sha256(data).hexdigest()
with open(os.path.join(out_dir, "fetch_receipt.json"), "w") as f:
    json.dump(receipt, f, indent=1)
print(json.dumps(receipt, indent=1))
sys.exit(0 if ok else 3)
