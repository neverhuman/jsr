"""Reference word timings for LibriSpeech dev/test (`ultra.word-timing/v1`), ruling O14.

Source: the public Montreal-Forced-Aligner LibriSpeech alignments released by Loren Lugosch on
Zenodo (record 2619474, DOI 10.5281/zenodo.2619474, version 1.0, CC BY 4.0). They were made with
MFA and its pretrained LibriSpeech acoustic model; they are independent of every system we test.

Only the dev/test members are fetched: the archive's central directory and the contiguous byte
spans holding dev-clean, dev-other, test-clean and test-other are read with HTTP range requests
(one connection), and every extracted member is checked against its zip CRC-32.

On-disk layout (data drive, never git):
  corpora/librispeech/mfa_alignments_zenodo2619474/<split>/**.TextGrid   raw MFA output
  corpora/librispeech/mfa_alignments_zenodo2619474/SOURCE.json          url, record, spans, hashes
  corpora/librispeech/word_timing/<split>.jsonl                          ultra.word-timing/v1
  corpora/librispeech/word_timing/MANIFEST.json                          sha256 of each jsonl

`ultra.word-timing/v1`: one JSON object per line, sorted by id:
  {"id": "<spk>-<chapter>-<utt>", "words": [{"w": "WORD", "start_s": 0.45, "end_s": 0.61}, ...]}
Words are upper-case LibriSpeech reference words; silences are dropped. MFA `<unk>` tokens take the
reference word at the same position. Utterances MFA could not align are absent.

CLI (run from python/ultra; heavy steps under `ionice -c3 nice -n 15 taskset -c 48-63`):
  python -m ultra.word_timing fetch | build | coverage | spotcheck
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass

SCHEMA = "ultra.word-timing/v1"
DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
RAW = os.path.join(DATA, "corpora", "librispeech", "mfa_alignments_zenodo2619474")
OUT = os.path.join(DATA, "corpora", "librispeech", "word_timing")
SPLITS = ("dev-clean", "dev-other", "test-clean", "test-other")

ZENODO_RECORD = "2619474"
ZENODO_DOI = "10.5281/zenodo.2619474"
ZENODO_URL = "https://zenodo.org/api/records/2619474/files/librispeech_alignments.zip/content"
ZENODO_ARCHIVE_MD5 = "2bab567d0ace651a4ba254e813629f46"  # whole archive, as published by Zenodo
ZENODO_ARCHIVE_SIZE = 623023332
LICENCE = "CC BY 4.0 (Zenodo record metadata); derived from LibriSpeech (CC BY 4.0)"

_SILENCE = {"", "sil", "sp", "spn", "<eps>"}
_UNK = "<unk>"


@dataclass(frozen=True)
class Word:
    w: str
    start_s: float
    end_s: float


# ---------------------------------------------------------------------------------------------
# TextGrid parsing and conversion


_INTERVAL = re.compile(r'xmin = ([0-9.eE+-]+)\s+xmax = ([0-9.eE+-]+)\s+text = "((?:[^"]|"")*)"')


def parse_textgrid_words(text: str) -> list[tuple[str, float, float]]:
    """Non-silence intervals of the `words` tier of an MFA long-format TextGrid."""
    m = re.search(r'name = "words"(.*?)(?:item \[\d+\]:|\Z)', text, re.S)
    if not m:
        raise ValueError("TextGrid has no 'words' tier")
    out = []
    for a, b, t in _INTERVAL.findall(m.group(1)):
        t = t.replace('""', '"').strip()
        if t.lower() in _SILENCE:
            continue
        out.append((t, float(a), float(b)))
    return out


def attach_reference(tokens: list[tuple[str, float, float]], ref_words: list[str]) -> list[Word] | None:
    """Map MFA tokens onto the upper-case reference words, position by position.

    MFA's dictionary splits some apostrophe words into pieces (WICKER'S -> wicker + s,
    D'AVRIGNY -> d + <unk>, BUSH' -> bush); such pieces are merged back into one reference word
    spanning first start to last end. `<unk>` matches any single reference word or piece.
    Returns None when the tokens cannot be mapped onto the whole reference; such utterances are
    reported as unmatched.
    """
    def same(tok: str, word: str) -> bool:
        return tok.lower() == _UNK or tok.upper() == word

    out, i = [], 0
    for r in ref_words:
        parts = [p for p in r.split("'") if p]
        if i < len(tokens) and tokens[i][0].upper() == r:
            k = 1
        elif parts and parts != [r] and i + len(parts) <= len(tokens) and all(
                same(tokens[i + j][0], p) for j, p in enumerate(parts)):
            k = len(parts)
        elif i < len(tokens) and same(tokens[i][0], r):
            k = 1
        else:
            return None
        a, b = tokens[i][1], tokens[i + k - 1][2]
        if not (0.0 <= a <= b):
            return None
        out.append(Word(r, round(a, 3), round(b, 3)))
        i += k
    return out if i == len(tokens) else None


def to_jsonl_line(uid: str, words: list[Word]) -> str:
    return json.dumps({"id": uid, "words": [{"w": w.w, "start_s": w.start_s, "end_s": w.end_s}
                                             for w in words]}, separators=(",", ":"))


def parse_jsonl_line(line: str) -> tuple[str, list[Word]]:
    d = json.loads(line)
    words = [Word(str(x["w"]), float(x["start_s"]), float(x["end_s"])) for x in d["words"]]
    return str(d["id"]), words


def validate(words: list[Word], audio_s: float | None = None) -> None:
    """Raise ValueError unless times are non-negative, ordered, and within the audio."""
    prev_end = 0.0
    for w in words:
        if not w.w or w.w != w.w.upper():
            raise ValueError(f"word must be non-empty upper-case: {w.w!r}")
        if w.start_s < 0 or w.end_s < w.start_s or w.start_s + 1e-6 < prev_end:
            raise ValueError(f"bad timing order at {w}")
        prev_end = w.end_s
    if audio_s is not None and words and words[-1].end_s > audio_s + 1e-3:
        raise ValueError(f"last word ends at {words[-1].end_s} s after audio end {audio_s} s")


# ---------------------------------------------------------------------------------------------
# Loader


def path(split: str, root: str = OUT) -> str:
    if split not in SPLITS:
        raise KeyError(f"no word timings for split {split!r}; have {SPLITS}")
    return os.path.join(root, f"{split}.jsonl")


def load(split: str, root: str = OUT) -> dict[str, list[Word]]:
    """id -> timed reference words, for one split."""
    out: dict[str, list[Word]] = {}
    with open(path(split, root), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uid, words = parse_jsonl_line(line)
                if uid in out:
                    raise ValueError(f"duplicate id {uid} in {split}")
                validate(words)
                out[uid] = words
    return out


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Fetch (HTTP range requests on the Zenodo archive; dev/test members only)


_MEMBER = re.compile(r"(dev|test)-(clean|other)/(\d+/\d+/\d+-\d+-\d+\.TextGrid|unaligned\.txt)")


def member_relpath(name: str) -> str:
    """Allowlisted archive member name -> relative path; anything else (zip slip) raises."""
    if not _MEMBER.fullmatch(name):
        raise ValueError(f"unexpected archive member {name!r}")
    return os.path.normpath(name)


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; each read is one HTTP Range request."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos, self.bytes_read = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}",
                                                        "User-Agent": "ultra-data-fetch"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:
                raise IOError(f"server ignored Range request (HTTP {r.status})")
            d = r.read()
        b[:len(d)] = d
        self.pos += len(d)
        self.bytes_read += len(d)
        return len(d)


def fetch(raw: str = RAW) -> dict:
    """Extract the dev/test TextGrids from the Zenodo archive into `raw` and write SOURCE.json."""
    rf = _RangeFile(ZENODO_URL, ZENODO_ARCHIVE_SIZE)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 20))
    infos = [i for i in z.infolist() if i.filename.split("/")[0] in SPLITS and not i.is_dir()]
    infos.sort(key=lambda i: i.header_offset)  # sequential reads, one span per split
    files = {}
    for i in infos:
        rel = member_relpath(i.filename)
        data = z.read(i)  # zipfile checks CRC-32 and raises BadZipFile on mismatch
        dest = os.path.join(raw, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    listing = "".join(f"{v}  {k}\n" for k, v in sorted(files.items()))
    src = {
        "source": "Montreal-Forced-Aligner LibriSpeech alignments (Lugosch et al. 2019)",
        "zenodo_record": ZENODO_RECORD, "doi": ZENODO_DOI, "version": "1.0", "url": ZENODO_URL,
        "archive_md5_published": ZENODO_ARCHIVE_MD5, "archive_size": ZENODO_ARCHIVE_SIZE,
        "archive_md5_verified": False,
        "note": "only dev/test members fetched by HTTP range; each member CRC-32-checked by zipfile",
        "licence": LICENCE, "splits": list(SPLITS), "members": len(files),
        "bytes_fetched": rf.bytes_read,
        "members_sha256_listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
    }
    with open(os.path.join(raw, "MEMBERS.sha256"), "w") as f:
        f.write(listing)
    with open(os.path.join(raw, "SOURCE.json"), "w") as f:
        json.dump(src, f, indent=1)
    return src


# ---------------------------------------------------------------------------------------------
# Build + coverage against our LibriSpeech transcripts


def build(split: str, raw: str = RAW, out: str = OUT) -> dict:
    """Convert one split to JSONL; return coverage statistics against libri transcripts."""
    from ultra import libri  # lazy: needs the decoded caches

    utts = libri.load_split(split)
    tg = {}
    for dirpath, _, names in os.walk(os.path.join(raw, split)):
        for n in names:
            if n.endswith(".TextGrid"):
                tg[n[:-len(".TextGrid")]] = os.path.join(dirpath, n)
    lines, n_words, timed_words, matched, overrun, max_overrun = [], 0, 0, 0, 0, 0.0
    unmatched = []
    for u in utts:
        ref = u.text.upper().split()
        n_words += len(ref)
        p = tg.get(u.uid)
        if p is None:
            unmatched.append((u.uid, "no alignment"))
            continue
        with open(p, encoding="utf-8") as f:
            words = attach_reference(parse_textgrid_words(f.read()), ref)
        if words is None:
            unmatched.append((u.uid, "token mismatch"))
            continue
        validate(words)
        over = words[-1].end_s - u.seconds if words else 0.0
        if over > 1e-3:
            overrun += 1
            max_overrun = max(max_overrun, over)
        matched += 1
        timed_words += len(words)
        lines.append((u.uid, to_jsonl_line(u.uid, words)))
    os.makedirs(out, exist_ok=True)
    dest = path(split, out)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for _, line in sorted(lines):
            f.write(line + "\n")
    os.replace(tmp, dest)
    return {"split": split, "utterances": len(utts), "matched": matched,
            "pct_utts_matched": round(100.0 * matched / len(utts), 3),
            "ref_words": n_words, "timed_words": timed_words,
            "pct_words_timed": round(100.0 * timed_words / n_words, 3),
            "last_word_after_audio_end": overrun, "max_overrun_s": round(max_overrun, 4),
            "unmatched": unmatched, "jsonl": dest, "sha256": sha256_file(dest)}


def write_manifest(stats: list[dict], raw: str = RAW, out: str = OUT) -> str:
    with open(os.path.join(raw, "SOURCE.json")) as f:
        src = json.load(f)
    man = {"schema": SCHEMA, "source": src,
           "splits": {s["split"]: {k: v for k, v in s.items() if k not in ("split", "jsonl")}
                      for s in stats}}
    p = os.path.join(out, "MANIFEST.json")
    with open(p, "w") as f:
        json.dump(man, f, indent=1)
    return p


# ---------------------------------------------------------------------------------------------
# Automated spot check: MFA last-word end vs energy-based speech offset (not human listening)


def energy_offset(pcm, sr: int = 16000, win_s: float = 0.025, hop_s: float = 0.010,
                  rel_db: float = -35.0, floor_db: float | None = 12.0) -> tuple[float, float]:
    """(onset_s, offset_s) of speech: first/last frame whose RMS exceeds the threshold.

    Threshold = max(p99 frame dB + rel_db, p10 frame dB + floor_db); floor_db=None uses the
    p99-relative term alone (fails on recordings whose noise floor is within 35 dB of the peak).
    """
    import numpy as np

    x = pcm.astype(np.float64) / 32768.0
    win, hop = int(win_s * sr), int(hop_s * sr)
    n = max(1, 1 + (len(x) - win) // hop)
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(rms)
    thr = np.percentile(db, 99) + rel_db
    if floor_db is not None:
        thr = max(thr, np.percentile(db, 10) + floor_db)
    idx = np.nonzero(db > thr)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    return idx[0] * hop / sr, (idx[-1] * hop + win) / sr


def spotcheck(split: str = "test-clean", n: int = 20, seed: int = 20261120,
              floor_db: float | None = 12.0) -> dict:
    import random

    import numpy as np

    from ultra import libri

    timing = load(split)
    utts = {u.uid: u for u in libri.load_split(split)}
    ids = sorted(timing)
    pick = random.Random(seed).sample(ids, n)
    rows = []
    for uid in pick:
        on, off = energy_offset(utts[uid].pcm, floor_db=floor_db)
        w = timing[uid]
        rows.append({"id": uid, "mfa_first_start": w[0].start_s, "rms_onset": round(on, 3),
                     "mfa_last_end": w[-1].end_s, "rms_offset": round(off, 3),
                     "d_end_ms": round(1000 * (w[-1].end_s - off), 1),
                     "d_start_ms": round(1000 * (w[0].start_s - on), 1),
                     "audio_s": round(utts[uid].seconds, 3)})
    d_end = np.abs([r["d_end_ms"] for r in rows])
    d_start = np.abs([r["d_start_ms"] for r in rows])
    return {"split": split, "n": n, "seed": seed, "method": "RMS 25/10 ms, thr=" + ("p99-35 dB" if floor_db is None
                                           else f"max(p99-35 dB, p10+{floor_db} dB)"),
            "median_abs_end_ms": float(np.median(d_end)), "max_abs_end_ms": float(d_end.max()),
            "median_abs_start_ms": float(np.median(d_start)), "rows": rows}


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        print(json.dumps(fetch(), indent=1))
    elif cmd == "build":
        stats = [build(s) for s in SPLITS]
        p = write_manifest(stats)
        for s in stats:
            print(json.dumps({k: v for k, v in s.items() if k != "unmatched"}),
                  "unmatched:", s["unmatched"][:10])
        print("manifest", p, sha256_file(p))
    elif cmd == "spotcheck":
        print(json.dumps({"noise_floor_aware": spotcheck(),
                          "p99_relative_only": spotcheck(floor_db=None)}, indent=1))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
