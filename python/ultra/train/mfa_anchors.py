"""H48c: MFA word-end anchors for training, on the 25 Hz encoder grid, per speed (owner-approved
2026-11-23: MFA alignments in training; results are labelled "MFA-anchored").

Source: the Montreal-Forced-Aligner LibriSpeech alignments of Lugosch et al. (Zenodo record 2619474,
DOI 10.5281/zenodo.2619474, v1.0, CC BY 4.0), the same release as the dev timings of
`ultra.word_timing`. Only the TRAIN splits are read here (allowlisted member names); dev/test
members are never opened by this module, and dev timings are for evaluation only.

Grid. Log-mel frame j covers samples [160 j, 160 j + 400); the causal stem makes encoder frame t
depend on log-mel frames <= 4 t (train.model.CausalStem), so its window ends at 40 t + 25 ms. The
evidence frame of a time tau (ms) is the first encoder frame whose window reaches tau:

    frame(tau) = max(0, ceil((tau - 25) / 40))

Speed perturbation (train.data: resample_poly(pcm, den, num) with Fraction(f).limit_denominator(100)
= num / den) maps tau to tau * den / num; frames are computed exactly in integers per speed.

Files (data drive, never git):
  mfa_anchors_<name>_v1.npz  `ultra.mfa-anchors/v1`: per utterance (sorted ids) the word start / end
      times in ms and their evidence frames at each speed, plus a transcript hash and the audio
      length MFA saw. This is all the trainer loads.
  mfa_phones_<name>_v1.npz   `ultra.mfa-phones/v1`: the phone tier (inventory ids, start / end ms,
      end frames per speed, word index), for finer anchors later.
  MANIFEST.json              coverage, unmatched ids, sha256 of both files and of the source zip.

CLI (from python/ultra):  python -m train.mfa_anchors build --zip <librispeech_alignments.zip> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from fractions import Fraction

import numpy as np

from train.errors import DataError

SCHEMA = "ultra.mfa-anchors/v1"
PHONE_SCHEMA = "ultra.mfa-phones/v1"
SPEEDS = (0.9, 1.0, 1.1)
TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
FRAME_MS = 40  # 25 Hz encoder frames
STEM_RIGHT_MS = 25  # encoder frame t's stem window ends at 40 t + 25 ms
FRAME_RULE = "frame = max(0, ceil((tau_ms * den / num - 25) / 40)), speed = num / den"
_SILENCE = {"", "sil", "sp", "<eps>"}
_TRAIN_MEMBER = re.compile(r"(train-clean-100|train-clean-360|train-other-500)/"
                           r"(\d+/\d+/(\d+-\d+-\d+)\.TextGrid|unaligned\.txt)")


# ---------------------------------------------------------------------------------------------
# grid


def speed_ratio(speed: float) -> tuple[int, int]:
    """(num, den) of the loader's resampling: Fraction(speed).limit_denominator(100)."""
    r = Fraction(speed).limit_denominator(100)
    return r.numerator, r.denominator


def evidence_frame(t_ms, speed: float = 1.0):
    """First 25 Hz encoder frame whose causal stem window reaches time t_ms (ms, original audio)
    of the audio played at `speed`; integer arithmetic, vectorised over numpy arrays."""
    num, den = speed_ratio(speed)
    x = np.asarray(t_ms, dtype=np.int64) * den - STEM_RIGHT_MS * num
    return np.maximum(0, -((-x) // (FRAME_MS * num)))


def text_hash(text: str) -> int:
    """64-bit transcript key: first 8 bytes (little endian) of sha256 of the upper-case words."""
    h = hashlib.sha256(" ".join(text.upper().split()).encode()).digest()
    return int.from_bytes(h[:8], "little")


# ---------------------------------------------------------------------------------------------
# token -> word mapping (SentencePiece: a piece starting with U+2581 opens a word)


def word_start_table(tokenizer) -> np.ndarray:
    """Bool over model output indices (blank 0, piece p at p + 1): True where the piece opens a
    word. Works with train.tokenizer.Tokenizer (its `.sp`) or a raw SentencePieceProcessor."""
    sp = getattr(tokenizer, "sp", tokenizer)
    n = sp.get_piece_size()
    t = np.zeros(n + 1, dtype=bool)
    for p in range(n):
        t[p + 1] = sp.id_to_piece(p).startswith("▁")
    return t


def token_word_index(ids, starts: np.ndarray) -> np.ndarray | None:
    """Word index of each token (0-based), or None if the first token does not open a word."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    s = starts[ids]
    if not s[0]:
        return None
    return np.cumsum(s) - 1


# ---------------------------------------------------------------------------------------------
# reader


class MfaAnchors:
    """Word-end anchors keyed by (utterance id, speed). Unknown ids give None (no anchors; the
    loss then treats the utterance as unanchored); an unknown speed or a transcript that differs
    from the one the file was built from raises DataError."""

    def __init__(self, path: str, tokenizer=None):
        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != SCHEMA:
                raise DataError(f"{path}: schema {z['schema']} != {SCHEMA}")
            self.speeds = tuple(float(s) for s in z["speeds"])
            uids = z["uids"]
            self.text_hash = z["text_hash"]
            self.word_offsets = z["word_offsets"]
            self.word_end_frame = z["word_end_frame"]
            self.word_end_ms = z["word_end_ms"]
        self.index = {u.decode(): i for i, u in enumerate(uids)}
        if len(self.index) != len(uids):
            raise DataError(f"{path}: duplicate utterance ids")
        self.path = path
        self.sha256 = sha256_file(path)  # provenance: printed by the trainer at start
        self.starts = word_start_table(tokenizer) if tokenizer is not None else None
        self.counts = {"anchored": 0, "missing": 0, "token_mismatch": 0}

    def __len__(self) -> int:
        return len(self.index)

    def _col(self, speed: float) -> int:
        for k, s in enumerate(self.speeds):
            if abs(s - float(speed)) < 1e-9:
                return k
        raise DataError(f"no MFA anchors for speed {speed!r}; file has {self.speeds}")

    def word_ends(self, uid: str, speed: float = 1.0) -> np.ndarray | None:
        """Evidence frames of the word ends of `uid` at `speed`, or None if not aligned."""
        k = self._col(speed)
        i = self.index.get(uid)
        if i is None:
            return None
        return self.word_end_frame[self.word_offsets[i]:self.word_offsets[i + 1], k]

    def token_anchors(self, uid: str, speed: float, rnnt_ids, text: str) -> np.ndarray | None:
        """Per-token anchor frame (the end of the token's word) or None (counted)."""
        if self.starts is None:
            raise DataError("MfaAnchors.token_anchors needs the RNN-T tokenizer")
        ends = self.word_ends(uid, speed)
        if ends is None:
            self.counts["missing"] += 1
            return None
        if int(self.text_hash[self.index[uid]]) != text_hash(text):
            raise DataError(f"{uid}: transcript differs from the one the MFA anchors were built from")
        words = token_word_index(rnnt_ids, self.starts)
        if words is None or len(ends) != len(text.split()) or (
                words.size and int(words[-1]) + 1 != len(ends)):
            self.counts["token_mismatch"] += 1
            return None
        self.counts["anchored"] += 1
        return ends[words].astype(np.int32)


# ---------------------------------------------------------------------------------------------
# builder (Zenodo zip -> side files); train splits only


_TIER = r'name = "{}"(.*?)(?:item \[\d+\]:|\Z)'
_INTERVAL = re.compile(r'xmin = ([0-9.eE+-]+)\s+xmax = ([0-9.eE+-]+)\s+text = "((?:[^"]|"")*)"')
_XMAX = re.compile(r"^xmax = ([0-9.eE+-]+)", re.M)


def parse_tier(text: str, name: str, keep_silence: bool = False) -> list[tuple[str, float, float]]:
    m = re.search(_TIER.format(name), text, re.S)
    if not m:
        raise ValueError(f"TextGrid has no {name!r} tier")
    out = []
    for a, b, t in _INTERVAL.findall(m.group(1)):
        t = t.replace('""', '"').strip()
        if not keep_silence and t.lower() in _SILENCE:
            continue
        out.append((t, float(a), float(b)))
    return out


def _ms(x: float) -> int:
    return int(round(x * 1000.0))


def convert(tg_text: str, ref_text: str):
    """One utterance: (audio_ms, words [(start_ms, end_ms)], phones [(phone, start_ms, end_ms,
    word_index)]) or None when MFA's word tokens cannot be mapped onto the reference words."""
    from ultra.word_timing import attach_reference

    ref = ref_text.upper().split()
    words = attach_reference(parse_tier(tg_text, "words"), ref)
    if words is None:
        return None
    m = _XMAX.search(tg_text)
    audio_ms = _ms(float(m.group(1))) if m else 0
    wt = [(_ms(w.start_s), _ms(w.end_s)) for w in words]
    phones, k = [], 0
    for p, a, b in parse_tier(tg_text, "phones"):
        a_ms, b_ms = _ms(a), _ms(b)
        while k < len(wt) and a_ms >= wt[k][1]:
            k += 1
        wi = k if (k < len(wt) and wt[k][0] <= a_ms and b_ms <= wt[k][1]) else -1
        phones.append((p, a_ms, b_ms, wi))
    return audio_ms, wt, phones


def _transcripts(splits) -> dict[str, tuple[str, str]]:
    """uid -> (LibriSpeech transcript, split) from the parquet mirror (id and text columns only)."""
    import glob

    import pyarrow.parquet as pq

    from ultra.libri import PARQUET, SPLIT_DIRS

    out = {}
    for s in splits:
        files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[s], "*.parquet")))
        if not files:
            raise DataError(f"no parquet shards for {s}")
        for f in files:
            t = pq.read_table(f, columns=["id", "text"])
            for uid, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True):
                if uid in out:
                    raise DataError(f"duplicate id {uid}")
                out[uid] = (text, s)
    return out


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(zip_path: str, out_dir: str, splits=TRAIN_SPLITS, name: str = "train960",
          transcripts: dict[str, tuple[str, str]] | None = None, progress: bool = False) -> dict:
    """Read the train TextGrids straight from the zip (no extraction) and write the side files.
    `transcripts` (uid -> (text, split)) defaults to the parquet mirror's train transcripts."""
    for s in splits:
        if s not in TRAIN_SPLITS:
            raise DataError(f"{s}: MFA anchors are built for training splits only "
                            "(dev is for evaluation, test is sealed)")
    refs = transcripts if transcripts is not None else _transcripts(splits)
    stats = {s: {"matched": 0, "timed_words": 0, "unaligned_listed": 0, "token_mismatch": [],
                 "no_transcript": [], "overrun": 0} for s in splits}
    rows = {}
    with open(zip_path, "rb") as f:  # one sequential read: 281k small members seek badly on a disk
        blob = io.BytesIO(f.read())
    with zipfile.ZipFile(blob) as z:
        members = {}
        for info in z.infolist():
            m = _TRAIN_MEMBER.fullmatch(info.filename)
            if not m or m.group(1) not in splits:
                continue  # dev / test / anything else: never opened
            if m.group(3):
                members[m.group(3)] = (m.group(1), info)
            else:
                listed = z.read(info).decode("utf-8", "replace").split("\n")
                stats[m.group(1)]["unaligned_listed"] = sum(1 for x in listed if x.strip())
        order = sorted(members.items(), key=lambda kv: kv[1][1].header_offset)
        for n, (uid, (split, info)) in enumerate(order):
            if progress and n % 20000 == 0:
                print(json.dumps({"kind": "progress", "done": n, "of": len(order)}), flush=True)
            ref = refs.get(uid)
            if ref is None:
                stats[split]["no_transcript"].append(uid)
                continue
            if ref[1] != split:
                raise DataError(f"{uid}: TextGrid under {split}, transcript under {ref[1]}")
            r = convert(z.read(info).decode("utf-8"), ref[0])  # zipfile checks CRC-32
            if r is None:
                stats[split]["token_mismatch"].append(uid)
                continue
            rows[uid] = (r, split)
    return _write(rows, refs, stats, splits, out_dir, name, zip_path)


def _write(rows, refs, stats, splits, out_dir, name, zip_path) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    uids = sorted(rows)
    n_words = sum(len(rows[u][0][1]) for u in uids)
    n_ph = sum(len(rows[u][0][2]) for u in uids)
    w_off = np.zeros(len(uids) + 1, np.int64)
    p_off = np.zeros(len(uids) + 1, np.int64)
    w_ms = np.zeros((n_words, 2), np.int64)
    p_ms = np.zeros((n_ph, 2), np.int64)
    p_word = np.zeros(n_ph, np.int32)
    p_name: list[str] = []
    audio_ms = np.zeros(len(uids), np.int64)
    hashes = np.zeros(len(uids), np.uint64)
    wi = pi = 0
    for i, u in enumerate(uids):
        (a_ms, wt, ph), split = rows[u]
        st = stats[split]
        st["matched"] += 1
        st["timed_words"] += len(wt)
        audio_ms[i] = a_ms
        hashes[i] = text_hash(refs[u][0])
        if wt and wt[-1][1] > a_ms + 1:
            st["overrun"] += 1
        w_ms[wi:wi + len(wt)] = wt
        for p, a, b, k in ph:
            p_ms[pi] = (a, b)
            p_word[pi] = k
            p_name.append(p)
            pi += 1
        wi += len(wt)
        w_off[i + 1], p_off[i + 1] = wi, pi
    if len(w_ms) and (w_ms.min() < 0 or w_ms.max() >= 65535 or p_ms.max() >= 65535):
        raise DataError("times outside uint16 ms")
    inventory = sorted(set(p_name))
    pid = {p: k for k, p in enumerate(inventory)}
    if len(inventory) > 255:
        raise DataError("phone inventory exceeds uint8")
    wsf = np.stack([evidence_frame(w_ms[:, 0], s) for s in SPEEDS], 1).astype(np.uint16)
    wef = np.stack([evidence_frame(w_ms[:, 1], s) for s in SPEEDS], 1).astype(np.uint16)
    pef = np.stack([evidence_frame(p_ms[:, 1], s) for s in SPEEDS], 1).astype(np.uint16)
    speeds = np.array(SPEEDS, np.float64)
    ratios = np.array([speed_ratio(s) for s in SPEEDS], np.int64)
    uid_arr = np.array([u.encode() for u in uids])
    a_path = os.path.join(out_dir, f"mfa_anchors_{name}_v1.npz")
    p_path = os.path.join(out_dir, f"mfa_phones_{name}_v1.npz")
    for path, arrays in (
            (a_path, dict(schema=np.array(SCHEMA), speeds=speeds, speed_num_den=ratios,
                          frame_rule=np.array(FRAME_RULE), uids=uid_arr, text_hash=hashes,
                          audio_ms=audio_ms.astype(np.uint32), word_offsets=w_off,
                          word_start_ms=w_ms[:, 0].astype(np.uint16),
                          word_end_ms=w_ms[:, 1].astype(np.uint16),
                          word_start_frame=wsf, word_end_frame=wef)),
            (p_path, dict(schema=np.array(PHONE_SCHEMA), speeds=speeds, speed_num_den=ratios,
                          frame_rule=np.array(FRAME_RULE), uids=uid_arr, phone_offsets=p_off,
                          phone_inventory=np.array(inventory),
                          phone_id=np.array([pid[p] for p in p_name], np.uint8),
                          phone_start_ms=p_ms[:, 0].astype(np.uint16),
                          phone_end_ms=p_ms[:, 1].astype(np.uint16),
                          phone_word=p_word, phone_end_frame=pef))):
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    man = {"schema": SCHEMA, "phone_schema": PHONE_SCHEMA, "name": name, "speeds": list(SPEEDS),
           "frame_rule": FRAME_RULE,
           "source": {"zenodo_record": "2619474", "doi": "10.5281/zenodo.2619474", "version": "1.0",
                      "licence": "CC BY 4.0", "zip": os.path.basename(zip_path),
                      "zip_sha256": sha256_file(zip_path)},
           "splits": {}, "files": {}}
    for s in splits:
        st = stats[s]
        utts = [u for u, (_, sp) in refs.items() if sp == s]
        words = sum(len(refs[u][0].split()) for u in utts)
        no_tg = sorted(u for u in utts if u not in rows and u not in set(st["token_mismatch"]))
        man["splits"][s] = {
            "utterances": len(utts), "matched": st["matched"],
            "pct_utts_matched": round(100.0 * st["matched"] / max(1, len(utts)), 3),
            "ref_words": words, "timed_words": st["timed_words"],
            "pct_words_timed": round(100.0 * st["timed_words"] / max(1, words), 3),
            "unaligned_listed": st["unaligned_listed"], "no_textgrid": len(no_tg),
            "token_mismatch": len(st["token_mismatch"]),
            "textgrid_without_transcript": len(st["no_transcript"]),
            "last_word_after_audio_end": st["overrun"],
            "no_textgrid_ids": no_tg, "token_mismatch_ids": sorted(st["token_mismatch"])}
    for path in (a_path, p_path):
        man["files"][os.path.basename(path)] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--zip", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--splits", nargs="+", default=list(TRAIN_SPLITS))
    b.add_argument("--name", default="train960")
    a = ap.parse_args(argv)
    man = build(a.zip, a.out, tuple(a.splits), a.name, progress=True)
    print(json.dumps({s: {k: v for k, v in d.items() if not k.endswith("_ids")}
                      for s, d in man["splits"].items()}, indent=1))
    print(json.dumps(man["files"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
