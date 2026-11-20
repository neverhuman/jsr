"""LibriSpeech access from the HF parquet mirror (openslr/librispeech_asr).

Decoded evaluation sets are cached as .npz (int16 PCM concatenated + offsets) on the data drive.
"""
from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
PARQUET = os.path.join(DATA, "corpora", "librispeech", "hf_parquet", "all")
CACHE = os.path.join(DATA, "cache", "librispeech_pcm")
SPLIT_DIRS = {"dev-clean": "validation.clean", "dev-other": "validation.other",
              "test-clean": "test.clean", "test-other": "test.other",
              "train-clean-100": "train.clean.100", "train-clean-360": "train.clean.360",
              "train-other-500": "train.other.500"}


@dataclass
class Utt:
    uid: str
    speaker: int
    text: str
    pcm: np.ndarray  # int16, 16 kHz mono

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 16000.0


def iter_parquet(split: str):
    """Yield (uid, speaker, text, flac_bytes) in a stable order (file, then row)."""
    files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards for {split} under {PARQUET}")
    for f in files:
        table = pq.read_table(f, columns=["id", "speaker_id", "text", "audio"])
        for row in table.to_pylist():
            yield row["id"], int(row["speaker_id"]), row["text"], row["audio"]["bytes"]


def decode_flac(b: bytes) -> np.ndarray:
    pcm, sr = sf.read(io.BytesIO(b), dtype="int16")
    if sr != 16000 or pcm.ndim != 1:
        raise ValueError(f"expected 16 kHz mono, got sr={sr} shape={pcm.shape}")
    return pcm


def load_split(split: str, limit: int | None = None) -> list[Utt]:
    """Decoded utterances of a split, cached as one .npz per split."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{split}.npz")
    if limit is not None and not os.path.exists(path):
        # Partial read without caching (screens on large splits).
        out = []
        for uid, s, t, b in iter_parquet(split):
            if len(out) >= limit:
                break
            out.append(Utt(uid, s, t, decode_flac(b)))
        return out
    if not os.path.exists(path):
        uids, spk, texts, pcms = [], [], [], []
        for uid, s, t, b in iter_parquet(split):
            uids.append(uid), spk.append(s), texts.append(t), pcms.append(decode_flac(b))
        offs = np.cumsum([0] + [len(p) for p in pcms]).astype(np.int64)
        tmp = path + ".tmp.npz"
        np.savez(tmp, pcm=np.concatenate(pcms), offsets=offs, uids=np.array(uids),
                 speakers=np.array(spk), texts=np.array(texts))
        os.replace(tmp, path)
    z = np.load(path)
    offs, pcm = z["offsets"], z["pcm"]
    n = len(z["uids"]) if limit is None else min(limit, len(z["uids"]))
    return [Utt(str(z["uids"][i]), int(z["speakers"][i]), str(z["texts"][i]), pcm[offs[i]:offs[i + 1]])
            for i in range(n)]
