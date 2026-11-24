"""Sealed transcript override for extra (pseudo-labelled) splits, default OFF.

A label map is a UTF-8 TSV of `uid<TAB>text` lines, pinned by its sha256. With it, a row of an extra
split whose `id` is in the map trains on the map's text instead of the parquet `text` column. With
`only`, extra-split rows that are not in the map are dropped before packing (LibriSpeech rows are
never touched). So two arms can share the exact same audio rows, packing and drops and differ only
in their transcripts (H53 Amendment 12, PL3 vs incumbent labels) without a second copy of the audio.

TrainConfig keys: `label_map` (path, "" = off), `label_map_sha256` (required with a path) and
`label_map_only` (0/1).
"""
from __future__ import annotations

import hashlib

from train.errors import DataError


class LabelMap:
    def __init__(self, texts: dict[str, str], only: bool, sha256: str):
        self.texts, self.only, self.sha256 = texts, bool(only), sha256

    @classmethod
    def load(cls, path: str, sha256: str, only: bool = False) -> LabelMap:
        if not sha256:
            raise DataError("label_map needs label_map_sha256")
        with open(path, "rb") as f:
            raw = f.read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha256:
            raise DataError(f"label_map {path}: sha256 {got} != declared {sha256}")
        texts: dict[str, str] = {}
        for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            uid, sep, text = line.partition("\t")
            if not sep or not uid or not text.strip() or "\t" in text:
                raise DataError(f"label_map {path}:{n}: expected 'uid<TAB>text' with non-empty text")
            if uid in texts:
                raise DataError(f"label_map {path}:{n}: duplicate uid {uid!r}")
            texts[uid] = text
        return cls(texts, only, got)

    def keep(self, uid: str) -> bool:
        return not self.only or uid in self.texts

    def apply(self, row: dict) -> dict:
        t = self.texts.get(row["id"])
        return row if t is None else {**row, "text": t}


def label_map_from_cfg(cfg) -> LabelMap | None:
    path = getattr(cfg, "label_map", "")
    if not path:
        return None
    return LabelMap.load(path, getattr(cfg, "label_map_sha256", ""), bool(getattr(cfg, "label_map_only", 0)))
