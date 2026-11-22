"""SentencePiece BPE tokenizers trained on LibriSpeech *training* transcripts only.

Two vocabularies (spec v1.0-B §17.3, master spec §4.2):
  * 1,024 lower-case pieces for the dense auxiliary CTC head;
  * 4,096 lower-case pieces for the transducer.
LibriSpeech references carry no case or punctuation, so both are style "plain" (master spec §4.6).

Model output index 0 is blank; SentencePiece id p maps to output index p + 1.
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pyarrow.parquet as pq
import sentencepiece as spm

from train.errors import DataError

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
TOKDIR = os.path.join(DATA, "tokenizers")
TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
BLANK = 0


def tokenizer_path(vocab: int) -> str:
    return os.path.join(TOKDIR, f"librispeech_train_bpe{vocab}.model")


def iter_train_text(splits=TRAIN_SPLITS):
    import glob

    from ultra.libri import PARQUET, SPLIT_DIRS

    for split in splits:
        files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))
        if not files:
            raise DataError(f"no parquet shards for {split}")
        for f in files:
            for t in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                yield t.lower()


def train_tokenizers(vocabs=(1024, 4096), splits=TRAIN_SPLITS) -> dict[int, str]:
    os.makedirs(TOKDIR, exist_ok=True)
    corpus = os.path.join(TOKDIR, "librispeech_train_text.txt")
    if not os.path.exists(corpus):
        tmp = corpus + ".tmp"
        with open(tmp, "w") as fh:
            for line in iter_train_text(splits):
                fh.write(line + "\n")
        os.replace(tmp, corpus)
    out = {}
    for v in vocabs:
        prefix = tokenizer_path(v)[: -len(".model")]
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=prefix, vocab_size=v, model_type="bpe",
            character_coverage=1.0, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
            input_sentence_size=300000, shuffle_input_sentence=True, num_threads=8)
        out[v] = prefix + ".model"
    return out


class Tokenizer:
    """Wraps a SentencePiece model; `encode` returns output indices (blank-shifted)."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise DataError(f"tokenizer model missing: {path} (run python -m train.tokenizer)")
        self.path = path
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.vocab = self.sp.get_piece_size()
        # Case of the model's pieces: ours are trained on lower-cased text, R0's (icefall) bpe.model
        # on upper case. Text is folded to the model's case before encoding.
        letters = "".join(self.sp.id_to_piece(i) for i in range(self.vocab) if not self.sp.is_control(i))
        upper = sum(c.isupper() for c in letters)
        self.upper = upper > sum(c.islower() for c in letters)
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()

    @property
    def num_outputs(self) -> int:
        return self.vocab + 1

    def encode(self, text: str) -> list[int]:
        return [p + 1 for p in self.sp.encode(text.upper() if self.upper else text.lower())]

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i - 1 for i in ids if i != BLANK])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, nargs="+", default=[1024, 4096])
    a = ap.parse_args()
    for v, p in train_tokenizers(tuple(a.vocab)).items():
        t = Tokenizer(p)
        print(v, p, t.sha256, t.sp.encode("hello world", out_type=str))


if __name__ == "__main__":
    main()
