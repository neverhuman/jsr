"""H46 item 3: `train.lm_decode beam` with neural token LMs as well as KenLM n-grams.

  python -m train.h46_nlm_decode beam --cache ENC --checkpoint CKPT --configs grid.json \
      --shard 3/14 --out DIR --fast

Identical to `python -m train.lm_decode beam` (same search, same outputs, same per-utterance row
cache rule), except that a config whose `lm` is a `train.nlm` run directory (it holds model.pt)
is fused through `train.nlm.NeuralFusion`: the trained FFNN through the exporter's int8 round
trip (its QAT forward), LODR still the KenLM bigram. Everything else goes to H42's code
unchanged.
"""
from __future__ import annotations

import os
import sys

import train.lm_fusion as lm_fusion

_TokenNgram, _NgramFusion = lm_fusion.TokenNgram, lm_fusion.NgramFusion


class NeuralLM:
    """Holds a train.nlm model where train.lm_decode expects a TokenNgram."""

    def __init__(self, path: str):
        from train.nlm import load_run

        self.path = path
        self.model, self.meta = load_run(path)
        self.order = self.model.ctx + 1
        self.lookups = 0  # rows are counted by NeuralFusion.contexts

    @staticmethod
    def is_run(path: str) -> bool:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, "model.pt"))


def token_lm(path: str, vocab: int, max_rows: int = 1 << 15):
    return NeuralLM(path) if NeuralLM.is_run(path) else _TokenNgram(path, vocab, max_rows)


def fusion(lm, lm_weight: float, lodr=None, lodr_weight: float = 0.0, max_rows: int = 1 << 15):
    if isinstance(lm, NeuralLM):
        from train.nlm import NeuralFusion

        return NeuralFusion(lm.model, lm_weight, lodr, lodr_weight, max_rows)
    return _NgramFusion(lm, lm_weight, lodr, lodr_weight, max_rows)


def main(argv=None) -> None:
    lm_fusion.TokenNgram, lm_fusion.NgramFusion = token_lm, fusion
    from train.lm_decode import main as decode_main

    decode_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
