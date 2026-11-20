"""The full student: CMVN -> SpecAugment (train only) -> encoder -> CTC head + transducer.

Loss (spec v1.0-B §26.2, lambda_distill = 0):
  L = RNNT + 0.3 * CTC
Averaging convention (frozen): each term is the sum of per-sequence NLL divided by the number of
target tokens it covers (transducer tokens for RNNT; tokens of CTC-feasible sequences for CTC).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from train.ctc import FeasibilityStats, ctc_loss
from train.features import GlobalCMVN
from train.model import Encoder, EncoderConfig
from train.transducer import Joiner, StatelessPredictor, greedy_decode, joiner_logprobs, rnnt_nll


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vocab_rnnt: int = 4097  # 4,096 pieces + blank
    vocab_ctc: int = 1025  # 1,024 pieces + blank
    pred_emb: int = 256
    joiner: int = 320
    joiner_act: str = "tanh"
    ctc_weight: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> ModelConfig:
        d = dict(d)
        d["encoder"] = EncoderConfig(**d["encoder"])
        return ModelConfig(**d)


PRESETS = {
    # Spec v1.0-B §17.3 S control: width 384, 16 blocks, 6 heads x 64, FFN x4, kernel 15.
    "S": ModelConfig(encoder=EncoderConfig(post_block_norm=False)),
    # The configuration of the first D1 attempt (collapsed); kept loadable.
    "S_POSTNORM": ModelConfig(),
    # Small pipeline-check model for D0 overfitting and tests.
    "D0": ModelConfig(encoder=EncoderConfig(d_model=192, n_blocks=4, n_heads=4, stem_channels=64,
                                            dropout=0.0)),
    "TINY": ModelConfig(encoder=EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8,
                                              conv_kernel=5, chunk=2, left_context=4, dropout=0.0),
                        vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16),
}


@dataclass(frozen=True)
class SpecAugConfig:
    enabled: bool = True
    freq_masks: int = 2
    freq_width: int = 27
    time_masks_per_10s: int = 2
    time_width: int = 40


def spec_augment(x: torch.Tensor, lens: torch.Tensor, cfg: SpecAugConfig, gen: torch.Generator):
    """In-place-free masking with an explicit generator (reproducible from the step seed)."""
    if not cfg.enabled:
        return x
    x = x.clone()
    B, T, F_ = x.shape
    for b in range(B):
        n = int(lens[b])
        for _ in range(cfg.freq_masks):
            w = int(torch.randint(0, cfg.freq_width + 1, (1,), generator=gen))
            f0 = int(torch.randint(0, max(1, F_ - w), (1,), generator=gen))
            x[b, :n, f0:f0 + w] = 0.0
        for _ in range(max(1, cfg.time_masks_per_10s * n // 1000)):
            w = int(torch.randint(0, min(cfg.time_width, max(1, n // 5)) + 1, (1,), generator=gen))
            t0 = int(torch.randint(0, max(1, n - w), (1,), generator=gen))
            x[b, t0:t0 + w] = 0.0
    return x


class ASRModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.cmvn = GlobalCMVN(cfg.encoder.n_mels)
        self.encoder = Encoder(cfg.encoder)
        self.ctc_head = nn.Linear(cfg.encoder.d_model, cfg.vocab_ctc)
        self.predictor = StatelessPredictor(cfg.vocab_rnnt, cfg.pred_emb, cfg.joiner)
        self.joiner = Joiner(cfg.encoder.d_model, cfg.joiner, cfg.vocab_rnnt, cfg.joiner_act)

    def encode(self, feats, feat_lens, chunk=None):
        return self.encoder(self.cmvn(feats), feat_lens, chunk)

    def losses(self, feats, feat_lens, rnnt_tgt, rnnt_lens, ctc_tgt, ctc_lens,
               stats: FeasibilityStats | None = None, augment: SpecAugConfig | None = None,
               gen: torch.Generator | None = None, chunk: int | None = None) -> dict:
        x = self.cmvn(feats)
        if augment is not None and gen is not None:
            x = spec_augment(x, feat_lens, augment, gen)
        enc, enc_lens = self.encoder(x, feat_lens, chunk)
        enc = enc.float()
        # Heads and losses run in fp32 even under autocast.
        with torch.autocast(device_type=enc.device.type, enabled=False):
            # CTC on the dense 25 Hz grid.
            ctc_lp = torch.log_softmax(self.ctc_head(enc).float(), dim=-1)
            ctc_sum, ctc_tokens, ok = ctc_loss(ctc_lp, enc_lens, ctc_tgt, ctc_lens, stats)
            # Transducer.
            enc_p = self.joiner.enc_proj(enc).float()
            pred_p = self.predictor(StatelessPredictor.contexts(rnnt_tgt)).float()
            lpb, lpl = joiner_logprobs(self.joiner, enc_p, pred_p, rnnt_tgt)
            nll = rnnt_nll(lpb, lpl, enc_lens, rnnt_lens)
        rnnt_tokens = int(rnnt_lens.sum())
        rnnt = nll.sum() / max(1, rnnt_tokens)
        ctc = ctc_sum / max(1, ctc_tokens)
        total = rnnt + self.cfg.ctc_weight * ctc
        return {"loss": total, "rnnt": rnnt, "ctc": ctc, "rnnt_tokens": rnnt_tokens,
                "ctc_tokens": ctc_tokens, "ctc_masked": int((~ok).sum()), "nll": nll.detach()}

    @torch.no_grad()
    def greedy(self, feats, feat_lens, chunk=None, max_symbols: int = 8):
        enc, lens = self.encode(feats, feat_lens, chunk)
        enc_p = self.joiner.enc_proj(enc.float())
        return greedy_decode(self.predictor, self.joiner, enc_p, lens, max_symbols)

    @torch.no_grad()
    def ctc_greedy(self, feats, feat_lens, chunk=None) -> list[list[int]]:
        """Best-path CTC on the dense 25 Hz grid: argmax, merge repeats, drop blanks."""
        enc, lens = self.encode(feats, feat_lens, chunk)
        best = self.ctc_head(enc.float()).argmax(dim=-1)
        out = []
        for b in range(best.shape[0]):
            seq = best[b, :int(lens[b])].tolist()
            out.append([k for i, k in enumerate(seq) if k != 0 and (i == 0 or seq[i - 1] != k)])
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
