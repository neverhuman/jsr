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
from train.transducer import (BLANK, Joiner, StatelessPredictor, greedy_decode, joiner_logprobs,
                              rnnt_nll)


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vocab_rnnt: int = 4097  # 4,096 pieces + blank
    vocab_ctc: int = 1025  # 1,024 pieces + blank
    pred_emb: int = 256
    joiner: int = 320
    joiner_act: str = "tanh"
    ctc_weight: float = 0.3
    qat: str = "none"  # "ternary": train.qat ternary weights + per-row int8 inputs
    qat_stem: str = "none"  # "int8": stem conv 2 + projection trained as the engine's int8 GEMMs
    # RNN-T training loss: "exact" (full lattice, train.transducer) or "pruned" (k2, H13b): a
    # training-only simple joiner (linear heads on the encoder and predictor outputs) gives a cheap
    # smoothed loss whose occupation gradients choose, per frame, a band of `prune_range` label
    # positions; the real joiner is evaluated only inside that band. Inference is unchanged.
    rnnt_loss: str = "exact"
    prune_range: int = 5
    simple_loss_scale: float = 0.5
    lm_only_scale: float = 0.25
    am_only_scale: float = 0.0

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
    # MULTIRATE S (spec v1.0-B §17.3): 4 blocks at 25 Hz, 10 at 12.5 Hz, 2 restored at 25 Hz.
    "S_MR": ModelConfig(encoder=EncoderConfig(post_block_norm=False, family="multirate",
                                              lower_blocks=4, middle_blocks=10, final_blocks=2)),
    # R4 pilot: S_MR with ternary QAT (train.qat).
    "S_MR_T": ModelConfig(encoder=EncoderConfig(post_block_norm=False, family="multirate",
                                                lower_blocks=4, middle_blocks=10, final_blocks=2),
                          qat="ternary"),
    # D3MR: S_MR with the stem trained as int8 (engine-lane: post-training int8 stem fails).
    "S_MR_I8S": ModelConfig(encoder=EncoderConfig(post_block_norm=False, family="multirate",
                                                  lower_blocks=4, middle_blocks=10,
                                                  final_blocks=2), qat_stem="int8"),
    # D4M: M-width multirate (d 512, 8 heads x 64, same 4/10/2 layout as S_MR), ~102M params.
    "M_MR": ModelConfig(encoder=EncoderConfig(d_model=512, n_heads=8, post_block_norm=False,
                                              family="multirate", lower_blocks=4,
                                              middle_blocks=10, final_blocks=2)),
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
        if cfg.rnnt_loss == "pruned":
            if cfg.prune_range < 2:
                raise ValueError("prune_range must be >= 2")
            self.simple_am = nn.Linear(cfg.encoder.d_model, cfg.vocab_rnnt)
            self.simple_lm = nn.Linear(cfg.joiner, cfg.vocab_rnnt)
        elif cfg.rnnt_loss != "exact":
            raise ValueError(f"unknown rnnt_loss {cfg.rnnt_loss}")
        if cfg.qat == "ternary":
            from train.qat import apply_ternary

            self.ternary_layers = apply_ternary(self)
        elif cfg.qat != "none":
            raise ValueError(f"unknown qat mode {cfg.qat}")
        if cfg.qat_stem == "int8":
            from train.qat import apply_int8_stem

            self.int8_stem_layers = apply_int8_stem(self)
        elif cfg.qat_stem != "none":
            raise ValueError(f"unknown qat_stem mode {cfg.qat_stem}")

    def encode(self, feats, feat_lens, chunk=None):
        return self.encoder(self.cmvn(feats), feat_lens, chunk)

    def losses(self, feats, feat_lens, rnnt_tgt, rnnt_lens, ctc_tgt, ctc_lens,
               stats: FeasibilityStats | None = None, augment: SpecAugConfig | None = None,
               gen: torch.Generator | None = None, chunk: int | None = None,
               pruned_scale: float = 1.0, kd: tuple | None = None, kd_weight: float = 0.0,
               simple_detach: bool = False, zloss_weight: float = 0.0) -> dict:
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
            zl = None
            if self.cfg.rnnt_loss == "pruned":
                nll, simple, zl = self._pruned_rnnt(enc, enc_p, pred_p, rnnt_tgt, enc_lens,
                                                    rnnt_lens, simple_detach,
                                                    zloss=zloss_weight > 0)
            else:
                lpb, lpl = joiner_logprobs(self.joiner, enc_p, pred_p, rnnt_tgt)
                nll, simple = rnnt_nll(lpb, lpl, enc_lens, rnnt_lens), None
                if zloss_weight > 0:  # H23: train/optim/zloss.py
                    from train.optim.zloss import diagonal_zloss

                    zl = diagonal_zloss(self.joiner, enc_p, pred_p, enc_lens, rnnt_lens)
        rnnt_tokens = int(rnnt_lens.sum())
        rnnt = nll.sum() / max(1, rnnt_tokens)
        ctc = ctc_sum / max(1, ctc_tokens)
        total = self.cfg.ctc_weight * ctc
        out = {"rnnt": rnnt, "ctc": ctc, "rnnt_tokens": rnnt_tokens,
               "ctc_tokens": ctc_tokens, "ctc_masked": int((~ok).sum()), "nll": nll.detach()}
        if kd is not None and kd_weight > 0:
            from train.kd import kd_loss

            with torch.autocast(device_type=enc.device.type, enabled=False):
                kl, n = kd_loss(self.joiner, enc_p, pred_p, enc_lens, rnnt_lens, *kd)
            total = total + kd_weight * kl
            out["kd"], out["kd_frames"] = kl, n
        if simple is None:
            total = total + rnnt
        else:
            rnnt_simple = simple.sum() / max(1, rnnt_tokens)
            total = total + pruned_scale * rnnt + self.cfg.simple_loss_scale * rnnt_simple
            out["rnnt_simple"] = rnnt_simple
        if zl is not None:
            total = total + zloss_weight * zl
            out["zloss"] = zl  # live: the DDP path rebuilds the loss from these parts
        out["loss"] = total
        return out

    def _pruned_rnnt(self, enc, enc_p, pred_p, tgt, enc_lens, tgt_lens, simple_detach=False,
                     zloss: bool = False):
        """(pruned per-sequence NLL, simple-joiner smoothed per-sequence loss, band z-loss or
        None) via k2."""
        try:
            import k2
        except ImportError as e:  # the exact loss needs no extra package
            raise ImportError("rnnt_loss='pruned' needs k2 (a build matching this torch); "
                              "see research/H12_accuracy_diagnostics") from e
        c = self.cfg
        bd = torch.zeros(tgt.shape[0], 4, dtype=torch.int64, device=enc.device)
        bd[:, 2], bd[:, 3] = tgt_lens, enc_lens
        symbols = tgt.long()
        simple, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=self.simple_lm(pred_p.detach() if simple_detach else pred_p).float(),
            am=self.simple_am(enc.detach() if simple_detach else enc).float(), symbols=symbols,
            termination_symbol=BLANK, lm_only_scale=c.lm_only_scale, am_only_scale=c.am_only_scale,
            boundary=bd, rnnt_type="regular", reduction="none", return_grad=True)
        ranges = k2.get_rnnt_prune_ranges(px_grad=px_grad, py_grad=py_grad, boundary=bd,
                                          s_range=min(c.prune_range, symbols.shape[1] + 1))
        am, lm = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
        logits = self.joiner.logits(am, lm).float()
        nll = k2.rnnt_loss_pruned(logits=logits, symbols=symbols, ranges=ranges,
                                  termination_symbol=BLANK, boundary=bd, rnnt_type="regular",
                                  reduction="none")
        if not zloss:
            return nll, simple, None
        from train.optim.zloss import band_zloss

        return nll, simple, band_zloss(logits, enc_lens)

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
