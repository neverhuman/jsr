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
    pred_ctx: int = 2  # stateless predictor context length (2 = v1 default, 4 = v2 BPE-500)
    joiner: int = 320
    joiner_act: str = "tanh"
    ctc_weight: float = 0.3
    inter_ctc: bool = False  # True: Self-Conditioned InterCTC head+feedback sized to vocab_ctc
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
        enc = dict(d["encoder"])
        if "stage_left" in enc and isinstance(enc["stage_left"], list):
            enc["stage_left"] = tuple(enc["stage_left"])
        d["encoder"] = EncoderConfig(**enc)
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
    # JSR v2 Wave H28: 0-lookahead 3-scale U-Net (25->12.5->6.25 Hz) + GQA + ReLU^2 + dwsep stem
    # + Pre-RoPE QK-RMSNorm + norm_affine=False + LayerScale + Self-Conditioned InterCTC + pred_ctx=4.
    "V2_S": ModelConfig(encoder=EncoderConfig(d_model=384, n_heads=6, kv_heads=2, ffn_act="relu2",
                                              stem_type="dwsep", qk_rmsnorm=True, norm_affine=False,
                                              layer_scale_init=0.1, stage_left=(32, 64, 64),
                                              post_block_norm=False, family="unet3",
                                              lower_blocks=2, middle_blocks=4, deep_blocks=8,
                                              up_blocks=2, final_blocks=2),
                        pred_ctx=4, inter_ctc=True),
    "V2_M": ModelConfig(encoder=EncoderConfig(d_model=512, n_heads=8, kv_heads=2, ffn_act="relu2",
                                              stem_type="dwsep", qk_rmsnorm=True, norm_affine=False,
                                              layer_scale_init=0.1, stage_left=(32, 64, 64),
                                              post_block_norm=False, family="unet3",
                                              lower_blocks=2, middle_blocks=4, deep_blocks=8,
                                              up_blocks=2, final_blocks=2),
                        pred_ctx=4, inter_ctc=True),
    "V2_L": ModelConfig(encoder=EncoderConfig(d_model=768, n_heads=12, kv_heads=2, ffn_act="relu2",
                                              stem_type="dwsep", qk_rmsnorm=True, norm_affine=False,
                                              layer_scale_init=0.1, stage_left=(32, 64, 64),
                                              post_block_norm=False, family="unet3",
                                              lower_blocks=2, middle_blocks=4, deep_blocks=10,
                                              up_blocks=2, final_blocks=2),
                        pred_ctx=4, inter_ctc=True),
    "V2_TINY": ModelConfig(encoder=EncoderConfig(d_model=32, n_heads=4, kv_heads=2, ffn_act="relu2",
                                                 stem_type="dwsep", qk_rmsnorm=True,
                                                 norm_affine=False, layer_scale_init=0.1,
                                                 stage_left=(4, 4, 4), stem_channels=8,
                                                 conv_kernel=5, chunk=4, left_context=8,
                                                 dropout=0.0, post_block_norm=False,
                                                 family="unet3", lower_blocks=1, middle_blocks=1,
                                                 deep_blocks=2, up_blocks=1, final_blocks=1),
                           vocab_rnnt=11, vocab_ctc=7, pred_emb=8, pred_ctx=4, joiner=16,
                           inter_ctc=True),
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
        import dataclasses

        self.cfg = cfg
        self.cmvn = GlobalCMVN(cfg.encoder.n_mels)
        enc_cfg = (dataclasses.replace(cfg.encoder, inter_ctc_vocab=cfg.vocab_ctc)
                   if (cfg.inter_ctc and cfg.encoder.inter_ctc_vocab == 0) else cfg.encoder)
        self.encoder = Encoder(enc_cfg)
        self.ctc_head = nn.Linear(cfg.encoder.d_model, cfg.vocab_ctc)
        self.predictor = StatelessPredictor(cfg.vocab_rnnt, cfg.pred_emb, cfg.joiner,
                                            context_len=cfg.pred_ctx)
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
               simple_detach: bool = False, zloss_weight: float = 0.0,
               cr_ctc_weight: float = 0.0, delay_penalty: float = 0.0,
               inter_ctc_weight: float = 0.0, balancer_weight: float = 0.0,
               view: torch.Tensor | None = None, cr_target: torch.Tensor | None = None,
               ctx: torch.Tensor | None = None) -> dict:
        """`view` / `cr_target` serve the full-form CR-CTC step (trainer.cr_ctc_full), which runs one
        view per call: `view` is that view's normalised, already augmented input (no augmentation
        here), `cr_target` the other view's detached CTC log-posteriors; the result then carries
        this view's own `ctc_lp`. Both None: the usual single call, unchanged."""
        x_norm = self.cmvn(feats)
        if view is not None:
            x = view
        else:
            x = spec_augment(x_norm, feat_lens, augment, gen) if (augment is not None and gen is not None) else x_norm
        need_aux = (self.cfg.inter_ctc and inter_ctc_weight > 0) or (balancer_weight > 0)
        if need_aux:
            enc, enc_lens, aux = self.encoder(x, feat_lens, chunk, return_aux=True)
        else:
            enc, enc_lens = self.encoder(x, feat_lens, chunk)
            aux = {}
        enc = enc.float()
        enc_lens0 = enc_lens  # with stream context: lengths including the prepended context
        if ctx is not None:  # H41 arm 11: every head and view sees only the utterance (H44)
            from train.stream_ctx import shift_rows

            enc, enc_lens = shift_rows(enc, enc_lens, ctx)
        # Heads and losses run in fp32 even under autocast.
        with torch.autocast(device_type=enc.device.type, enabled=False):
            # CTC on the dense 25 Hz grid.
            ctc_lp = torch.log_softmax(self.ctc_head(enc).float(), dim=-1)
            ctc_sum, ctc_tokens, ok = ctc_loss(ctc_lp, enc_lens, ctc_tgt, ctc_lens, stats)
            # Transducer.
            enc_p = self.joiner.enc_proj(enc).float()
            pred_p = self.predictor(StatelessPredictor.contexts(rnnt_tgt, self.cfg.pred_ctx)).float()
            zl = None
            if self.cfg.rnnt_loss == "pruned":
                nll, simple, zl = self._pruned_rnnt(enc, enc_p, pred_p, rnnt_tgt, enc_lens,
                                                    rnnt_lens, simple_detach,
                                                    zloss=zloss_weight > 0,
                                                    delay_penalty=delay_penalty)
            else:
                if delay_penalty:
                    raise ValueError("delay_penalty needs rnnt_loss='pruned' (k2)")
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
               "ctc_tokens": ctc_tokens, "ctc_masked": int((~ok).sum()), "nll": nll.detach(),
               "frames": int(enc_lens.sum())}  # denominator of the per-frame means under DDP
        if "inter_logits" in aux and inter_ctc_weight > 0:
            with torch.autocast(device_type=enc.device.type, enabled=False):
                il = aux["inter_logits"].float()
                if ctx is not None:  # the same utterance frames as the other heads
                    il, _ = shift_rows(il, enc_lens0, ctx)
                inter_lp = torch.log_softmax(il, dim=-1)
                isum, itok, _ = ctc_loss(inter_lp, enc_lens, ctc_tgt, ctc_lens, None)
                inter_ctc = isum / max(1, itok)
            total = total + inter_ctc_weight * inter_ctc
            out["inter_ctc"] = inter_ctc
        if "balancer" in aux and balancer_weight > 0:
            bal = aux["balancer"]
            total = total + balancer_weight * bal
            out["balancer"] = bal
        if view is not None:
            out["ctc_lp"] = ctc_lp.detach()
            if cr_target is not None and cr_ctc_weight > 0:  # KL(other view || this view), valid frames
                with torch.autocast(device_type=enc.device.type, enabled=False):
                    kl = (cr_target.exp() * (cr_target - ctc_lp)).sum(-1)
                    valid = torch.arange(ctc_lp.shape[1], device=enc.device)[None] < enc_lens[:, None]
                    cr_ctc = (kl * valid).sum() / valid.sum().clamp(min=1)
                total = total + cr_ctc_weight * cr_ctc
                out["cr_ctc"] = cr_ctc
        elif cr_ctc_weight > 0 and augment is not None and gen is not None:
            # H26 CR-CTC (Yao et al. 2024): supervise BOTH augmented views + symmetric stop-grad KL.
            x2 = spec_augment(x_norm, feat_lens, augment, gen)
            enc2, _ = self.encoder(x2, feat_lens, chunk)
            if ctx is not None:  # view 2 drops the same prepended context as view 1
                enc2, _ = shift_rows(enc2.float(), enc_lens0, ctx)
            with torch.autocast(device_type=enc.device.type, enabled=False):
                ctc_lp2 = torch.log_softmax(self.ctc_head(enc2.float()), dim=-1)
                ctc_sum2, ctc_tok2, _ = ctc_loss(ctc_lp2, enc_lens, ctc_tgt, ctc_lens, None)
                ctc2 = ctc_sum2 / max(1, ctc_tok2)
                p1, p2 = ctc_lp.exp(), ctc_lp2.exp()
                kl12 = (p2.detach() * (ctc_lp2.detach() - ctc_lp)).sum(-1)
                kl21 = (p1.detach() * (ctc_lp.detach() - ctc_lp2)).sum(-1)
                valid = torch.arange(ctc_lp.shape[1], device=enc.device)[None] < enc_lens[:, None]
                cr_ctc = (0.5 * (kl12 + kl21) * valid).sum() / valid.sum().clamp(min=1)
            ctc_avg = 0.5 * (ctc + ctc2)
            total = total + self.cfg.ctc_weight * (ctc_avg - ctc) + cr_ctc_weight * cr_ctc
            out["ctc"] = ctc_avg
            out["cr_ctc"] = cr_ctc
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
                     zloss: bool = False, delay_penalty: float = 0.0):
        """(pruned per-sequence NLL, simple-joiner smoothed per-sequence loss, band z-loss or
        None) via k2. delay_penalty > 0 (H34) applies k2's emission-delay regulariser to both
        the smoothed and the pruned loss (as icefall does); 0 leaves the calls unchanged."""
        dp = {"delay_penalty": delay_penalty} if delay_penalty else {}
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
            boundary=bd, rnnt_type="regular", reduction="none", return_grad=True, **dp)
        ranges = k2.get_rnnt_prune_ranges(px_grad=px_grad, py_grad=py_grad, boundary=bd,
                                          s_range=min(c.prune_range, symbols.shape[1] + 1))
        am, lm = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
        logits = self.joiner.logits(am, lm).float()
        nll = k2.rnnt_loss_pruned(logits=logits, symbols=symbols, ranges=ranges,
                                  termination_symbol=BLANK, boundary=bd, rnnt_type="regular",
                                  reduction="none", **dp)
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
