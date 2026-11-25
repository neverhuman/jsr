"""H41 arm 11: stream-context augmentation (always-on streaming).

Every training utterance gets a random LEFT context of 0 to `max_s` seconds (40 ms steps, so it
maps to whole 25 Hz encoder frames) prepended at feature level, drawn per example from
  * digital silence (1/3),
  * synthetic white / pink / brown noise at -40 / -30 / -20 dBFS RMS (1/3),
  * the tail of another utterance of the same batch, with an optional 0-2 s silence gap (1/3).
The encoder attends over the context (as it does in a stream), but the losses see only the
utterance's own frames: ASRModel.losses shifts each row of the encoder output left by its context
(`Batch.ctx`), so the prepended audio's words are never targets. Seeded per (seed, epoch,
utterance); no external data. Each row's utterance + context stays within `cap` x the batch's
longest utterance, which bounds the padded batch at `cap` x its plain size (GPU memory).
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import torch

from train.features import SAMPLE_RATE

STEP = 4  # feature frames per encoder frame (two stride-2 convolutions)


def _noise(color: str, n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float64)
    if color != "white":
        spec = np.fft.rfft(w)
        f = np.arange(1, len(spec) + 1, dtype=np.float64)
        spec /= np.sqrt(f) if color == "pink" else f  # pink 1/f power, brown 1/f^2 power
        w = np.fft.irfft(spec, n=n)
    w /= np.sqrt(np.mean(w * w)) + 1e-12
    return (w * 10.0 ** (dbfs / 20.0)).astype(np.float32)


class ContextBank:
    """Precomputed context features: one silence frame and a long clip per noise kind."""

    COLORS, LEVELS = ("white", "pink", "brown"), (-40.0, -30.0, -20.0)

    def __init__(self, logmel, seconds: float = 12.0, seed: int = 41, noise: bool = True):
        self.silence = logmel(torch.zeros(4 * 400))[0]  # (80,): every digital-zero frame is equal
        if noise:
            rng = np.random.default_rng(seed)
            n = int(seconds * SAMPLE_RATE)
            self.noise = {(c, lv): logmel(torch.from_numpy(_noise(c, n, lv, rng)))
                          for c in self.COLORS for lv in self.LEVELS}
        else:
            self.noise = {}


def _speaker_id(uid: str) -> str:
    parts = uid.split("-")
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 and not parts[0].isdigit() else parts[0]


def _rng(seed: int, epoch: int, uid: str) -> np.random.Generator:
    h = hashlib.sha256(f"ctx:{seed}:{epoch}:{uid}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def add_context(exs: list, bank: ContextBank, seed: int, epoch: int, max_s: float = 5.12,
                cap: float = 1.5, p: float = 1.0, mode: str = "h41", gap_s: float = 0.0) -> list:
    """New examples with context frames prepended and `ctx` (feature frames) set."""
    if mode not in ("h41", "turn_switch", "turn_switch_mixed"):
        raise ValueError(
            f"unknown stream_ctx mode {mode!r}; expected 'h41', 'turn_switch', or 'turn_switch_mixed'"
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"stream_ctx p must be in [0, 1], got {p}")
    if gap_s < 0.0:
        raise ValueError(f"stream_ctx gap_s must be >= 0, got {gap_s}")
    longest = max(e.feats.shape[0] for e in exs)
    kmax = int(round(max_s * 100)) // STEP
    out = []
    if mode == "turn_switch_mixed":
        gap_steps = max(0, int(round(gap_s * 100 / STEP)))
        gap = gap_steps * STEP
        spks = [_speaker_id(e.uid) for e in exs]
        for i, e in enumerate(exs):
            r = _rng(seed, epoch, e.uid)
            n = e.feats.shape[0]
            room = max(0, int(cap * longest) - n) // STEP
            hi = min(kmax - gap_steps, room - gap_steps)
            if len(exs) >= 2 and room >= gap_steps + 12 and hi >= 12 and float(r.random()) < p:
                want_same = float(r.random()) < 0.5
                if want_same:
                    cands = [j for j in range(len(exs)) if j != i and spks[j] == spks[i]]
                else:
                    cands = [j for j in range(len(exs)) if j != i and spks[j] != spks[i]]
                if not cands:
                    out.append(e)
                    continue
                j = cands[int(r.integers(0, len(cands)))]
                k_tail = int(r.integers(12, hi + 1))
                tail_len = k_tail * STEP
                tail = exs[j].feats[-tail_len:]
                pad = tail_len - tail.shape[0]
                parts = []
                if pad > 0:
                    parts.append(bank.silence.expand(pad, -1))
                parts.append(tail)
                if gap > 0:
                    parts.append(bank.silence.expand(gap, -1))
                ctx = torch.cat(parts)
                L = tail_len + gap
                out.append(dataclasses.replace(e, feats=torch.cat([ctx.to(e.feats.dtype), e.feats]), ctx=L))
            else:
                out.append(e)
        return out
    if mode == "turn_switch":
        gap_steps = max(0, int(round(gap_s * 100 / STEP)))
        gap = gap_steps * STEP
        spks = [_speaker_id(e.uid) for e in exs]
        for i, e in enumerate(exs):
            r = _rng(seed, epoch, e.uid)
            n = e.feats.shape[0]
            room = max(0, int(cap * longest) - n) // STEP
            hi = min(kmax - gap_steps, room - gap_steps)
            if len(exs) >= 2 and room >= gap_steps + 12 and hi >= 12 and float(r.random()) < p:
                cands = [j for j in range(len(exs)) if j != i and spks[j] != spks[i]]
                if not cands:
                    cands = [j for j in range(len(exs)) if j != i]
                j = cands[int(r.integers(0, len(cands)))]
                k_tail = int(r.integers(12, hi + 1))
                tail_len = k_tail * STEP
                tail = exs[j].feats[-tail_len:]
                pad = tail_len - tail.shape[0]
                parts = []
                if pad > 0:
                    parts.append(bank.silence.expand(pad, -1))
                parts.append(tail)
                if gap > 0:
                    parts.append(bank.silence.expand(gap, -1))
                ctx = torch.cat(parts)
                L = tail_len + gap
                out.append(dataclasses.replace(e, feats=torch.cat([ctx.to(e.feats.dtype), e.feats]), ctx=L))
            else:
                out.append(e)
        return out
    for i, e in enumerate(exs):
        r = _rng(seed, epoch, e.uid)
        n = e.feats.shape[0]
        room = max(0, int(cap * longest) - n) // STEP
        k = min(int(r.integers(0, kmax + 1)), room)
        L = k * STEP
        kind = int(r.integers(0, 3))
        if L == 0:
            ctx = e.feats.new_zeros(0, e.feats.shape[1])
        elif kind == 0 or len(exs) < 2 and kind == 2:  # digital silence
            ctx = bank.silence.expand(L, -1).clone()
        elif kind == 1:  # synthetic noise
            clip = bank.noise[(bank.COLORS[int(r.integers(0, 3))], bank.LEVELS[int(r.integers(0, 3))])]
            s = int(r.integers(0, clip.shape[0] - L + 1))
            ctx = clip[s:s + L].clone()
        else:  # the tail of another utterance of the batch, then an optional silence gap
            j = int(r.integers(0, len(exs) - 1))
            j += j >= i
            gap = min(L, int(r.integers(0, 51)) * STEP)  # 0-2 s
            tail = exs[j].feats[-(L - gap):] if L > gap else exs[j].feats[:0]
            pad = L - gap - tail.shape[0]  # the other utterance may be shorter than the tail
            ctx = torch.cat([bank.silence.expand(pad, -1), tail, bank.silence.expand(gap, -1)])
        out.append(dataclasses.replace(e, feats=torch.cat([ctx.to(e.feats.dtype), e.feats]), ctx=L))
    return out


def shift_rows(enc: torch.Tensor, lens: torch.Tensor, ctx: torch.Tensor):
    """Drop each row's first ctx // STEP encoder frames (its prepended context)."""
    c = torch.div(ctx.to(enc.device), STEP, rounding_mode="floor")
    t = torch.arange(enc.shape[1], device=enc.device)[None] + c[:, None]
    t = t.clamp(max=enc.shape[1] - 1)
    return torch.gather(enc, 1, t[..., None].expand(-1, -1, enc.shape[-1])), lens - c
