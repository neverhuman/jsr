"""H22 distillation from R0: the cached-teacher store and the collapsed-support KL loss.

Teacher cache files come from train/teacher_cache.py (one .npz per (train shard, speed)). For
every cached utterance the store keeps views into the concatenated arrays:
  ids (Tt, k) R0 token ids of the top-k joiner log-probs, lp (Tt, k) those log-probs,
  u (Tt,) R0 tokens emitted before each frame (the lattice row R0's greedy path is on),
  ok: R0's 1-best equals the normalised reference (only then is (t, u_t) on the reference lattice).
The student shares R0's bpe.model through train.tokenizer, so student index = R0 id + 1 for
pieces and 0 for blank.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ultra.wer import normalise


class TeacherStore:
    def __init__(self, cache_dir: str):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"no teacher cache files under {cache_dir}")
        self._entries: dict[tuple[str, float], tuple] = {}
        self._arrays = []  # keep the big arrays alive
        self.topk = None
        for f in files:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            sp = float(meta["speed"])
            ids, lp, emit = z["topk_ids"], z["topk_lp"], z["emitted"]
            self.topk = ids.shape[1] if self.topk is None else self.topk
            if ids.shape[1] != self.topk:
                raise ValueError(f"{f}: top-k {ids.shape[1]} != {self.topk}")
            self._arrays.append((ids, lp, emit))
            off = 0
            for uid, text, hyp, n in zip(z["uid"], z["text"], z["hyp"], z["n_frames"], strict=True):
                n = int(n)
                e = emit[off:off + n]
                u = np.concatenate([[0], np.cumsum(e[:-1] != 0)]).astype(np.int64) if n else e.astype(np.int64)
                ok = normalise(str(hyp)) == normalise(str(text))
                self._entries[(str(uid), sp)] = (ids[off:off + n], lp[off:off + n], u, ok)
                off += n

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, uid: str, speed: float):
        return self._entries.get((uid, round(float(speed), 1)))


def r0_to_student(ids: torch.Tensor) -> torch.Tensor:
    """R0 ids (blank 0, piece p) -> student output indices (blank 0, piece p + 1)."""
    return torch.where(ids > 0, ids + 1, torch.zeros_like(ids))


def verify_emitted_path(emitted: np.ndarray, rnnt_tgt: list[int] | np.ndarray) -> bool:
    """H26 forensic guard: verify teacher's non-blank emitted subword sequence matches the
    student's exact RNN-T target tokens (stronger than normalised text equality)."""
    nz = emitted[emitted > 0]
    student_emitted = (nz.astype(np.int64) + 1).tolist()
    target = list(rnnt_tgt) if not isinstance(rnnt_tgt, np.ndarray) else rnnt_tgt.astype(np.int64).tolist()
    return student_emitted == target


def kd_loss(joiner, enc_p: torch.Tensor, pred_p: torch.Tensor, enc_lens: torch.Tensor,
            rnnt_lens: torch.Tensor, kd_ids: torch.Tensor, kd_lp: torch.Tensor,
            kd_u: torch.Tensor, kd_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean over used frames of KL(teacher || student) on the collapsed support {top-k, OTHER},
    at each frame's teacher lattice cell (t, u_t). Returns (loss, number of frames used)."""
    T = min(enc_p.shape[1], kd_ids.shape[1])
    if T == 0:
        return enc_p.sum() * 0.0, 0
    t_idx = torch.arange(T, device=enc_p.device)[None]
    u = kd_u[:, :T].clamp(min=0)
    mask = kd_mask[:, :T] & (t_idx < enc_lens[:, None]) & (u <= rnnt_lens[:, None])
    n = int(mask.sum())
    if n == 0:
        return enc_p.sum() * 0.0, 0
    u = torch.minimum(u, rnnt_lens[:, None])
    pp = torch.gather(pred_p, 1, u[..., None].expand(-1, -1, pred_p.shape[-1]))  # (B, T, J)
    logp = F.log_softmax(joiner.logits(enc_p[:, :T], pp).float(), dim=-1)          # (B, T, V)
    sid = r0_to_student(kd_ids[:, :T].long())
    p_top = torch.gather(logp, -1, sid)                                              # (B, T, k)
    q = kd_lp[:, :T].float().exp()
    q_other = (1.0 - q.sum(-1)).clamp(min=1e-6)
    p_other = torch.log((1.0 - p_top.exp().sum(-1)).clamp(min=1e-6))
    kl = (q * (kd_lp[:, :T].float() - p_top)).sum(-1) + q_other * (torch.log(q_other) - p_other)
    return (kl * mask).sum() / n, n
