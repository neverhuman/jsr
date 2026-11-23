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


def _speed_key(speed) -> float:
    """Exact speed factor key (to 1e-3, so 0.9 / 1.0 / 1.1 match however they were parsed)."""
    return round(float(speed), 3)


KD_META = "KD_META.json"  # cache identity side file (format ultra.kd-cache/v1), see write_meta()
COUNT_KEYS = ("accepted", "missing", "text_mismatch", "path_mismatch", "frame_mismatch")


class KdIdentityError(ValueError):
    """The teacher cache's identity does not match the student run (tokenizer, shard files)."""


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_meta(cache_dir: str, tokenizer_path: str, teacher: str) -> dict:
    """Write `KD_META.json`: the tokenizer the teacher's token ids belong to, the teacher, and every
    shard's sha256 (checked against the producer's SHA256SUMS when present)."""
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(cache_dir, "*.npz")))
    sums = {}
    sums_path = os.path.join(cache_dir, "SHA256SUMS")
    if os.path.exists(sums_path):
        for line in open(sums_path):
            if line.strip():
                h, name = line.split()
                sums[name.lstrip("*")] = h
    shards = {}
    for name in files:
        h = _sha256_file(os.path.join(cache_dir, name))
        if name in sums and sums[name] != h:
            raise KdIdentityError(f"{name}: sha256 {h} differs from SHA256SUMS {sums[name]}")
        shards[name] = h
    speeds = sorted({_speed_key(json.loads(str(np.load(os.path.join(cache_dir, n))["meta"]))["speed"]) for n in files})
    meta = {"format": "ultra.kd-cache/v1", "teacher": teacher, "tokenizer_sha256": _sha256_file(tokenizer_path),
            "speeds": speeds, "shards": shards}
    with open(os.path.join(cache_dir, KD_META), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    return meta


class TeacherStore:
    """Frame-level R0 targets keyed by (utterance, exact speed).

    Identity: with `tokenizer_sha256`, the cache must prove which tokenizer its ids belong to, via
    `KD_META.json` (or the same field in every shard's meta). A mismatch, an unlisted shard, or a
    shard whose sha256 differs refuses the whole store. `counts` tallies every lookup outcome; only
    `accepted` rows carry a KD term."""

    def __init__(self, cache_dir: str, tokenizer_sha256: str | None = None, verify_shards: bool = True):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"no teacher cache files under {cache_dir}")
        self.identity = self._check_identity(cache_dir, files, tokenizer_sha256, verify_shards)
        self.counts = dict.fromkeys(COUNT_KEYS, 0)
        self._entries: dict[tuple[str, float], tuple] = {}
        self._arrays = []  # keep the big arrays alive
        self.topk = None
        for f in files:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            sp = _speed_key(meta["speed"])
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
                self._entries[(str(uid), sp)] = (ids[off:off + n], lp[off:off + n], u, ok, e)
                off += n

    @staticmethod
    def _check_identity(cache_dir, files, tokenizer_sha256, verify_shards) -> dict:
        side = os.path.join(cache_dir, KD_META)
        if os.path.exists(side):
            meta = json.load(open(side))
            listed = meta.get("shards", {})
            for f in files:
                name = os.path.basename(f)
                if name not in listed:
                    raise KdIdentityError(f"{name} is not listed in {KD_META}")
                if verify_shards and _sha256_file(f) != listed[name]:
                    raise KdIdentityError(f"{name}: sha256 differs from {KD_META}")
            got = meta.get("tokenizer_sha256")
        else:
            shard_toks = {json.loads(str(np.load(f)["meta"])).get("tokenizer_sha256") for f in files}
            got = shard_toks.pop() if len(shard_toks) == 1 else None
        if tokenizer_sha256 is not None and got != tokenizer_sha256:
            raise KdIdentityError(f"teacher cache tokenizer {got} != student tokenizer {tokenizer_sha256} "
                                  f"(write {KD_META} with train.kd_meta, or rebuild the cache)")
        return {"kd_tokenizer_sha256": got, "kd_identity": KD_META if os.path.exists(side) else "shard-meta"}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, uid: str, speed: float):
        """The entry cached at exactly this speed, else None: a sample at a speed with no cache
        trains without the KD term (never with another speed's targets)."""
        row = self._entries.get((uid, _speed_key(speed)))
        if row is None:
            return None
        ids, lp, u, ok, _ = row
        return ids, lp, u, ok

    def get_checked(self, uid: str, speed: float, target_ids: list[int] | np.ndarray,
                    n_frames: int | None = None):
        """`(ids, lp, u, ok)` with ok only if R0's text matches, its emitted subword path equals the
        student's RNN-T targets (`verify_emitted_path`, PR #72's B2) and, given `n_frames`, its frame
        count is within one of the student's encoder frames. Every outcome is counted."""
        row = self._entries.get((uid, _speed_key(speed)))
        if row is None:
            self.counts["missing"] += 1
            return None
        ids, lp, u, ok, emitted = row
        if not ok:
            self.counts["text_mismatch"] += 1
        elif not verify_emitted_path(emitted, target_ids):
            self.counts["path_mismatch"] += 1
            ok = False
        elif n_frames is not None and abs(len(u) - int(n_frames)) > 1:
            self.counts["frame_mismatch"] += 1
            ok = False
        else:
            self.counts["accepted"] += 1
        return ids, lp, u, bool(ok)


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
