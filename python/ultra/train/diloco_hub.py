"""Hardened Safetensors Cross-Node DiLoCo Coordinator (`DiLoCoHub`, ULTRA v3)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import namedtuple
from pathlib import Path
from pickle import UnpicklingError
from typing import Callable, Mapping

from safetensors.torch import load as st_load, load_file as st_load_file
import torch

from train.diloco import DiLoCoOuterOptimizer, unpack_bf16_delta
from train.diloco_client import serialize_tensors
from train.diloco_promote import _atomic_write as _atomic_write_bytes
from train.ops.diloco_sync import _cohort_key, extract_weights

_BUFFER_HINTS = ("cmvn", "running_mean", "running_var", "num_batches_tracked", "inv_std", "buffer")
PendingSubmission = namedtuple(
    "PendingSubmission",
    "worker_id host base_round anchor_sha256 data_rank data_world payload_sha256 "
    "local_steps audio_hours local_loss delta probe delta_norm worker_pos ts",
)

class DiLoCoValidationError(ValueError): """Malformed or invalid delta submission (HTTP 400)."""
class DiLoCoConflictError(ValueError): """Round, anchor, rank, buffer, or duplicate payload conflict (HTTP 409)."""

def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def load_init_anchor(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix == ".safetensors":
        sd = st_load_file(str(p), device="cpu")
        meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").is_file() else {}
        return {"model": sd, "ema": sd, **meta}
    if p.suffix not in (".pt", ".pth"): raise DiLoCoValidationError(f"unsupported init anchor format {p.suffix!r}")
    try: payload = torch.load(str(p), map_location="cpu", weights_only=True)
    except (UnpicklingError, RuntimeError, ValueError, EOFError) as exc:
        raise DiLoCoValidationError(f"unsafe or invalid init anchor {p.name}: {exc}") from exc
    if not isinstance(payload, Mapping) or "model" not in payload: raise DiLoCoValidationError(f"init anchor {p.name} missing 'model'")
    return dict(payload)

def _split_keys(weights: Mapping[str, torch.Tensor], trainable_keys=None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tset = set(trainable_keys) if trainable_keys is not None else None
    t = tuple(k for k in sorted(weights) if weights[k].is_floating_point() and (k in tset if tset is not None else not any(h in k.lower() for h in _BUFFER_HINTS)))
    return t, tuple(sorted(k for k in weights if k not in t))

def _probe_vector(delta: Mapping[str, torch.Tensor], keys: tuple[str, ...], max_elems: int = 4096) -> tuple[torch.Tensor, float]:
    sq_sum, parts, per_t = 0.0, [], max(1, max_elems // max(1, len(keys)))
    for k in keys:
        vf = delta[k].detach().float().cpu().reshape(-1)
        sq_sum += float(vf.dot(vf).item())
        parts.append(vf[::max(1, vf.numel() // per_t)][:per_t])
    return (torch.cat(parts) if parts else torch.zeros(1)), sq_sum ** 0.5

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm().item()), float(b.norm().item())
    return 1.0 if (na == 0.0 or nb == 0.0) else float(a.dot(b).item() / (na * nb))

class DiLoCoHub:
    """Thread-safe exact-sync (default) or opt-in experimental async DiLoCo coordinator."""

    def __init__(
        self, cohort: str, anchor_payload: Mapping, work_dir: str | Path, *,
        expected_roster: tuple[str, ...] | None = None, allow_async_experimental: bool = False,
        quorum_k: int = 2, quorum_timeout_s: float = 900.0, outer_lr: float = 0.7,
        outer_momentum: float = 0.9, staleness_alpha: float = 0.5, min_cosine: float = -0.1,
        use_ema: bool = False, hub_ema_decay: float = 0.0, keep_last_k: int = 5,
        keep_every_m: int = 10, trainable_keys=None, _skip_initial_publish: bool = False,
    ):
        self.allow_async_experimental = bool(allow_async_experimental)
        self.expected_roster = tuple(sorted(set(expected_roster))) if expected_roster else None
        if self.expected_roster is None and not self.allow_async_experimental:
            raise ValueError("fixed expected_roster is required unless allow_async_experimental=True")
        self.cohort, self.work_dir = cohort, Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.quorum_k = len(self.expected_roster) if self.expected_roster else max(1, int(quorum_k))
        self.quorum_timeout_s, self.staleness_alpha, self.min_cosine, self.hub_ema_decay = float(quorum_timeout_s), float(staleness_alpha), float(min_cosine), float(hub_ema_decay)
        self.keep_last_k, self.keep_every_m = max(1, int(keep_last_k)), max(1, int(keep_every_m))
        self.template = {k: anchor_payload.get(k, 0 if k == "step" else None) for k in ("model_config", "tokenizer_sha256", "step")}
        self._cohort_sig, init_w = _cohort_key(anchor_payload), extract_weights(anchor_payload, use_ema=use_ema)
        self._trainable_keys, self._buffer_keys = _split_keys(init_w, trainable_keys if trainable_keys is not None else anchor_payload.get("trainable_keys"))
        t_init = {k: init_w[k].detach().float().cpu().clone() for k in self._trainable_keys}
        self._buffers = {k: init_w[k].detach().cpu().clone() for k in self._buffer_keys}
        self.opt = DiLoCoOuterOptimizer(t_init, lr=outer_lr, momentum=outer_momentum, nesterov=True)
        self._hub_ema = {k: v.clone() for k, v in t_init.items()} if 0.0 < self.hub_ema_decay < 1.0 else None
        self._lock, self._on_round, self._pending, self._workers, self._worker_positions, self._recent_rounds = threading.Lock(), [], {}, {}, {}, []
        self._round_started_at, self._total_audio_h, self._last_delta_dtype, self._anchor_bytes, self._anchor_sha256 = time.time(), 0.0, "float32", b"", ""
        if not _skip_initial_publish:
            self._anchor_bytes, self._anchor_sha256 = self._persist_round(self.opt, self._hub_ema, 0, self._workers, self._worker_positions, self._recent_rounds, 0.0)
            self._write_status_unlocked()

    @property
    def sync_mode(self) -> bool: return (self.expected_roster is not None) and (not self.allow_async_experimental)
    @property
    def round(self) -> int: return self.opt.round
    @property
    def anchor_sha256(self) -> str:
        with self._lock: return self._anchor_sha256
    @property
    def hub_ema(self) -> dict[str, torch.Tensor] | None:
        with self._lock: return None if self._hub_ema is None else {k: v.clone() for k, v in self._hub_ema.items()}
    def add_round_listener(self, cb: Callable[[int], None]) -> None:
        with self._lock: self._on_round.append(cb)
    def anchor_state(self) -> tuple[int, str, bytes]:
        with self._lock: return self.opt.round, self._anchor_sha256, self._anchor_bytes
    def anchor_bytes(self) -> tuple[int, bytes]: return self.anchor_state()[0], self.anchor_state()[2]
    def status(self) -> dict:
        with self._lock: return self._build_status_unlocked()

    def _persist_round(self, opt: DiLoCoOuterOptimizer, ema, rnd: int, workers: dict, positions: dict, recent: list[dict], total_ah: float) -> tuple[bytes, str]:
        anchor_bytes = serialize_tensors({**opt.anchor, **self._buffers})
        anchor_sha = _sha256(anchor_bytes)
        state_bytes = serialize_tensors({
            **{f"anchor.{k}": v for k, v in opt.anchor.items()}, **{f"velocity.{k}": v for k, v in opt.velocity.items()},
            **{f"buffer.{k}": v for k, v in self._buffers.items()}, **({f"ema.{k}": v for k, v in ema.items()} if ema else {}),
        })
        meta = {
            "round": int(rnd), "cohort": self.cohort, "anchor_sha256": anchor_sha, "outer_state_sha256": _sha256(state_bytes),
            "config": {"expected_roster": list(self.expected_roster) if self.expected_roster else None, "allow_async_experimental": self.allow_async_experimental,
                       "quorum_k": self.quorum_k, "quorum_timeout_s": self.quorum_timeout_s, "outer_lr": opt.lr, "outer_momentum": opt.momentum,
                       "staleness_alpha": self.staleness_alpha, "min_cosine": self.min_cosine, "hub_ema_decay": self.hub_ema_decay,
                       "keep_last_k": self.keep_last_k, "keep_every_m": self.keep_every_m},
            "trainable_keys": list(self._trainable_keys), "buffer_keys": list(self._buffer_keys), "template": self.template,
            "total_audio_h": total_ah, "roster_meta": {"size": len(self.expected_roster or ())},
            "worker_positions": dict(positions), "workers": dict(workers), "recent_rounds": list(recent[-20:]), "updated_at": time.time(),
        }
        for name, raw in ((f"anchor_r{rnd}.safetensors", anchor_bytes), (f"outer_state_r{rnd}.safetensors", state_bytes),
                          (f"outer_meta_r{rnd}.json", json.dumps(meta, indent=1).encode("utf-8")), ("latest_anchor.safetensors", anchor_bytes)):
            _atomic_write_bytes(self.work_dir / name, raw)
        return anchor_bytes, anchor_sha

    @classmethod
    def from_work_dir(cls, work_dir: str | Path) -> "DiLoCoHub":
        wd = Path(work_dir)
        for mp in sorted(wd.glob("outer_meta_r*.json"), key=lambda p: int(p.stem.split("_r", 1)[1]), reverse=True):
            try:
                meta = json.loads(mp.read_text())
                rnd, raw_state = int(meta["round"]), (wd / f"outer_state_r{int(meta['round'])}.safetensors").read_bytes()
                if _sha256(raw_state) != meta["outer_state_sha256"]: continue
                tensors = st_load(raw_state)
            except (OSError, ValueError, KeyError, TypeError): continue
            anchor, buffers = {k: tensors[f"anchor.{k}"] for k in meta["trainable_keys"]}, {k: tensors[f"buffer.{k}"] for k in meta["buffer_keys"]}
            cfg = {**meta["config"], "expected_roster": tuple(meta["config"]["expected_roster"]) if meta["config"]["expected_roster"] else None}
            hub = cls(meta["cohort"], {"model": {**anchor, **buffers}, "trainable_keys": meta["trainable_keys"], **meta["template"]}, wd, _skip_initial_publish=True, **cfg)
            hub.opt.load_state_dict({"round": rnd, "lr": hub.opt.lr, "momentum": hub.opt.momentum, "nesterov": True,
                                     "anchor": anchor, "velocity": {k: tensors[f"velocity.{k}"] for k in meta["trainable_keys"]}})
            hub._hub_ema = {k: tensors[f"ema.{k}"].clone() for k in meta["trainable_keys"]} if 0.0 < hub.hub_ema_decay < 1.0 else None
            hub._workers, hub._worker_positions = dict(meta.get("workers") or {}), dict(meta.get("worker_positions") or {})
            hub._recent_rounds, hub._total_audio_h = list(meta.get("recent_rounds") or []), float(meta.get("total_audio_h") or 0.0)
            hub._anchor_bytes = serialize_tensors({**anchor, **buffers}); hub._anchor_sha256 = _sha256(hub._anchor_bytes)
            hub._write_status_unlocked()
            return hub
        raise FileNotFoundError(f"no intact outer_meta_r*.json found in {wd}")

    def _prune_old_rounds(self, cur_round: int) -> None:
        for mp in list(self.work_dir.glob("outer_meta_r*.json")):
            rnd = int(mp.stem.split("_r", 1)[1])
            if 0 < rnd <= (cur_round - self.keep_last_k) and (rnd % self.keep_every_m != 0):
                for p in (mp, self.work_dir / f"outer_state_r{rnd}.safetensors", self.work_dir / f"anchor_r{rnd}.safetensors"):
                    if p.exists(): p.unlink()

    def _build_status_unlocked(self) -> dict:
        now, dtype_tag = time.time(), "FP32" if self._last_delta_dtype == "float32" else "BF16"
        active = [w for w in self._workers.values() if now - float(w.get("last_seen") or 0.0) < 1800.0 and w.get("status") != "quarantined"]
        norms = [float(w["delta_norm"]) for w in self._workers.values() if w.get("delta_norm") is not None]
        return {"cohort": self.cohort, "mode": "sync_roster" if self.sync_mode else "async_quorum", "delta_dtype": self._last_delta_dtype,
                "anchor_sha256": self._anchor_sha256, "mode_label": f"{'Sync' if self.sync_mode else 'Async'}-{dtype_tag} {'Roster' if self.sync_mode else 'Quorum'}",
                "expected_roster": list(self.expected_roster) if self.expected_roster else None, "round": self.opt.round, "quorum_k": self.quorum_k,
                "pending_count": len(self._pending), "pending_workers": sorted(self._pending), "total_audio_h": round(self._total_audio_h, 2),
                "effective_audio_h_per_h": round(sum(float(w.get("audio_h_per_h") or 0.0) for w in active), 1),
                "mean_delta_norm": round(sum(norms) / len(norms), 4) if norms else None, "max_staleness": max((int(w.get("staleness") or 0) for w in self._workers.values()), default=0),
                "quarantined_workers": sorted(wid for wid, w in self._workers.items() if w.get("status") == "quarantined"),
                "outer_lr": self.opt.lr, "outer_momentum": self.opt.momentum, "workers": dict(self._workers),
                "worker_positions": dict(self._worker_positions), "recent_rounds": list(self._recent_rounds[-20:]), "updated_at": now}
    def _write_status_unlocked(self) -> None:
        _atomic_write_bytes(self.work_dir / "diloco_status.json", json.dumps(self._build_status_unlocked(), indent=1).encode("utf-8"))

    def _validate_tensors(self, delta_sd: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
        if set(delta_sd) != (set(self._trainable_keys) | set(self._buffer_keys)):
            raise DiLoCoValidationError("delta key set does not match anchor key set")
        for k in self._buffer_keys:
            t, ref = delta_sd[k], self._buffers[k]
            if not isinstance(t, torch.Tensor) or t.shape != ref.shape or t.dtype != ref.dtype or (t.is_floating_point() and not torch.isfinite(t).all()):
                raise DiLoCoValidationError(f"invalid buffer {k!r}")
            if not torch.equal(t.detach().cpu(), ref): raise DiLoCoConflictError(f"buffer {k!r} is not bit-equal across roster")
        dtype_name, trainable = "float32", {}
        for k in self._trainable_keys:
            t, ref = delta_sd[k], self.opt.anchor[k]
            if not isinstance(t, torch.Tensor) or t.shape != ref.shape: raise DiLoCoValidationError(f"tensor {k!r} shape mismatch")
            if self.sync_mode and t.dtype != torch.float32: raise DiLoCoValidationError(f"sync mode requires float32 for {k!r}, got {t.dtype}")
            if t.dtype not in (torch.float32, torch.bfloat16): raise DiLoCoValidationError(f"unsupported dtype {t.dtype} for {k!r}")
            dtype_name = "bfloat16" if t.dtype == torch.bfloat16 else dtype_name
            tf = t.detach().float().cpu()
            if not torch.isfinite(tf).all(): raise DiLoCoValidationError(f"non-finite values in delta tensor {k!r}")
            trainable[k] = tf
        return trainable, dtype_name

    def submit_delta(
        self, *, worker_id: str, host: str, base_round: int, local_steps: int, audio_hours: float, local_loss: float,
        delta_sd: Mapping[str, torch.Tensor], cohort_meta: Mapping | None = None, audio_h_per_h: float = 0.0,
        anchor_sha256: str | None = None, data_rank: int | None = None, data_world: int | None = None,
        payload_sha256: str | None = None, worker_pos: dict | None = None,
    ) -> dict:
        if cohort_meta is not None and _cohort_key(cohort_meta) != self._cohort_sig: raise DiLoCoValidationError(f"cohort mismatch from {worker_id}")
        if self.expected_roster is not None and worker_id not in self.expected_roster: raise DiLoCoValidationError(f"worker {worker_id!r} not in fixed roster")
        if audio_hours <= 0.0: raise DiLoCoValidationError(f"audio_hours must be > 0 (got {audio_hours})")
        trainable, dtype_name = self._validate_tensors(delta_sd)
        unpacked = unpack_bf16_delta(trainable) if dtype_name == "bfloat16" else trainable
        probe, dnorm = _probe_vector(unpacked, self._trainable_keys)
        psha, now, notify_round, listeners = payload_sha256 or _sha256(serialize_tensors(delta_sd)), time.time(), None, []
        with self._lock:
            if self.sync_mode:
                if int(base_round) != self.opt.round or anchor_sha256 != self._anchor_sha256:
                    raise DiLoCoConflictError(f"stale/mismatched anchor: round={base_round} expected={self.opt.round}")
                n_r = len(self.expected_roster or ())
                if data_world is None or int(data_world) != n_r or data_rank is None or not (0 <= int(data_rank) < n_r):
                    raise DiLoCoValidationError(f"invalid data_rank={data_rank} data_world={data_world} for roster size {n_r}")
            if worker_id in self._pending:
                if self._pending[worker_id].payload_sha256 == psha:
                    return {"accepted": True, "duplicate": True, "worker_id": worker_id, "current_round": self.opt.round,
                            "anchor_sha256": self._anchor_sha256, "pending_count": len(self._pending), "triggered_outer": False, "round_record": None}
                raise DiLoCoConflictError(f"conflicting duplicate submission from {worker_id}")
            if self.sync_mode and any(p.data_rank == int(data_rank) for p in self._pending.values()):
                raise DiLoCoConflictError(f"duplicate data_rank={data_rank} in round {self.opt.round}")
            prev_w, prev_pos = self._workers.get(worker_id), self._worker_positions.get(worker_id)
            self._workers[worker_id] = {"worker_id": worker_id, "host": host, "base_round": int(base_round), "staleness": max(0, self.opt.round - int(base_round)),
                "local_steps": int(local_steps), "audio_hours_last": round(float(audio_hours), 3),
                "audio_hours_total": round(float((prev_w or {}).get("audio_hours_total", 0.0)) + float(audio_hours), 3),
                "audio_h_per_h": round(float(audio_h_per_h), 1), "local_loss": round(float(local_loss), 4), "delta_norm": round(dnorm, 4), "status": "pending", "last_seen": now}
            if worker_pos is not None: self._worker_positions[worker_id] = dict(worker_pos)
            self._pending[worker_id] = PendingSubmission(worker_id, host, int(base_round), anchor_sha256 or "", int(data_rank) if data_rank is not None else None,
                int(data_world) if data_world is not None else None, psha, int(local_steps), float(audio_hours), float(local_loss), unpacked, probe, dnorm, worker_pos, now)
            self._last_delta_dtype = dtype_name
            ready = (set(self._pending) == set(self.expected_roster) if self.sync_mode
                     else (len(self._pending) >= self.quorum_k or ((now - self._round_started_at) >= self.quorum_timeout_s and len(self._pending) >= 1)))
            try:
                round_rec = self._commit_outer_round_unlocked(now) if ready else None
                self._write_status_unlocked()
            except OSError:
                self._pending.pop(worker_id, None)
                self._workers.pop(worker_id, None); self._worker_positions.pop(worker_id, None)
                if prev_w is not None: self._workers[worker_id] = prev_w
                if prev_pos is not None: self._worker_positions[worker_id] = prev_pos
                raise
            if round_rec is not None: notify_round, listeners = self.opt.round, list(self._on_round)
            resp = {"accepted": True, "duplicate": False, "worker_id": worker_id, "current_round": self.opt.round,
                    "anchor_sha256": self._anchor_sha256, "pending_count": len(self._pending), "triggered_outer": round_rec is not None, "round_record": round_rec}
        for cb in listeners:
            if notify_round is not None: cb(notify_round)
        return resp

    def _commit_outer_round_unlocked(self, now: float) -> dict:
        subs = [self._pending[w] for w in (self.expected_roster if self.sync_mode else sorted(self._pending))]
        if self.sync_mode and {s.data_rank for s in subs} != set(range(len(subs))):
            raise DiLoCoConflictError("incomplete data_rank coverage across roster")
        valid, quarantined, cos_scores = [], [], {}
        for s in subs:
            peer = sum(((o.probe / max(float(o.probe.norm().item()), 1e-12)) * (1.0 if self.sync_mode else o.audio_hours) for o in subs if o.worker_id != s.worker_id), torch.zeros_like(s.probe))
            c = _cosine(s.probe, peer) if len(subs) >= 2 else 1.0
            cos_scores[s.worker_id] = round(c, 4)
            (quarantined.append(s.worker_id) if (not self.sync_mode and len(subs) >= 3 and c < self.min_cosine) else valid.append(s))
        if not valid: valid, quarantined = subs, []
        weights = None if self.sync_mode else [max(1e-6, s.audio_hours * ((1.0 + max(0, self.opt.round - s.base_round)) ** (-self.staleness_alpha))) for s in valid]
        trial_opt = DiLoCoOuterOptimizer(self.opt.anchor, lr=self.opt.lr, momentum=self.opt.momentum, nesterov=self.opt.nesterov)
        trial_opt.load_state_dict(self.opt.state_dict())
        new_anchor = trial_opt.step_from_deltas([s.delta for s in valid], weights=weights)
        trial_ema = None if self._hub_ema is None else {k: self._hub_ema[k] * self.hub_ema_decay + new_anchor[k] * (1.0 - self.hub_ema_decay) for k in self._trainable_keys}
        next_workers = {k: dict(v) for k, v in self._workers.items()}
        for wid, c in cos_scores.items(): next_workers[wid]["cosine"] = c
        for wid in quarantined: next_workers[wid]["status"] = "quarantined"
        for s in valid: next_workers[s.worker_id]["status"] = "synced"
        round_ah = sum(s.audio_hours for s in valid)
        rec = {"round": trial_opt.round, "time": now, "cohort": self.cohort, "mode": "sync_roster" if self.sync_mode else "async_quorum",
               "delta_dtype": self._last_delta_dtype, "workers": [s.worker_id for s in valid], "hosts": sorted({s.host for s in valid}),
               "quarantined": quarantined, "audio_h_round": round(round_ah, 3), "total_audio_h": round(self._total_audio_h + round_ah, 3),
               "mean_cos": round(sum(cos_scores[s.worker_id] for s in valid) / max(1, len(valid)), 4),
               "mean_delta_norm": round(sum(s.delta_norm for s in valid) / max(1, len(valid)), 4),
               "max_staleness": max(max(0, trial_opt.round - 1 - s.base_round) for s in valid),
               "delta_norms": {s.worker_id: round(s.delta_norm, 4) for s in valid}, "quorum_wait_s": round(now - self._round_started_at, 1)}
        next_recent = (self._recent_rounds + [rec])[-100:]
        anchor_bytes, anchor_sha = self._persist_round(trial_opt, trial_ema, trial_opt.round, next_workers, self._worker_positions, next_recent, self._total_audio_h + round_ah)
        rec["anchor_sha256"], rec["anchor_path"] = anchor_sha, str(self.work_dir / f"anchor_r{trial_opt.round}.safetensors")
        self.opt, self._hub_ema, self._anchor_bytes, self._anchor_sha256 = trial_opt, trial_ema, anchor_bytes, anchor_sha
        self._workers, self._recent_rounds, self._total_audio_h, self._round_started_at = next_workers, next_recent, self._total_audio_h + round_ah, now
        self._pending.clear()
        with open(self.work_dir / "diloco_rounds.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
        self._prune_old_rounds(self.opt.round)
        return rec

from train.diloco_http import make_hub_app  # noqa: E402,F401
if __name__ == "__main__":
    from train.diloco_http import main
    raise SystemExit(main())
