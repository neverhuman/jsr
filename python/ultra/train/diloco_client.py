"""Hardened safetensors HTTP worker client & trainer hook for live DiLoCo (`trainer.py`)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Mapping

import torch

from train.diloco import pack_bf16_delta

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class DiLoCoBarrierError(RuntimeError):
    """Raised when a DiLoCo sync, auth, conflict, or blocking round barrier fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def sign_request(token: str, worker_id: str, round_idx: int, anchor_sha256: str, payload_sha256: str) -> str:
    """Compute HMAC-SHA256 hex digest over `(worker_id, round, anchor_sha256, payload_sha256)`."""
    msg = f"{worker_id}|{int(round_idx)}|{anchor_sha256}|{payload_sha256}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def serialize_tensors(tensors: Mapping[str, torch.Tensor]) -> bytes:
    """Serialize CPU contiguous tensors deterministically via safetensors."""
    from safetensors.torch import save as save_safetensors

    return save_safetensors({k: tensors[k].detach().cpu().contiguous().clone() for k in sorted(tensors)})


class DiLoCoClient:
    """Pushes HMAC-signed safetensors deltas to `DiLoCoHub` and blocks on the round barrier."""

    def __init__(
        self, hub_url: str, worker_id: str, model_state: Mapping[str, torch.Tensor], *,
        auth_token: str | None = None, cohort_meta: dict | None = None, host: str | None = None,
        timeout_s: float = 60.0, barrier_timeout_s: float = 600.0, poll_interval_s: float = 0.25,
        delta_dtype: str = "float32", data_rank: int = 0, data_world: int = 1,
        trainable_keys: set[str] | tuple[str, ...] | None = None, allow_async_experimental: bool = False,
    ):
        token = auth_token if auth_token is not None else os.environ.get("DILOCO_AUTH_TOKEN", "")
        if not token:
            raise ValueError("DiLoCoClient requires auth_token or DILOCO_AUTH_TOKEN")
        if delta_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"unsupported delta_dtype={delta_dtype!r}")
        if delta_dtype != "float32" and not allow_async_experimental:
            raise ValueError("exact sync mode requires delta_dtype='float32'")
        self.hub_url, self.worker_id, self.auth_token = hub_url.rstrip("/"), worker_id, token
        self.host, self.cohort_meta = host or socket.gethostname(), cohort_meta
        self.timeout_s, self.barrier_timeout_s = float(timeout_s), float(barrier_timeout_s)
        self.poll_interval_s, self.delta_dtype = max(0.05, float(poll_interval_s)), delta_dtype
        self.data_rank, self.data_world = int(data_rank), int(data_world)
        self.allow_async_experimental = bool(allow_async_experimental)
        self.trainable_keys = set(trainable_keys) if trainable_keys is not None else None
        self.base_round = 0
        self.anchor = {
            k: v.detach().float().cpu().clone() if v.is_floating_point() else v.detach().cpu().clone()
            for k, v in model_state.items()
        }
        self.anchor_sha256 = hashlib.sha256(serialize_tensors(self.anchor)).hexdigest()

    def _auth_headers(self, round_idx: int, anchor_sha256: str, payload_sha256: str) -> dict[str, str]:
        return {
            "X-DiLoCo-Worker": self.worker_id,
            "X-DiLoCo-Round": str(int(round_idx)),
            "X-DiLoCo-Anchor-SHA256": anchor_sha256,
            "X-DiLoCo-Auth": sign_request(self.auth_token, self.worker_id, round_idx, anchor_sha256, payload_sha256),
        }

    def pull_anchor(
        self, model: torch.nn.Module, force: bool = False, *, wait_round: int | None = None, timeout_s: float | None = None
    ) -> tuple[int, bool]:
        """Download safetensors anchor (optionally long-polling for `wait_round`)."""
        headers = self._auth_headers(self.base_round, self.anchor_sha256, EMPTY_SHA256)
        if not force and wait_round is None:
            headers["If-None-Match"] = self.anchor_sha256
        url, req_timeout = f"{self.hub_url}/anchor", self.timeout_s
        if wait_round is not None:
            wait_s = max(0.05, min(30.0, float(timeout_s if timeout_s is not None else self.timeout_s)))
            url = f"{url}?{urllib.parse.urlencode({'wait_round': int(wait_round), 'timeout_s': round(wait_s, 3)})}"
            req_timeout = max(self.timeout_s, wait_s + 5.0)
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                if resp.status == 202:
                    return self.base_round, False
                raw = resp.read()
                rnd = int(resp.headers.get("X-DiLoCo-Round", "0"))
                hdr_sha = resp.headers.get("X-DiLoCo-Anchor-SHA256", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return self.base_round, False
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise DiLoCoBarrierError(f"GET /anchor HTTP {exc.code}: {detail}", status_code=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise DiLoCoBarrierError(f"GET /anchor failed: {exc}") from exc

        actual_sha = hashlib.sha256(raw).hexdigest()
        if not hdr_sha or not hmac.compare_digest(actual_sha, hdr_sha):
            raise DiLoCoBarrierError(f"anchor sha256 mismatch: header={hdr_sha!r} actual={actual_sha!r}")
        import safetensors
        from safetensors.torch import load as load_safetensors

        try:
            sd = load_safetensors(raw)
        except (safetensors.SafetensorError, ValueError) as exc:
            raise DiLoCoBarrierError(f"invalid safetensors anchor payload: {exc}") from exc
        model.load_state_dict(sd, strict=True)
        self.anchor = {k: v.detach().cpu().clone() for k, v in sd.items()}
        self.base_round, self.anchor_sha256 = rnd, actual_sha
        return rnd, True

    def _compute_delta(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        local_sd = model.state_dict()
        tkeys = self.trainable_keys if self.trainable_keys is not None else {n for n, _ in model.named_parameters()}
        if self.delta_dtype == "bfloat16":
            bf16 = pack_bf16_delta(self.anchor, local_sd)
            return {k: (bf16[k] if k in tkeys else local_sd[k].detach().cpu().clone()) for k in self.anchor}
        return {
            k: ((self.anchor[k].float() - local_sd[k].detach().float().cpu()).to(torch.float32)
                if (k in tkeys and self.anchor[k].is_floating_point()) else local_sd[k].detach().cpu().clone())
            for k in self.anchor
        }

    def sync(
        self, model: torch.nn.Module, *, local_steps: int, audio_hours: float, local_loss: float,
        audio_h_per_h: float = 0.0, worker_pos: dict | None = None, broadcast_fn: Callable[[], None] | None = None,
    ) -> dict:
        """Push HMAC-signed safetensors delta and block until the next round anchor commits."""
        t0 = time.monotonic()
        payload = serialize_tensors(self._compute_delta(model))
        payload_sha = hashlib.sha256(payload).hexdigest()
        meta = {
            "worker_id": self.worker_id, "host": self.host, "base_round": self.base_round,
            "anchor_sha256": self.anchor_sha256, "data_rank": self.data_rank, "data_world": self.data_world,
            "local_steps": int(local_steps), "audio_hours": float(max(audio_hours, 1e-6)),
            "local_loss": float(local_loss), "audio_h_per_h": float(audio_h_per_h),
            "cohort_meta": self.cohort_meta, "worker_pos": worker_pos,
        }
        headers = {"Content-Type": "application/octet-stream", "X-DiLoCo-Meta": json.dumps(meta, sort_keys=True),
                   **self._auth_headers(self.base_round, self.anchor_sha256, payload_sha)}
        req = urllib.request.Request(f"{self.hub_url}/delta", data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise DiLoCoBarrierError(f"POST /delta HTTP {exc.code}: {detail}", status_code=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise DiLoCoBarrierError(f"POST /delta failed: {exc}") from exc

        updated, target_round = False, self.base_round + 1
        if not self.allow_async_experimental:
            deadline = t0 + self.barrier_timeout_s
            while self.base_round < target_round:
                rem = deadline - time.monotonic()
                if rem <= 0.0:
                    raise DiLoCoBarrierError(f"barrier timeout after {self.barrier_timeout_s:.1f}s waiting for round {target_round}")
                rnd, upd = self.pull_anchor(model, force=True, wait_round=target_round, timeout_s=min(rem, 15.0))
                if upd and rnd >= target_round:
                    updated = True
                    break
                sleep_s = min(self.poll_interval_s, max(0.0, deadline - time.monotonic()))
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        elif int(res.get("current_round", self.base_round)) > self.base_round:
            _, updated = self.pull_anchor(model, force=False)

        if updated and broadcast_fn is not None:
            broadcast_fn()
        return {
            "diloco_round": self.base_round, "diloco_anchor_sha256": self.anchor_sha256,
            "diloco_updated": updated, "diloco_triggered": bool(res.get("triggered_outer", False)),
            "diloco_sync_s": round(time.monotonic() - t0, 3),
        }


class TrainerDiLoCoHook:
    """Encapsulates `Trainer.run()` DiLoCo lifecycle & D1 boundary gating so `trainer.py` stays compact."""

    def __init__(self, client: DiLoCoClient | None, every: int) -> None:
        from train.diloco_d1 import DiLoCoBoundaryGate

        self.client, self.every, self.round = client, int(every), None
        self.just_synced, self.gate = False, DiLoCoBoundaryGate()
        self._audio_s, self._t0 = 0.0, time.time()

    @classmethod
    def from_trainer(cls, trainer) -> "TrainerDiLoCoHook":
        from train.diloco_d1 import make_diloco_state, verify_hub_resume_alignment

        cfg = trainer.cfg
        if not (cfg.diloco_hub and cfg.diloco_every > 0):
            return cls(None, 0)
        hook = cls(None, cfg.diloco_every)
        dr = int(trainer.data_rank) if getattr(cfg, "data_world", 0) > 0 else int(os.environ.get("DILOCO_DATA_RANK", str(getattr(trainer, "data_rank", 0))))
        dw = int(trainer.data_world) if getattr(cfg, "data_world", 0) > 0 else int(os.environ.get("DILOCO_DATA_WORLD", str(getattr(trainer, "data_world", 1))))
        if trainer.rank == 0:
            wid = cfg.diloco_worker or os.path.basename(os.path.normpath(trainer.run_dir))
            hook.client = DiLoCoClient(
                cfg.diloco_hub, wid, trainer.model.state_dict(),
                auth_token=os.environ.get("DILOCO_AUTH_TOKEN", ""),
                cohort_meta={"model_config": trainer.model_cfg.to_dict(), "tokenizer_sha256": getattr(trainer.tok, "sha256", None)},
                delta_dtype=cfg.diloco_dtype, data_rank=dr, data_world=dw,
                barrier_timeout_s=float(os.environ.get("DILOCO_BARRIER_TIMEOUT_S", "600.0")),
                trainable_keys={n for n, _ in trainer.model.named_parameters()},
                allow_async_experimental=os.environ.get("DILOCO_ALLOW_ASYNC", "0") == "1",
            )
            rnd, upd = hook.client.pull_anchor(trainer.model, force=True)
            verify_hub_resume_alignment(trainer, rnd, hook.client.anchor_sha256)
            hook.round = rnd
            trainer._diloco_state = make_diloco_state(trainer, rnd, hook.client.anchor_sha256, dr, dw, hook.every)
            trainer.log({"kind": "diloco_init", "step": trainer.step, "diloco_round": rnd,
                         "diloco_anchor_sha256": hook.client.anchor_sha256, "diloco_updated": upd, "hub": cfg.diloco_hub, "worker": wid})
        if trainer.world > 1:
            from train.dist import broadcast_state

            broadcast_state(trainer.model)
        return hook

    def on_step(self, trainer, step_rec: Mapping) -> None:
        self.just_synced = False
        if self.every <= 0:
            return
        self._audio_s += float(step_rec.get("audio_s", 0.0))
        if trainer.step % self.every != 0:
            return
        updated = False
        if trainer.rank == 0 and self.client is not None:
            from train.diloco_d1 import make_diloco_state

            dt = max(time.time() - self._t0, 1e-6)
            pos_dict = trainer.pos.to_dict() if hasattr(trainer, "pos") and hasattr(trainer.pos, "to_dict") else None
            res = self.client.sync(
                trainer.model, local_steps=self.every, audio_hours=self._audio_s / 3600.0,
                local_loss=float(step_rec["loss"]), audio_h_per_h=self._audio_s / dt,
                worker_pos={"step": int(trainer.step), "position": pos_dict},
            )
            self.round, updated, self.just_synced = res["diloco_round"], bool(res["diloco_updated"]), True
            trainer._diloco_state = make_diloco_state(
                trainer, self.round, res["diloco_anchor_sha256"], self.client.data_rank, self.client.data_world, self.every
            )
            trainer.log({"kind": "diloco_sync", "step": trainer.step, **res})
        if trainer.world > 1:
            from train.dist import broadcast_state, broadcast_value

            if bool(broadcast_value(updated, trainer.device)):
                broadcast_state(trainer.model)
                self.just_synced = True
        self._audio_s, self._t0 = 0.0, time.time()

    def log_fields(self) -> dict:
        return {"diloco_round": self.round} if self.round is not None else {}
