"""Hardened safetensors + HMAC HTTP server and CLI for live Cross-Node DiLoCo (ULTRA v3)."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import safetensors
from safetensors.torch import load as load_safetensors

from train.diloco_client import EMPTY_SHA256, sign_request
from train.diloco_hub import DiLoCoConflictError, DiLoCoHub, DiLoCoValidationError, load_init_anchor


def _resolve_token(auth_token: str | None, token_file: str = "") -> str:
    if auth_token is not None:
        return auth_token.strip()
    if token_file:
        return Path(token_file).read_text().strip()
    return os.environ.get("DILOCO_AUTH_TOKEN", "").strip()


def make_hub_app(hub: DiLoCoHub, *, auth_token: str | None = None):
    from aiohttp import web

    token = _resolve_token(auth_token)
    if not token:
        raise ValueError("DiLoCo HTTP hub requires non-empty auth_token (--auth-token-file or DILOCO_AUTH_TOKEN)")

    loop_state: dict = {"loop": None, "event": None}

    def _bump_event() -> None:
        ev = loop_state["event"]
        loop_state["event"] = asyncio.Event()
        if ev is not None:
            ev.set()

    def _on_round_commit(_rnd: int) -> None:
        lp = loop_state["loop"]
        if lp is not None and not lp.is_closed():
            lp.call_soon_threadsafe(_bump_event)

    async def _on_startup(_app: web.Application) -> None:
        loop_state["loop"] = asyncio.get_running_loop()
        loop_state["event"] = asyncio.Event()

    hub.add_round_listener(_on_round_commit)

    def _verify_auth(req: web.Request, worker_id: str, round_idx: int, anchor_sha: str, payload_sha: str) -> bool:
        got = req.headers.get("X-DiLoCo-Auth", "")
        if not got:
            return False
        expected = sign_request(token, worker_id, round_idx, anchor_sha, payload_sha)
        return hmac.compare_digest(got, expected)

    def _verify_get_auth(req: web.Request) -> bool:
        wid = req.headers.get("X-DiLoCo-Worker", "")
        anchor_sha = req.headers.get("X-DiLoCo-Anchor-SHA256", "")
        try:
            rnd = int(req.headers.get("X-DiLoCo-Round", "0"))
        except ValueError:
            return False
        return _verify_auth(req, wid, rnd, anchor_sha, EMPTY_SHA256)

    async def get_status(req: web.Request) -> web.Response:
        if not _verify_get_auth(req):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(hub.status())

    async def get_anchor(req: web.Request) -> web.Response:
        if not _verify_get_auth(req):
            return web.json_response({"error": "unauthorized"}, status=401)
        wait_raw = req.query.get("wait_round")
        if wait_raw is not None:
            try:
                target = int(wait_raw)
                wait_s = max(0.0, min(60.0, float(req.query.get("timeout_s", "30.0"))))
            except ValueError as exc:
                return web.json_response({"error": f"invalid wait params: {exc}"}, status=400)
            deadline = time.monotonic() + wait_s
            while hub.round < target:
                rem = deadline - time.monotonic()
                if rem <= 0.0:
                    return web.json_response(
                        {"waiting": True, "current_round": hub.round, "wait_round": target}, status=202
                    )
                ev = loop_state["event"]
                if hub.round >= target or ev is None:
                    break
                try:
                    await asyncio.wait_for(ev.wait(), timeout=rem)
                except asyncio.TimeoutError:
                    break
            if hub.round < target:
                return web.json_response(
                    {"waiting": True, "current_round": hub.round, "wait_round": target}, status=202
                )

        rnd, sha, payload = hub.anchor_state()
        if wait_raw is None and req.headers.get("If-None-Match") in (sha, str(rnd)):
            return web.Response(
                status=304, headers={"X-DiLoCo-Round": str(rnd), "X-DiLoCo-Anchor-SHA256": sha}
            )
        return web.Response(
            body=payload,
            content_type="application/octet-stream",
            headers={
                "X-DiLoCo-Round": str(rnd),
                "X-DiLoCo-Anchor-SHA256": sha,
                "X-DiLoCo-Cohort": hub.cohort,
            },
        )

    async def post_delta(req: web.Request) -> web.Response:
        try:
            meta = json.loads(req.headers.get("X-DiLoCo-Meta", "{}"))
            worker_id = str(meta["worker_id"])
            base_round = int(meta["base_round"])
            anchor_sha = str(meta.get("anchor_sha256", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return web.json_response({"accepted": False, "error": f"invalid X-DiLoCo-Meta: {exc}"}, status=400)

        body = await req.read()
        payload_sha = hashlib.sha256(body).hexdigest()
        if not _verify_auth(req, worker_id, base_round, anchor_sha, payload_sha):
            return web.json_response({"accepted": False, "error": "unauthorized"}, status=401)

        def _process() -> dict:
            delta_sd = load_safetensors(body)
            return hub.submit_delta(
                worker_id=worker_id,
                host=str(meta.get("host", "unknown")),
                base_round=base_round,
                local_steps=int(meta.get("local_steps", 0)),
                audio_hours=float(meta.get("audio_hours", 1.0)),
                local_loss=float(meta.get("local_loss", 0.0)),
                delta_sd=delta_sd,
                cohort_meta=meta.get("cohort_meta"),
                audio_h_per_h=float(meta.get("audio_h_per_h", 0.0)),
                anchor_sha256=anchor_sha,
                data_rank=meta.get("data_rank"),
                data_world=meta.get("data_world"),
                payload_sha256=payload_sha,
                worker_pos=meta.get("worker_pos"),
            )

        try:
            res = await asyncio.to_thread(_process)
        except DiLoCoConflictError as exc:
            return web.json_response({"accepted": False, "error": str(exc)}, status=409)
        except (safetensors.SafetensorError, DiLoCoValidationError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return web.json_response({"accepted": False, "error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response({"accepted": False, "error": f"storage commit failed: {exc}"}, status=500)
        return web.json_response(res)

    app = web.Application(client_max_size=1024 * 1024 * 1024)
    app.on_startup.append(_on_startup)
    app.router.add_get("/status", get_status)
    app.router.add_get("/anchor", get_anchor)
    app.router.add_post("/delta", post_delta)
    return app


def main(argv: list[str] | None = None) -> int:
    from aiohttp import web

    ap = argparse.ArgumentParser(description="Hardened Safetensors DiLoCo Hub (ULTRA v3)")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--init-anchor", default="", help="initial anchor .safetensors or .pt checkpoint")
    ap.add_argument("--work-dir", required=True, help="directory for anchors, outer state, and telemetry")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--roster", default="", help="comma-separated fixed worker roster (required by default)")
    ap.add_argument("--allow-async-experimental", action="store_true", help="opt-in experimental async quorum/BF16")
    ap.add_argument("--auth-token-file", default="", help="path to shared HMAC secret (else DILOCO_AUTH_TOKEN)")
    ap.add_argument("--quorum-k", type=int, default=2)
    ap.add_argument("--quorum-timeout-s", type=float, default=900.0)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--staleness-alpha", type=float, default=0.5)
    ap.add_argument("--min-cosine", type=float, default=-0.1)
    ap.add_argument("--hub-ema-decay", type=float, default=0.0, help="per-round anchor EMA decay d^H (0=disabled)")
    ap.add_argument("--keep-last-k", type=int, default=5)
    ap.add_argument("--keep-every-m", type=int, default=10)
    ap.add_argument("--ema", action="store_true", help="seed anchor from checkpoint EMA weights")
    ap.add_argument("--resume", action="store_true", help="resume from latest durable outer_meta_rN.json")
    a = ap.parse_args(argv)

    token = _resolve_token(None, a.auth_token_file)
    if not token:
        ap.error("HMAC auth token required via --auth-token-file or DILOCO_AUTH_TOKEN")
    roster = tuple(w.strip() for w in a.roster.split(",") if w.strip()) or None
    if roster is None and not a.allow_async_experimental and not a.resume:
        ap.error("--roster is required unless --allow-async-experimental is explicitly set")

    wd = Path(a.work_dir)
    if a.resume and list(wd.glob("outer_meta_r*.json")):
        hub = DiLoCoHub.from_work_dir(wd)
    else:
        if not a.init_anchor:
            ap.error("--init-anchor is required when not resuming an existing work-dir")
        hub = DiLoCoHub(
            a.cohort, load_init_anchor(a.init_anchor), wd,
            expected_roster=roster, allow_async_experimental=a.allow_async_experimental,
            quorum_k=a.quorum_k, quorum_timeout_s=a.quorum_timeout_s,
            outer_lr=a.outer_lr, outer_momentum=a.outer_momentum,
            staleness_alpha=a.staleness_alpha, min_cosine=a.min_cosine,
            use_ema=a.ema, hub_ema_decay=a.hub_ema_decay,
            keep_last_k=a.keep_last_k, keep_every_m=a.keep_every_m,
        )
    mode = "sync-roster" if hub.sync_mode else f"async-experimental(k={hub.quorum_k})"
    print(f"DiLoCoHub [{hub.cohort}] round={hub.round} on http://{a.host}:{a.port} ({mode})", flush=True)
    web.run_app(make_hub_app(hub, auth_token=token), host=a.host, port=a.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
