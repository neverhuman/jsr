"""Regression checks for the fleet rate and WebSocket contract."""

import asyncio
from collections import deque
import time

import pytest

from tools.training_dashboard_metrics import fleet_summary, host_view, probe_view


def sample(now, audio, wall, wer, *, split="dev-clean", overfit=False, dev_limit=400):
    return {
        "updated_at": now,
        "runs": [{
            "id": "one",
            "name": "Dense",
            "active": True,
            "dev_limit": dev_limit,
            "overfit": overfit,
            "train": [{"step": 50, "time": now, "loss": 1, "audio_s": audio, "wall_s": wall}],
            "eval": [{
                "step": 50, "split": split, "wer": wer, "decoder": "greedy", "ref_words": 100,
                "examples": [{"ref": "reference text", "hyp": "hypothesis text"}],
            }],
        }],
    }


def test_fleet_adds_independent_rates_and_expires_disconnected_hosts():
    a = probe_view(sample(1000, 12000, 60, 9))
    b = probe_view(sample(1000, 3000, 30, 14, split="dev-other"))
    assert "examples" not in a["runs"][0]["eval"][0]
    history = deque(maxlen=6)
    hosts = [host_view(name, row, 1000, "", 1000) for name, row in (("a", a), ("b", b))]
    fleet = fleet_summary(hosts, history, 1000)
    assert fleet["audio_per_second"] == 300
    assert [(row["split"], row["wer"]) for row in fleet["wer_by_split"]] == [
        ("dev-clean", 9), ("dev-other", 14),
    ]
    hosts = [host_view("a", a, 1000, "", 1016), host_view("b", b, 1016, "", 1016)]
    assert hosts[0]["state"] == "offline"
    assert fleet_summary(hosts, history, 1016)["audio_per_second"] == 100


def test_wer_uses_population_and_excludes_overfit_and_invalid_values():
    full = probe_view(sample(1000, 200, 2, 10, dev_limit=2703))
    subset = probe_view(sample(1000, 200, 2, 12))
    overfit = probe_view(sample(1000, 200, 0, 0, overfit=True))
    invalid = probe_view(sample(1000, 200, 2, float("nan")))
    assert overfit["runs"][0]["train"][0]["audio_per_second"] is None
    assert not invalid["runs"][0]["eval"]
    hosts = [host_view(str(index), row, 1000, "", 1000)
             for index, row in enumerate((full, subset, overfit, invalid))]
    best = fleet_summary(hosts, deque(), 1000)["wer_by_split"]
    assert {row["scope"]: row["wer"] for row in best} == {
        "First 400 utterances": 12, "First 2,703 utterances": 10,
    }


def test_rate_includes_wall_time_when_evaluation_resets_the_block_timer():
    payload = sample(1000, 12000, 60, 9)
    payload["runs"][0]["train"].append(
        {"step": 100, "time": 1060, "loss": 1, "audio_s": 6000, "wall_s": 5}
    )
    point = probe_view(payload)["runs"][0]["train"][-1]
    assert point["audio_per_second"] == 100
    assert point["rate_wall_s"] == 60


def test_distributed_workers_follow_fresh_rank_zero_without_double_counting():
    from tools.training_dashboard import Fleet

    now = time.time()
    fleet = Fleet(("xbabe5", "xbabe8", "xbabe9"))
    for host, ranks in (("xbabe5", (0, 1)), ("xbabe8", (3,)), ("xbabe9", (2,))):
        runs = []
        for rank in ranks:
            run = sample(now, 300, 3, 6)["runs"][0]
            run.update(id=f"run-{rank}", ddp={"group": "job", "rank": rank, "world_size": 4,
                                             "started_at": now - 1})
            runs.append(run)
        fleet.feed(host, {"updated_at": now, "runs": runs})
    fleet.pulse()
    assert fleet.summary["nodes_live"] == 3
    assert fleet.summary["active_runs"] == 1
    assert fleet.summary["audio_per_second"] == 100
    assert len(fleet.summary["wer_by_split"]) == 1
    worker = fleet.hosts["xbabe8"]["runs"][0]
    assert worker["shared_from"]["host"] == "xbabe5"
    assert not worker["train"] and not worker["eval"]
    fleet.lost("xbabe5", "Collector disconnected")
    fleet.pulse()
    assert fleet.hosts["xbabe8"]["state"] == "stale"
    assert fleet.hosts["xbabe8"]["runs"][0]["shared_from"] is None
    assert fleet.summary["audio_per_second"] is None


def test_old_leader_log_cannot_mark_workers_live_after_restart():
    from tools.training_dashboard import Fleet

    now = time.time()
    fleet = Fleet(("leader", "worker"))
    for rank, host in enumerate(("leader", "worker")):
        payload = sample(now, 300, 3, 6)
        payload["runs"][0]["ddp"] = {"group": "reused", "rank": rank, "world_size": 2, "started_at": now + 1}
        fleet.feed(host, payload)
    fleet.pulse()
    assert fleet.hosts["leader"]["state"] == fleet.hosts["worker"]["state"] == "stale"


def test_websocket_snapshot_delta_reconnect_and_origin():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from tools.training_dashboard import Fleet, handle_websocket

    async def check():
        fleet = Fleet(("xbabe3",))
        app = web.Application()

        async def socket(request):
            return await handle_websocket(request, fleet, ())

        app.router.add_get("/ws", socket)
        async with TestServer(app) as server, aiohttp.ClientSession() as client:
            url = server.make_url("/ws")
            with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                await client.ws_connect(url, origin="https://another-origin.example")
            assert error.value.status == 403
            async with client.ws_connect(url) as ws:
                initial = await ws.receive_json(timeout=2)
                assert initial["type"] == "snapshot"
                assert initial["hosts"][0]["state"] == "connecting"
                fleet.feed("xbabe3", sample(time.time(), 300, 3, 9))
                fleet.pulse()
                data = await ws.receive_json(timeout=2)
                assert data["type"] == "update"
                assert data["host"]["state"] == "live"
                assert data["fleet"]["audio_per_second"] == 100
                fleet.pulse()
                assert (await ws.receive_json(timeout=2))["type"] == "tick"
            async with client.ws_connect(url) as ws:
                assert (await ws.receive_json(timeout=2))["hosts"][0]["state"] == "live"
                await ws.send_str("change a job")
                assert (await ws.receive(timeout=2)).data == 1008

    asyncio.run(check())
