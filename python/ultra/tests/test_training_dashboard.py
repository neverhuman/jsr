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


def test_probe_attributes_same_host_ddp_ranks_to_both_gpus(monkeypatch, tmp_path, capsys):
    import json
    from types import SimpleNamespace

    from tools import training_dashboard_probe as probe
    from tools.training_dashboard import Fleet

    now = time.time()
    run_dir = tmp_path / "shared"
    run_dir.mkdir()
    train = {"kind": "train", "step": 100, "time": now, "audio_s": 300, "wall_s": 3}
    (run_dir / "train.jsonl").write_text(json.dumps(train) + "\n")
    listing = "\n".join(
        f"{pid} {parent} Mon Nov 23 06:01:00 2026 python3 -m train.trainer --run-dir {run_dir}"
        for pid, parent in ((101, 99), (100, 99), (102, 101))
    )
    devices = "\n".join(f"{index}, GPU-test{index}, A100, 83, 15000, 40960, 220" for index in (0, 1))

    def command(args, timeout=1.5):
        if args[0] == "ps":
            return listing
        if args[1].startswith("--query-gpu="):
            return devices
        assert args[1] == "--query-compute-apps=pid,gpu_uuid"
        return "100, GPU-test0\n101, GPU-test1"

    monkeypatch.setattr(probe, "command", command)
    monkeypatch.setattr(probe, "linux_process", lambda pid: (
        ["python3", "-m", "train.trainer", "--run-dir", str(run_dir)], str(pid)
    ))
    monkeypatch.setattr(probe.shutil, "which", lambda name: name)
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe, "distributed_process", lambda pid, start: {
        "group": "job", "rank": pid - 100, "world_size": 2, "started_at": now - 1
    })

    probe.live(SimpleNamespace(run_dir=[], disk_path=None, once=True))
    payload = json.loads(capsys.readouterr().out)
    assert sorted(run["ddp"]["rank"] for run in payload["runs"]) == [0, 1]
    fleet = Fleet(("xbabe5",))
    fleet.feed("xbabe5", payload)
    fleet.pulse()
    host = fleet.hosts["xbabe5"]
    assert [(gpu["index"], gpu["uuid"]) for gpu in host["gpus"]] == [(0, "GPU-test0"), (1, "GPU-test1")]
    leader, follower = sorted(host["runs"], key=lambda run: run["ddp"]["rank"])
    assert (leader["gpu_index"], leader["gpu_uuid"]) == (0, "GPU-test0")
    assert (follower["gpu_index"], follower["gpu_uuid"]) == (1, "GPU-test1")
    assert leader["id"] != follower["id"]
    assert follower["shared_from"]["id"] == leader["id"]
    assert not follower["train"]
    assert fleet.summary["active_runs"] == 1
    assert fleet.summary["audio_per_second"] == 100


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


def test_probe_reads_full_run_from_step_zero_and_attaches_train_to_eval(tmp_path):
    import json
    from tools.training_dashboard_probe import History

    log = tmp_path / "train.jsonl"
    lines = [json.dumps({"kind": "start", "step": 0, "time": 100.0})]
    for step in range(1, 1501):
        lines.append(json.dumps({
            "kind": "train", "step": step, "time": 100.0 + step,
            "loss": round(5.0 / (1.0 + step / 100.0), 4),
            "rnnt": round(3.0 / (1.0 + step / 100.0), 4),
            "ctc": round(2.0 / (1.0 + step / 100.0), 4),
            "audio_s": 100.0, "wall_s": 1.0, "epoch": step // 500,
        }))
        if step in (100, 500, 1000, 1500):
            lines.append(json.dumps({
                "kind": "eval", "step": step, "time": 100.5 + step,
                "split": "dev-clean", "decoder": "greedy",
                "wer": 25.0 - step / 100.0, "ema_wer": 24.5 - step / 100.0,
                "ref_words": 4000,
            }))
    log.write_text("\n".join(lines) + "\n")
    history = History()
    assert history.read_log(log)
    global_rows = history.global_train_rows()
    assert global_rows[0]["step"] == 1
    assert global_rows[-1]["step"] == 1500
    assert len(global_rows) <= 401
    assert [row["step"] for row in global_rows] == sorted(row["step"] for row in global_rows)
    assert len(history.train) == 160
    eval_rows = sorted(history.evals.values(), key=lambda r: r["step"])
    assert len(eval_rows) == 4
    assert eval_rows[-1]["ema_wer"] == pytest.approx(9.5)
    assert eval_rows[-1]["train_loss"] is not None


def test_overfit_summary_detects_improving_vs_overfitting():
    now = 2000.0
    improving = {
        "updated_at": now,
        "runs": [{
            "id": "imp", "name": "Improving", "active": True,
            "train_global": [
                {"step": 100, "time": now - 30, "loss": 2.0, "audio_s": 100, "wall_s": 1},
                {"step": 200, "time": now - 20, "loss": 1.5, "audio_s": 100, "wall_s": 1},
                {"step": 300, "time": now - 10, "loss": 1.1, "audio_s": 100, "wall_s": 1},
            ],
            "train": [{"step": 300, "time": now - 10, "loss": 1.1, "audio_s": 100, "wall_s": 1}],
            "eval": [
                {"step": 100, "split": "dev-clean", "decoder": "greedy", "wer": 12.0, "ref_words": 4000},
                {"step": 200, "split": "dev-clean", "decoder": "greedy", "wer": 9.5, "ref_words": 4000},
                {"step": 300, "split": "dev-clean", "decoder": "greedy", "wer": 7.8, "ref_words": 4000},
            ],
        }],
    }
    imp_run = probe_view(improving)["runs"][0]
    assert imp_run["overfit_check"]["status"] == "improving"
    assert imp_run["overfit_check"]["best_wer"] == 7.8

    overfitting = {
        "updated_at": now,
        "runs": [{
            "id": "ovf", "name": "Overfit", "active": True,
            "train_global": [
                {"step": 100, "time": now - 30, "loss": 1.0, "audio_s": 100, "wall_s": 1},
                {"step": 200, "time": now - 20, "loss": 0.7, "audio_s": 100, "wall_s": 1},
                {"step": 300, "time": now - 10, "loss": 0.4, "audio_s": 100, "wall_s": 1},
            ],
            "train": [{"step": 300, "time": now - 10, "loss": 0.4, "audio_s": 100, "wall_s": 1}],
            "eval": [
                {"step": 100, "split": "dev-clean", "decoder": "greedy", "wer": 6.0, "ref_words": 4000},
                {"step": 200, "split": "dev-clean", "decoder": "greedy", "wer": 6.5, "ref_words": 4000},
                {"step": 300, "split": "dev-clean", "decoder": "greedy", "wer": 7.1, "ref_words": 4000},
            ],
        }],
    }
    ovf_run = probe_view(overfitting)["runs"][0]
    assert ovf_run["overfit_check"]["status"] == "overfitting"
    assert ovf_run["overfit_check"]["delta_pp"] == pytest.approx(1.1)
    assert ovf_run["overfit_check"]["train_loss_ratio"] == pytest.approx(0.4)
