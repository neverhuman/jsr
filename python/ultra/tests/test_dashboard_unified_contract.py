"""Unified contract & adversarial proof tests for BOTH :8766 and :8765 dashboards."""
import asyncio
import json
import time

import pytest

from tools.dashboard_events import (
    DEFAULT_MEASUREMENT_DOMAIN, LEGACY_MEASUREMENT_DOMAIN, MonotonicEventStore, SCHEMA_VERSION,
    annotate_host_hardware, read_timing_status, read_v3_scorecard,
)
from tools.fleet_dashboard.parse import (
    RunState, annotate_node, global_summary, parse_line, parse_nvidia, parse_other_load, parse_timing_lock,
)
from tools.fleet_dashboard.server import Fleet as Fleet8765, make_app as make_app_8765
from tools.training_dashboard import Fleet as Fleet8766, application as make_app_8766, arguments
from tools.training_dashboard_metrics import host_view, probe_view
from tools.training_dashboard_probe import History

FULL_SHA256, FULL_COMMIT, FULL_TREE = "a" * 64, "b" * 40, "c" * 40


def test_unknown_schema_and_rejected_event_never_mutate_run_id_or_state():
    store = MonotonicEventStore()
    ok, _, _ = store.ingest({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_CANON",
                             "attempt_id": 2, "rank": 0, "seq": 10, "step": 500, "planned_exposure_ahh": 100.0}, stream_key="V3_CANON")
    assert ok
    ok_bad, reason, _ = store.ingest({"schema": "ultra.dashboard.event/v99", "kind": "train", "run_id": "POISON",
                                      "attempt_id": 9, "rank": 0, "seq": 99, "step": 9999, "planned_exposure_ahh": 999.0}, stream_key="V3_CANON")
    assert not ok_bad and reason == "unknown_schema"
    ok_old, reason_old, _ = store.ingest({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "OLD_POISON",
                                          "attempt_id": 1, "rank": 0, "seq": 99, "step": 100, "planned_exposure_ahh": 50.0}, stream_key="V3_CANON")
    assert not ok_old and reason_old == "older_attempt"
    view = store.canonical_view("V3_CANON")
    assert view["run_id"] == "V3_CANON" and view["step"] == 500 and view["planned_exposure_ahh"] == 100.0
    assert view["rejected_events"] == 2 and view["rank"] == 0
    assert parse_line(json.dumps({"schema": "ultra.dashboard.event/v99", "kind": "train", "step": 1})) is None
    rs = RunState("V3_CANON")
    assert rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_CANON",
                                         "attempt_id": 2, "rank": 0, "seq": 10, "step": 500, "audio_h_per_h": 100.0, "time": 2000})))
    assert not rs.add({"schema": "ultra.dashboard.event/v99", "k": "t", "run_id": "BAD", "attempt_id": 9, "seq": 99, "step": 999})
    assert not rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "OLD_POISON",
                                             "attempt_id": 1, "rank": 0, "seq": 99, "step": 100, "audio_h_per_h": 50.0, "time": 1900})))
    s = rs.summary(now=2005.0)
    assert s["run_id"] == "V3_CANON" and s["step"] == 500 and s["rank"] == 0 and s["schema"] == SCHEMA_VERSION
    assert [row[0] for row in rs.train] == [500]
    with pytest.raises(ValueError, match="unknown dashboard event schema"):
        probe_view({"schema": "ultra.dashboard.event/v99", "updated_at": 1000, "runs": []})


def test_stale_terminal_preserves_rank_seq_and_cannot_poison_next_attempt():
    rs = RunState("V3_TERM")
    assert rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_TERM",
                                         "attempt_id": 2, "rank": 0, "seq": 10, "step": 500, "audio_h_per_h": 100.0, "time": 2000})))
    stale_div_att1 = parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "diverged", "run_id": "V3_TERM",
                                            "attempt_id": 1, "rank": 0, "seq": 50, "step": 400, "time": 1900, "error": "nan"}))
    assert stale_div_att1["rank"] == 0 and stale_div_att1["seq"] == 50 and not rs.add(stale_div_att1)
    assert not rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "diverged", "run_id": "V3_TERM",
                                             "attempt_id": 2, "rank": 0, "seq": 9, "step": 450, "time": 1990, "error": "nan"})))
    assert rs.summary(now=2010.0)["status"] == "running"
    assert rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "diverged", "run_id": "V3_TERM",
                                         "attempt_id": 2, "rank": 0, "seq": 11, "step": 501, "time": 2005, "error": "boom"})))
    assert rs.summary(now=2010.0)["status"] == "diverged"
    assert rs.add(parse_line(json.dumps({"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_TERM",
                                         "attempt_id": 3, "rank": 0, "seq": 1, "step": 10, "audio_h_per_h": 120.0, "time": 2020})))
    s3 = rs.summary(now=2025.0)
    assert s3["attempt_id"] == 3 and s3["status"] == "running" and s3["step"] == 10 and [r[0] for r in rs.train] == [10]


def test_out_of_order_prior_attempt_rejected_and_history_clean_on_8766_and_8765(tmp_path):
    now = time.time()
    log = tmp_path / "train.jsonl"
    events = [
        {"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_A1", "attempt_id": 2, "rank": 0, "seq": 10,
         "step": 500, "loss": 1.1, "audio_s": 100.0, "wall_s": 1.0, "planned_exposure_ahh": 100.0, "time": now,
         "ckpt_sha256": FULL_SHA256, "git_commit": FULL_COMMIT, "git_tree": FULL_TREE,
         "cohort": {"name": "prod", "diloco_round": 4, "roster": ["xbabe5", "xbabe8"], "mode": "sync_roster"}},
        {"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_A1", "attempt_id": 1, "rank": 0, "seq": 99,
         "step": 100, "loss": 9.9, "audio_s": 50.0, "wall_s": 1.0, "time": now - 100},
        {"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_A1", "attempt_id": 2, "rank": 0, "seq": 9,
         "step": 450, "loss": 8.8, "audio_s": 50.0, "wall_s": 1.0, "time": now - 10},
        {"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_A1", "attempt_id": 2, "rank": 0, "seq": 10,
         "step": 500, "loss": 7.7, "audio_s": 50.0, "wall_s": 1.0, "time": now},
        {"schema": SCHEMA_VERSION, "kind": "train", "run_id": "V3_A1", "attempt_id": 2, "rank": 1, "seq": 10,
         "step": 500, "loss": 1.2, "planned_exposure_ahh": 999.0, "audio_h_per_h": 50.0, "time": now},
        {"schema": SCHEMA_VERSION, "kind": "eval", "run_id": "V3_A1", "attempt_id": 2, "rank": 0, "seq": 11,
         "step": 500, "split": "dev-clean", "wer": 3.12, "s": 42, "d": 5, "i": 3, "ref_words": 1600,
         "measurement_domain": DEFAULT_MEASUREMENT_DOMAIN, "wave_gate": "PASS", "time": now + 1},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    hist = History()
    assert hist.read_log(log) and [r["step"] for r in hist.train.values()] == [500] and [r["step"] for r in hist.global_train_rows()] == [500]
    assert hist.rejected_events == 3 and hist.artifact == {"ckpt_sha256": FULL_SHA256, "git_commit": FULL_COMMIT, "git_tree": FULL_TREE}
    f8766 = Fleet8766(("xbabe5",), diloco_hub="")
    f8766.feed("xbabe5", {"schema": SCHEMA_VERSION, "updated_at": now, "runs": [{
        "schema": SCHEMA_VERSION, "id": "r0", "run_id": "V3_A1", "name": "V3_A1", "active": True,
        "attempt_id": 2, "rank": 0, "world_size": 2, "seq": 11, "artifact": hist.artifact, "cohort": hist.cohort,
        "sdi": hist.sdi, "measurement_domain": hist.measurement_domain, "wave_gate": hist.wave_gate,
        "train": list(hist.train.values()), "eval": list(hist.evals.values()),
    }]})
    f8766.feed("xbabe5", {"schema": SCHEMA_VERSION, "updated_at": now + 2, "runs": [{
        "schema": SCHEMA_VERSION, "id": "r0", "run_id": "V3_A1", "name": "V3_A1", "active": True,
        "attempt_id": 1, "rank": 0, "seq": 99,
        "train": [{"step": 100, "time": now - 100, "loss": 9.9, "audio_s": 50.0, "wall_s": 1.0, "attempt_id": 1, "seq": 99}],
    }]})
    f8766.pulse()
    r8766 = f8766.hosts["xbabe5"]["runs"][0]
    assert r8766["step"] == 500 and r8766["attempt_id"] == 2 and r8766["rank"] == 0 and [r["step"] for r in r8766["train"]] == [500]
    assert r8766["sdi"] == {"s": 42, "d": 5, "i": 3, "ref_words": 1600}
    assert r8766["artifact"] == {"ckpt_sha256": FULL_SHA256, "git_commit": FULL_COMMIT, "git_tree": FULL_TREE}
    assert r8766["cohort"] == {"name": "prod", "diloco_round": 4, "roster": ["xbabe5", "xbabe8"], "mode": "sync_roster"}
    assert r8766["measurement_domain"] == DEFAULT_MEASUREMENT_DOMAIN and r8766["wave_gate"] == "PASS"
    rs = RunState("V3_A1", host="xbabe5")
    for e in events:
        rs.add(parse_line(json.dumps(e)))
    s8765 = rs.summary(now=now + 2)
    assert s8765["schema"] == SCHEMA_VERSION and s8765["step"] == 500 and s8765["rank"] == 0
    assert s8765["planned_exposure_ahh"] == 100.0 and s8765["rank_local_ahh"] == {1: 999.0} and [r[0] for r in rs.train] == [500]
    assert s8765["sdi"] == {"s": 42, "d": 5, "i": 3, "ref_words": 1600}
    assert s8765["artifact"] == {"ckpt_sha256": FULL_SHA256, "git_commit": FULL_COMMIT, "git_tree": FULL_TREE}
    assert s8765["cohort_meta"] == {"name": "prod", "diloco_round": 4, "roster": ["xbabe5", "xbabe8"], "mode": "sync_roster"}


def test_legacy_sdi_preserved_and_same_host_multi_rank_never_double_counts():
    now = time.time()
    leg = {"kind": "eval", "step": 200, "split": "dev-clean", "wer": 5.0, "sub": 10, "del": 2, "ins": 1, "ref_words": 260, "time": now}
    rs = RunState("LEG")
    rs.add(parse_line(json.dumps(leg)))
    s_leg = rs.summary(now=now + 1)
    assert s_leg["sdi"] == {"s": 10, "d": 2, "i": 1, "ref_words": 260} and s_leg["measurement_domain"] == LEGACY_MEASUREMENT_DOMAIN
    pv = probe_view({"updated_at": now, "runs": [{"id": "leg", "name": "LEG", "active": True, "eval": [leg]}]})
    assert pv["runs"][0]["sdi"] == {"s": 10, "d": 2, "i": 1, "ref_words": 260} and pv["runs"][0]["measurement_domain"] == LEGACY_MEASUREMENT_DOMAIN
    f8766 = Fleet8766(("xbabe5",), diloco_hub="")
    f8766.feed("xbabe5", {"schema": SCHEMA_VERSION, "updated_at": now, "runs": [
        {"id": "d1-r0", "run_id": "D1", "name": "D1_rank0", "rank": 0, "world_size": 2, "active": True,
         "train": [{"step": 50, "time": now, "loss": 1.0, "audio_s": 100.0, "wall_s": 1.0}]},
        {"id": "d1-r1", "run_id": "D1", "name": "D1_rank1", "rank": 1, "world_size": 2, "active": True,
         "train": [{"step": 50, "time": now, "loss": 1.0, "audio_s": 100.0, "wall_s": 1.0}]},
    ]})
    f8766.pulse()
    assert f8766.hosts["xbabe5"]["audio_per_second"] == 100.0 and f8766.summary["active_runs"] == 1 and f8766.summary["audio_per_second"] == 100.0
    g8765 = global_summary([
        {"name": "D1_rank0", "run_id": "D1", "rank": 0, "status": "running", "cohort": "prod", "ahh": 100.0},
        {"name": "D1_rank1", "run_id": "D1", "rank": 1, "status": "running", "cohort": "prod", "ahh": 100.0},
    ], {})
    assert g8765["live_runs"] == 1 and g8765["audio_h_per_h"] == 100.0


def test_dual_gpu_imbalance_target_bands_and_cpu_load_gauge_on_both_dashboards():
    now = time.time()
    pv = probe_view({"schema": SCHEMA_VERSION, "updated_at": now, "cpu_load": 16.0, "cpu_count": 32, "runs": [], "gpus": [
        {"index": 0, "name": "A100", "util_pct": 0.0, "memory_used_mib": 100, "memory_total_mib": 40960},
        {"index": 1, "name": "A100", "util_pct": 84.0, "memory_used_mib": 28000, "memory_total_mib": 40960},
    ]})
    hv = host_view("xbabe5", pv, now, "", now)
    assert hv["dual_gpu_imbalance"] is True and hv["cpu_load_pct"] == 50.0
    assert [(g["index"], g["util_pct"], g["target_band"]) for g in hv["gpus"]] == [(0, 0.0, "UNDER <70%"), (1, 84.0, "TARGET 80-85%")]
    gpus_8765 = parse_nvidia("0, NVIDIA A100, 0, 100, 40960, 40, 35\n1, NVIDIA A100, 84, 28000, 40960, 220, 62\n")
    node = annotate_node({"name": "xbabe5", "up": True, "load": 16.0, "ncpu": 32, "gpus": gpus_8765})
    assert node["dual_gpu_imbalance"] is True and node["cpu_load_pct"] == 50.0
    assert [(g["index"], g["util"], g["target_band"]) for g in node["gpus"]] == [(0, 0.0, "UNDER <70%"), (1, 84.0, "TARGET 80-85%")]
    assert annotate_host_hardware([{"index": 0, "util_pct": 95.0}])["gpus"][0]["target_band"] == "OVER >90%"
    assert annotate_host_hardware([{"index": 0, "util_pct": 50.0}], other_load={"gpu": True})["gpus"][0]["target_band"] == "OTHER PROJECT"
    assert annotate_host_hardware([{"index": 0, "util_pct": 0.0}], timing_lock={"status": "held"})["gpus"][0]["target_band"] == "QUIET TIMING"


def test_http_ws_zero_runs_scorecard_timing_other_project_and_diloco_hub(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    scorecard_file = tmp_path / "scoreboard.json"
    scorecard_file.write_text(json.dumps({
        "schema": "ultra.v3.scoreboard/v1", "version": 20, "generated_utc": "2026-11-23T07:48:27Z",
        "checked_utc": "2026-11-23T07:52:37Z", "label": "Exploratory, LibriSpeech DEV only",
        "evidence_classes": {"measured": "receipt", "estimated": "basis", "pending": "owner"},
        "columns": [
            {"key": "wer_pooled", "label": "WER pooled", "unit": "%", "better": "lower", "group": "accuracy"},
            {"key": "wer_clean", "label": "WER dev-clean", "unit": "%", "better": "lower", "group": "accuracy"},
            {"key": "wer_other", "label": "WER dev-other", "unit": "%", "better": "lower", "group": "accuracy"},
            {"key": "ftl_end_ms", "label": "First token, end clock", "unit": "ms", "better": "lower", "group": "latency"},
            {"key": "ftl_onset_ms", "label": "First token, onset clock", "unit": "ms", "better": "lower", "group": "latency"},
            {"key": "x_rt", "label": "× real time", "unit": "×", "better": "higher", "group": "cost"},
            {"key": "w_per_stream", "label": "W per stream", "unit": "W", "better": "lower", "group": "cost"},
        ],
        "headline": {"best_wer_native_engine": {"row": "b0", "name": "B0 continuous", "value": 8.272, "display": "8.272", "unit": "%", "evidence": "measured", "row_wer_display": "8.272"}},
        "release_gate": {"target_edits": 7895, "best_native_candidate": {"row": "b0", "status": "PASS", "progress_pct": 100.0}},
        "rows": [{"id": "b0", "name": "B0 continuous", "labels": ["exploratory, DEV only"], "wer_guard": "baseline", "cells": {
            "wer_pooled": {"value": 8.272, "display": "8.272", "evidence": "measured", "receipts": [{"path": "evals/b0.json", "sha256": FULL_SHA256}]},
            "wer_clean": {"value": 4.26, "display": "4.26", "evidence": "measured"},
            "wer_other": {"value": 11.86, "display": "11.86", "evidence": "measured"},
            "ftl_end_ms": {"value": 312.2, "display": "312", "evidence": "measured", "inherited_from": "a1l", "note": "design clock"},
            "ftl_onset_ms": {"value": 581.1, "display": "581", "evidence": "measured"},
            "x_rt": {"value": 28.45, "display": "28.45×", "evidence": "estimated", "range_display": "27–29×", "basis": "scaled"},
            "w_per_stream": {"value": None, "display": "pending", "evidence": "pending", "owner": "bench-lane", "eta_utc": "2026-11-23T08:00Z"},
        }}],
    }))
    bad_sc = tmp_path / "bad_scoreboard.json"
    bad_sc.write_text(json.dumps({"schema": "ultra.v3.scoreboard/v99", "version": 1, "rows": []}))
    no_schema_sc = tmp_path / "no_schema_scoreboard.json"
    no_schema_sc.write_text(json.dumps({"version": 1, "rows": []}))
    assert read_v3_scorecard(bad_sc) is None and read_v3_scorecard(no_schema_sc) is None
    opp = tmp_path / "opponents"
    (opp / "TIMING.lock").mkdir(parents=True)
    (opp / "TIMING.lock" / "owner").write_text("bench-open 14d\n")
    (opp / "TIMING.quiet").write_text("quiet cpu=1%\n")
    assert read_timing_status(opp) == {"status": "held", "owner": "bench-open 14d", "quiet": "quiet cpu=1%"}
    sc_loaded = read_v3_scorecard(scorecard_file)
    assert sc_loaded["schema"] == "ultra.v3.scoreboard/v1" and sc_loaded["version"] == 20
    assert sc_loaded["rows"][0]["cells"]["ftl_end_ms"]["display"] == "312" and sc_loaded["rows"][0]["cells"]["w_per_stream"]["evidence"] == "pending"
    assert "cotenant-worker" in parse_other_load("OTHER_GPU 24000 cotenant-worker\n", [], set())["label"]
    assert parse_timing_lock("TIMING_LOCK held bench-open\nTIMING_QUIET quiet\n")["status"] == "held"

    async def check():
        async def hub_status(_r):
            return web.json_response({
                "cohort": "V3_Sync", "round": 5, "quorum_k": 2, "pending_count": 1,
                "expected_roster": ["xbabe5", "xbabe8"], "total_audio_h": 42.0, "effective_audio_h_per_h": 6200.0,
                "workers": {"xbabe5": {"host": "xbabe5", "base_round": 5, "staleness": 0, "status": "synced", "delta_norm": 0.8, "cosine": 0.91}},
                "recent_rounds": [{"round": 5, "mean_cos": 0.91, "quarantined": []}],
            })

        hub_app = web.Application()
        hub_app.router.add_get("/status", hub_status)
        async with TestServer(hub_app) as hub_srv:
            hub_url = str(hub_srv.make_url("/")).rstrip("/")
            args = arguments(["--hosts", "xbabe0", "xbabe5", "--scoreboard", str(scorecard_file), "--diloco-hub", hub_url])
            app8766 = make_app_8766(args, start_collectors=False, run_root=tmp_path)
            f65 = Fleet8765(scorecard_path=scorecard_file, diloco_hub=hub_url, run_root=tmp_path)
            app8765 = make_app_8765(fleet=f65, start_background_tasks=False)
            f65.scan_diloco()
            await f65.poll_diloco_hub()
            async with TestServer(app8766) as s66, TestServer(app8765) as s65, aiohttp.ClientSession() as cli:
                await app8766["fleet"].poll_v3(cli)
                app8766["fleet"].pulse()
                html66 = await (await cli.get(s66.make_url("/"))).text()
                assert "ULTRA v3 Best-Model Scorecard" in html66 and "(sha256 \"+String(r.sha256||\"\").slice(0,12)" in html66
                assert "¹ inherited from base row · † 100 MFA clips" in html66 and "cell.inherited_from?\"¹\":\"\"" in html66
                snap66 = await (await cli.get(s66.make_url("/api/snapshot"))).json()
                sc66 = await (await cli.get(s66.make_url("/v3/scoreboard.json"))).json()
                assert snap66["schema"] == SCHEMA_VERSION and snap66["fleet"]["active_runs"] == 0
                assert sc66["schema"] == "ultra.v3.scoreboard/v1" and sc66["version"] == 20 and sc66["checked_utc"] == "2026-11-23T07:52:37Z"
                assert sc66["rows"][0]["cells"]["wer_pooled"]["receipts"][0]["sha256"] == FULL_SHA256
                assert sc66["rows"][0]["cells"]["wer_pooled"]["display"] == "8.272" and snap66["diloco"][0]["cohort"] == "V3_Sync"
                html65 = await (await cli.get(s65.make_url("/"))).text()
                assert "ULTRA v3 Best-Model Scorecard" in html65 and "(sha256 \"+String(r.sha256||\"\").slice(0,12)" in html65
                assert "¹ inherited from base row · † 100 MFA clips" in html65 and "cell.inherited_from?\"¹\":\"\"" in html65
                snap65 = await (await cli.get(s65.make_url("/api/snapshot"))).json()
                sc65 = await (await cli.get(s65.make_url("/v3/scoreboard.json"))).json()
                assert snap65["schema"] == SCHEMA_VERSION and snap65["global"]["live_runs"] == 0
                assert sc65 == sc66 and snap65["v3_scorecard"]["evidence_classes"]["measured"] == "receipt"
                app8766["fleet"].scorecard_path = tmp_path / "missing.json"
                assert (await cli.get(s66.make_url("/v3/scoreboard.json"))).status == 503
                async with cli.ws_connect(s66.make_url("/ws")) as ws66, cli.ws_connect(s65.make_url("/ws")) as ws65:
                    m66, m65 = await ws66.receive_json(timeout=2), await ws65.receive_json(timeout=2)
                    assert m66["schema"] == m65["schema"] == SCHEMA_VERSION
                    assert m66["v3_scorecard"]["version"] == m65["v3_scorecard"]["version"] == 20
                    assert m66["diloco"][0]["mode_label"] == m65["diloco"][0]["mode_label"] == "Sync-FP32 Roster"

    asyncio.run(check())
