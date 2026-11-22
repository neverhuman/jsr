from tools.fleet_dashboard.parse import RunState, classify_cohort, global_summary, parse_line, parse_nvidia
from tools.fleet_dashboard.remote_tail import infer_total_steps_from_cmd, parse_remote_payload
from tools.fleet_dashboard.server import host_for


def test_parse_train_and_eval_lines():
    t = parse_line(
        '{"kind": "train", "step": 50, "pos": {"epoch": 0}, "loss": 2.1, "rnnt": 0.37, '
        '"audio_h_per_h": 372.7, "audio_s": 3600, "pad_pct": 4.8, "data_wait_pct": 2.3, '
        '"max_qkv_norm": 94.1, "time": 10}'
    )
    assert t["k"] == "t" and t["step"] == 50 and t["ahh"] == 372.7 and t["ep"] == 0
    assert t["rnnt"] == 0.37 and t["pad_pct"] == 4.8 and t["wait_pct"] == 2.3 and t["qkv"] == 94.1
    e = parse_line('{"kind": "eval", "step": 5000, "split": "dev-clean", "epoch": 0, "wer": 6.99, "ema_wer": 6.56, "time": 11}')
    assert e == {"k": "e", "step": 5000, "ep": 0, "split": "dev-clean", "wer": 6.99, "ema_wer": 6.56, "time": 11}
    assert parse_line("{torn") is None and parse_line("") is None
    assert parse_line('{"kind": "diverged", "error": "boom"}')["k"] == "div"


def test_runstate_decimates_and_summarises():
    rs = RunState("D2c_S_warm_pruned_960h", host="xbabe5")
    for s in range(1, 4001):
        rs.add({"k": "t", "step": s, "loss": 1.6, "rnnt": 0.35, "ahh": 2000.0, "time": float(s), "lr": 2e-4, "audio_s": 36.0, "ep": 0})
    assert len(rs.train) <= 1500 and rs.train[-1][0] <= 4000 and rs.train[-1][5] == 0.35
    rs.add({"k": "e", "step": 4000, "split": "dev-clean", "wer": 6.99, "ema_wer": 6.56, "time": 4000.0})
    rs.add({"k": "e", "step": 3000, "split": "dev-clean", "wer": 8.0, "time": 3000.0})
    s = rs.summary(now=4100.0)
    assert s["cohort"] == "prod" and s["best"][2] == 6.99 and s["best_ema"][4] == 6.56
    assert s["status"] == "running" and round(s["audio_h_total"]) == 40
    assert rs.summary(now=10_000.0)["status"] == "stopped"


def test_cohort_host_rules_and_remote_tail():
    assert classify_cohort("D2c_S_warm_pruned_960h") == "prod"
    assert classify_cohort("L1w_S_spm512_warmH13a_960h") == "prod"
    assert classify_cohort("H23a_adamw") == "ablation"
    assert classify_cohort("L3_M_MR_spm512_pruned_960h") == "scratch"
    assert host_for("D2c_S_warm_pruned_960h") == "xbabe5"
    assert host_for("H23a_adamw") == "xbabe6"
    assert host_for("L1w_S_spm512_warmH13a_960h") == "xbabe8"
    assert infer_total_steps_from_cmd("python -m train.trainer --set total_steps=24000 lr=1e-3") == 24000
    assert parse_remote_payload('[{"run":"H23a_adamw","off":120,"lines":[]}]')[0]["run"] == "H23a_adamw"
    assert parse_remote_payload("broken") == []


def test_nvidia_and_global():
    g = parse_nvidia("0, NVIDIA A100-PCIE-40GB, 80, 32000, 40960, 250.1, 61\n1, NVIDIA A100-PCIE-40GB, 40, 8000, 40960, 120, 50\n")
    assert len(g) == 2 and g[0]["name"] == "A100-PCIE-40GB" and g[1]["util"] == 40.0
    nodes = {"a": {"up": True, "ncpu": 100, "cpu_pct": 50.0, "gpus": g}, "b": {"up": False}}
    runs = [
        {"status": "running", "cohort": "prod", "ahh": 500.0, "best": [1, "dev-clean", 6.9, 0, 6.5], "best_ema": [1, "dev-clean", 6.9, 0, 6.5]},
        {"status": "stale", "cohort": "ablation", "ahh": 100.0, "best": None, "best_ema": None},
    ]
    out = global_summary(runs, nodes)
    assert out["audio_h_per_h"] == 500.0 and out["gpus"] == 2 and out["gpus_busy"] == 1
    assert out["best_wer"] == 6.5 and out["best_prod_wer"] == 6.5
    assert out["gpu_mem_pct"] == round(100 * 40000 / 81920, 1) and out["cpu_pct"] == 50.0 and out["nodes_up"] == 1


def test_live_hosts_status_and_idle_nodes():
    from tools.fleet_dashboard.parse import parse_live_runs

    probe = "CPU 500\nRUN D2c_S_warm_pruned_960h\nRUN [^\nRUN D2c_x5g1\n"
    assert parse_live_runs(probe) == {"D2c_S_warm_pruned_960h", "D2c_x5g1"}
    rs = RunState("D2c_S_warm_pruned_960h", host="xbabe5")
    rs.add({"k": "t", "step": 1, "loss": 1.0, "ahh": 5000.0, "time": 1000.0, "lr": 1e-4, "audio_s": 1.0, "ep": 0})
    rs.live_hosts.update({"xbabe9", "xbabe5", "xbabe8"})
    s = rs.summary(now=1100.0)
    assert s["status"] == "running" and s["host"] == "xbabe5,xbabe8,xbabe9" and s["hosts"] == ["xbabe5", "xbabe8", "xbabe9"]
    assert rs.summary(now=1000.0 + 601)["status"] == "stalled"  # live process, silent log
    rs.live_hosts.clear()
    assert rs.summary(now=1100.0)["status"] == "running"  # fresh log, process not seen (e.g. unprobed host)
    assert rs.summary(now=1000.0 + 301)["status"] == "stopped"
    nodes = {"x8": {"up": True, "gpus": [{"util": 0.0, "mem_used": 0, "mem_total": 1}], "runs": []},
             "x9": {"up": True, "gpus": [{"util": 90.0, "mem_used": 1, "mem_total": 1}], "runs": ["D2c"]}}
    g = global_summary([s], nodes)
    assert g["idle_nodes"] == ["x8"] and g["stalled"] == []


def test_starting_and_authoritative_inventory():
    from tools.fleet_dashboard.parse import classify_cohort

    assert classify_cohort("D2d_S_blend_960h") == "prod"
    old = RunState("D2c")
    old.add({"k": "t", "step": 9, "loss": 1.0, "ahh": 3000.0, "time": 1000.0, "lr": 1e-4, "audio_s": 1.0, "ep": 0})
    assert old.summary(now=1100.0, inventory_complete=True)["status"] == "stopped"  # fresh log, no process anywhere
    assert old.summary(now=1100.0, inventory_complete=False)["status"] == "running"  # a node is down: trust the log
    new = RunState("D2d")
    new.set_live("xbabe5", True, now=2000.0)
    assert new.summary(now=2100.0)["status"] == "starting"  # live, no block logged yet
    assert new.summary(now=2000.0 + 601)["status"] == "stalled"
    new.add({"k": "t", "step": 1, "loss": 1.0, "ahh": 6000.0, "time": 2650.0, "lr": 1e-4, "audio_s": 1.0, "ep": 0})
    assert new.summary(now=2660.0)["status"] == "running"


def test_first_probe_does_not_mark_starting():
    rs = RunState("D3MRc")
    rs.add({"k": "t", "step": 5, "loss": 1.0, "ahh": 500.0, "time": 1000.0, "lr": 1e-4, "audio_s": 1.0, "ep": 0})
    rs.set_live("xbabe1", True, now=0.0)  # dashboard just started: launch time unknown
    assert rs.summary(now=1200.0, inventory_complete=True)["status"] == "running"


def test_rank_heartbeat_summary():
    from tools.fleet_dashboard.parse import summarize_rank

    r = summarize_rank({"run": "D2d_S_960h", "rank": 3, "world": 4, "gpu": 0, "step": 500, "time": 990.0,
                        "audio_s": 3600.0 * 5, "interval_s": 3.6, "loss": 1.4, "compute_s": 2.1, "comm_s": 0.9,
                        "wait_s": 0.1, "sleep_s": 0.5}, now=1000.0)
    assert r["ahh"] == 5000.0 and r["rank"] == 3 and r["age"] == 10.0 and r["comm_s"] == 0.9
