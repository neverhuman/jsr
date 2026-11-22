from tools.fleet_dashboard.parse import RunState, global_summary, parse_line, parse_nvidia


def test_parse_train_and_eval_lines():
    t = parse_line('{"kind": "train", "step": 50, "pos": {"epoch": 0}, "loss": 2.1, "audio_h_per_h": 372.7, "audio_s": 3600, "time": 10}')
    assert t["k"] == "t" and t["step"] == 50 and t["ahh"] == 372.7 and t["ep"] == 0
    e = parse_line('{"kind": "eval", "step": 5000, "split": "dev-clean", "epoch": 0, "wer": 21.9, "time": 11}')
    assert e == {"k": "e", "step": 5000, "ep": 0, "split": "dev-clean", "wer": 21.9, "time": 11}
    assert parse_line("{torn") is None and parse_line("") is None
    assert parse_line('{"kind": "diverged", "error": "boom"}')["k"] == "div"


def test_runstate_decimates_and_summarises():
    rs = RunState("D2")
    for s in range(1, 4001):
        rs.add({"k": "t", "step": s, "loss": 1.0, "ahh": 300.0, "time": float(s), "lr": 1e-3, "audio_s": 36.0, "ep": 0})
    assert len(rs.train) <= 1500 and rs.train[-1][0] <= 4000
    rs.add({"k": "e", "step": 4000, "split": "dev-clean", "wer": 7.0, "time": 4000.0})
    rs.add({"k": "e", "step": 3000, "split": "dev-clean", "wer": 8.0, "time": 3000.0})
    s = rs.summary(now=4100.0)
    assert s["best"][2] == 7.0 and s["status"] == "running" and round(s["audio_h_total"]) == 40
    assert rs.summary(now=10_000.0)["status"] == "stale"


def test_nvidia_and_global():
    g = parse_nvidia("0, NVIDIA A100-PCIE-40GB, 80, 32000, 40960, 250.1, 61\n1, NVIDIA A100-PCIE-40GB, 40, 8000, 40960, 120, 50\n")
    assert len(g) == 2 and g[0]["name"] == "A100-PCIE-40GB" and g[1]["util"] == 40.0
    nodes = {"a": {"up": True, "ncpu": 100, "cpu_pct": 50.0, "gpus": g}, "b": {"up": False}}
    runs = [{"status": "running", "ahh": 500.0, "best": [1, "dev-clean", 9.0, 0]},
            {"status": "stale", "ahh": 100.0, "best": None}]
    out = global_summary(runs, nodes)
    assert out["audio_h_per_h"] == 500.0 and out["gpus"] == 2 and out["gpus_busy"] == 1
    assert out["gpu_mem_pct"] == round(100 * 40000 / 81920, 1) and out["cpu_pct"] == 50.0 and out["nodes_up"] == 1
