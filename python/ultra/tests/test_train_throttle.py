from train.dist import StepThrottle


def test_throttle_uses_time_averaged_util_and_live_target(tmp_path):
    t = StepThrottle(0.0, 0, every=1, live_dir=str(tmp_path))  # target 0: no sampler thread
    assert t.sleep_for(1, 1.0) == 0.0
    t.target = 0.85
    t.samples.extend([1.0] * 20)  # busy: duty steps down, sleep appears
    t.sleep_for(1, 1.0)
    assert t.duty < 1.0 and t.sleep_for(2, 1.0) > 0
    t.samples.clear()
    t.samples.extend([0.70] * 20)  # well under target: duty recovers to 1 and the sleep vanishes
    for k in range(3, 40):
        t.sleep_for(k, 1.0)
    assert t.duty == 1.0 and t.sleep_for(41, 1.0) == 0.0
    (tmp_path / "UTIL_TARGET").write_text("0.6\n")  # live override
    t.sleep_for(42, 1.0)
    assert t.target == 0.6 and t.duty < 1.0
    t.pause(0.01)
    assert t.stats()["slept_s"] > 0 and t.stats()["target"] == 0.6


def test_physical_gpu_follows_cuda_visible_devices(monkeypatch):
    from train.dist import physical_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert physical_gpu(0) == "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert physical_gpu(1) == "3"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert physical_gpu(0) == "0"
