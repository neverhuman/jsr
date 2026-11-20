import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.fleet_guard import MAX_DUTY, MIN_DUTY, Policy, ema, next_duty  # noqa: E402


def test_duty_falls_when_host_is_over_target_and_rises_with_headroom():
    p = Policy(gpu_target=0.85, cpu_target=0.85)
    assert next_duty(0.8, 0.95, 0.5, p) < 0.8          # GPU over
    assert next_duty(0.8, None, 0.92, p) < 0.8         # CPU over, no GPU
    assert next_duty(0.8, 0.5, 0.5, p) > 0.8           # headroom
    assert next_duty(0.8, 0.82, 0.5, p) == 0.8         # inside the margin: hold
    assert next_duty(MIN_DUTY, 1.0, 1.0, p) == MIN_DUTY
    assert next_duty(MAX_DUTY, 0.1, 0.1, p) == MAX_DUTY


def test_ema():
    assert ema(None, 0.5, 0.3) == 0.5
    assert abs(ema(1.0, 0.0, 0.3) - 0.7) < 1e-12


def _state(pid: int) -> str:
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split(")")[1].split()[0]


def test_guard_pauses_then_always_resumes_the_job():
    job = subprocess.Popen(["sleep", "30"], start_new_session=True)
    g = subprocess.Popen([sys.executable, "-m", "ultra.fleet_guard", "--pgid", str(job.pid), "--gpu", "-1",
                          "--period", "0.4", "--sample", "0.1", "--start-duty", "0.1"],
                         cwd=os.path.join(os.path.dirname(__file__), ".."))
    try:
        seen = set()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            seen.add(_state(job.pid))
            time.sleep(0.02)
        g.send_signal(signal.SIGTERM)
        g.wait(timeout=5)
        after = _state(job.pid)
    finally:
        job.kill()
        job.wait()
    assert "T" in seen, f"the guard never paused the job (states seen: {sorted(seen)})"
    assert after == "S", f"job must be running/sleeping after the guard exits, got {after!r}"


def test_zero_floor_lets_the_job_yield_completely():
    p = Policy(gpu_target=0.74, cpu_target=0.85)
    d = 0.05
    for _ in range(3):
        d = next_duty(d, 0.95, 0.2, p, min_duty=0.0)
    assert d == 0.0
