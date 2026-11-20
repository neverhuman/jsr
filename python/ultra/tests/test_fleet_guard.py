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


def test_guard_always_resumes_the_job_on_exit():
    job = subprocess.Popen(["sleep", "30"], start_new_session=True)
    g = subprocess.Popen([sys.executable, "-m", "ultra.fleet_guard", "--pgid", str(job.pid), "--gpu", "-1",
                          "--period", "0.4", "--sample", "0.1", "--start-duty", "0.3"],
                         cwd=os.path.join(os.path.dirname(__file__), ".."))
    time.sleep(1.0)
    g.send_signal(signal.SIGTERM)
    g.wait(timeout=5)
    with open(f"/proc/{job.pid}/stat") as f:
        state = f.read().split(")")[1].split()[0]
    job.kill()
    job.wait()
    assert state != "T", "job left stopped after the guard exited"
