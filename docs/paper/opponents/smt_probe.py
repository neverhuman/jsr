"""Probe whether two vCPUs share a physical core: time a fixed loop alone and beside a busy neighbour."""
import os, subprocess, sys, time

def work():
    t = time.perf_counter(); s = 0
    for i in range(12_000_000):
        s += i * i
    return time.perf_counter() - t

if len(sys.argv) == 2 and sys.argv[1] == "spin":
    while True:
        pass
a, b = int(sys.argv[1]), int(sys.argv[2])
os.sched_setaffinity(0, {a})
alone = min(work() for _ in range(3))
spin = subprocess.Popen([sys.executable, __file__, "spin"], preexec_fn=lambda: os.sched_setaffinity(0, {b}))
time.sleep(0.3)
beside = min(work() for _ in range(3))
spin.kill(); spin.wait()
print(f"cpu{a} alone {alone:.3f}s, beside busy cpu{b} {beside:.3f}s, slowdown {beside/alone:.2f}x")
