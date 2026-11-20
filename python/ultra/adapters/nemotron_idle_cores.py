"""Print per-CPU busy fraction over a short window from /proc/stat (idle-core picking for timing)."""
import sys
import time


def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out


def busy(window_s: float = 5.0) -> dict[int, float]:
    a = snap()
    time.sleep(window_s)
    b = snap()
    return {c: 1.0 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in a}


if __name__ == "__main__":
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    for c, u in sorted(busy(w).items(), key=lambda x: x[1]):
        print(c, round(u, 3))
