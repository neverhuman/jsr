"""Generate the Z1 RMS witness fixture with the reference arithmetic.

The normalisation block is copied from the z1-rc2 / z1-clean-guard probe scripts
(`run_z1.py`, `run_guard.py`). Output: z1.bin (int16 LE: for each case, input then expected
output) and z1.tsv (name, n, gain, clipped, in_offset_samples, out_offset_samples).
Run: python3 gen_z1_fixture.py   (numpy required)
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def z1(x):
    f = x.astype(np.float64)
    rms = np.sqrt(np.mean(f * f))
    gain = float(np.clip(1928 / max(rms, 1e-9), 10 ** (-12 / 20), 10 ** (40 / 20)))
    y = np.round(f * gain)
    n_clipped = int(np.count_nonzero((y > 32767) | (y < -32768)))
    return np.clip(y, -32768, 32767).astype(np.int16), gain, n_clipped


def xorshift_signal(n, seed=0x9E3779B97F4A7C15):
    m = (1 << 64) - 1
    s, out = seed, []
    for _ in range(n):
        s ^= s >> 12
        s ^= (s << 25) & m
        s ^= s >> 27
        v = ((s * 0x2545F4914F6CDD1D) & m) >> 48
        v = v - 65536 if v >= 32768 else v
        out.append(int(v / 8))  # truncate toward zero, like Rust i16 / 8
    return np.array(out, dtype=np.int16)


def sine(n, amp, hz=440.0):
    return np.rint(amp * np.sin(2 * np.pi * hz * np.arange(n) / 16000)).astype(np.int16)


burst = np.where(np.arange(4000) % 2 == 0, 100, -100).astype(np.int16)
burst[::400] = np.where(np.arange(0, 4000, 400) % 800 == 0, 32767, -32768)
cases = {
    "silence": np.zeros(4000, dtype=np.int16),
    "dc": np.full(8000, 1000, dtype=np.int16),
    "clipped_sine": np.clip(np.rint(60000 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)), -32768, 32767).astype(np.int16),
    "quiet_sine_m30db": sine(8000, 32767 * 10 ** (-30 / 20)),
    "rail_burst": burst,
    "random_3s": xorshift_signal(48000),
}
chunks, rows, pos = [], [], 0
for name, x in cases.items():
    y, gain, clipped = z1(x)
    rows.append(f"{name}\t{len(x)}\t{gain!r}\t{clipped}\t{pos}\t{pos + len(x)}")
    chunks += [x, y]
    pos += 2 * len(x)
np.concatenate(chunks).astype("<i2").tofile(HERE / "z1.bin")
(HERE / "z1.tsv").write_text("\n".join(rows) + "\n")
