"""Whole-engine projection per assignment, from (1) bench-open's quiet xbabe0 profile (14h, provisional)
and (2) PQ kernel times: swarm #6's measured staged-format receipt (H0, non-quiet, scalar LUT build),
the same with a 4x faster (SIMD) LUT build, and H24's measured pqscan ratios (SIMD table build).

Model per step: T = T_nonGEMM(chunk) + sum_INT8 t_int8(l) + sum_PQ max(t_pq(l), bytes_pq(l) / BW).
INT8 encoder GEMM is DRAM-bound at every chunk (14h), so t_int8(l) = T_gemm8(chunk) * bytes(l) / bytes_all
and BW = bytes_all / T_gemm8(M=1). Prints a table; writes results/projection/projection.json.
"""
from __future__ import annotations

import json

R = "/mnt/xbabe3-external/jsr/h51-pq/results/projection/"
prof = json.load(open(R + "quiet_profile_14h.json"))
sw = json.load(open(R + "swarm6_h0_pqscan_microbench.json"))["results"]

# per-step quiet numbers (ms)
CH = {}
for c, cfg in ((4, "c4"), (2, "c2"), (1, "c1")):
    d = prof[cfg]
    step = d["step_ms"]
    g8 = d["enc_gemm_step_ms"]
    CH[c] = {"m": c, "step": step, "gemm8": g8, "other": step - g8}

# layers: (kind, n, k, count)
L = [("ff.w1", 1536, 384, 32), ("ff.w2", 384, 1536, 32), ("att.qkv", 1152, 384, 16),
     ("att.out", 384, 384, 16), ("conv.pw1", 768, 384, 16), ("conv.pw2", 384, 384, 16),
     ("stem.out", 384, 2432, 1), ("enc_proj", 320, 384, 1)]
BYTES_ALL = sum(n * k * c for _, n, k, c in L)
LN_INPUT = {"ff.w1", "att.qkv", "att.out", "conv.pw1", "enc_proj"}  # LUT-safe inputs (R3)


def sw_time(m, k, n, stages, part):
    for r in sw:
        if r["m"] == m and r["k"] == k and r["n"] == n and r["stages"] == stages:
            return r[part] * 1e-6  # ns -> ms
    return None


def t_pq_swarm(kind, n, k, m, bpw, lut_speedup=1.0):
    """Swarm staged format: 0.5 bpw per stage. Measured shapes 384<->1536 at M 1 and 4; others
    scaled (LUT cost depends on K, scan on N*K)."""
    stages = int(round(bpw / 0.5))
    mm = 1 if m == 1 else 4
    ref_k = 384 if k <= 384 else 1536
    ref_n = 1536 if ref_k == 384 else 384
    lut = sw_time(mm, ref_k, ref_n, stages, "lut_ns") * (k / ref_k) / lut_speedup
    scan = sw_time(mm, ref_k, ref_n, stages, "scan_ns") * (n * k) / (ref_n * ref_k)
    scale = m / mm  # M = 2 between the measured M 1 and M 4 points
    return (lut + scan) * (scale if m == 2 else 1.0)


H24 = {  # INT8(K0) / PQ time ratio, s4R1, table build included (H24 README, H0 hot)
    (384, 1536): {1: 3.36, 2: 2.87, 4: 2.68}, (1536, 384): {1: 1.79, 2: 1.83, 4: 1.88},
    (384, 1152): {1: 3.21, 2: 2.85, 4: 2.55}, (384, 384): {1: 2.40, 2: 2.1, 4: 1.83}}
K0_PS = {1: 34.13, 2: 24.10, 4: 19.31}  # ps/MAC, K0 hot, 384->1536


def t_pq_h24(kind, n, k, m, bpw):
    key = (384, 1536) if (k == 384 and n == 1536) else (1536, 384) if k >= 1536 else \
        (384, 1152) if n in (1152, 768) else (384, 384)
    ratio = H24[key][m] / (bpw / 1.0)  # R stages scale both table build and scan
    return n * k * m * K0_PS[m] * 1e-9 / ratio  # ms


def project(assign, kernel, ch):
    t = ch["other"]
    bw = BYTES_ALL / CH[1]["gemm8"]  # bytes per ms (DRAM-bound INT8 at M 1)
    det = {}
    for kind, n, k, c in L:
        fmt = assign(kind)
        if fmt == "int8":
            tl = ch["gemm8"] * n * k * c / BYTES_ALL
        else:
            bpw = fmt
            if kernel == "swarm_scalar_lut":
                comp = t_pq_swarm(kind, n, k, ch["m"], bpw) * c
            elif kernel == "swarm_simd_lut_x4":
                comp = t_pq_swarm(kind, n, k, ch["m"], bpw, lut_speedup=4.0) * c
            else:
                comp = t_pq_h24(kind, n, k, ch["m"], bpw) * c
            byts = n * k * c * bpw / 8 + 32 * k * c * (bpw / 2) + 12 * n * c
            tl = max(comp, byts / bw)
        det[kind] = round(tl, 3)
        t += tl
    return t, det


ASSIGN = {
    "all INT8": lambda kd: "int8",
    "all 2 bpw": lambda kd: 2.0,
    "all 1 bpw": lambda kd: 1.0,
    "LN-input 2 bpw, rest INT8": lambda kd: 2.0 if kd in LN_INPUT else "int8",
    "LN-input 1 bpw, rest INT8": lambda kd: 1.0 if kd in LN_INPUT else "int8",
    "LN-input 1 bpw, post-SiLU 2 bpw": lambda kd: 1.0 if kd in LN_INPUT else 2.0,
}
out = []
for kernel in ("swarm_scalar_lut", "swarm_simd_lut_x4", "h24_simd_table"):
    for name, fn in ASSIGN.items():
        row = {"kernel": kernel, "assignment": name}
        for c in (1, 2, 4):
            t, det = project(fn, kernel, CH[c])
            row[f"chunk{c}_step_ms"] = round(t, 3)
            row[f"chunk{c}_speedup"] = round(CH[c]["step"] / t, 2)
        out.append(row)
        print(f"{kernel:18s} {name:32s} c1 {row['chunk1_speedup']:.2f}x  c2 {row['chunk2_speedup']:.2f}x  c4 {row['chunk4_speedup']:.2f}x")
json.dump({"quiet_profile": CH, "bytes_int8": BYTES_ALL, "rows": out}, open(R + "projection.json", "w"), indent=1)
