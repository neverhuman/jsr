"""Whole-engine speed projection for a PQ assignment (Amdahl + bandwidth / L3 residency).

Inputs (JSON): quiet per-step profile per chunk (INT8 GEMM time, other time), per-shape PQ/INT8
kernel time ratios per M (hot cache, table build included), and an assignment (layer -> format).
GEMM time per step for a layer set = max(compute, bytes / BW); BW = L3 if the whole working set
fits the residency budget, else the effective DRAM bandwidth calibrated from the INT8 run at M = 1
(INT8 at one row is bandwidth-bound: 55.6 MB per step).

usage: python -I project.py <inputs.json>  (prints a table and writes <inputs>.out.json)
"""
from __future__ import annotations

import json
import sys

MB = 1e6


def project(cfg):
    out = []
    shapes = cfg["shapes"]  # name -> {"n","k","count"}
    for chunk in cfg["chunks"]:
        m = chunk["m"]
        t_step = chunk["t_step_ms"]
        t_gemm8 = chunk["t_gemm_int8_ms"]
        t_other = t_step - t_gemm8
        bw_dram = cfg["int8_bytes_mb"] * MB / (cfg["t_gemm_int8_m1_ms"] * 1e-3)  # B/s
        for arm in cfg["arms"]:
            comp, bytes_ = 0.0, 0.0
            for name, sh in shapes.items():
                fmt = arm["assign"].get(name, arm.get("default", "int8"))
                macs = sh["n"] * sh["k"] * m * sh["count"]
                if fmt == "int8":
                    # INT8 compute at this M from the hot kernel rate (ps/MAC), bytes = N*K
                    comp += macs * cfg["int8_ps_per_mac"][str(m)] * 1e-12
                    bytes_ += sh["n"] * sh["k"] * sh["count"]
                else:
                    r = cfg["pq_ratio"][fmt][name][str(m)]  # INT8 hot time / PQ hot time (incl. table)
                    comp += macs * cfg["int8_ps_per_mac"][str(m)] * 1e-12 / r
                    bpw = cfg["formats"][fmt]["bits_per_weight"]
                    bytes_ += sh["n"] * sh["k"] * sh["count"] * bpw / 8 + cfg["formats"][fmt]["overhead_bytes_per_k"] * sh["k"] * sh["count"]
            ws = bytes_ + cfg["other_resident_bytes"]
            resident = ws <= cfg["l3_budget_mb"] * MB
            bw = cfg["l3_bw_gbs"] * 1e9 if resident else bw_dram
            t_g = max(comp, bytes_ / bw) * 1e3
            t = t_other + t_g
            out.append({"chunk": chunk["chunk"], "arm": arm["name"], "t_other_ms": round(t_other, 3),
                        "t_gemm_ms": round(t_g, 3), "compute_ms": round(comp * 1e3, 3),
                        "bytes_mb": round(bytes_ / MB, 2), "working_set_mb": round(ws / MB, 2),
                        "l3_resident": resident, "t_step_ms": round(t, 3),
                        "speedup": round(t_step / t, 3)})
    return out


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    res = project(cfg)
    for r in res:
        print(r)
    json.dump(res, open(sys.argv[1].replace(".json", ".out.json"), "w"), indent=1)
