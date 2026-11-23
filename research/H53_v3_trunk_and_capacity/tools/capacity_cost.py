"""H53 capacity sizing: CPU cost, real-time margin and gated always-on CPU of candidate encoders.

Cost law (docs/ultra/costlaw/W01_H0.md, Zen 2 AVX2 int8, weights streamed from DRAM): a GEMM pass
over W weights for F frames costs W (a + c F), a ~= 51 ps/weight (DRAM streaming), c ~= 14 ps/MAC.
A layer at frame rate r (Hz) run in chunks of F frames costs W (a + c F) r / F per audio-second.
runtime-lane's profile: encoder.gemm_i8 is 88-91% of B0's step time. So, calibrated on B0's
measured quiet-host cost (bench-open TIMING 14br, xbabe0, 1 core, chunk 4, greedy):
    c(model) = c_B0 * (0.10 + 0.90 * G(model) / G(B0)),   G = sum_l W_l (a + c F_l) r_l / F_l.
For dense encoders at one chunk size G is proportional to the encoder weights, so the ratio is
the encoder-parameter ratio; multirate middle blocks (12.5 Hz, F = 2) are cheaper per weight.

Gated always-on CPU at speech duty s (H47 sleep gate): G_s = a_s * c + f, with a_s the gate's
CPU-weighted awake fraction and f the non-speech floor (H47 gate: f <= 2% of ungated). The capacity
model K must satisfy G_15(K) <= c_B0 (B0 UNGATED). Measured form (no model of a_s needed):
    r = c_K / c_B0 <= r_max = min(0.1 / c_B0, (1 - phi) / (g15 - phi)),
    g15 = G_15(B0) / c_B0 (H47, measured on B0 with the real gate), phi = f / c_B0.
The first term is the active real-time margin >= 10x (c_K <= 0.1 CPU-s per audio-s).
"""
from __future__ import annotations

import json
import math
import sys

A_PS, C_PS = 51.0, 14.0  # W01 int8_avx2 out of cache (1536M / 4096M fits)
C_B0 = 0.0351  # CPU-s per audio-s, B0 = A1-49000 EMA INT8, xbabe0 1 core, chunk 4 (TIMING 14br)
GEMM_SHARE = 0.90
PHI = 0.02  # H47 non-speech gate: floor <= 2% of ungated

# Encoder weights per block / extra (from train.asr presets, counted with torch; see results JSON)
BLOCK = {384: 3_408_000, 448: 4_635_456, 512: 6_051_328}
STEM_PLUS = {384: 55_611_904 - 16 * 3_408_000, 512: 98_216_832 - 16 * 6_051_328,
             448: 112_490_688 - 24 * 4_635_456}  # stem + final norm (+ cmvn) per width


def g_dense(d: int, blocks: int, chunk: int = 4) -> float:
    w = blocks * BLOCK[d] + STEM_PLUS[d]
    return w * (A_PS + C_PS * chunk) * 25.0 / chunk


def g_multirate(d: int, lower: int, middle: int, final: int, chunk: int = 4) -> float:
    w25 = (lower + final) * BLOCK[d] + STEM_PLUS[d]
    w12 = middle * BLOCK[d] + 2 * (2 * d * d + d)  # middle blocks + pair / restore projections
    f12 = chunk // 2
    return w25 * (A_PS + C_PS * chunk) * 25.0 / chunk + w12 * (A_PS + C_PS * f12) * 12.5 / f12


CANDIDATES = [
    # name, G, inference params (incl. predictor / joiner / CTC), engine support today, warm start
    ("S (B0) d384x16", g_dense(384, 16), 56_381_226, "yes (B0)", "-"),
    ("S-24 d384x24", g_dense(384, 24), 83_645_226, "dense path; verify", "exact growth from B0"),
    ("S-32 d384x32", g_dense(384, 32), 110_909_226, "dense path; verify", "exact growth from B0"),
    ("M-16 d512x16", g_dense(512, 16), 99_091_242, "dense path; verify", "from scratch"),
    ("M-18 d512x18", g_dense(512, 18), 111_193_898, "dense path; verify", "from scratch"),
    ("M-20 d512x20", g_dense(512, 20), 123_296_554, "dense path; verify", "from scratch"),
    ("W448-24 d448x24", g_dense(448, 24), 113_312_554, "dense path; verify", "from scratch"),
    ("M_MR (L3) d512 4/10/2", g_multirate(512, 4, 10, 2), 100_141_354, "yes (L3 INT8 verified)",
     "L3-86400 exists (spm512 tokenizer)"),
]

G15_SCENARIOS = {
    "0.19 (eval-lane stream items, mean 6.8 s, R 1 s batched replay, hangover 1.5 s)": 0.19,
    "0.24 (AO-bench items, mean 3.0 s)": 0.24,
    "0.35 (with false wakes)": 0.35,
    "0.50 (approved plan's 'about half on a 15%-speech stream')": 0.50,
}


def main() -> int:
    g0 = CANDIDATES[0][1]
    rt_cap = 0.1 / C_B0
    rows = []
    for name, g, params, engine, warm in CANDIDATES:
        r = GEMM_SHARE * g / g0 + (1 - GEMM_SHARE)
        c = C_B0 * r
        gated = {k: round((v * r - PHI * r + PHI) if False else (r * (v - PHI) + PHI), 4)
                 for k, v in G15_SCENARIOS.items()}  # G15(K)/c_B0 = r (g15 - phi) + phi
        rows.append({"model": name, "inference_params": params, "int8_MB": round(params / 1e6, 1),
                     "cpu_ratio_vs_B0": round(r, 3), "cpu_s_per_audio_s": round(c, 4),
                     "rt_margin_x": round(1 / c, 1), "gated15_over_B0_ungated": gated,
                     "passes_rt_10x": c <= 0.1,
                     "passes_gated_all_scenarios": all(v <= 1.0 for v in gated.values()) and c <= 0.1,
                     "engine": engine, "warm_start": warm})
    rmax = {k: round(min(rt_cap, (1 - PHI) / (v - PHI)), 3) for k, v in G15_SCENARIOS.items()}
    out = {"schema": "h53.capacity_sizing/v1", "c_B0": C_B0, "cost_law_ps": {"a": A_PS, "c": C_PS},
           "gemm_share": GEMM_SHARE, "phi": PHI, "rt_cap_ratio": round(rt_cap, 3),
           "r_max_by_g15": rmax, "rows": rows,
           "note": "estimates; the decision uses r measured on xbabe0 (grown-at-init export) and "
                   "g15 measured by H47 on B0 with the real gate"}
    json.dump(out, sys.stdout, indent=1)
    print()
    print("| model | params | CPU ratio | CPU-s/audio-s | RT margin | gated@15% / B0 ungated (g15 = .19 / .24 / .35 / .50) |",
          file=sys.stderr)
    for r in rows:
        g = list(r["gated15_over_B0_ungated"].values())
        print(f"| {r['model']} | {r['inference_params'] / 1e6:.1f} M | {r['cpu_ratio_vs_B0']:.2f}x | "
              f"{r['cpu_s_per_audio_s']:.4f} | {r['rt_margin_x']:.1f}x | "
              + " / ".join(f"{x:.2f}" for x in g) + " |", file=sys.stderr)
    print("r_max by g15:", rmax, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
