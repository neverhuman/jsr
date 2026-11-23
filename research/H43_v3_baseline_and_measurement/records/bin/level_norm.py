#!/usr/bin/env python3
"""eval-lane: offline level normalisation of 16 kHz int16 WAVs before the engine (a stand-in for a front-end AGC; engine untouched).
  utt : scale each utterance to the target RMS (non-causal: uses the whole utterance's RMS; gain clipped to [-12, +40] dB).
  agcwarm : the same causal AGC, but its state (level, gain) carries from one utterance to the next in sorted-id order, as in a
        continuous stream at constant microphone gain (warm at each word onset except the very first file).
  agc : causal AGC, cold start at unity gain. 10 ms frames; level = exponentially smoothed mean square of SPEECH frames
        (frames above -65 dBFS update it; tau AGC_TAU_S, default 0.4 s); gain = target / sqrt(level), clipped to [-12, +40] dB, smoothed per frame
        (tau 0.05 s) and linearly interpolated across each frame; output rounded and clipped to int16.
  level_norm.py METHOD TARGET_RMS IN_DIR OUT_DIR"""
import numpy as np, wave, os, sys
method, target, src, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(dst, exist_ok=True)
GMIN, GMAX = 10 ** (-12 / 20), 10 ** (40 / 20); FLOOR = (32768 * 10 ** (-65 / 20)) ** 2
TAU = float(os.environ.get("AGC_TAU_S", "0.4"))   # level time constant (s); the slow variant uses 5
H = 160; a_lvl = np.exp(-H / (TAU * 16000)); a_g = np.exp(-H / (0.05 * 16000))
state = {"lvl": target * target, "g": 1.0}
for f in sorted(os.listdir(src)):
    if not f.endswith(".wav"):
        continue
    with wave.open(f"{src}/{f}", "rb") as w:
        params = w.getparams(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    if method == "utt":
        rms = np.sqrt(np.mean(x * x)) if len(x) else target
        y = x * float(np.clip(target / max(rms, 1e-9), GMIN, GMAX))
    else:
        n = len(x); nf = (n + H - 1) // H
        lvl, g = (state["lvl"], state["g"]) if method == "agcwarm" else (target * target, 1.0)
        gains = np.empty(nf + 1); gains[0] = g
        for i in range(nf):
            fr = x[i * H:(i + 1) * H]; ms = float(np.mean(fr * fr)) if len(fr) else 0.0
            if ms > FLOOR:
                lvl = a_lvl * lvl + (1 - a_lvl) * ms
            g_t = float(np.clip(target / np.sqrt(lvl), GMIN, GMAX)); g = a_g * g + (1 - a_g) * g_t; gains[i + 1] = g
        state["lvl"], state["g"] = lvl, g
        idx = np.arange(n) / H                       # gain applied to frame i ramps from gains[i] to gains[i+1] (causal: uses frames <= i)
        y = x * np.interp(idx, np.arange(nf + 1), gains)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    with wave.open(f"{dst}/{f}", "wb") as w:
        w.setparams(params); w.writeframes(y.tobytes())
print(method, src, "->", dst, len(os.listdir(dst)))
