"""H52 Libri-Light segmentation core: VAD metadata -> utterances <= MAX_LEN with real tail silence.

Rules (pre-registered in the H52 README):
  * Units: Libri-Light VAD segments, sorted, overlaps merged; units separated by < MIN_GAP (0.3 s)
    are merged (no cut may fall in a pause shorter than the minimum tail pad).
  * Units too long to fit in one utterance are split only at internal *energy* pauses of
    >= MIN_GAP (frame log-energy below a per-file threshold between the file's non-speech floor
    and its speech level). Pieces that still do not fit are dropped and counted.
  * Utterances are contiguous spans of the original audio (internal pauses kept; nothing is
    concatenated, unlike Libri-Light's cut_by_vad.py). Cuts happen only between units; a cut is
    forced at any gap >= FORCE_GAP. Within a run, cut points come from a DP that minimises
    sum ((L - TARGET) / 4)^2 - 0.5 * min(gap, 1.5) / 1.5 (prefers ~TARGET s and longer pauses).
  * Padding: head = min(HEAD, gap before), tail = min(TAIL_MAX, gap after) and tail >= TAIL_MIN
    always (an utterance whose real tail silence would be < TAIL_MIN, i.e. at a file end, is
    dropped and counted). Length incl. padding <= MAX_LEN (18.0 s, so speed 0.9 stays <= 20 s).
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf

SR = 16000
MAX_LEN = 18.0
MIN_LEN = 1.0
TARGET = 14.0
HEAD = 0.4
TAIL_MAX = 0.6
TAIL_MIN = 0.3
MIN_GAP = 0.3
FORCE_GAP = 2.5
BLOCK = 320  # 20 ms energy blocks
PARAMS = dict(SR=SR, MAX_LEN=MAX_LEN, MIN_LEN=MIN_LEN, TARGET=TARGET, HEAD=HEAD, TAIL_MAX=TAIL_MAX,
              TAIL_MIN=TAIL_MIN, MIN_GAP=MIN_GAP, FORCE_GAP=FORCE_GAP, BLOCK_MS=BLOCK * 1000 // SR,
              DP="sum((L-14)/4)^2 - 0.5*min(gap,1.5)/1.5", ENERGY_THR="floor+clip(0.3*(speech-floor),3,10) dB")


def merge_units(va, dur):
    segs = sorted((max(0.0, float(s)), min(dur, float(e))) for s, e in va)
    out = []
    for s, e in segs:
        if e <= s:
            continue
        if out and s - out[-1][1] < MIN_GAP:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def block_energy_db(pcm):
    n = len(pcm) // BLOCK
    x = pcm[:n * BLOCK].astype(np.float32).reshape(n, BLOCK)
    return 10.0 * np.log10(np.mean(x * x, axis=1) + 1.0)


def energy_threshold(edb, units):
    """Per-file pause threshold between the non-speech floor and the speech level (dB)."""
    n = len(edb)
    speech = np.zeros(n, bool)
    for s, e in units:
        speech[int(s * SR / BLOCK):int(np.ceil(e * SR / BLOCK))] = True
    sp = edb[speech]
    ns = edb[~speech]
    floor = float(np.median(ns)) if len(ns) >= 50 else float(np.percentile(edb, 5))
    level = float(np.median(sp)) if len(sp) else floor + 20.0
    return floor + float(np.clip(0.3 * (level - floor), 3.0, 10.0)), floor, level


HEAD_MIN = 0.15  # head may shrink to this (or the natural gap) to fit MAX_LEN
DROP_COST = 50.0  # DP: dropping a unit only when nothing feasible exists


def split_long(s, e, edb, thr, max_speech):
    """Split one too-long unit at internal energy pauses >= MIN_GAP (>= 0.5 s from its edges).
    Returns (pieces as dicts with eh / et = boundary is an energy pause, seconds dropped)."""
    b0, b1 = int(s * SR / BLOCK), int(e * SR / BLOCK)
    low = edb[b0:b1] < thr
    need = int(np.ceil(MIN_GAP * SR / BLOCK))
    pauses = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if j - i >= need:
                ps, pe = s + i * BLOCK / SR, s + j * BLOCK / SR
                if ps - s >= 0.5 and e - pe >= 0.5:
                    pauses.append((ps, pe))
            i = j
        else:
            i += 1
    bounds = [s] + [x for p in pauses for x in p] + [e]
    pieces, dropped = [], 0.0
    for k in range(0, len(bounds), 2):
        ps, pe = bounds[k], bounds[k + 1]
        if pe - ps <= max_speech:
            pieces.append(dict(s=ps, e=pe, eh=k > 0, et=k + 2 < len(bounds)))
        else:
            dropped += pe - ps
    return pieces, dropped


def plan(va, dur, pcm=None):
    """Return (utterances, stats). Utterance: dict(t0, t1, v0, v1, head, tail, n_units, eh, et, speech)."""
    units = merge_units(va, dur)
    stats = {"vad_s": round(sum(e - s for s, e in units), 3), "dropped_long_s": 0.0, "dropped_tail_s": 0.0,
             "dropped_short_s": 0.0, "dropped_dp_s": 0.0, "energy_splits": 0, "n_long_units": 0}
    max_unit = MAX_LEN - HEAD_MIN - TAIL_MIN
    pieces = []
    edb = thr = None
    if any(e - s > max_unit for s, e in units):
        if pcm is not None:
            edb = block_energy_db(pcm)
            thr, floor, level = energy_threshold(edb, units)
            stats.update(floor_db=round(floor, 1), level_db=round(level, 1), thr_db=round(thr, 1))
    for s, e in units:
        if e - s <= max_unit:
            pieces.append(dict(s=s, e=e, eh=False, et=False))
            continue
        stats["n_long_units"] += 1
        if thr is None:
            stats["dropped_long_s"] += e - s
            continue
        ps, dropped = split_long(s, e, edb, thr, max_unit)
        stats["dropped_long_s"] += dropped
        stats["energy_splits"] += sum(1 for p in ps if p["et"])
        pieces.extend(ps)
    n = len(pieces)
    if n == 0:
        return [], stats
    gb = [pieces[0]["s"]] + [pieces[i]["s"] - pieces[i - 1]["e"] for i in range(1, n)]
    ga = [pieces[i + 1]["s"] - pieces[i]["e"] for i in range(n - 1)] + [dur - pieces[-1]["e"]]

    def fit(ai, bi):
        head, tail = min(HEAD, gb[ai]), min(TAIL_MAX, ga[bi])
        if tail < TAIL_MIN - 1e-9:
            return None
        span = pieces[bi]["e"] - pieces[ai]["s"]
        excess = span + head + tail - MAX_LEN
        if excess > 0:
            dh = min(excess, head - min(head, HEAD_MIN))
            head -= dh
            excess -= dh
            dt = min(excess, tail - TAIL_MIN)
            tail -= dt
            excess -= dt
            if excess > 1e-9:
                return None
        return head, tail, span + head + tail

    runs, start = [], 0
    for i in range(n - 1):
        if ga[i] >= FORCE_GAP:
            runs.append((start, i))
            start = i + 1
    runs.append((start, n - 1))
    utts = []
    for a0, b0 in runs:
        m = b0 - a0 + 1
        INF = float("inf")
        best = [INF] * (m + 1)
        back = [None] * (m + 1)
        best[0] = 0.0
        for j in range(1, m + 1):
            bi = a0 + j - 1
            best[j], back[j] = best[j - 1] + DROP_COST, ("drop", j - 1)
            for i in range(j - 1, -1, -1):
                ai = a0 + i
                if pieces[bi]["e"] - pieces[ai]["s"] > MAX_LEN:
                    break
                f = fit(ai, bi)
                if f is None or best[i] == INF:
                    continue
                c = ((f[2] - TARGET) / 4.0) ** 2
                if j < m:
                    c -= 0.5 * min(ga[bi], 1.5) / 1.5
                if best[i] + c < best[j]:
                    best[j], back[j] = best[i] + c, ("utt", i)
        j, chosen = m, []
        while j > 0:
            kind, i = back[j]
            if kind == "drop":
                stats["dropped_dp_s"] += pieces[a0 + j - 1]["e"] - pieces[a0 + j - 1]["s"]
            else:
                chosen.append((a0 + i, a0 + j - 1))
            j = i
        for ai, bi in reversed(chosen):
            head, tail, L = fit(ai, bi)
            speech = sum(pieces[k]["e"] - pieces[k]["s"] for k in range(ai, bi + 1))
            if L < MIN_LEN:
                stats["dropped_short_s"] += speech
                continue
            v0, v1 = pieces[ai]["s"], pieces[bi]["e"]
            utts.append(dict(t0=round(v0 - head, 3), t1=round(v1 + tail, 3), v0=round(v0, 3), v1=round(v1, 3),
                             head=round(head, 3), tail=round(tail, 3), n_units=bi - ai + 1,
                             eh=int(pieces[ai]["eh"]), et=int(pieces[bi]["et"]), speech=round(speech, 3)))
    for k in ("dropped_long_s", "dropped_tail_s", "dropped_short_s", "dropped_dp_s"):
        stats[k] = round(stats[k], 3)
    return utts, stats


def encode_flac(pcm16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm16, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_flac(b: bytes) -> np.ndarray:
    pcm, sr = sf.read(io.BytesIO(b), dtype="int16")
    if sr != SR or pcm.ndim != 1:
        raise ValueError(f"expected {SR} Hz mono, got {sr} Hz shape {pcm.shape}")
    return pcm


def cut(pcm: np.ndarray, u: dict) -> np.ndarray:
    return pcm[int(round(u["t0"] * SR)):int(round(u["t1"] * SR))]
