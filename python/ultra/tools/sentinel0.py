"""Sentinel-0: exact integer-arithmetic two-speed wake gate (spec section 5, Appendix A).

Pure Python integer decision path: zero floating point, cross-ISA bit-exact.
"""
from __future__ import annotations

import struct

HOP = 160
V, U = 25, 5
KF, KS = 2, 8
THETA_F, THETA_S = 640, 256
NABS_F, NABS_S = 3943, 4455
BIG = 1 << 30

LOG2_LUT = (
    0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
    32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59,
    60, 61, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85,
    86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 178, 178, 179, 180, 181, 182, 183, 184, 185, 185, 186, 187, 188, 189,
    190, 191, 192, 192, 193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 203,
    204, 205, 206, 207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217,
    218, 219, 220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231,
    231, 232, 233, 234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244,
    244, 245, 246, 247, 247, 248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
)


def ilog2_q8(v: int) -> int:
    """Q8 log2 of integer v >= 1: (e << 8) + LUT[8 bits below leading one]."""
    if v < 1:
        raise ValueError(f"ilog2_q8 requires v >= 1, got {v}")
    e = v.bit_length() - 1
    return (e << 8) + LOG2_LUT[((v << 8) >> e) & 0xFF]


class MinStat:
    """Martin minimum-statistics sliding sub-window floor tracker (1.25 s)."""

    __slots__ = ("sub", "cur", "k", "i")

    def __init__(self) -> None:
        self.sub = [BIG] * U
        self.cur = BIG
        self.k = 0
        self.i = 0

    def floor(self) -> int:
        m = self.cur
        for v in self.sub:
            if v < m:
                m = v
        return m

    def push(self, level: int) -> None:
        if level < self.cur:
            self.cur = level
        self.k += 1
        if self.k == V:
            self.sub[self.i] = self.cur
            self.i = (self.i + 1) % U
            self.cur = BIG
            self.k = 0


class Sentinel0:
    """Two-speed pure-integer wake gate (Band A 16 kHz diff, Band B 4 kHz box diff)."""

    def __init__(self, theta_f: int = THETA_F, theta_s: int = THETA_S,
                 nabs_f: int = NABS_F, nabs_s: int = NABS_S) -> None:
        self.theta_f, self.theta_s = theta_f, theta_s
        self.nabs_f, self.nabs_s = nabs_f, nabs_s
        self.prev_x, self.prev_b = 0, 0
        self.ea, self.eb = [0] * KS, [0] * KS
        self.n = 0
        self.fl = [MinStat() for _ in range(4)]

    def push_hop(self, x) -> tuple[bool, int, list[int], list[int]]:
        if len(x) != HOP:
            raise ValueError(f"Expected {HOP} samples, got {len(x)}")
        ea, px = 0, self.prev_x
        for s in x:
            si = int(s)
            d = si - px
            ea += d * d
            px = si
        self.prev_x = px
        eb, pb = 0, self.prev_b
        for m in range(HOP // 4):
            o = 4 * m
            b = int(x[o]) + int(x[o + 1]) + int(x[o + 2]) + int(x[o + 3])
            d = b - pb
            eb += d * d
            pb = b
        self.prev_b = pb
        return self.step_energies(ea, eb)

    def step_energies(self, ea: int, eb: int) -> tuple[bool, int, list[int], list[int]]:
        i = self.n % KS
        self.ea[i], self.eb[i] = ea, eb
        j = (self.n - 1) % KS
        fa = ea + (self.ea[j] if self.n >= 1 else 0)
        fb = eb + (self.eb[j] if self.n >= 1 else 0)
        levels = [ilog2_q8(fa + 1), ilog2_q8(fb + 1),
                  ilog2_q8(sum(self.ea) + 1), ilog2_q8(sum(self.eb) + 1)]
        if self.n < KS - 1:
            self.n += 1
            return False, 0, levels, [BIG, BIG, BIG, BIG]
        floors = [max(self.fl[0].floor(), self.nabs_f), max(self.fl[1].floor(), self.nabs_f),
                  max(self.fl[2].floor(), self.nabs_s), max(self.fl[3].floor(), self.nabs_s)]
        fast = (levels[0] - floors[0] >= self.theta_f) or (levels[1] - floors[1] >= self.theta_f)
        slow = (levels[2] - floors[2] >= self.theta_s) or (levels[3] - floors[3] >= self.theta_s)
        for k in range(4):
            self.fl[k].push(levels[k])
        self.n += 1
        return (fast or slow), (1 if fast else 0) | (2 if slow else 0), levels, floors


def golden_pcm(n: int = 96000) -> list[int]:
    """Platform-independent LCG noise + two 200 Hz square bursts (spec Appendix C)."""
    s = 12345
    out: list[int] = []
    for i in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        v = (s >> 16) & 0xFFFF
        if v >= 32768:
            v -= 65536
        v >>= 7
        sq = 1 if (i // 40) % 2 == 0 else -1
        if 32000 <= i < 40000:
            v += 2000 * sq
        if 64000 <= i < 72000:
            v += 260 * sq
        out.append(v)
    return out


def _fnv1a64(h: int, data: bytes) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def trace_fnv1a64(pcm: list[int]) -> tuple[int, int, list[int]]:
    """Return (pcm_fnv1a64, trace_fnv1a64, hot_hop_indices) matching Appendix C."""
    hp = _fnv1a64(0xCBF29CE484222325, struct.pack(f"<{len(pcm)}h", *pcm))
    ht = 0xCBF29CE484222325
    gate = Sentinel0()
    hot_hops: list[int] = []
    for i in range(len(pcm) // HOP):
        hot, path, levels, floors = gate.push_hop(pcm[i * HOP:(i + 1) * HOP])
        ht = _fnv1a64(ht, struct.pack("<8iB", *levels, *floors, path))
        if hot:
            hot_hops.append(i)
    return hp, ht, hot_hops
