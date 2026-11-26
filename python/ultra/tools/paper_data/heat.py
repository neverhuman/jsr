"""Heat-map cell colours shared with the briefs: five stops, best to worst, missing in grey. Computed here, never by hand."""

from __future__ import annotations

import math

STOPS = ("A7F3D0", "D9F99D", "FEF9C3", "FED7AA", "FECACA")
MISSING = "F3F4F6"
SHOW_FINAL = False  # owner direction: tables show Isosyll live only; set True to restore the Isosyll final rows


def _rgb(h: str) -> tuple[int, int, int]:
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colour(t: float) -> str:
    """t in [0, 1], 0 = best. Linear interpolation between the five stops."""
    t = min(1.0, max(0.0, t)) * (len(STOPS) - 1)
    i = min(int(t), len(STOPS) - 2)
    a, b, f = _rgb(STOPS[i]), _rgb(STOPS[i + 1]), t - i
    return "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))


def scale(values: list[float | None], *, log: bool, higher_better: bool) -> list[str]:
    """One hex colour per value (MISSING for None), normalised over the column."""
    pts = [v for v in values if v is not None]
    f = (lambda v: math.log(v)) if log else (lambda v: v)
    lo, hi = (min(map(f, pts)), max(map(f, pts))) if pts else (0.0, 1.0)
    out = []
    for v in values:
        if v is None:
            out.append(MISSING)
            continue
        t = 0.0 if hi == lo else (f(v) - lo) / (hi - lo)
        out.append(colour(1.0 - t if higher_better else t))
    return out


def cell(hex_colour: str, body: str) -> str:
    return f"\\hmcell{{{hex_colour}}}{{{body}}}"
