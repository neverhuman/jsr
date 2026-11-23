"""Attention-chunk modes for training, monitoring and evaluation (H44 training fixes).

Two layers spell the same choices differently, and the trainer must translate between them:
- Config / CLI level (`TrainConfig.chunk`, `chunk_modes`, `eval_chunks`): 0 means the model's
  trained chunk ("inherit"), and the token `offline` means unrestricted attention.
- Model level (`Encoder.forward(chunk=...)`): None means the trained chunk, and 0 (or less) means
  the whole utterance, i.e. offline.
`resolve()` maps the first onto the second, so a configured 0 can never become an offline decode.
`check_legal()` refuses chunks a family cannot run causally: multirate pairs frames, so it needs an
even chunk, and unet3 groups four frames, so it needs a multiple of four.
"""
from __future__ import annotations

OFFLINE = "offline"


def parse_list(spec: str) -> list[int | str]:
    """'8,16,offline' -> [8, 16, 'offline']: positive frame counts and the offline token only."""
    out: list[int | str] = []
    for v in spec.split(","):
        v = v.strip()
        if not v:
            continue
        if v.lower() == OFFLINE:
            out.append(OFFLINE)
            continue
        c = int(v)
        if c <= 0:
            raise ValueError(f"chunk {c}: give a positive number of frames, or '{OFFLINE}'")
        out.append(c)
    return out


def resolve(requested: int | str | None, configured: int = 0) -> int | None:
    """Config-level chunk -> the model's argument. None or 0: the configured chunk when one is set,
    else the model's own (None). 'offline': 0, the whole utterance. A positive integer is itself."""
    if requested == OFFLINE:
        return 0
    if requested is None or requested == 0:
        return configured if configured and configured > 0 else None
    if int(requested) < 0:
        raise ValueError(f"chunk {requested}: negative chunks are not a mode")
    return int(requested)


def key(c: int | str) -> str:
    """Suffix for monitor fields: 'c8', or 'offline'."""
    return OFFLINE if c == OFFLINE else f"c{int(c)}"


def check_legal(family: str, chunks) -> None:
    """Raise unless every chunk can run causally on this encoder family (offline / None pass)."""
    for c in chunks:
        if c is None or c == OFFLINE:
            continue
        c = int(c)
        if c <= 0:
            raise ValueError(f"chunk {c} is not a positive frame count (use '{OFFLINE}' for offline)")
        if family in ("multirate", "unet3") and c % 2:
            raise ValueError(f"{family} needs an even chunk: {c} would put a pair's second frame in the future")
        if family == "unet3" and c % 4:
            raise ValueError(f"unet3 needs a chunk divisible by 4, got {c}")


def check_plan(cfg, model_cfg) -> None:
    """Startup check of every chunk a run will use: the fixed chunk, each `chunk_modes` draw and each
    `eval_chunks` entry, so an illegal value fails before step 0 instead of at its first draw."""
    plan = [cfg.chunk or None] + parse_list(cfg.chunk_modes or "") + parse_list(cfg.eval_chunks or "")
    if OFFLINE in parse_list(cfg.chunk_modes or ""):
        raise ValueError("chunk_modes trains causal chunks only; 'offline' is an eval_chunks entry")
    check_legal(model_cfg.encoder.family, plan)
