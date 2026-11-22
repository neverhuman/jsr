"""Shared constants, the error type, formatting and raw-file helpers (sha256 manifest)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
import csv
import hashlib
import io
import json
import re



REPO = Path(__file__).resolve().parents[4]
DATA = Path("docs/paper/data")
RAW = DATA / "raw"
LABELS = ("exploratory", "reported", "qualified")
NAME_RE = re.compile(r"^[A-Za-z]+$")
REQUIRED = ("id", "value", "unit", "label", "configuration", "workload", "workload_digest", "source")

# The single-harness rehearsal (exploratory): one binary per job, same host, same workload.
CANDIDATES = (
    # key, candidate string in the raw files, display name, macro stem
    ("rzero", "zipformer-int8", "R0 Zipformer INT8", "Rzero"),
    ("dense", "/home/ubuntu/jsr-engine-data/runs/d2-avg3-45-50-55/weights-int8prep", "D2 avg-3 (45/50/55k) dense INT8", "Dense"),
    ("mr", "/home/ubuntu/jsr-data/runs/engine/d2mr-50000/weights-int8prep", "D2MR-50k MR INT8", "MR"),
)
# Job C (panel frontier) and Job B (ABAB) timed the dense step-45k checkpoint, not the avg-3 finalist.
FRONT_DENSE = "/home/ubuntu/jsr-data/runs/engine/d2-45000/weights-int8prep"
# Every single-harness ladder run on xbabe0: (raw dir, dense candidate, build note).
LADDER_RUNS = (
    ("ladder_a_57fc3c5", FRONT_DENSE, "Job A, ultra 57fc3c5 (pre-H15 mel), dense step 45k"),
    ("ladder_d_ce58952", FRONT_DENSE, "Job D, ultra ce58952, dense step 45k"),
    ("ladder_e_ce58952_avg3", "/home/ubuntu/jsr-engine-data/runs/d2-avg3-45-50-55/weights-int8prep",
     "Job E, ultra ce58952, dense avg-3 finalist"),
)
MATCHED_RUNG = 32  # highest total-streams rung every rehearsal candidate sustained
LADDER_WORKLOAD = "dev-clean.npz, first 10 utt cycled per stream (ladder --limit 10)"
ABAB_WORKLOAD = "dev-clean.npz, first 50 utt, 232.005 audio-s per block (--limit 50)"
DEV400_TRAINER = "dev-clean first 400 utt (trainer DEV400, 7,678 ref words)"
PANEL_DIGEST = "76fead746d1f8b8ae57a30aa341e81364538eb69759af813d6f13d6d1f57fc36"
SWEEP_WORKLOAD = "DEV selection panel, 730 utt (400 dev-clean + 330 dev-other)"

EVIDENCE_REQUIRED = (
    "id", "system", "role", "configuration", "workload", "workload_digest", "scope",
    "hardware", "cpu_s_per_audio_s", "cpu_reps", "wer_pct", "label", "source", "receipt_sha256",
)


class PaperDataError(RuntimeError):
    """A paper source is inconsistent with its manifest, or a required value is missing or malformed.

    purpose: every paper number must come from a pinned, repo-tracked source; nothing is defaulted.
    common_fixes: re-copy the raw artifact and run `--write-manifest`; add the missing field to
    docs/paper/data/curated.json or the evidence row; never edit generated outputs by hand.
    docs_url: docs/paper/data/README.md
    repair_hint: cd python/ultra && ../../.venv/bin/python -m tools.paper_data && \
    ../../.venv/bin/python -m pytest -q tests/test_paper_data.py
    """

    purpose = "paper numbers come only from pinned sources"
    docs_url = "docs/paper/data/README.md"
    repair_hint = "cd python/ultra && ../../.venv/bin/python -m tools.paper_data"

    def __init__(self, reason: str):
        super().__init__(f"paper data refused: {reason}. repair: fix the source named above, then run "
                         f"`{self.repair_hint}` (docs: {self.docs_url})")
        self.reason = reason


# ---------------------------------------------------------------- helpers


def fmt(x: float, nd: int) -> str:
    """Round half away from zero on the shortest decimal repr (so 7.654-7.029 prints 0.63)."""
    d = Decimal(repr(round(float(x), 9)))
    q = Decimal(1) if nd == 0 else Decimal(1).scaleb(-nd)
    return str(d.quantize(q, rounding=ROUND_HALF_UP))


def thousands(s: str) -> str:
    """'7300' -> '7{,}300' (integers only)."""
    if not s.isdigit() or len(s) < 5:
        return s
    out = []
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return "{,}".join(out)


def sci_tex(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_json(root: Path, rel: str):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def read_csv(root: Path, rel: str) -> list[dict]:
    with (root / rel).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def verify_manifest(root: Path) -> dict[str, str]:
    """Every raw file must be listed in MANIFEST.txt with a matching sha256, and vice versa."""
    shas: dict[str, str] = {}
    for line in (root / RAW / "MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, rel = line.split()[:2]
        p = root / RAW / rel
        if not p.is_file():
            raise PaperDataError(f"MANIFEST lists missing file {rel}")
        got = sha256_file(p)
        if got != sha:
            raise PaperDataError(f"sha256 mismatch for raw/{rel}: manifest {sha}, file {got}")
        shas[rel] = sha
    on_disk = {
        str(p.relative_to(root / RAW)) for p in (root / RAW).rglob("*") if p.is_file() and p.name != "MANIFEST.txt"
    }
    extra = sorted(on_disk - set(shas))
    if extra:
        raise PaperDataError(f"raw files not in MANIFEST.txt: {extra}")
    return shas


def write_manifest(root: Path, origins: dict[str, str] | None = None) -> str:
    """Rebuild MANIFEST.txt from the files on disk (keeps recorded origins)."""
    mpath = root / RAW / "MANIFEST.txt"
    old: dict[str, str] = {}
    if mpath.is_file():
        for line in mpath.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and not line.startswith("#"):
                old[parts[1]] = parts[2]
    old.update(origins or {})
    lines = ["# sha256  path-under-raw  origin (copied 2026-11-22; exploratory rehearsal + trainer sweeps)"]
    for p in sorted(x for x in (root / RAW).rglob("*") if x.is_file() and x.name != "MANIFEST.txt"):
        rel = str(p.relative_to(root / RAW))
        lines.append(f"{sha256_file(p)}  {rel}  {old.get(rel, 'unknown-origin')}")
    text = "\n".join(lines) + "\n"
    mpath.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- record registry
