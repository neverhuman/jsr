"""Regenerate every number and figure table of the ULTRA paper from repo-tracked sources.

Standard library only. Deterministic: sorted keys, fixed formatting, LF line endings.

Inputs (all under the repo root):
    docs/paper/data/raw/**           small raw bench artifacts, sha256-pinned in raw/MANIFEST.txt
    docs/paper/data/curated.json     numbers that live only in docs, each with its exact source
    docs/paper/data/evidence/*.jsonl matched cold-DEV comparator rows (format: data/README.md)
    research/curves/index.json, research/curves/<run>/{train,eval}.csv

Outputs (docs/paper/data/ unless --out):
    metrics.tex      one \\newcommand per record; a missing value prints a red ?? and fails
    records.jsonl    every record: id, value, unit, label, configuration, workload, digest, source
    SOURCES.md       macro -> value -> label -> source table
    *.csv            figure tables read by docs/paper/figures/*.tex

Usage (from python/ultra):
    ../../.venv/bin/python -m tools.paper_data            # write docs/paper/data
    ../../.venv/bin/python -m tools.paper_data --check    # exit 1 if committed files differ
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from decimal import ROUND_HALF_UP, Decimal
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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


class Records:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def add(self, rid: str, value, *, unit: str, label: str, configuration: str, workload: str,
            workload_digest: str, source: str, raw=None, line=None, quote=None) -> None:
        if not NAME_RE.match(rid):
            raise PaperDataError(f"macro name must be letters only: {rid!r}")
        if rid in self.rows:
            raise PaperDataError(f"duplicate record id {rid}")
        if label not in LABELS:
            raise PaperDataError(f"{rid}: label {label!r} not in {LABELS}")
        self.rows[rid] = {
            "id": rid, "value": value, "raw": raw, "unit": unit, "label": label,
            "configuration": configuration, "workload": workload, "workload_digest": workload_digest,
            "source": source, "line": line, "quote": quote,
        }

    def raw(self, rid: str) -> float:
        r = self.rows[rid]["raw"]
        if r is None:
            raise PaperDataError(f"{rid} has no numeric raw value")
        return float(r)

    def pending(self) -> list[str]:
        return sorted(rid for rid, r in self.rows.items() if r.get("pending"))

    def missing(self) -> list[str]:
        out = []
        for rid, r in self.rows.items():
            if r.get("pending"):
                continue
            if any(r.get(k) in (None, "") for k in REQUIRED):
                out.append(rid)
        return sorted(out)


# ---------------------------------------------------------------- inputs → records


def add_curated(root: Path, rec: Records) -> dict:
    cur = read_json(root, str(DATA / "curated.json"))
    for rid in sorted(cur["metrics"]):
        e = cur["metrics"][rid]
        if "text" in e:
            value, raw = e["text"], None
        elif e.get("raw") is None:
            value, raw = None, None
        else:
            raw = e["raw"]
            value = thousands(fmt(float(raw), int(e["decimals"])))
        rec.add(
            rid, value, raw=raw, unit=e["unit"], label=e["label"], configuration=e.get("configuration"),
            workload=e.get("workload"), workload_digest=e.get("workload_digest"),
            source=f"{e['source']} \u00a7 {e['section']}", line=e.get("line"), quote=e.get("quote"),
        )
        if e.get("pending"):
            rec.rows[rid]["pending"] = e["pending"]
    return cur


# Codex Evidence: Rust-rescored DEV accuracy rows (schema jsr.paper-evidence/v1), pinned in raw/.
CODEX_ROWS = (
    # macro stem, measurement id, which WER fields
    ("WerDtwoGreedy", "d2-60k-greedy", ("Clean", "Other")),
    ("WerDtwoBeam", "d2-60k-beam4", ("Clean", "Other")),
    ("WerDtwoAvgGreedy", "d2-avg3-60k-greedy", ("Clean", "Other")),
    ("WerDtwoBest", "d2-avg3-60k-beam4", ("Clean", "Other", "Pooled")),
    ("WerDtwoBestPanel", "panel-d2-avg3-60k-beam4", ("Clean", "Other", "Pooled")),
    ("WerDtwoAvgGreedyPanel", "panel-d2-avg3-60k-greedy", ("Pooled",)),
    ("WerDtwoAvgOldGreedyPanel", "panel-d2-avg3-55k-greedy", ("Clean", "Other", "Pooled")),
    ("WerDtwoSnapGreedyPanel", "panel-d2-55k-greedy", ("Pooled",)),
    ("WerDtwoMRAvgThreePanel", "panel-smr-avg3-50k-greedy", ("Pooled",)),
    ("WerDtwoMRAvgFivePanel", "panel-smr-avg5-50k-greedy", ("Pooled",)),
    ("MRBeamPanelGreedyFP", "panel-d2mr-greedy-fp32", ("",)),
    ("MRBeamPanelBeamFP", "panel-d2mr-beam4-fp32", ("",)),
    ("MRBeamPanelBeamInt", "panel-d2mr-beam4-int8", ("",)),
    ("MRBeamPanelGreedyInt", "panel-d2mr-greedy-int8", ("",)),
    ("WerDtwoMRBeamPanel", "panel-d2mr-beam4-int8", ("Clean", "Other")),
)


def add_codex_evidence(root: Path, rec: Records) -> None:
    rel = RAW / "evidence_codex.json"
    ev = read_json(root, str(rel))
    if ev.get("schema") != "jsr.paper-evidence/v1":
        raise PaperDataError(f"{rel}: unexpected schema {ev.get('schema')!r}")
    meas = {m["id"]: m for m in ev["measurements"]}
    norm = ev["normalizer"]
    for stem, mid, fields in CODEX_ROWS:
        if mid not in meas:
            raise PaperDataError(f"{rel}: measurement {mid} missing")
        m = meas[mid]
        wl = ev["workloads"][m["workload"]]
        conf = ev["configurations"][m["configuration"]]
        cdesc = (f"{conf['label']}; {conf.get('numeric_mode', '')}; {conf.get('decoder', '')}; {conf.get('runtime', '')}; "
                 f"weights {str(conf.get('weights_sha256', ''))[:12]}; rescored by ultra-score {norm['scorer']['sha256'][:12]} "
                 f"({norm['id']})")
        for f in fields:
            key = {"Clean": "wer_clean_pct", "Other": "wer_other_pct", "Pooled": "wer_pooled_pct", "": "wer_pooled_pct"}[f]
            v = m.get(key)
            rec.add(f"{stem}{f}", None if v is None else fmt(v, 2), raw=v, unit="% WER", label="exploratory",
                    configuration=cdesc, workload=f"{wl['label']} ({wl['utterances']} utt)",
                    workload_digest=wl["digest"], source=f"docs/paper/data/raw/evidence_codex.json measurements[id={mid}].{key}")


PANEL5 = (
    # rust score report under raw/panel5_rust_scores/, system, split
    ("rzero_dev-clean.json", "R0 Zipformer INT8", "dev-clean"),
    ("rzero_dev-other.json", "R0 Zipformer INT8", "dev-other"),
    ("dense_avg3_dev-clean.json", "D2 avg-3 (45/50/55k) dense INT8", "dev-clean"),
    ("dense_avg3_dev-other.json", "D2 avg-3 (45/50/55k) dense INT8", "dev-other"),
    ("mr_dev-clean_jobc.json", "D2MR-50k MR INT8", "dev-clean"),
    ("mr_dev-other.json", "D2MR-50k MR INT8", "dev-other"),
    ("moontiny_dev-other.json", "Moonshine v2 tiny INT8", "dev-other"),
)


def add_panel5(root: Path, rec: Records) -> None:
    """Runtime Jobs C/F1/F2: same-harness (ultra ce58952) panel runs, rescored by ultra-score 38eacc04."""
    rep = {}
    for name, system, split in PANEL5:
        d = read_json(root, str(RAW / "panel5_rust_scores" / name))
        if d.get("normaliser_id") != "ultra-en-wer-norm/v1":
            raise PaperDataError(f"{name}: unexpected normaliser {d.get('normaliser_id')}")
        ds = d["system_a"]["datasets"][split]
        c = ds["counts"]
        want = {"dev-clean": 400, "dev-other": 330}[split]
        if ds["utterances"] != want:
            raise PaperDataError(f"{name}: {ds['utterances']} utterances, expected the full panel half ({want})")
        rep[name] = (c["substitutions"] + c["deletions"] + c["insertions"], c["ref_words"])
    conf = ("bench-suite --panel, threads 1, grid 320 ms, ultra ce58952 (same executable as Jobs C/D/E); "
            "scored by ultra-score 38eacc04 (ultra-en-wer-norm/v1)")
    src = "docs/paper/data/raw/panel5_rust_scores/"
    halves = {"Clean": ("DEV selection panel, dev-clean half (400 utt)", "clean"),
              "Other": ("DEV selection panel, dev-other half (330 utt)", "other")}

    def add(rid, files, system):
        e = sum(rep[f][0] for f in files)
        n = sum(rep[f][1] for f in files)
        wl = "DEV selection panel, pooled (730 utt)" if len(files) == 2 else halves["Clean" if "clean" in files[0] else "Other"][0]
        rec.add(rid, fmt(100 * e / n, 2), raw=100 * e / n, unit="% WER", label="exploratory",
                configuration=f"{system}; {'pooled errors / ref words over both halves; ' if len(files) == 2 else ''}{conf}",
                workload=wl, workload_digest=PANEL_DIGEST, source=src + " + ".join(files))

    add("WerRzeroPanel", ["rzero_dev-clean.json", "rzero_dev-other.json"], "R0 Zipformer INT8")
    add("WerRzeroPanelClean", ["rzero_dev-clean.json"], "R0 Zipformer INT8")
    add("WerRzeroPanelOther", ["rzero_dev-other.json"], "R0 Zipformer INT8")
    add("WerRehearsalDense", ["dense_avg3_dev-clean.json"], "D2 avg-3 (45/50/55k) dense prepared INT8, the ladder's dense finalist")
    add("WerRehearsalDenseOther", ["dense_avg3_dev-other.json"], "D2 avg-3 (45/50/55k) dense prepared INT8")
    add("WerRehearsalDensePooled", ["dense_avg3_dev-clean.json", "dense_avg3_dev-other.json"], "D2 avg-3 dense prepared INT8")
    add("WerRehearsalMROther", ["mr_dev-other.json"], "D2MR-50k multirate prepared INT8")
    add("WerRehearsalMRPooled", ["mr_dev-clean_jobc.json", "mr_dev-other.json"], "D2MR-50k multirate prepared INT8")
    add("WerMoonTinyPanelOther", ["moontiny_dev-other.json"], "Moonshine v2 tiny INT8 (sherpa offline)")


def add_cluster_gap(root: Path, rec: Records) -> None:
    """Codex Evidence: paired speaker-cluster bootstrap of the pooled WER gap, MR (b) vs R0 (a), 730 panel."""
    rel = RAW / "tier1_dev_cluster_gap.json"
    d = read_json(root, str(rel))
    pd_ = d["comparison"]["pooled_delta"]
    # the two systems must be exactly the same-harness rows used elsewhere
    for side, rid in (("a", "WerRzeroPanel"), ("b", "WerRehearsalMRPooled")):
        if abs(100 * pd_[f"observed_{side}"] - rec.raw(rid)) > 1e-6:
            raise PaperDataError(f"{rel}: observed_{side} does not match {rid}")
    strata = ", ".join(f"{s_['dataset']} {s_['clusters']} speakers" for s_ in pd_["strata"])
    conf = (f"pooled WER, multirate 50k INT8 minus R0 INT8, same-harness 730-utt panel; paired speaker-cluster bootstrap "
            f"({strata}), {pd_['replicates']} replicates, seed {pd_['seed']}, two-sided 95%; ultra-score "
            f"({d['normaliser_id']})")
    common = dict(label="exploratory", configuration=conf, workload="DEV selection panel, pooled (730 utt)",
                  workload_digest=PANEL_DIGEST)
    src = f"docs/paper/data/{rel.relative_to(DATA)} comparison.pooled_delta"
    for rid, v in (("GapMRvsRzeroPooled", pd_["delta"]), ("GapMRvsRzeroPooledLo", pd_["lower"]),
                   ("GapMRvsRzeroPooledHi", pd_["upper"])):
        rec.add(rid, fmt(100 * v, 2), raw=100 * v, unit="pp WER", source=f"{src}", **common)
    rec.add("GapReplicates", thousands(str(pd_["replicates"])), raw=pd_["replicates"], unit="resamples",
            source=f"{src}.replicates", **common)


def add_derived(rec: Records) -> None:
    """Ratios and differences of curated numbers (same workload on both sides)."""
    def ratio(rid, a, b, nd, conf):
        x = rec.raw(a) / rec.raw(b)
        ra, rb = rec.rows[a], rec.rows[b]
        rec.add(rid, fmt(x, nd), raw=round(x, 6), unit="x", label="exploratory",
                configuration=f"{conf}; derived {a}/{b}", workload=ra["workload"],
                workload_digest=ra["workload_digest"], source=f"derived: {ra['source']} ; {rb['source']}")

    def diff(rid, a, b, nd, conf):
        x = rec.raw(a) - rec.raw(b)
        ra = rec.rows[a]
        rec.add(rid, fmt(x, nd), raw=round(x, 6), unit="pp WER", label="exploratory",
                configuration=f"{conf}; derived {a}-{b}", workload=ra["workload"],
                workload_digest=ra["workload_digest"], source=f"derived: {ra['source']}")

    ratio("SpeedupVsRzero", "CostRzero", "CostEngineIntEight", 1,
          "R0 vs engine INT8 whole-service CPU, same executable 6df5c8f")
    x = 1.0 / rec.raw("SingleMthreeMR")
    r0 = rec.rows["SingleMthreeMR"]
    rec.add("RTFxMthreeMR", fmt(x, 0), raw=round(x, 6), unit="x real time per CPU core", label="exploratory",
            configuration="1 / SingleMthreeMR: D2MR-final INT8 single stream on M3 Ultra (dev tool)",
            workload=r0["workload"], workload_digest=r0["workload_digest"], source=f"derived: {r0['source']}")
    # No cross-run ratios: ORT FP32 exists only in Run B (db5c14a), so it is compared only with Run B's engine FP32.
    ratio("SpeedupEngineFPVsORT", "CostORTRunB", "CostEngineFPRunB", 2,
          "Run B (db5c14a, one executable): ORT FP32 vs fused engine FP32, 400/400 identical transcripts")
    diff("BeamGainClean", "WerDtwoGreedyClean", "WerDtwoBeamClean", 2, "D2-60k beam 4 vs greedy, full dev-clean")
    diff("BeamGainOther", "WerDtwoGreedyOther", "WerDtwoBeamOther", 2, "D2-60k beam 4 vs greedy, full dev-other")
    diff("AvgGainClean", "WerDtwoGreedyClean", "WerDtwoAvgGreedyClean", 2,
         "D2 avg-3 (50/55/60k) vs single 60k, greedy, full dev-clean")
    diff("AvgGainOther", "WerDtwoGreedyOther", "WerDtwoAvgGreedyOther", 2,
         "D2 avg-3 vs single 60k, greedy, full dev-other")
    diff("StackGainClean", "WerDtwoGreedyClean", "WerDtwoBestClean", 2,
         "D2 avg-3 + beam 4 vs single 60k greedy, full dev-clean")
    diff("StackGainOther", "WerDtwoGreedyOther", "WerDtwoBestOther", 2,
         "D2 avg-3 + beam 4 vs single 60k greedy, full dev-other")


def curve_rows(root: Path, run: str) -> tuple[list[dict], list[dict]]:
    """Eval rows (deduplicated by step; rows without an epoch are init/final duplicates) with
    the run-local cumulative audio-hours linearly interpolated from train.csv."""
    train = [(int(r["step"]), float(r["audio_h_cum"])) for r in read_csv(root, f"research/curves/{run}/train.csv")]
    train.sort()

    def audio_h(step: int) -> float:
        if step <= train[0][0]:
            return train[0][1] * step / train[0][0]
        for (s0, h0), (s1, h1) in zip(train, train[1:]):
            if s0 <= step <= s1:
                return h0 + (h1 - h0) * (step - s0) / (s1 - s0)
        raise PaperDataError(f"{run}: step {step} beyond train.csv")

    seen, rows = set(), []
    for r in read_csv(root, f"research/curves/{run}/eval.csv"):
        if r["split"] != "dev-clean" or r["epoch"] == "":
            continue
        step = int(r["step"])
        if step in seen:
            continue
        seen.add(step)
        rows.append({"step": step, "epoch": int(r["epoch"]), "wer": float(r["wer"]), "audio_h": audio_h(step)})
    rows.sort(key=lambda r: r["step"])
    # epoch-boundary evals are the off-grid steps (not a multiple of the 250-step log interval)
    bounds = [r for r in rows if r["step"] % 250 != 0]
    return rows, bounds


def curve_wer(root: Path, run: str, step: int) -> float:
    for r in read_csv(root, f"research/curves/{run}/eval.csv"):
        if r["split"] == "dev-clean" and int(r["step"]) == step:
            return float(r["wer"])
    raise PaperDataError(f"{run}: no dev-clean eval at step {step}")


def add_rehearsal(root: Path, rec: Records, out: dict[str, str]) -> None:
    """Warm single-harness rehearsal (exploratory): Job E ladder (main), Jobs A/D (dense step 45k, range),
    Job C panel frontier, Job B ABAB.

    Jobs C, D and E ran the same `ultra` executable; Job B ran 4fafb45 and Job A 57fc3c5."""
    ldir = RAW / "ladder_e_ce58952_avg3"
    ladder = read_json(root, str(ldir / "density_ladder.json"))
    points = [json.loads(x) for x in (root / ldir / "density_ladder.points.jsonl").read_text().splitlines() if x]
    abab = read_json(root, str(RAW / "abab_timing.json"))
    panel = read_json(root, str(RAW / "frontier_panel.json"))
    lsrc = f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.json"
    psrc = f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.points.jsonl"
    asrc = "docs/paper/data/raw/abab_timing.json"
    fsrc = "docs/paper/data/raw/frontier_panel.json"
    man = {}
    for sub in ("abab_manifests", "ladder_d_ce58952", "ladder_e_ce58952_avg3", "panel_manifests"):
        for p in sorted((root / RAW / sub).glob("*.job_manifest.json")):
            j = json.loads(p.read_text())
            man[(sub, j["candidate"])] = j
    npz = {j["npz_sha256"] for j in man.values()}
    if len(npz) != 1:
        raise PaperDataError(f"rehearsal jobs ran on different audio caches: {npz}")
    npz_sha = npz.pop()
    exe = {j["executable_sha256"] for (sub, _), j in man.items() if sub != "abab_manifests"}
    if len(exe) != 1:
        raise PaperDataError(f"ladder and panel frontier must share one executable, got {exe}")
    if panel.get("panel_manifest_sha256") != PANEL_DIGEST:
        raise PaperDataError("frontier_panel.json is not on the frozen DEV panel")
    for d in (ladder, abab, panel):
        if d["completion_status"] != "exploratory":
            raise PaperDataError("rehearsal inputs must be labelled exploratory")
    hw = "xbabe0 AMD TR PRO 3995WX (Zen 2), shared host, nice 10"
    lconf = (f"density ladder 4:48:4, {ladder['workers']} single-thread workers on one CCX, observer on a separate CCX, "
             f"{ladder['step_seconds']:.0f} s rungs, one repeat, grid {ladder['grid_ms']} ms, whole-service fleet CPU; "
             f"ultra ce58952 (post-H15 sparse mel); {hw}")
    pconf = f"bench-suite --panel, dev-clean half, threads 1, grid {panel['grid_ms']} ms, warm in-process rusage; ultra ce58952; {hw}"
    pwl = "DEV selection panel, dev-clean half (400 utt, 3,052.165 audio-s)"

    by_cand = {r["candidate"]: r for r in ladder["results"]}
    rows_cost, rows_ladder = [], []
    claims, early, fails = {}, {}, {}
    for key, cand, disp, stem in CANDIDATES:
        res = by_cand[cand]
        pts = sorted((p for p in points if p["candidate"] == cand), key=lambda p: p["total_streams"])
        ok = [p for p in pts if p["all_sustained"] and p["drained"]]
        claim = res["claimed_total_streams"]
        if max(p["total_streams"] for p in ok) != claim:
            raise PaperDataError(f"{cand}: claimed {claim} disagrees with the points file")
        first_fail = next((p for p in pts if not (p["all_sustained"] and p["drained"])), None)
        before_fail = max(p["total_streams"] for p in ok if first_fail is None or p["total_streams"] < first_fail["total_streams"])
        above = [p for p in pts if p["total_streams"] > claim]
        at_claim = next(p for p in pts if p["total_streams"] == claim)
        at_match = next(p for p in pts if p["total_streams"] == MATCHED_RUNG)
        if not (at_match["all_sustained"] and at_match["drained"]):
            raise PaperDataError(f"{cand} did not sustain the matched rung {MATCHED_RUNG}")
        claims[key] = (claim, at_match["rtf_service_cpu"])
        early[key] = before_fail
        wl, dg = LADDER_WORKLOAD, npz_sha
        common = dict(label="exploratory", workload=wl, workload_digest=dg)
        rec.add(f"Den{stem}CCX", str(claim), raw=claim, unit="streams/CCX",
                configuration=f"{disp}; highest sustained rung (tool claim); {lconf}",
                source=f"{lsrc} results[candidate={cand}].claimed_total_streams", **common)
        rec.add(f"Den{stem}BeforeFailCCX", str(before_fail), raw=before_fail, unit="streams/CCX",
                configuration=f"{disp}; last sustained rung before the first failing rung (conservative)",
                source=f"{psrc} candidate={cand}", **common)
        rec.add(f"Den{stem}Core", fmt(claim / ladder["workers"], 0), raw=claim / ladder["workers"],
                unit="streams/core", configuration=f"{disp}; claimed streams / {ladder['workers']} workers",
                source=f"{lsrc} results[candidate={cand}]", **common)
        rec.add(f"DenPninetyNine{stem}", fmt(at_claim["max_worker_p99_ms"], 0), raw=at_claim["max_worker_p99_ms"],
                unit="ms", configuration=f"{disp}; max worker p99 lateness at the claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        rec.add(f"DenRss{stem}", fmt(at_claim["peak_rss_bytes"] / 1e9, 2), raw=at_claim["peak_rss_bytes"], unit="GB",
                configuration=f"{disp}; peak RSS of the whole 4-worker fleet at the claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        if above:
            rec.add(f"DenFail{stem}CCX", str(above[0]["total_streams"]), raw=above[0]["total_streams"],
                    unit="streams/CCX", configuration=f"{disp}; first rung above the claim (failed)",
                    source=f"{psrc} candidate={cand} total_streams={above[0]['total_streams']}", **common)
            rec.add(f"DenFailPninetyNine{stem}", fmt(above[0]["max_worker_p99_ms"], 0), raw=above[0]["max_worker_p99_ms"],
                    unit="ms", configuration=f"{disp}; max worker p99 lateness at the first failed rung",
                    source=f"{psrc} candidate={cand} total_streams={above[0]['total_streams']}", **common)
        inv = 1.0 / at_match["rtf_service_cpu"]
        rec.add(f"AudioPerCpu{stem}", fmt(inv, 1), raw=inv, unit="audio-s per CPU-s",
                configuration=f"{disp}; 1 / whole-service CPU-s per audio-s at the matched {MATCHED_RUNG}-stream rung",
                source=f"derived: {psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        rec.add(f"RTFxLadder{stem}", fmt(inv, 1), raw=inv, unit="x real time per CPU core",
                configuration=f"{disp}; audio-seconds processed per CPU-second at the matched {MATCHED_RUNG}-stream rung "
                              "(1 / CostLadder)",
                source=f"derived: {psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        util = 100.0 * claim * at_claim["rtf_service_cpu"] / ladder["workers"]
        rec.add(f"UtilClaim{stem}", fmt(util, 0), raw=util, unit="% of the CCX's 4 cores",
                configuration=f"{disp}; claimed streams x CPU-s per audio-s at the claimed rung / {ladder['workers']} cores",
                source=f"derived: {psrc} candidate={cand} total_streams={claim}", **common)
        fails[key] = (claim, above[0]["total_streams"] if above else None)
        rec.add(f"CostLadder{stem}", fmt(at_match["rtf_service_cpu"], 4), raw=at_match["rtf_service_cpu"],
                unit="CPU-s/audio-s", configuration=f"{disp}; whole-service CPU at the matched {MATCHED_RUNG}-stream rung; {lconf}",
                source=f"{psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        rec.add(f"CostClaim{stem}", fmt(at_claim["rtf_service_cpu"], 4), raw=at_claim["rtf_service_cpu"],
                unit="CPU-s/audio-s", configuration=f"{disp}; whole-service CPU at its claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        fcand = FRONT_DENSE if key == "dense" else cand
        fdisp = "D2-45k dense INT8" if key == "dense" else disp
        fr = next(r for r in panel["rows"] if r["candidate"] == fcand)
        pcommon = dict(label="exploratory", workload=pwl, workload_digest=PANEL_DIGEST)
        fsel = f"{fsrc} rows[candidate={fcand}]"
        rec.add(f"Frontier{stem}", fmt(fr["batch_rtf_cpu"], 4), raw=fr["batch_rtf_cpu"], unit="CPU-s/audio-s",
                configuration=f"{fdisp}; batch-16; {pconf}", source=f"{fsel}.batch_rtf_cpu", **pcommon)
        rec.add(f"FrontierPerCore{stem}", fmt(fr["batch_streams_per_core"], 1), raw=fr["batch_streams_per_core"],
                unit="streams/core", configuration=f"{fdisp}; 1/batch-16 cost", source=f"{fsel}.batch_streams_per_core", **pcommon)
        rec.add(f"FrontierSingle{stem}", fmt(fr["single_rtf_cpu"], 3), raw=fr["single_rtf_cpu"], unit="CPU-s/audio-s",
                configuration=f"{fdisp}; single stream; {pconf}", source=f"{fsel}.single_rtf_cpu", **pcommon)
        rec.add(f"FrontierChunkPninetyNine{stem}", fmt(fr["chunk_ms"]["p99"], 0), raw=fr["chunk_ms"]["p99"], unit="ms",
                configuration=f"{fdisp}; per-chunk compute p99, single stream", source=f"{fsel}.chunk_ms.p99", **pcommon)
        rec.add(f"WerRehearsal{stem}" + ("FortyFive" if key == "dense" else ""), fmt(fr["wer_pct"], 2), raw=fr["wer_pct"], unit="% WER",
                configuration=f"{fdisp}; same harness and executable as the ladder, Rust scorer, greedy",
                source=f"{fsel}.wer_pct", **pcommon)
        if key == "dense":
            wer = rec.raw("WerRehearsalDense")
        else:
            wer = fr["wer_pct"]
        rows_cost.append([key, disp, fmt(at_match["rtf_service_cpu"], 5), fmt(wer, 3), claim, before_fail,
                          fmt(at_claim["rtf_service_cpu"], 5),
                          # the panel frontier timed the 45k dense checkpoint: never put it on the avg-3 row
                          "" if key == "dense" else fmt(fr["batch_rtf_cpu"], 5),
                          "" if key == "dense" else fmt(fr["single_rtf_cpu"], 5)])
        for p in pts:
            rows_ladder.append([key, p["total_streams"], fmt(p["rtf_service_cpu"], 5), fmt(p["max_worker_p99_ms"], 2),
                                int(bool(p["all_sustained"] and p["drained"])), fmt(p["peak_rss_bytes"] / 1e9, 3)])

    base_claim, base_cost = claims["rzero"]
    base_fb = next(r for r in panel["rows"] if r["candidate"] == CANDIDATES[0][1])["batch_rtf_cpu"]
    for key, stem in (("dense", "Dense"), ("mr", "MR")):
        c, cost = claims[key]
        common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha)
        rec.add(f"DenRatio{stem}", fmt(c / base_claim, 1), raw=c / base_claim, unit="x",
                configuration=f"claimed streams/CCX {key} over R0, same ladder", source=f"derived: {lsrc}", **common)
        rec.add(f"DenRatio{stem}BeforeFail", fmt(early[key] / early["rzero"], 1), raw=early[key] / early["rzero"], unit="x",
                configuration=f"last-rung-before-first-failure streams/CCX {key} over R0", source=f"derived: {psrc}", **common)
        rec.add(f"CostRatio{stem}", fmt(base_cost / cost, 2), raw=base_cost / cost, unit="x",
                configuration=f"R0 over {key} CPU-s/audio-s at the {MATCHED_RUNG}-stream rung", source=f"derived: {psrc}", **common)
        fcand = FRONT_DENSE if key == "dense" else dict((k, c) for k, c, _, _ in CANDIDATES)[key]
        fb = next(r for r in panel["rows"] if r["candidate"] == fcand)["batch_rtf_cpu"]
        rec.add(f"FrontierRatio{stem}", fmt(base_fb / fb, 2), raw=base_fb / fb, unit="x", label="exploratory",
                configuration=f"R0 over {key} batch-16 CPU-s/audio-s, panel frontier", workload=pwl,
                workload_digest=PANEL_DIGEST, source=f"derived: {fsrc}")
    (rc, rf), (mc, mf) = fails["rzero"], fails["mr"]
    if rf is None or mf is None:
        raise PaperDataError("density bracket needs a failed rung for R0 and MR")
    bcommon = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=f"derived: {psrc}")
    lo, hi = mc / (rf - 1), (mf - 1) / rc
    rec.add("DenRatioMRBracketLo", fmt(lo, 1), raw=lo, unit="x",
            configuration=f"rung-spacing bracket, low end: MR claim {mc} / (R0 first failed rung {rf} - 1)", **bcommon)
    rec.add("DenRatioMRBracketHi", fmt(hi, 1), raw=hi, unit="x",
            configuration=f"rung-spacing bracket, high end: (MR first failed rung {mf} - 1) / R0 claim {rc}", **bcommon)
    largs = man[("ladder_e_ce58952_avg3", CANDIDATES[0][1])]["cli_args"]
    limits = {tuple(j["cli_args"][j["cli_args"].index("--limit") + 1:j["cli_args"].index("--limit") + 2])
              for (sub, _), j in man.items() if sub == "ladder_e_ce58952_avg3"}
    if len(limits) != 1:
        raise PaperDataError(f"ladder jobs used different --limit values: {limits}")
    lim = largs[largs.index("--limit") + 1]
    rec.add("LadderWorkload", f"first {lim} utterances cycled per stream (\\texttt{{--limit {lim}}})", raw=int(lim),
            unit="text", label="reported", configuration="ladder workload, read from the Job E job manifests (cli_args)",
            workload=LADDER_WORKLOAD, workload_digest=npz_sha,
            source=f"docs/paper/data/{ldir.relative_to(DATA)}/*.job_manifest.json cli_args --limit")
    add_ladder_runs(root, rec, npz_sha)
    add_equal_load_probe(root, rec, npz_sha, ldir, ladder)
    for rid, v, unit, conf, src, wl in (
        ("DenWorkers", ladder["workers"], "workers", "single-threaded workers on one 4-core CCX", lsrc, LADDER_WORKLOAD),
        ("DenRungSeconds", int(ladder["step_seconds"]), "s", "duration of each ladder rung", lsrc, LADDER_WORKLOAD),
        ("MatchedRung", MATCHED_RUNG, "streams/CCX", "highest rung all three candidates sustained", psrc, LADDER_WORKLOAD),
        ("AbabBlocks", abab["blocks_count"], "blocks", "counterbalanced ABAB blocks", asrc, ABAB_WORKLOAD),
        ("FrontierStreams", 16, "streams", "batch size of the frontier batch row", fsrc, pwl),
    ):
        rec.add(rid, str(v), raw=v, unit=unit, label="reported", configuration=conf, workload=wl,
                workload_digest=PANEL_DIGEST if wl == pwl else npz_sha, source=src)

    for pair in abab["paired_vs_baseline"]:
        stem = "Dense" if "d2-45000" in pair["candidate"] else "MR"
        g = pair["geometric_mean_ratio"]
        lo, hi = pair["geometric_mean_ratio_ci95_t"]
        conf = (f"{stem} vs R0 single-stream warm in-process CPU, {pair['blocks_paired']} paired blocks, "
                f"geometric-mean ratio (secondary), Student-t 95% CI, Holm; ultra 4fafb45; {hw}")
        src = f"{asrc} paired_vs_baseline[candidate~{stem}]"
        common = dict(label="exploratory", configuration=conf, workload=ABAB_WORKLOAD, workload_digest=npz_sha, source=src)
        for rid, v, nd in ((f"AbabRatio{stem}", g, 3), (f"AbabRatio{stem}Lo", lo, 3), (f"AbabRatio{stem}Hi", hi, 3),
                           (f"AbabSpeed{stem}", 1 / g, 2)):
            rec.add(rid, fmt(v, nd), raw=v, unit="x", **common)
        rec.add(f"AbabPholm{stem}", sci_tex(pair["holm_p_value"]), raw=pair["holm_p_value"], unit="p", **common)
        # pre-registered primary statistic (FREEZE section 5): median paired ratio with a bootstrap 95% CI
        med = pair["median_ratio_cand_over_base"]
        mlo, mhi = pair["ratio_ci95_bootstrap"]
        mconf = (f"{stem} vs R0 single-stream warm in-process CPU, median of {pair['blocks_paired']} paired block ratios, "
                 f"bootstrap 95% CI ({pair['bootstrap_replicates']} replicates, seed {pair['seed']}); PRIMARY statistic; "
                 f"ultra 4fafb45; {hw}")
        mcommon = dict(label="exploratory", configuration=mconf, workload=ABAB_WORKLOAD, workload_digest=npz_sha, source=src)
        for rid, v, nd in ((f"AbabMedian{stem}", med, 3), (f"AbabMedian{stem}Lo", mlo, 3),
                           (f"AbabMedian{stem}Hi", mhi, 3), (f"AbabMedianSpeed{stem}", 1 / med, 2)):
            rec.add(rid, fmt(v, nd), raw=v, unit="x", **mcommon)

    out["rehearsal_cost.csv"] = csv_text(
        ["key", "system", "cpu_matched", "wer_panel_clean", "streams_ccx", "streams_before_fail",
         "cpu_at_claim", "cpu_batch16", "cpu_single"], rows_cost)
    out["rehearsal_ladder.csv"] = csv_text(["key", "total_streams", "cpu", "p99_ms", "sustained", "fleet_rss_gb"], rows_ladder)
    # one file per candidate for pgfplots (no filtering needed in TeX)
    for key, *_ in CANDIDATES:
        rows = [r for r in rows_ladder if r[0] == key]
        out[f"ladder_{key}.csv"] = csv_text(["total_streams", "p99_ms", "sustained"], [[r[1], r[3], r[4]] for r in rows])


def add_equal_load_probe(root: Path, rec: Records, npz_sha: str, ldir: Path, ladder: dict) -> None:
    """Job E's fixed equal-load probe (same stream count for every system) and the ladder scope."""
    front = read_json(root, str(ldir / "frontier.json"))
    fsrc = f"docs/paper/data/{ldir.relative_to(DATA)}/frontier.json"
    probes = {}
    for key, cand, disp, stem in CANDIDATES:
        row = next(r for r in front["rows"] if r["candidate"] == cand)
        pr = row["sustained_probe"]
        man = json.loads((root / ldir / f"{cand.replace('/', '_')}.job_manifest.json").read_text())
        args = man["cli_args"]
        secs = int(args[args.index("--sustained-seconds") + 1])
        rcpt_name = f"{cand.replace('/', '_')}.sustained.receipt.json"
        rcpt = json.loads((root / ldir / rcpt_name).read_text())
        probes[key] = (pr["total_streams"], secs)
        common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha)
        conf = (f"{disp}; fixed equal-load probe, {pr['workers']} workers x {pr['streams_per_worker']} streams = "
                f"{pr['total_streams']} streams on one CCX for {secs} s (Job E, ultra ce58952)")
        src = f"{fsrc} rows[candidate={cand}].sustained_probe"
        rec.add(f"ProbePninetyNine{stem}", fmt(pr["lateness_p99_ms"], 0), raw=pr["lateness_p99_ms"], unit="ms",
                configuration=f"{conf}; worst-worker p99 lateness", source=src, **common)
        rec.add(f"ProbeSustained{stem}", "yes" if pr["sustained"] else "no", raw=int(bool(pr["sustained"])), unit="flag",
                configuration=f"{conf}; sustained under the A1 rule", source=src, **common)
        rec.add(f"ProbeDrained{stem}", "yes" if pr["drained"] else "no", raw=int(bool(pr["drained"])), unit="flag",
                configuration=f"{conf}; all accepted audio processed at cutoff", source=src, **common)
        cost = rcpt["process_tree_cpu_seconds"] / rcpt["audio_seconds"]
        rec.add(f"ProbeCost{stem}", fmt(cost, 4), raw=cost, unit="CPU-s/audio-s",
                configuration=f"{conf}; process-tree CPU over processed audio (observer excluded)",
                source=f"docs/paper/data/{ldir.relative_to(DATA)}/{rcpt_name}", **common)
    if len(set(probes.values())) != 1:
        raise PaperDataError(f"equal-load probe differs between systems: {probes}")
    n, secs = next(iter(probes.values()))
    common = dict(label="reported", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=fsrc)
    rec.add("ProbeStreams", str(n), raw=n, unit="streams/CCX", configuration="equal-load probe stream count, every system", **common)
    rec.add("ProbeSeconds", str(secs), raw=secs, unit="s", configuration="equal-load probe duration", **common)
    if ladder["confirm_seconds"] or ladder["repeats_required"] != 1:
        raise PaperDataError("ladder scope changed (confirmation or repeats); update LadderScope")
    scope = f"{ladder['step_seconds']:.0f}~s rungs, one repeat per rung, no confirmation"
    rec.add("LadderScope", scope, raw=None, unit="text", label="reported",
            configuration="ladder protocol of the exploratory rehearsal, from density_ladder.json "
                          "(step_seconds, repeats_required, confirm_seconds)",
            workload=LADDER_WORKLOAD, workload_digest=npz_sha,
            source=f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.json")


def add_ladder_runs(root: Path, rec: Records, npz_sha: str) -> None:
    """Reproducibility across the three single-harness ladder runs (A, D, E)."""
    claims: dict[str, list[tuple[int, str]]] = {"rzero": [], "dense": [], "mr": []}
    for sub, dense, note in LADDER_RUNS:
        lad = read_json(root, str(RAW / sub / "density_ladder.json"))
        by = {r["candidate"]: r["claimed_total_streams"] for r in lad["results"]}
        claims["rzero"].append((by[CANDIDATES[0][1]], note))
        claims["dense"].append((by[dense], note))
        claims["mr"].append((by[CANDIDATES[2][1]], note))
    srcs = "; ".join(f"docs/paper/data/raw/{sub}/density_ladder.json" for sub, _, _ in LADDER_RUNS)
    common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=srcs)
    runs = len(LADDER_RUNS)
    for key, stem in (("rzero", "Rzero"), ("mr", "MR")):
        vals = {v for v, _ in claims[key]}
        if len(vals) != 1:
            raise PaperDataError(f"{key} density differs across ladder runs: {claims[key]}; replace the reproducibility "
                                 "macros with a range")
    rec.add("DenReproRuns", str(runs), raw=runs, unit="runs",
            configuration=(f"single-harness ladder runs in which R0 ({claims['rzero'][0][0]}/CCX) and MR "
                           f"({claims['mr'][0][0]}/CCX) gave identical densities: " + "; ".join(n for _, n in claims["rzero"])),
            **common)
    base = claims["rzero"][0][0]
    # Step-45k dense was timed in two runs (A, D); the avg-3 finalist in one (E). Never pool the two checkpoints.
    d45 = [v for (v, n), (_, dense, _) in zip(claims["dense"], LADDER_RUNS) if dense == FRONT_DENSE]
    d45_notes = "; ".join(f"{n}: {v}" for (v, n), (_, dense, _) in zip(claims["dense"], LADDER_RUNS) if dense == FRONT_DENSE)
    src45 = "; ".join(f"docs/paper/data/raw/{sub}/density_ladder.json" for sub, dense, _ in LADDER_RUNS if dense == FRONT_DENSE)
    c45 = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=src45)
    rec.add("DenDenseFortyFiveRangeLo", str(min(d45)), raw=min(d45), unit="streams/CCX",
            configuration=f"dense step-45k checkpoint, lowest over its {len(d45)} ladder runs ({d45_notes})", **c45)
    rec.add("DenDenseFortyFiveRangeHi", str(max(d45)), raw=max(d45), unit="streams/CCX",
            configuration=f"dense step-45k checkpoint, highest over its {len(d45)} ladder runs ({d45_notes})", **c45)
    rec.add("DenRatioDenseFortyFiveRangeLo", fmt(min(d45) / base, 1), raw=min(d45) / base, unit="x",
            configuration="dense step-45k over R0, lowest over runs A/D", **c45)
    rec.add("DenRatioDenseFortyFiveRangeHi", fmt(max(d45) / base, 1), raw=max(d45) / base, unit="x",
            configuration="dense step-45k over R0, highest over runs A/D", **c45)
    rec.add("DenDenseFortyFiveRuns", str(len(d45)), raw=len(d45), unit="runs",
            configuration=f"ladder runs that timed dense step 45k ({d45_notes})", **c45)
    dD = dict((sub, v) for (v, _), (sub, _, _) in zip(claims["dense"], LADDER_RUNS))["ladder_d_ce58952"]
    cD = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha,
              source="docs/paper/data/raw/ladder_d_ce58952/density_ladder.json")
    rec.add("DenDenseFortyFiveCCX", str(dD), raw=dD, unit="streams/CCX",
            configuration=("dense step 45k in Job D (ultra ce58952, the executable of the Job C panel-frontier row); "
                           f"the same checkpoint gave {min(d45)} in Job A, see DenDenseFortyFiveRange*"), **cD)
    rec.add("DenRatioDenseFortyFive", fmt(dD / base, 1), raw=dD / base, unit="x",
            configuration="dense step 45k over R0 in Job D", **cD)
    # MR over dense: different training runs (dense S vs multirate S_MR) and different dense checkpoints
    mr = claims["mr"][0][0]
    alld = [v for v, _ in claims["dense"]]
    cmv = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=srcs)
    rec.add("DenRatioMRvsDenseLo", fmt(mr / max(alld), 1), raw=mr / max(alld), unit="x",
            configuration=(f"MR {mr}/CCX over the highest dense value {max(alld)}; different training runs (dense S vs "
                           "multirate S_MR), not a matched architecture ablation"), **cmv)
    rec.add("DenRatioMRvsDenseHi", fmt(mr / min(alld), 1), raw=mr / min(alld), unit="x",
            configuration=(f"MR {mr}/CCX over the lowest dense value {min(alld)}; different training runs (dense S vs "
                           "multirate S_MR), not a matched architecture ablation"), **cmv)


def add_curves(root: Path, rec: Records, out: dict[str, str]) -> None:
    idx = read_json(root, "research/curves/index.json")
    for run, fname, stem in (("D2_S_960h", "curve_d2.csv", "Dtwo"), ("D2MR_S_MR_960h", "curve_d2mr.csv", "DtwoMR")):
        rows, bounds = curve_rows(root, run)
        out[fname] = csv_text(["step", "epoch", "audio_h", "wer"],
                              [[r["step"], r["epoch"], fmt(r["audio_h"], 1), fmt(r["wer"], 3)] for r in rows])
        out[fname.replace("curve_", "epochs_")] = csv_text(
            ["epoch", "step", "audio_h", "wer"],
            [[b["epoch"], b["step"], fmt(b["audio_h"], 1), fmt(b["wer"], 3)] for b in bounds])
        params = idx[run]["params"]
        rec.add(f"Params{stem}", fmt(params / 1e6, 1), raw=params, unit="M parameters", label="reported",
                configuration=run, workload="none (model size)", workload_digest="n/a",
                source=f"research/curves/index.json {run}.params")
    rec.add("ParamsRone", fmt(idx["D2_S_960h"]["params"] / 1e6, 1), raw=idx["D2_S_960h"]["params"], unit="M parameters",
            label="reported", configuration="R1 dense S (same as ParamsDtwo)", workload="none (model size)",
            workload_digest="n/a", source="research/curves/index.json D2_S_960h.params")
    d2rows, _ = curve_rows(root, "D2_S_960h")
    r45 = next(r for r in d2rows if r["step"] == 45000)
    rec.add("DtwoAudioHoursRehearsal", thousands(fmt(r45["audio_h"], 0)), raw=round(r45["audio_h"], 3),
            unit="audio-h", label="reported", configuration="D2 run-local audio-hours seen at step 45000 (excl. D1d warm start)",
            workload="training run", workload_digest="n/a", source="research/curves/D2_S_960h/train.csv")
    rec.add("DtwoEpochRehearsal", str(r45["epoch"]), raw=r45["epoch"], unit="epoch (0-indexed)", label="reported",
            configuration="D2 epoch at step 45000", workload="training run", workload_digest="n/a",
            source="research/curves/D2_S_960h/eval.csv step 45000")


def sweep_cells(root: Path, sub: str) -> dict:
    cells = {}
    for p in sorted((root / RAW / sub).glob("*.json")):
        j = json.loads(p.read_text())
        split = j["panel"]["split"]
        if j["panel"]["manifest_sha256"] != PANEL_DIGEST:
            raise PaperDataError(f"{p.name}: panel manifest is not the frozen DEV panel")
        bw = j["beam_wer"]
        key = (j["beam"], j["blank_penalty"], "on" if j["length_norm"] else "off", split)
        cells[key] = {"err": bw["sub"] + bw["del"] + bw["ins"], "ref": bw["ref_words"], "wer": bw["wer"],
                      "greedy_err": j["sub"] + j["del"] + j["ins"], "greedy_wer": j["wer"], "step": j["step"]}
    return cells


def add_sweeps(root: Path, rec: Records, out: dict[str, str]) -> None:
    summary_d2 = read_json(root, str(RAW / "d2_decode_sweep_summary.json"))
    for sub, stem, model, conf in (
        ("d2_decode_sweep", "Dtwo", "d2", "D2 avg of 45/50/55k (saved as step 55000), trainer beam search, chunk 4, left 128"),
        ("d2mr_decode_sweep", "DtwoMR", "d2mr", "D2MR-final ckpt-50000, trainer beam search, chunk 4, left 128"),
    ):
        cells = sweep_cells(root, sub)
        beams = sorted({k[0] for k in cells})
        bps = sorted({k[1] for k in cells})
        if len(cells) != len(beams) * len(bps) * 2 * 2:
            raise PaperDataError(f"{sub}: incomplete grid ({len(cells)} cells)")
        greedy = {s: next(c for k, c in cells.items() if k[3] == s) for s in ("dev-clean", "dev-other")}

        def pooled(errs, refs):
            return 100.0 * errs / refs

        g_pooled = pooled(greedy["dev-clean"]["greedy_err"] + greedy["dev-other"]["greedy_err"],
                          greedy["dev-clean"]["ref"] + greedy["dev-other"]["ref"])
        ranked = []
        for b in beams:
            for bp in bps:
                for ln in ("off", "on"):
                    c, o = cells[(b, bp, ln, "dev-clean")], cells[(b, bp, ln, "dev-other")]
                    ranked.append((round(pooled(c["err"] + o["err"], c["ref"] + o["ref"]), 3), b, bp, ln, c["wer"], o["wer"]))
        ranked.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        if model == "d2":
            # cross-check against the trainer's own summary
            if abs(summary_d2["greedy"]["pooled"] - g_pooled) > 0.0015:
                raise PaperDataError("D2 sweep greedy pooled disagrees with decode_sweep_summary.json")
            if abs(summary_d2["ranked"][0]["pooled"] - ranked[0][0]) > 0.0015:
                raise PaperDataError("D2 sweep best pooled disagrees with decode_sweep_summary.json")
        for split in ("dev-clean", "dev-other"):
            rows = [[1] + [fmt(greedy[split]["greedy_wer"], 3)] * (2 * len(bps))]
            for b in beams:
                rows.append([b] + [fmt(cells[(b, bp, ln, split)]["wer"], 3) for ln in ("off", "on") for bp in bps])
            header = ["beam"] + [f"bp{int(bp * 10):02d}_ln{ln}" for ln in ("off", "on") for bp in bps]
            out[f"sweep_{model}_{split.replace('-', '')}.csv"] = csv_text(header, rows)
        best = ranked[0]
        b4 = next(t for t in ranked if t[1] == 4 and t[2] == 0.0 and t[3] == "off")
        src = f"docs/paper/data/raw/{sub}/*.json"
        common = dict(label="exploratory", workload=SWEEP_WORKLOAD, workload_digest=PANEL_DIGEST, source=src)
        rec.add(f"Sweep{stem}GreedyPooled", fmt(g_pooled, 2), raw=round(g_pooled, 4), unit="% WER",
                configuration=f"{conf}; greedy; pooled over the panel", **common)
        rec.add(f"Sweep{stem}BeamFourPooled", fmt(b4[0], 2), raw=b4[0], unit="% WER",
                configuration=f"{conf}; beam 4, blank penalty 0, no length norm; pooled", **common)
        rec.add(f"Sweep{stem}BestPooled", fmt(best[0], 2), raw=best[0], unit="% WER",
                configuration=f"{conf}; best grid cell: beam {best[1]}, blank penalty {best[2]}, length norm {best[3]}",
                **common)
        rec.add(f"Sweep{stem}BestBeam", str(best[1]), raw=best[1], unit="beam",
                configuration=f"{conf}; beam of the best grid cell", **common)
        out[f"sweep_{model}_ranked.csv"] = csv_text(
            ["pooled", "beam", "blank_penalty", "length_norm", "dev_clean", "dev_other"],
            [[fmt(t[0], 3), t[1], fmt(t[2], 1), t[3], fmt(t[4], 3), fmt(t[5], 3)] for t in ranked])


def add_tables(cur: dict, out: dict[str, str]) -> None:
    t = cur["tables"]["left_context"]
    out["left_context.csv"] = csv_text(t["columns"], [[fmt(v, 3) if isinstance(v, float) else v for v in r] for r in t["rows"]])


def accuracy_table(rec: Records, out: dict[str, str]) -> None:
    rows = []
    for name, c, o in (
        ("D2-60k greedy", "WerDtwoGreedyClean", "WerDtwoGreedyOther"),
        ("D2-60k beam 4", "WerDtwoBeamClean", "WerDtwoBeamOther"),
        ("D2 avg-3 greedy", "WerDtwoAvgGreedyClean", "WerDtwoAvgGreedyOther"),
        ("D2 avg-3 + beam 4", "WerDtwoBestClean", "WerDtwoBestOther"),
        ("D2MR-50k greedy", "WerDtwoMRClean", "WerDtwoMROther"),
    ):
        rows.append([name, rec.rows[c]["value"], rec.rows[o]["value"], c, o])
    out["accuracy_full_dev.csv"] = csv_text(["system", "dev_clean", "dev_other", "macro_clean", "macro_other"], rows)


def load_evidence(root: Path, rec: Records, out: dict[str, str]) -> None:
    """Cold-DEV comparator rows for Figure 1. All rows must share one workload digest and scope."""
    rows = []
    edir = root / DATA / "evidence"
    for p in sorted(edir.glob("*.jsonl")) if edir.is_dir() else []:
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            miss = [k for k in EVIDENCE_REQUIRED if r.get(k) in (None, "", [])]
            if miss:
                raise PaperDataError(f"{p.name}:{n}: evidence row missing {miss}")
            if r["label"] not in LABELS:
                raise PaperDataError(f"{p.name}:{n}: bad label {r['label']!r}")
            med = statistics.median(float(x) for x in r["cpu_reps"])
            if abs(med - float(r["cpu_s_per_audio_s"])) > 1e-9 * max(1.0, med):
                raise PaperDataError(f"{p.name}:{n}: cpu_s_per_audio_s is not the median of cpu_reps")
            rows.append(r)
    if len({(r["workload_digest"], r["scope"], r["hardware"]) for r in rows}) > 1:
        raise PaperDataError("evidence rows mix workloads, scopes or hardware; Figure 1 needs one matched set")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise PaperDataError("duplicate evidence ids")
    rows.sort(key=lambda r: r["id"])
    # label anchor for Figure 1: right of the point, except near the right edge of the log axis
    out["cold_dev.csv"] = csv_text(
        ["id", "system", "role", "cpu", "wer", "label", "anchor"],
        [[r["id"], r["system"], r["role"], f"{float(r['cpu_s_per_audio_s']):.5f}", f"{float(r['wer_pct']):.2f}", r["label"],
          "east" if float(r["cpu_s_per_audio_s"]) > 0.3 else "west"] for r in rows])
    first = rows[0] if rows else None
    if rows:
        ours = [r for r in rows if r["role"] == "ours"]
        sentence = (f"Figure~1 places {len(rows)} systems measured on one matched cold workload"
                    f"{'' if ours else ' (no R1 row yet)'}.")
    else:
        sentence = "The matched cold-boundary rows are still pending, so Figure~1 shows its pending state."
    rec.add("ColdSummarySentence", sentence, unit="sentence", label="reported",
            configuration="generated from the number of admitted cold-DEV evidence rows",
            workload=first["workload"] if first else "pending (no evidence rows yet)",
            workload_digest=first["workload_digest"] if first else "pending",
            source="docs/paper/data/evidence/*.jsonl")
    rec.add("ColdRowsCount", str(len(rows)), raw=len(rows), unit="rows", label="reported",
            configuration="number of matched cold-DEV evidence rows in Figure 1 (0 = pending)",
            workload=first["workload"] if first else "pending (no evidence rows yet)",
            workload_digest=first["workload_digest"] if first else "pending",
            source="docs/paper/data/evidence/*.jsonl")


# ---------------------------------------------------------------- outputs


def tex_escape(s: str) -> str:
    return s.replace("\\", "/").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("_", "\\_")


def metrics_tex(rec: Records) -> str:
    lines = [
        "% Generated by python/ultra/tools/paper_data.py. Do not edit by hand.",
        "% Every macro is a bare number (no unit, no %). Provenance: docs/paper/data/SOURCES.md.",
        "% Labels: exploratory = shared-host / non-qualifying measurement; reported = configuration",
        "% or descriptive constant; qualified = from a qualifying window (none yet).",
        "\\providecommand{\\PaperMissing}[1]{\\textcolor{red}{??}}",
        "",
    ]
    for rid in sorted(rec.rows):
        r = rec.rows[rid]
        missing = r.get("pending") or any(r.get(k) in (None, "") for k in REQUIRED)
        body = f"\\PaperMissing{{{rid}}}" if missing else r["value"]
        lines.append(f"\\newcommand{{\\{rid}}}{{{body}}}% [{r['label']}] {r['unit']}")
    cap_r = ("Warm single-harness rehearsal (exploratory): R0 Zipformer INT8, the dense finalist (average of steps "
             "45/50/55k) and the multirate model at step 50k, all prepared INT8, in one \\texttt{ultra bench-suite} "
             "executable on a shared AMD Threadripper PRO 3995WX (Zen~2). Density: \\DenWorkers{} single-threaded workers "
             "on one CCX, \\DenRungSeconds{}~s ladder rungs, one repeat; R0 and multirate densities were identical in "
             "\\DenReproRuns{} runs; the earlier dense step-45k checkpoint reached \\DenDenseFortyFiveRangeLo--"
             "\\DenDenseFortyFiveRangeHi{} per CCX over two runs. Left: whole-service CPU-seconds per audio-second at the "
             "matched \\MatchedRung-stream rung against WER on the panel's 400 dev-clean utterances, measured in the same "
             "harness and executable. "
             "Right: highest sustained streams per CCX, with each checkpoint's WER.")
    cap_c = ("Cold process-tree CPU cost against pooled DEV WER on the frozen 730-utterance DEV panel, "
             "one pinned physical core, median of three repetitions, AMD Threadripper PRO 3995WX (Zen~2).")
    cap_f = ("Warm, single harness, exploratory, Zen~2 (AMD Threadripper PRO 3995WX, shared host). Live streams per "
             "core sustained at p99 lateness of at most \\LatenessBoundMs\\,ms (top axis: per CCX, \\DenWorkers{} "
             "single-threaded workers; \\LadderScope) against WER on the 400 dev-clean utterances of the DEV panel, "
             "both measured in one \\texttt{ultra bench-suite} executable for the same checkpoints. Whiskers run to "
             "the first failed rung. Hollow: the earlier dense step-45k checkpoint, "
             "\\DenDenseFortyFiveRangeLo--\\DenDenseFortyFiveRangeHi{} streams per CCX over two runs. Dashed: a later "
             "dense checkpoint's WER, not timed.")
    lines += ["", f"\\newcommand{{\\FrontierWarmCaption}}{{{cap_f}}}"]
    lines += ["", f"\\newcommand{{\\RehearsalCaption}}{{{cap_r}}}", f"\\newcommand{{\\ColdCaption}}{{{cap_c}}}", ""]
    return "\n".join(lines)


def sources_md(rec: Records) -> str:
    out = [
        "# Paper number sources",
        "",
        "Generated by `python/ultra/tools/paper_data.py`; do not edit. Each row is one `metrics.tex` macro.",
        "Labels: **exploratory** = measured on a shared host or outside a qualifying window (all measurements so far);",
        "**reported** = a configuration or descriptive constant; **qualified** = from a qualifying window (none yet).",
        "`AGENT_CHAT.md` is the gitignored coordination log of the main checkout; its lines are append-only.",
        "",
        "| Macro | Value | Unit | Label | Configuration | Workload (digest) | Source | Line |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def cell(s) -> str:
        return ("" if s is None else str(s)).replace("|", "\\|").replace("\n", " ")

    for rid in sorted(rec.rows):
        r = rec.rows[rid]
        dig = r["workload_digest"] or "MISSING"
        dig = dig if len(dig) < 20 else dig[:12] + "\u2026"
        out.append("| " + " | ".join(cell(x) for x in (
            f"`\\{rid}`", r["value"] if r["value"] not in (None, "") else ("**PENDING**: " + r["pending"] if r.get("pending") else "**MISSING**"), r["unit"], r["label"],
            r["configuration"], f"{r['workload']} ({dig})", r["source"], r["line"] or "")) + " |")
    return "\n".join(out) + "\n"


def records_jsonl(rec: Records) -> str:
    return "".join(json.dumps(rec.rows[k], sort_keys=True, ensure_ascii=False) + "\n" for k in sorted(rec.rows))


def build(root: Path = REPO) -> tuple[dict[str, str], list[str]]:
    """Return ({relative file name: content}, missing record ids)."""
    verify_manifest(root)
    rec = Records()
    out: dict[str, str] = {}
    cur = add_curated(root, rec)
    add_codex_evidence(root, rec)
    add_panel5(root, rec)
    add_cluster_gap(root, rec)
    add_derived(rec)
    add_rehearsal(root, rec, out)
    add_curves(root, rec, out)
    add_sweeps(root, rec, out)
    add_tables(cur, out)
    accuracy_table(rec, out)
    load_evidence(root, rec, out)
    out["metrics.tex"] = metrics_tex(rec)
    out["SOURCES.md"] = sources_md(rec)
    out["records.jsonl"] = records_jsonl(rec)
    return dict(sorted(out.items())), rec.missing()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / DATA, help="output directory")
    ap.add_argument("--check", action="store_true", help="compare against --out instead of writing")
    ap.add_argument("--write-manifest", action="store_true", help="rebuild raw/MANIFEST.txt from disk")
    args = ap.parse_args(argv)
    if args.write_manifest:
        write_manifest(REPO)
    files, missing = build(REPO)
    if args.check:
        bad = [n for n, t in files.items() if not (args.out / n).is_file() or (args.out / n).read_text(encoding="utf-8") != t]
        for n in bad:
            print(f"stale: {n}", file=sys.stderr)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for n, t in files.items():
            (args.out / n).write_text(t, encoding="utf-8", newline="\n")
        bad = []
    for m in missing:
        print(f"MISSING required value: {m}", file=sys.stderr)
    for line in (files["records.jsonl"].splitlines()):
        r = json.loads(line)
        if r.get("pending"):
            print(f"PENDING (prints red ?? in the PDF): {r['id']}: {r['pending']}", file=sys.stderr)
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
