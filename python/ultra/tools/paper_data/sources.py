"""Readers for evidence files: Codex evidence.json, same-harness panel scores, cluster gap, cold-DEV rows."""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import (
    DATA,
    EVIDENCE_REQUIRED,
    LABELS,
    PANEL_DIGEST,
    PaperDataError,
    RAW,
    csv_text,
    fmt,
    read_json,
    thousands,
)
from .records import Records


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
