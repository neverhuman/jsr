"""Readers for evidence files: Codex evidence.json, same-harness panel scores and the cluster gap."""

from __future__ import annotations

from pathlib import Path

from .common import DATA, PANEL_DIGEST, PaperDataError, RAW, fmt, read_json, thousands
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
    ("dense_avg3new_clean-greedy.json", "D2 avg-3 (55/60/65k) dense INT8 greedy", "dev-clean"),
    ("dense_avg3new_clean-beam4.json", "D2 avg-3 (55/60/65k) dense INT8 beam 4", "dev-clean"),
    ("dense_avg3new_other-greedy.json", "D2 avg-3 (55/60/65k) dense INT8 greedy", "dev-other"),
    ("dense_avg3new_other-beam4.json", "D2 avg-3 (55/60/65k) dense INT8 beam 4", "dev-other"),
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
    score = "scored by ultra-score 38eacc04 (ultra-en-wer-norm/v1)"
    exe_old = "ultra ce58952 (same executable as Jobs C/D/E)"
    src = "docs/paper/data/raw/panel5_rust_scores/"
    halves = {"Clean": ("DEV selection panel, dev-clean half (400 utt)", "clean"),
              "Other": ("DEV selection panel, dev-other half (330 utt)", "other")}

    def add(rid, files, system):
        exe = "ultra 43f31a1 (Job G panel runs g3a/g3b)" if files[0].startswith("dense_avg3new") else exe_old
        conf = f"bench-suite --panel, threads 1, grid 320 ms, {exe}; {score}"
        e = sum(rep[f][0] for f in files)
        n = sum(rep[f][1] for f in files)
        wl = "DEV selection panel, pooled (730 utt)" if len(files) == 2 else halves["Clean" if "clean" in files[0] else "Other"][0]
        rec.add(rid, fmt(100 * e / n, 2), raw=100 * e / n, unit="% WER", label="exploratory",
                configuration=f"{system}; {'pooled errors / ref words over both halves; ' if len(files) == 2 else ''}{conf}",
                workload=wl, workload_digest=PANEL_DIGEST, source=src + " + ".join(files))

    add("WerRzeroPanel", ["rzero_dev-clean.json", "rzero_dev-other.json"], "R0 Zipformer INT8")
    add("WerRzeroPanelClean", ["rzero_dev-clean.json"], "R0 Zipformer INT8")
    add("WerRzeroPanelOther", ["rzero_dev-other.json"], "R0 Zipformer INT8")
    old = "D2 avg-3 (45/50/55k) dense prepared INT8, the Job E ladder's dense row (history)"
    add("WerRehearsalDenseOld", ["dense_avg3_dev-clean.json"], old)
    add("WerRehearsalDenseOldOther", ["dense_avg3_dev-other.json"], old)
    add("WerRehearsalDenseOldPooled", ["dense_avg3_dev-clean.json", "dense_avg3_dev-other.json"], old)
    new = "D2 avg-3 (55/60/65k) dense prepared INT8, the headline dense checkpoint (Job G, ultra 43f31a1)"
    for stem, tag in (("Dense", "greedy"), ("DenseBeam", "beam4")):
        sysd = f"{new}, {'greedy' if tag == 'greedy' else 'beam 4'}"
        add(f"WerRehearsal{stem}", [f"dense_avg3new_clean-{tag}.json"], sysd)
        add(f"WerRehearsal{stem}Other", [f"dense_avg3new_other-{tag}.json"], sysd)
        add(f"WerRehearsal{stem}Pooled", [f"dense_avg3new_clean-{tag}.json", f"dense_avg3new_other-{tag}.json"], sysd)
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
