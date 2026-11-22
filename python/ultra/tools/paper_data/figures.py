"""Figure tables: training curves, decode sweeps, left-context and accuracy tables."""

from __future__ import annotations

from pathlib import Path
import json

from .common import (
    PANEL_DIGEST,
    PaperDataError,
    RAW,
    SWEEP_WORKLOAD,
    csv_text,
    fmt,
    read_csv,
    read_json,
    thousands,
)
from .records import Records


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
