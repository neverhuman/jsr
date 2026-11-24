"""Long-run training dynamics of the release lineage: full-DEV pooled WER against cumulative training audio.

Stage 1 trains on LibriSpeech 960 h only (EMA checkpoints, PyTorch streaming decoder, `ultra-score`).
Stage 2 adds Libri-Light pseudo-labelled by R0, then a cooldown; its points are the INT8 engine
(release decoder, same scorer): live greedy c=4 and the utterance-segmented final pass.
Inputs: raw/training/ (sha256-pinned). Outputs: data/training_curve.csv and \\Train* macros.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

from .common import DATA, RAW, PaperDataError, fmt, sha256_file, thousands
from .records import Records

TR = RAW / "training"
STAGE1_STEPS = (10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 49000)
# Stage-2 points: (file, exposure after stage 1 in audio-h, label, live reader, final reader).
# Exposures are the branch-point / cut definitions of docs/ultra/releases/v3.0.0.md (Training, lines 224-226):
# small BP1 10k, small BP2 20k (S2a ends at 20,504.5), medium BP1 = S2a + 10k, cooldown = 41,658 + 10,000.
S2A_H, S2B_H, COOL_H = 20504.5, 21153.7, 10000.0


def _rr(d: dict) -> tuple[float, float]:  # release-row verdict: live = system_a only for the stage-1 end row
    return 100.0 * d["score_report"]["system_a"]["pooled_wer"], float(d["pooled_wer_pct"])


def _row(d: dict) -> tuple[float, float]:
    return float(d["provisional"]["wer_pooled"]), float(d["committed"]["wer_pooled"])


def _summary(key: str):
    def read(d: dict) -> tuple[float, float]:
        fd = d["full_dev"]
        return float(fd[f"{key}_provisional"]["pooled"]["wer"]), float(fd[f"{key}_committed"]["pooled"]["wer"])
    return read


def add_training(root: Path, rec: Records, out: dict[str, str]) -> None:
    base = root / TR
    if not base.is_dir():
        return
    expo = {int(r["step"]): float(r["audio_h_cum"]) for r in csv.DictReader((base / "stage1" / "exposure.csv").open())}
    rows = []
    for s in STAGE1_STEPS:
        for c in (4, 16):
            p = base / "stage1" / f"step{s}.c{c}.score.json"
            if not p.is_file():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d["system_a"]["utterances"] != 5567:
                raise PaperDataError(f"{p}: not full DEV")
            rows.append({"stage": 1, "series": f"s1c{c}", "audio_kh": expo[s] / 1000.0, "wer": 100.0 * d["system_a"]["pooled_wer"]})
    s1_end = expo[STAGE1_STEPS[-1]]
    p0 = json.loads((base / "stage2" / "p0-stage1-end.release-row.json").read_text(encoding="utf-8"))
    p4 = json.loads((base / "stage2" / "p4-cooldown.release-row.json").read_text(encoding="utf-8"))
    p4i = json.loads((base / "stage2" / "p4-cooldown.int8.json").read_text(encoding="utf-8"))
    s2 = [
        (0.0, _rr(p0)[0], _rr(p0)[1]),
        (10000.0, *_row(json.loads((base / "stage2" / "p1-small-10k.row.json").read_text(encoding="utf-8")))),
        (20000.0, *_summary("s2a_bp2")(json.loads((base / "stage2" / "p2-small-20k.summary.json").read_text(encoding="utf-8")))),
        (S2A_H + 10000.0, *_summary("s2b_bp1")(json.loads((base / "stage2" / "p3-medium-10k.summary.json").read_text(encoding="utf-8")))),
        (S2A_H + S2B_H + COOL_H, float(p4i["int8_pooled_wer_pct"]), float(p4["pooled_wer_pct"])),
    ]
    for h, live, final in s2:
        rows.append({"stage": 2, "series": "s2live", "audio_kh": (s1_end + h) / 1000.0, "wer": live})
        rows.append({"stage": 2, "series": "s2final", "audio_kh": (s1_end + h) / 1000.0, "wer": final})
    lines = ["stage,series,audio_kh,wer"] + [f"{r['stage']},{r['series']},{r['audio_kh']:.3f},{r['wer']:.3f}" for r in rows]
    out["training_curve.csv"] = "\n".join(lines) + "\n"

    def s1(step: int, c: int) -> float:
        return next(r["wer"] for r in rows if r["series"] == f"s1c{c}" and abs(r["audio_kh"] - expo[step] / 1000.0) < 1e-9)

    dig = sha256_file(base / "stage1" / "exposure.csv")
    src1 = f"{TR}/stage1/ (score reports) + exposure.csv (cumulative audio_s of the stage-1 train.jsonl, sha256 89d75fbd)"
    src2 = f"{TR}/stage2/ (release decoder bb1071a, ultra-score d784f8ca)"

    def put(rid, value, raw, unit, what, src):
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what,
                workload="full LibriSpeech DEV (5,567 utt)", workload_digest=dig, source=src)

    put("TrainStageOneEndKhRaw", fmt(s1_end / 1000.0, 3), s1_end / 1000.0, "kh", "stage 1 end (plot coordinate)", src1)
    total = s1_end + S2A_H + S2B_H + COOL_H
    put("TrainStageOneKh", thousands(fmt(s1_end, 0)), s1_end, "audio-h", "stage 1 logged training audio (960 h only)", src1)
    put("TrainStageOneEpochs", fmt(s1_end / 960.0, 0), s1_end / 960.0, "epochs", "stage 1 passes over 960 h", src1)
    put("TrainStageTwoKh", thousands(fmt(S2A_H + S2B_H, 0)), S2A_H + S2B_H, "audio-h", "stage 2 audio (960 h + Libri-Light)", src2)
    put("TrainCoolKh", thousands(fmt(COOL_H, 0)), COOL_H, "audio-h", "EMA cooldown audio", src2)
    put("TrainTotalKh", thousands(fmt(total, 0)), total, "audio-h", "total logged training audio of the release", src2)
    mid, end = s1(30000, 4), s1(49000, 4)
    put("TrainPlateauMid", fmt(mid, 2), mid, "% WER", "stage 1 live c=4 pooled at its 30k-step checkpoint", src1)
    put("TrainPlateauEnd", fmt(end, 2), end, "% WER", "stage 1 live c=4 pooled at its last checkpoint", src1)
    put("TrainPlateauMidKh", thousands(fmt(expo[30000], 0)), expo[30000], "audio-h", "exposure at the 30k-step checkpoint", src1)
    put("TrainEngineEnd", fmt(s2[0][1], 2), s2[0][1], "% WER", "stage 1 end, INT8 engine live c=4 pooled", src2)
    put("TrainLiveGain", fmt(s2[0][1] - s2[-1][1], 2), s2[0][1] - s2[-1][1], "pp", "stage 2 + cooldown live WER reduction", src2)
    put("TrainFinalGain", fmt(s2[0][2] - s2[-1][2], 2), s2[0][2] - s2[-1][2], "pp", "stage 2 + cooldown final WER reduction", src2)
    put("TrainSmallLive", fmt(s2[2][1], 2), s2[2][1], "% WER", "after 20k h with Libri-Light small, live", src2)
    put("TrainMediumLive", fmt(s2[3][1], 2), s2[3][1], "% WER", "after adding Libri-Light medium (10k h), live", src2)
