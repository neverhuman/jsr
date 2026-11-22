"""Export compact training and validation curves from trainer `train.jsonl` logs.

The owner wants the training and validation score history of every finalist kept. Full logs
stay on the data drive. This tool writes small, committable CSVs per run:

    research/curves/<run>/train.csv   decimated train rows (every --every steps, plus the last)
    research/curves/<run>/eval.csv    every eval row (step, split, epoch, WER, S/D/I)
    research/curves/index.json        per-run summary and the sha256 of the source log

Usage:
    python -m tools.export_curves --out research/curves RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

TRAIN_FIELDS = (
    "step", "epoch", "loss", "rnnt", "ctc", "lr", "grad_norm",
    "audio_h_cum", "audio_h_per_h", "time",
)
EVAL_FIELDS = ("step", "split", "epoch", "wer", "sub", "del", "ins", "ref_words", "forced_advances", "time")


def _records(path: Path):
    """Yield the parseable JSON objects of a JSONL log; skip torn or partial lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def extract(path: Path, every: int) -> tuple[list[dict], list[dict], dict]:
    """Return (train_rows, eval_rows, summary) for one train.jsonl."""
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    audio_s = 0.0
    last_train: dict | None = None
    start: dict = {}
    for rec in _records(path):
        kind = rec.get("kind")
        if kind == "start" and not start:
            start = rec
        elif kind == "train":
            audio_s += float(rec.get("audio_s", 0.0))
            row = {
                "step": rec.get("step"),
                "epoch": (rec.get("pos") or {}).get("epoch"),
                "loss": rec.get("loss"),
                "rnnt": rec.get("rnnt"),
                "ctc": rec.get("ctc"),
                "lr": rec.get("lr"),
                "grad_norm": rec.get("grad_norm"),
                "audio_h_cum": round(audio_s / 3600.0, 3),
                "audio_h_per_h": rec.get("audio_h_per_h"),
                "time": rec.get("time"),
            }
            last_train = row
            step = rec.get("step")
            if isinstance(step, int) and every > 0 and step % every == 0:
                train_rows.append(row)
        elif kind == "eval":
            eval_rows.append({k: rec.get(k) for k in EVAL_FIELDS})
    if last_train is not None and (not train_rows or train_rows[-1]["step"] != last_train["step"]):
        train_rows.append(last_train)
    best: dict[str, dict] = {}
    for row in eval_rows:
        wer = row.get("wer")
        split = row.get("split")
        if wer is None or split is None:
            continue
        if split not in best or wer < best[split]["wer"]:
            best[split] = {"wer": wer, "step": row.get("step"), "epoch": row.get("epoch")}
    summary = {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "init_from": start.get("init_from"),
        "params": start.get("params"),
        "last_step": last_train["step"] if last_train else None,
        "last_epoch": last_train["epoch"] if last_train else None,
        "audio_h_seen": round(audio_s / 3600.0, 1),
        "n_eval": len(eval_rows),
        "best_eval": best,
    }
    return train_rows, eval_rows, summary


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def export(run_dirs: list[Path], out: Path, every: int) -> dict:
    """Write per-run CSVs under `out` and merge summaries into `out/index.json`."""
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for run_dir in run_dirs:
        log = run_dir / "train.jsonl"
        if not log.is_file():
            raise FileNotFoundError(f"{log} not found")
        name = run_dir.name.removesuffix("_copy")
        train_rows, eval_rows, summary = extract(log, every)
        dest = out / name
        dest.mkdir(exist_ok=True)
        _write_csv(dest / "train.csv", TRAIN_FIELDS, train_rows)
        _write_csv(dest / "eval.csv", EVAL_FIELDS, eval_rows)
        summary["source"] = str(log)
        index[name] = summary
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("research/curves"))
    ap.add_argument("--every", type=int, default=250, help="keep train rows at multiples of this step")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    index = export(args.run_dirs, args.out, args.every)
    for name in (d.name.removesuffix("_copy") for d in args.run_dirs):
        s = index[name]
        print(f"{name}: step {s['last_step']} epoch {s['last_epoch']} evals {s['n_eval']} best {s['best_eval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
