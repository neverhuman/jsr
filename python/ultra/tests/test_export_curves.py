import csv
import json

from tools.export_curves import export, extract


def _log(tmp_path):
    run = tmp_path / "RUN_copy"
    run.mkdir()
    lines = [
        {"kind": "start", "init_from": "x.pt", "params": 10, "time": 0},
        *[
            {"kind": "train", "step": s, "pos": {"epoch": s // 300}, "loss": 1.0 / s, "audio_s": 3600.0, "time": s}
            for s in range(50, 1050, 50)
        ],
        {"kind": "eval", "step": 500, "split": "dev-clean", "epoch": 1, "wer": 20.0},
        {"kind": "eval", "step": 1000, "split": "dev-clean", "epoch": 3, "wer": 15.5},
    ]
    text = "\n".join(json.dumps(r) for r in lines) + "\n{torn"
    (run / "train.jsonl").write_text(text)
    return run


def test_extract_decimates_and_keeps_every_eval(tmp_path):
    run = _log(tmp_path)
    train, evals, summary = extract(run / "train.jsonl", every=250)
    assert [r["step"] for r in train] == [250, 500, 750, 1000]
    assert len(evals) == 2
    assert summary["best_eval"]["dev-clean"] == {"wer": 15.5, "step": 1000, "epoch": 3}
    assert summary["audio_h_seen"] == 20.0
    assert summary["last_step"] == 1000


def test_last_train_row_is_kept_when_off_grid(tmp_path):
    run = _log(tmp_path)
    train, _, _ = extract(run / "train.jsonl", every=300)
    assert train[-1]["step"] == 1000


def test_export_writes_csvs_and_index(tmp_path):
    run = _log(tmp_path)
    out = tmp_path / "curves"
    index = export([run], out, every=250)
    assert "RUN" in index and len(index["RUN"]["source_sha256"]) == 64
    with (out / "RUN" / "eval.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["wer"] for r in rows] == ["20.0", "15.5"]
