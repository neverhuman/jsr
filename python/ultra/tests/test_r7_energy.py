"""R7 energy ingest: summarizer validity guards, staging custody, and the real evidence loader's rejections."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tools import paper_data as pd
from tools.paper_data import evidence
from tools.paper_data.common import PaperDataError

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "paper_data"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r7 = load_tool("r7_summarize")
stage = load_tool("r7_stage")


def receipt(j, clean=True, sib=0.0, elsewhere=0):
    return {
        "joules_per_audio_s": j, "cpu_s_per_audio_s": 0.05, "flags": [] if clean else ["idle_drift_core"],
        "clean_repetitions": {"core": 1 if clean else 0, "package": 0},
        "reps": [{"wall_seconds": 10.0, "audio_seconds": 100.0, "core": {"idle_watts": 0.1},
                  "package": {"idle_watts": 60.0}, "mean_mhz": 4200,
                  "busy": {"sibling_busy_fraction_run": sib, "elsewhere_busy_cpus_run": elsewhere}}],
    }


def put(root, block, key, j, clean=True):
    d = root / f"block{block}"
    d.mkdir(exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(receipt(j, clean)))


def full(root, overrides=None, drop=()):
    """A complete campaign: every arm in blocks 1-3 plus both anchors. overrides: {(block, key): (j, clean)}."""
    overrides = overrides or {}
    for key in r7.IDS:
        for b in (1, 2, 3):
            if (b, key) not in drop:
                put(root, b, key, *overrides.get((b, key), (1.0, True)))
    for key, j in (("anchor730-live", 0.328), ("anchor730-zip", 0.573)):
        if (1, key) not in drop:
            put(root, 1, key, *overrides.get((1, key), (j, True)))


def run(root, monkeypatch, out):
    monkeypatch.setattr(sys, "argv", ["r7_summarize.py", str(root), "--evidence", str(out)])
    r7.main()


def test_valid_run_median_over_clean_blocks_and_grey_arm(tmp_path, monkeypatch):
    full(tmp_path, {(2, "vosk-small"): (9.0, False), (3, "vosk-small"): (1.2, True),
                    (1, "vosk-large"): (5.0, False), (2, "vosk-large"): (6.0, False), (3, "vosk-large"): (7.0, False)})
    out = tmp_path / "evidence.jsonl"
    run(tmp_path, monkeypatch, out)
    rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert "vosk-large-zen2-t1" not in rows
    row = rows["vosk-small-zen2-t1"]
    assert row["kind"] == "r7_energy" and row["n_clean"] == 2 and row["clean_blocks"] == [1, 3]
    assert row["energy_j_per_audio_s"] == pytest.approx(1.1)
    assert row["anchor_check"] == "VALID"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["arms"]["vosk-large"]["grey"] is True and summary["validity"] == "VALID"
    assert not list(tmp_path.glob("evidence.jsonl.tmp"))


BAD_RUNS = {
    "missing_anchor": dict(drop=[(1, "anchor730-zip")]),
    "unclean_anchor": dict(overrides={(1, "anchor730-live"): (0.328, False)}),
    "out_of_band_anchor": dict(overrides={(1, "anchor730-zip"): (0.573 * 1.06, True)}),
    "partial_blocks": dict(drop=[(3, "vosk-small"), (2, "whispercpp-stream-base")]),
}


@pytest.mark.parametrize("case", sorted(BAD_RUNS))
def test_void_or_partial_run_exits_nonzero_and_keeps_existing_evidence(tmp_path, monkeypatch, case):
    root = tmp_path / "receipts"
    root.mkdir()
    full(root, **BAD_RUNS[case])
    out = tmp_path / "evidence.jsonl"
    out.write_bytes(b'{"keep": "me"}\n')
    with pytest.raises(SystemExit) as exc:
        run(root, monkeypatch, out)
    assert exc.value.code not in (0, None)
    assert out.read_bytes() == b'{"keep": "me"}\n'
    assert json.loads((root / "summary.json").read_text())["validity"] == "VOID"


def test_void_run_creates_no_evidence_file(tmp_path, monkeypatch):
    full(tmp_path, drop=[(1, "anchor730-live")])
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert not out.exists()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_or_non_finite(tmp_path, monkeypatch, bad):
    full(tmp_path, {(b, "vosk-small"): (bad, True) for b in (1, 2, 3)})
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert not out.exists()


def test_rejects_duplicate_targets(tmp_path, monkeypatch):
    full(tmp_path)
    monkeypatch.setitem(r7.IDS, "vosk-large", r7.IDS["vosk-small"])
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert not out.exists()


# ---- staging custody ----

def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "b1-vosk-small.json").write_text(json.dumps(receipt(1.0)))
    (raw / "b1-vosk-small.unclean1.json").write_text(json.dumps(receipt(2.0, clean=False, sib=0.3)))
    (raw / "b1-vosk-large.failed1.json").write_text("not json at all")
    (raw / "b2-vosk-large.failed1.json").write_text(json.dumps({**receipt(3.0), "flags": ["nonzero_exit"]}))
    (raw / "b2-vosk-large.failed2.json").write_text(json.dumps({**receipt(3.0), "flags": ["nonzero_exit"]}))
    return raw


def test_stage_retains_failed_only_groups(tmp_path):
    raw, root = raw_dir(tmp_path), tmp_path / "root"
    stage.main([str(raw), str(root)])
    kept = sorted(p.name for p in (root / "attempts").iterdir())
    assert kept == sorted(p.name for p in raw.glob("*.json"))
    sel = json.loads((root / "selection.json").read_text())
    assert sel["b1-vosk-large"]["used"] is None and sel["b2-vosk-large"]["used"] is None
    assert len(sel["b2-vosk-large"]["attempts"]) == 2
    assert not (root / "block1" / "vosk-large.json").exists()
    assert sel["b1-vosk-small"]["used"] == "final" and (root / "block1" / "vosk-small.json").exists()


def test_stage_binds_sha_to_the_parsed_bytes(tmp_path):
    raw, root = raw_dir(tmp_path), tmp_path / "root"
    stage.main([str(raw), str(root)])
    sel = json.loads((root / "selection.json").read_text())
    staged = (root / "block1" / "vosk-small.json").read_bytes()
    assert sel["b1-vosk-small"]["used_sha256"] == hashlib.sha256(staged).hexdigest()
    for group in sel.values():
        for a in group["attempts"]:
            assert a["sha256"] == hashlib.sha256((root / "attempts" / a["file"]).read_bytes()).hexdigest()


def write_sums(raw, tamper=None, omit=None):
    lines = []
    for p in sorted(raw.glob("*.json")):
        if p.name == omit:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{'0' * 64 if p.name == tamper else h}  {p.name}\n")
    (raw / "SHA256SUMS").write_text("".join(lines))


def test_stage_verifies_archive_sums(tmp_path):
    raw = raw_dir(tmp_path)
    write_sums(raw)
    stage.main([str(raw), str(tmp_path / "ok")])
    write_sums(raw, tamper="b1-vosk-small.json")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        stage.main([str(raw), str(tmp_path / "bad")])
    write_sums(raw, omit="b2-vosk-large.failed2.json")
    with pytest.raises(SystemExit, match="not listed"):
        stage.main([str(raw), str(tmp_path / "unlisted")])


def test_stage_refuses_a_used_output_dir_unless_asked(tmp_path):
    raw, root = raw_dir(tmp_path), tmp_path / "root"
    stage.main([str(raw), str(root)])
    stray = root / "notes.txt"
    stray.write_text("mine")
    (root / "block1" / "stale.json").write_text("{}")
    with pytest.raises(SystemExit, match="already holds"):
        stage.main([str(raw), str(root)])
    assert (root / "block1" / "stale.json").exists()
    stage.main([str(raw), str(root), "--clear-owned"])
    assert stray.read_text() == "mine" and not (root / "block1" / "stale.json").exists()


# ---- the real loader ----

def r7_line(**kw):
    r = {"kind": "r7_energy", "id": "sys-a-zen2-t1", "energy_method": "core_idle_subtracted",
         "energy_j_per_audio_s": 0.5, "n_clean": 3, "_where": "r7.jsonl:1"}
    r.update(kw)
    return r


def cost_row(i="sys-a-zen2-t1"):
    return {"id": i, "_where": "cold.jsonl:1"}


def merge(*sidecars):
    rows = [cost_row()]
    evidence._merge(rows, list(sidecars))
    return rows[0]


def test_loader_accepts_a_valid_r7_row():
    row = merge(r7_line())
    assert row["energy_idle_subtracted"] == 0.5


@pytest.mark.parametrize("bad", [0, 0.0, -0.1, float("nan"), float("inf"), "0.5", True, None])
def test_loader_rejects_bad_energy(bad):
    with pytest.raises(PaperDataError, match="finite positive"):
        merge(r7_line(energy_j_per_audio_s=bad))


def test_loader_rejects_duplicate_targets():
    with pytest.raises(PaperDataError, match="duplicate R7"):
        merge(r7_line(), r7_line(energy_j_per_audio_s=0.6, _where="r7.jsonl:2"))


@pytest.mark.parametrize("marker", [{"anchor_check": "VOID"}, {"validity": "void"}, {"void": True}])
def test_loader_rejects_void_rows(marker):
    with pytest.raises(PaperDataError, match="VOID"):
        merge(r7_line(**marker))


def write_evidence(tmp_path, name, lines):
    ev = tmp_path / pd.DATA / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / name).write_text("".join(json.dumps(x) + "\n" for x in lines))
    return tmp_path


@pytest.mark.parametrize("kind", [None, "energy", "cost", "lag"])
def test_loader_rejects_missing_or_wrong_kind(tmp_path, kind):
    line = {k: v for k, v in r7_line().items() if k not in ("kind", "_where")}
    if kind:
        line["kind"] = kind
    with pytest.raises(PaperDataError, match="needs kind 'r7_energy'"):
        evidence._read(write_evidence(tmp_path, "r7.jsonl", [line]))


def test_loader_rejects_void_file_and_void_file_name(tmp_path):
    body = {k: v for k, v in r7_line().items() if k != "_where"}
    with pytest.raises(PaperDataError, match="not VALID"):
        evidence._read(write_evidence(tmp_path, "r7.jsonl", [{"kind": "r7_validity", "validity": "VOID"}, body]))
    with pytest.raises(PaperDataError, match="VOID by its name"):
        evidence._read(write_evidence(tmp_path / "b", "r7-VOID.jsonl", [body]))


def test_loader_reads_a_valid_file(tmp_path):
    body = {k: v for k, v in r7_line().items() if k != "_where"}
    _, sidecars, _ = evidence._read(write_evidence(tmp_path, "r7.jsonl", [{"kind": "r7_validity", "validity": "VALID"}, body]))
    assert len(sidecars) == 1
