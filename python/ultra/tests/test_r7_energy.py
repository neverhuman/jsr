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


FULL_PANEL_S = 5259.56  # audio-seconds of the 730-utterance DEV panel, what an anchor must decode


def receipt(j, clean=True, sib=0.0, elsewhere=0, audio=100.0):
    return {
        "joules_per_audio_s": j, "cpu_s_per_audio_s": 0.05, "flags": [] if clean else ["idle_drift_core"],
        "clean_repetitions": {"core": 1 if clean else 0, "package": 0},
        "reps": [{"wall_seconds": 10.0, "audio_seconds": audio, "core": {"idle_watts": 0.1},
                  "package": {"idle_watts": 60.0}, "mean_mhz": 4200,
                  "busy": {"sibling_busy_fraction_run": sib, "elsewhere_busy_cpus_run": elsewhere}}],
    }


def put(root, block, key, j, clean=True, audio=None):
    d = root / f"block{block}"
    d.mkdir(exist_ok=True)
    if audio is None:
        audio = FULL_PANEL_S if key.startswith("anchor730") else 100.0
    (d / f"{key}.json").write_text(json.dumps(receipt(j, clean, audio=audio)))


def manifest(root):
    """selection.json as r7_stage.py writes it: the sha256 of every staged receipt."""
    sel = {}
    for f in root.glob("block[123]/*.json"):
        sel[f"b{f.parent.name[5:]}-{f.stem}"] = {"used_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (root / "selection.json").write_text(json.dumps(sel))


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
    manifest(root)


def run(root, monkeypatch, out):
    monkeypatch.setattr(sys, "argv", ["r7_summarize.py", str(root), "--evidence", str(out)])
    r7.main()


def arm_rows(out):
    return {r["id"]: r for r in map(json.loads, out.read_text().splitlines()) if r["kind"] == "r7_energy"}


def assert_void_marker(out):
    """After a failed run the evidence file holds one VOID line, and the paper loader refuses it."""
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    head = json.loads(lines[0])
    assert head["kind"] == "r7_validity" and head["validity"] == "VOID" and head["problems"]
    root = out.parent / "paper"
    ev = root / pd.DATA / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "xbabe0-r7-energy.jsonl").write_bytes(out.read_bytes())
    with pytest.raises(PaperDataError, match="not VALID"):
        evidence._read(root)


def test_valid_run_median_over_clean_blocks_and_grey_arm(tmp_path, monkeypatch):
    full(tmp_path, {(2, "vosk-small"): (9.0, False), (3, "vosk-small"): (1.2, True),
                    (1, "vosk-large"): (5.0, False), (2, "vosk-large"): (6.0, False), (3, "vosk-large"): (7.0, False)})
    out = tmp_path / "evidence.jsonl"
    run(tmp_path, monkeypatch, out)
    rows = arm_rows(out)
    head = json.loads(out.read_text().splitlines()[0])
    assert head["kind"] == "r7_validity" and head["validity"] == "VALID" and head["arms"] == len(rows)
    assert head["anchors"]["live"]["audio_seconds"] == FULL_PANEL_S
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
def test_void_or_partial_run_exits_nonzero_and_replaces_stale_valid_evidence(tmp_path, monkeypatch, case):
    good = tmp_path / "good"
    good.mkdir()
    full(good)
    out = tmp_path / "evidence.jsonl"
    run(good, monkeypatch, out)  # a VALID file from an earlier run, able to colour the figure
    assert arm_rows(out)
    root = tmp_path / "receipts"
    root.mkdir()
    full(root, **BAD_RUNS[case])
    with pytest.raises(SystemExit) as exc:
        run(root, monkeypatch, out)
    assert exc.value.code not in (0, None)
    assert_void_marker(out)
    assert not list(tmp_path.glob("evidence.jsonl.tmp"))
    assert json.loads((root / "summary.json").read_text())["validity"] == "VOID"


@pytest.mark.parametrize("audio", [597.09, FULL_PANEL_S - 0.06, FULL_PANEL_S + 0.06])
def test_an_anchor_that_is_not_the_full_panel_voids_the_run(tmp_path, monkeypatch, audio):
    full(tmp_path)
    put(tmp_path, 1, "anchor730-zip", 0.573, audio=audio)
    manifest(tmp_path)
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert_void_marker(out)
    assert any("not the full panel" in p for p in json.loads((tmp_path / "summary.json").read_text())["problems"])


def test_an_anchor_within_the_audio_tolerance_is_the_full_panel(tmp_path, monkeypatch):
    full(tmp_path)
    put(tmp_path, 1, "anchor730-zip", 0.573, audio=FULL_PANEL_S + 0.04)
    manifest(tmp_path)
    run(tmp_path, monkeypatch, tmp_path / "evidence.jsonl")


def test_receipt_changed_after_staging_is_refused_before_any_output(tmp_path, monkeypatch):
    root = tmp_path / "receipts"
    root.mkdir()
    full(root)
    out = tmp_path / "evidence.jsonl"
    out.write_bytes(b"old\n")
    put(root, 1, "anchor730-live", 0.329)  # an anchor edited after the manifest was written
    with pytest.raises(SystemExit, match="do not match selection.json"):
        run(root, monkeypatch, out)
    assert out.read_bytes() == b"old\n" and not (root / "summary.json").exists()


def test_missing_manifest_is_refused(tmp_path, monkeypatch):
    full(tmp_path)
    (tmp_path / "selection.json").unlink()
    with pytest.raises(SystemExit, match="selection.json"):
        run(tmp_path, monkeypatch, tmp_path / "evidence.jsonl")


def test_evidence_hashes_are_of_the_bytes_parsed(tmp_path, monkeypatch):
    full(tmp_path)
    out = tmp_path / "evidence.jsonl"
    run(tmp_path, monkeypatch, out)
    for row in arm_rows(out).values():
        for rc in row["receipts"]:
            assert rc["sha256"] == hashlib.sha256((tmp_path / rc["path"]).read_bytes()).hexdigest()
    summary = json.loads((tmp_path / "summary.json").read_text())
    live = tmp_path / summary["anchors"]["live"]["receipt"]
    assert summary["anchors"]["live"]["receipt_sha256"] == hashlib.sha256(live.read_bytes()).hexdigest()


def test_receipts_swapped_after_verification_do_not_reach_the_output(tmp_path, monkeypatch):
    full(tmp_path)
    sel = json.loads((tmp_path / "selection.json").read_text())
    real_check = r7.check_manifest

    def check_then_swap(root, keys):
        snap = real_check(root, keys)
        for b in (1, 2, 3):
            put(root, b, "vosk-small", 99.0)  # the bytes change after the check, before the values are used
        put(root, 1, "anchor730-live", 9.9)
        return snap

    monkeypatch.setattr(r7, "check_manifest", check_then_swap)
    out = tmp_path / "evidence.jsonl"
    run(tmp_path, monkeypatch, out)
    row = arm_rows(out)["vosk-small-zen2-t1"]
    assert row["energy_j_per_audio_s"] == pytest.approx(1.0) and row["blocks"] == [1.0, 1.0, 1.0]
    assert row["receipt_sha256"] == sel["b1-vosk-small"]["used_sha256"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["validity"] == "VALID"
    assert summary["anchors"]["live"]["j_per_audio_s"] == pytest.approx(0.328)
    assert summary["anchors"]["live"]["receipt_sha256"] == sel["b1-anchor730-live"]["used_sha256"]


def test_each_receipt_is_read_once(tmp_path, monkeypatch):
    full(tmp_path)
    reads = []
    real = r7.load
    monkeypatch.setattr(r7, "load", lambda root, b, k: reads.append((b, k)) or real(root, b, k))
    run(tmp_path, monkeypatch, tmp_path / "evidence.jsonl")
    assert len(reads) == len(set(reads))


def test_void_run_writes_only_the_void_marker(tmp_path, monkeypatch):
    full(tmp_path, drop=[(1, "anchor730-live")])
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert_void_marker(out)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_or_non_finite(tmp_path, monkeypatch, bad):
    full(tmp_path, {(b, "vosk-small"): (bad, True) for b in (1, 2, 3)})
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert_void_marker(out)


def test_rejects_duplicate_targets(tmp_path, monkeypatch):
    full(tmp_path)
    monkeypatch.setitem(r7.IDS, "vosk-large", r7.IDS["vosk-small"])
    out = tmp_path / "evidence.jsonl"
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, out)
    assert_void_marker(out)


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
         "energy_j_per_audio_s": 0.5, "n_clean": 3, "anchor_check": "VALID", "_where": "r7.jsonl:1"}
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


@pytest.mark.parametrize("bad", ["valid", "Valid", "", None, True, "PENDING"])
def test_loader_needs_anchor_check_valid_exactly(bad):
    with pytest.raises(PaperDataError, match="anchor_check 'VALID'"):
        merge(r7_line(anchor_check=bad))


def test_loader_rejects_a_missing_anchor_check():
    line = {k: v for k, v in r7_line().items() if k != "anchor_check"}
    with pytest.raises(PaperDataError, match="anchor_check 'VALID'"):
        merge(line)


def test_loader_rejects_disagreeing_aliases_and_resolves_one_key():
    with pytest.raises(PaperDataError, match="two targets"):
        merge(r7_line(row="sys-a-zen2-t1", id="other"))
    assert merge(r7_line(row="sys-a-zen2-t1", id="sys-a-zen2-t1"))["energy_idle_subtracted"] == 0.5
    only_row = {k: v for k, v in r7_line(row="sys-a-zen2-t1").items() if k != "id"}
    assert merge(only_row)["energy_idle_subtracted"] == 0.5


def test_loader_rejects_the_same_target_through_different_aliases():
    by_row = {k: v for k, v in r7_line(row="sys-a-zen2-t1").items() if k != "id"}
    with pytest.raises(PaperDataError, match="duplicate R7"):
        merge(by_row, r7_line(_where="r7.jsonl:2"))


def test_loader_rejects_a_missing_energy_field():
    line = {k: v for k, v in r7_line().items() if k != "energy_j_per_audio_s"}
    with pytest.raises(PaperDataError, match="no energy_j_per_audio_s"):
        merge(line)


@pytest.mark.parametrize("bad", [0, -1, "3", 2.5, True, None, [3]])
def test_loader_rejects_a_malformed_n_clean(bad):
    with pytest.raises(PaperDataError, match="n_clean"):
        merge(r7_line(n_clean=bad))


def test_loader_rejects_an_absent_n_clean():
    line = {k: v for k, v in r7_line().items() if k != "n_clean"}
    with pytest.raises(PaperDataError, match="n_clean"):
        merge(line)


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
