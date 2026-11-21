"""evaluate --panel verifies the dev-panel manifest and cache hashes and refuses non-dev splits;
the plain --split path refuses test-* splits; outputs record the panel manifest sha256."""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.evaluate import check_split, load_panel, main, panel_record, panel_utts  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture
def cache(monkeypatch, request):
    from ultra import libri

    d = os.path.join(REPO, "target", "test-train", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for split in ("dev-clean", "test-clean"):
        uids = np.array([f"{split}-u{i}" for i in range(4)])
        pcm = np.zeros(4 * 160, dtype=np.int16)
        np.savez(os.path.join(d, f"{split}.npz"), pcm=pcm, offsets=np.arange(0, 641, 160),
                 uids=uids, speakers=np.array([1, 1, 2, 2]), texts=np.array(["a"] * 4))
    monkeypatch.setattr(libri, "CACHE", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _panel(d, split="dev-clean", uids=None, tamper=None):
    path = os.path.join(d, f"{split}.npz")
    sp = {"split": split, "uids": uids or [f"{split}-u0", f"{split}-u2"], "speakers": 2,
          "utterances": 2, "speaker_of": {},
          "cache_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    panel = {"schema": "ultra.dev-panel/v1", "seed": "s", "per_speaker": 1, "splits": [sp]}
    panel["manifest_sha256"] = hashlib.sha256(
        json.dumps(panel, indent=1, sort_keys=True).encode()).hexdigest()
    if tamper:
        tamper(panel)
    out = os.path.join(d, "panel.json")
    with open(out, "w") as fh:
        json.dump(panel, fh, indent=1, sort_keys=True)
    return out


def test_valid_panel_loads(cache):
    assert sorted(u.uid for u in panel_utts(_panel(cache))) == ["dev-clean-u0", "dev-clean-u2"]


def test_manifest_tamper_refused(cache):
    p = _panel(cache, tamper=lambda x: x["splits"][0]["uids"].append("dev-clean-u1"))
    with pytest.raises(ValueError, match="manifest_sha256"):
        panel_utts(p)


def test_cache_mismatch_refused(cache):
    p = _panel(cache)
    with open(os.path.join(cache, "dev-clean.npz"), "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ValueError, match="cache sha256"):
        panel_utts(p)


def test_test_split_refused(cache):
    with pytest.raises(ValueError, match="only dev"):
        panel_utts(_panel(cache, split="test-clean"))


def test_missing_uid_refused(cache):
    with pytest.raises(ValueError, match="missing"):
        panel_utts(_panel(cache, uids=["dev-clean-u0", "dev-clean-zz"]))


def test_real_panel_manifest():
    path = os.path.join(REPO, "docs", "ultra", "data", "dev_selection_panel.json")
    assert os.path.isfile(path), f"selection panel missing: {path}"
    with open(path) as fh:
        spec = json.load(fh)
    assert spec["schema"] == "ultra.dev-panel/v1"
    assert {s["split"] for s in spec["splits"]} == {"dev-clean", "dev-other"}
    rest = {k: v for k, v in spec.items() if k != "manifest_sha256"}
    assert hashlib.sha256(json.dumps(rest, indent=1, sort_keys=True).encode()).hexdigest() == \
        spec["manifest_sha256"]


def test_plain_split_path_refuses_test_splits():
    for split in ("test-clean", "test-other"):
        with pytest.raises(ValueError, match="test-\\* splits are held out"):
            check_split(split)
        # refused before any checkpoint or data is touched
        with pytest.raises(ValueError, match="held out"):
            main(["--run-dir", "/nonexistent", "--split", split])
    assert check_split("dev-other") == "dev-other"
    assert check_split("train-clean-100") == "train-clean-100"  # D0 overfit checks


def test_panel_manifest_is_recorded(cache):
    path = _panel(cache)
    utts, sha = load_panel(path, "dev-clean")
    with open(path) as fh:
        assert sha == json.load(fh)["manifest_sha256"]
    assert [u.uid for u in utts] == ["dev-clean-u0", "dev-clean-u2"]
    assert panel_record(path, "dev-clean", sha) == {
        "file": "panel.json", "split": "dev-clean", "manifest_sha256": sha}
    assert panel_record(path, None, sha)["split"] == "all"
