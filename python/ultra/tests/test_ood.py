import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra import ood


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A throwaway git repo with docs/ultra/LOOKS.jsonl and a fake 'ami' cache."""
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    (repo / "docs" / "ultra").mkdir(parents=True)
    (repo / "docs" / "ultra" / "LOOKS.jsonl").write_text("")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    pcms = [np.full(1600, 7, np.int16), np.arange(800, dtype=np.int16)]
    ood.write_caches("ami", ["a-1", "a-2"], pcms, ["SECRET ONE", "SECRET TWO"], cache=str(cache))
    monkeypatch.delenv(ood.ENV_LOOK, raising=False)
    return repo, cache


def _looks(repo):
    return [json.loads(x) for x in (repo / "docs" / "ultra" / "LOOKS.jsonl").read_text().splitlines() if x]


def test_refused_without_tag_even_with_look_id(world, monkeypatch):
    repo, cache = world
    monkeypatch.setenv(ood.ENV_LOOK, "look-1")
    with pytest.raises(ood.SealedReadRefused, match="does not exist"):
        ood.load_split("ami", cache=str(cache), repo=str(repo))
    assert _looks(repo) == []


def test_refused_without_look_id_even_with_tag(world):
    repo, cache = world
    _git(repo, "tag", ood.FREEZE_TAG)
    with pytest.raises(ood.SealedReadRefused, match=ood.ENV_LOOK):
        ood.load_split("ami", cache=str(cache), repo=str(repo))
    assert _looks(repo) == []


def test_audio_only_allowed_without_tag(world):
    repo, cache = world
    a = ood.load_audio("ami", cache=str(cache))
    assert [x.uid for x in a] == ["a-1", "a-2"]
    assert a[0].seconds == 0.1 and not hasattr(a[0], "text")
    assert _looks(repo) == []


def test_planted_fake_freeze_tag_on_unrelated_commit_refused(world, monkeypatch):
    repo, cache = world
    _git(repo, "checkout", "-q", "--orphan", "fake")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "planted")
    _git(repo, "tag", ood.FREEZE_TAG)
    _git(repo, "checkout", "-q", "main")
    monkeypatch.setenv(ood.ENV_LOOK, "planted-look")
    with pytest.raises(ood.SealedReadRefused, match="not an ancestor"):
        ood.load_split("ami", cache=str(cache), repo=str(repo))
    assert _looks(repo) == []


def test_permitted_read_logs_and_second_look_refused(world, monkeypatch):
    repo, cache = world
    _git(repo, "tag", ood.FREEZE_TAG)
    monkeypatch.setenv(ood.ENV_LOOK, "look-A")
    utts = ood.load_split("ami", cache=str(cache), repo=str(repo))
    assert [u.text for u in utts] == ["SECRET ONE", "SECRET TWO"]
    log = _looks(repo)
    assert len(log) == 1 and log[0]["look_id"] == "look-A" and log[0]["set"] == "ami"
    assert log[0]["sealed_sha256"] == ood.sha256_file(ood.sealed_path("ami", str(cache)))
    monkeypatch.setenv(ood.ENV_LOOK, "look-B")
    with pytest.raises(ood.SealedReadRefused, match="one read"):
        ood.load_split("ami", cache=str(cache), repo=str(repo))
    assert len(_looks(repo)) == 1


def test_blocked_and_unknown_sets():
    with pytest.raises(LookupError, match="BLOCKED_RIGHTS"):
        ood.load_audio("tedlium")
    with pytest.raises(KeyError):
        ood.load_audio("librispeech")


def test_write_caches_rejects_misaligned(tmp_path):
    with pytest.raises(ValueError):
        ood.write_caches("ami", ["x", "x"], [np.zeros(1, np.int16)] * 2, ["a", "b"], cache=str(tmp_path))


def test_tampered_cache_is_refused(world):
    repo, cache = world
    p = ood.audio_path("ami", str(cache))
    with open(p, "r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 0xFF]))
    ood._VERIFIED.clear()
    with pytest.raises(ood.CacheIntegrityError):
        ood.load_audio("ami", cache=str(cache))


def test_clusters_from_ids_only():
    assert ood.cluster_of("ami", "AMI_IS1009b_H03_FIO089_0023485_0026102") == "IS1009b"
    assert ood.cluster_of("voxpopuli", "20180613-0900-PLENARY-24-en_20180613-21:40:52_2") == "20180613-0900-PLENARY"
    with pytest.raises(ValueError):
        ood.cluster_of("ami", "garbage")


def _race(args):
    look, cache, repo = args
    os.environ[ood.ENV_LOOK] = look  # per-process environment
    try:
        ood.load_split("ami", cache=cache, repo=repo)
        return "ok"
    except ood.SealedReadRefused:
        return "refused"


def test_concurrent_looks_serialised(world):
    """Two processes racing with different look-ids: exactly one read is permitted."""
    import multiprocessing as mp

    repo, cache = world
    _git(repo, "tag", ood.FREEZE_TAG)
    with mp.get_context("fork").Pool(2) as pool:
        results = pool.map(_race, [(f"look-{i}", str(cache), str(repo)) for i in range(2)])
    assert sorted(results) == ["ok", "refused"]
    assert len({d["look_id"] for d in _looks(repo)}) == 1
